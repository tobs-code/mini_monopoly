"""
GPT-Training für Mini-Monopoly (History-only, S0-Smoke-Test).

Trainiert einen kleinen HookedTransformer (TransformerLens) als
nächsten-Token-Vorhersager auf der tokenisierten Event-Historie
(§6 Schritt 5). Das Modell sieht NUR den Event-Token-Stream; Position,
Cash und Ownership sind nur implizit aus den Events erschließbar (§5.3).

Merkmale:
- Checkpoint-Resume: speichert Modell + Optimizer + Schrittnummer + Bestwerte
  **und die RNG-Zustände** (torch + Python) regelmäßig. Resume vom selben
  Checkpoint ist deterministisch und reproduziert den Verlauf bitweise
  (verifiziert auf GPU: gleicher Checkpoint -> identische Folge-Losses).
  Hinweis: exakte Bit-Gleichheit mit einem *nie unterbrochenen* Lauf ist auf
  GPU wegen inherenten FP-Nichtdeterminismus zwischen frischen Prozessen nicht
  garantiert (mit Dropout=0, dem Default, ist das Training ansonsten
  deterministisch); auf CPU ist volle Bit-Gleichheit erreichbar. Wissenschaftlich
  relevant ist die Reproduzierbarkeit von Daten, Split und Config — die ist
  garantiert (game_ids im Manifest, S0-Split aus data.splits).
- S0-Split kommt aus `data.splits.split_s0` (Single Source of Truth, §8), nicht
  aus einer zweiten Implementierung in train.py; die beteiligten game_ids werden
  im Manifest festgehalten.
- Reshuffling der Trainingssequenzen zwischen den Epochen (keine künstliche
  Korrelation über Batch-Kompositionen).
- Eigenständiges Best-Checkpoint ("ckpt_best.pt") zusätzlich zu den
  periodischen Intervall-Checkpoints.
- FP16-Mixed-Precision auf CUDA, FP32 auf CPU (automatisch, Skalierer nur bei
  CUDA).
- Config-Manifest (Kernprinzip §8): alle Hyperparameter + Daten + Split-Seed
  + Trainings-/Val-game_ids in <run_dir>/manifest.json.

Nutzung (im Conda-Env `iprop`, das TransformerLens + CUDA hat):
    python training/train.py --run-dir runs/run1
    python training/train.py --run-dir runs/run1 --resume   # fortsetzen
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformer_lens import HookedTransformer, HookedTransformerConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.tokenizer import MiniTokenizer
from data.splits import load_games, split_s0


# ----------------------------------------------------------------------
# Architektur (§5.2) + Tokenizer (§5.6)
# ----------------------------------------------------------------------
def default_config() -> dict:
    return {
        "n_layers": 4,
        "d_model": 256,
        "d_head": 64,          # d_model // n_heads
        "n_heads": 4,
        "d_mlp": 1024,         # 4 * d_model
        "n_ctx": 288,          # >= max. tokenisierte Länge 284 (§5.2)
        "act_fn": "gelu",
        "attention_dir": "causal",
        "use_attn_result": True,
        "use_split_qkv_input": True,
    }


def override_config(base: dict, args) -> dict:
    """Erlaubt Architektur-Änderungen über die CLI (Kapazitäts-Test, §7).

    Wird NUR beim frischen Lauf angewandt (beim Resume bestimmt die Config
    im Checkpoint-Manifest, sonst bricht Resume-Nutzung). Consistenz-Helfer:
    d_head und d_mlp werden aus d_model/n_heads abgeleitet, wenn nicht
    explizit gesetzt.
    """
    arch = dict(base)
    # Analyse-only-Flags abschalten: use_attn_result / use_split_qkv_input werden
    # NUR von den Probes (run_with_cache) benötigt, sind aber reine
    # Cache-Einstellungen und ändern NICHT die trainierten Gewichte. Beim Training
    # blähen sie den Aktivierungs-Cache massiv auf (bei 8L/384/batch32 -> mehrere
    # GB RAM, siehe h2_8l384_mixed-Fehlstart). Die Probes laden die Architektur
    # aus dem Manifest und greifen ohnehin nur auf "resid_post" zu -> Flags dort
    # nicht nötig. Rückwärtskompatibel: alte Manifests enthalten die Flags=True
    # und bleiben unverändert nutzbar.
    arch["use_attn_result"] = False
    arch["use_split_qkv_input"] = False
    if args.n_layers is not None:
        arch["n_layers"] = args.n_layers
    if args.d_model is not None:
        arch["d_model"] = args.d_model
    if args.n_heads is not None:
        arch["n_heads"] = args.n_heads
    if args.d_head is not None:
        arch["d_head"] = args.d_head
    else:
        arch["d_head"] = arch["d_model"] // arch["n_heads"]
    if args.d_mlp is not None:
        arch["d_mlp"] = args.d_mlp
    else:
        arch["d_mlp"] = 4 * arch["d_model"]
    return arch


# ----------------------------------------------------------------------
# Daten -> gepaddete Batches
# ----------------------------------------------------------------------
def tokenize_games(games, tokenizer: MiniTokenizer) -> list[list[int]]:
    """Game-Dicts -> Token-ID-Sequenzen (volle Länge, ungepaddet)."""
    return [tokenizer.tokenize_events(g["events"]) for g in games]


def make_batch(seqs: list[list[int]], n_ctx: int, pad_id: int, device) -> torch.Tensor:
    """Sequenzen auf n_ctx auffüllen -> (B, n_ctx) LongTensor."""
    B = len(seqs)
    batch = torch.full((B, n_ctx), pad_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        L = min(len(seq), n_ctx)
        batch[i, :L] = torch.tensor(seq[:L], dtype=torch.long)
    return batch.to(device)


def make_batches(seqs, batch_size, n_ctx, pad_id, device):
    """Gleichmäßige Batches aus der Sequenzliste erzeugen."""
    return [
        make_batch(seqs[i:i + batch_size], n_ctx, pad_id, device)
        for i in range(0, len(seqs), batch_size)
    ]


# ----------------------------------------------------------------------
# Loss mit Padding-Maske (predict-token-t+1, ignore PAD)
# ----------------------------------------------------------------------
def compute_loss(model, batch, pad_id):
    """Cross-Entropy auf allen Nicht-PAD-Zielpositionen. Shift um eins."""
    logits = model(batch, return_type="logits")       # (B, T, V)
    V = logits.shape[-1]
    targets = batch[:, 1:]                            # (B, T-1)
    logits = logits[:, :-1]                           # (B, T-1, V)
    return F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), ignore_index=pad_id
    )


# ----------------------------------------------------------------------
# Logging / Checkpointing
# ----------------------------------------------------------------------
def log_line(run_dir, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(run_dir, "train.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _snapshot(model, optimizer, scaler, step, epoch, best_val, manifest,
              train_seqs, val_seqs, schedule_total):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val,
        "manifest": manifest,
        "schedule_total": schedule_total,
        # RNG-Stände für bitweise Reproduzierbarkeit beim Resume
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "py_rng": random.getstate(),
        "train_seq_ids": train_seqs,   # aktuelle (ggf. reshufflete) Reihenfolge
        "val_seq_ids": val_seqs,
    }


def save_ckpt(path, run_dir, **kw):
    torch.save(_snapshot(**kw), path)
    log_line(run_dir, f"Checkpoint gespeichert: {os.path.basename(path)} (step {kw['step']})")


def load_ckpt(path, device):
    # weights_only=False: eigene, kontrollierte Checkpoints (manifest, RNG).
    return torch.load(path, map_location=device, weights_only=False)


def prune_old_ckpts(run_dir, keep):
    """Löscht überschüssige periodische Checkpoints (Rotation).

    WICHTIG: numerisch nach Schrittnummer sortieren (nicht lexikalisch —
    'ckpt_step10' < 'ckpt_step5' lexikalisch wäre falsch).
    """
    if keep <= 0:
        return

    def step_num(f):
        return int(f.removeprefix("ckpt_step").removesuffix(".pt"))

    periodic = sorted(
        (f for f in os.listdir(run_dir)
         if f.startswith("ckpt_step") and f.endswith(".pt")),
        key=step_num,
    )
    excess = periodic[:-keep] if keep < len(periodic) else []
    for f in excess:
        try:
            os.remove(os.path.join(run_dir, f))
        except OSError:
            pass


def restore_rng(ckpt, device):
    """Setzt die RNG-Zustände aus dem Checkpoint zurück (bitweise Resume)."""
    # torch_rng wurde per map_location=device evtl. nach CUDA verschoben;
    # set_rng_state erwartet aber einen CPU-ByteTensor.
    torch.set_rng_state(ckpt["torch_rng"].to("cpu"))
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(
                [t.to("cuda") for t in ckpt["cuda_rng"]]
            )
        except Exception as e:  # CUDA-RNG nicht immer 1:1 wiederherstellbar
            # Bei Dropout=0 (Default) irrelevant; relevant erst wenn Dropout aktiv.
            print(f"HINWEIS: CUDA-RNG nicht wiederhergestellt (kein Effekt bei "
                  f"Dropout=0): {e}", flush=True)
    random.setstate(ckpt["py_rng"])


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Mini-Monopoly GPT-Training")
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--run-dir", default="runs/smoke")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--keep-ckpts", type=int, default=0,
                    help="Nur die letzten N periodischen ckpt_step*.pt behalten "
                         "(0 = alle behalten). ckpt_best.pt und der finale "
                         "Checkpoint werden nie gelöscht. Schützt run_dir vor "
                         "Plattenfüllung bei langen Läufen.")
    ap.add_argument("--schedule-total", type=int, default=None,
                    help="Gesamtlänge der LR-Cosine-Kurve (Default: max_steps). "
                         "Beim Resume konstant halten, um LR-Sprünge zu vermeiden.")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-layers", type=int, default=None,
                    help="Architektur-Override (Kapazitäts-Test): Anzahl Layer")
    ap.add_argument("--d-model", type=int, default=None,
                    help="Architektur-Override: d_model")
    ap.add_argument("--n-heads", type=int, default=None,
                    help="Architektur-Override: Anzahl Heads")
    ap.add_argument("--d-head", type=int, default=None,
                    help="Architektur-Override: d_head (Default: d_model // n_heads)")
    ap.add_argument("--d-mlp", type=int, default=None,
                    help="Architektur-Override: d_mlp (Default: 4 * d_model)")
    args = ap.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = device.type == "cuda"
    log_line(args.run_dir, f"Device: {device} | Mixed-Precision: {amp}")

    tokenizer = MiniTokenizer()
    arch = default_config()
    arch["d_vocab"] = tokenizer.d_vocab

    # ------------------------- Daten / Resume -------------------------
    resume = args.resume and os.path.isdir(args.run_dir)
    if resume:
        ckpt_files = sorted(f for f in os.listdir(args.run_dir)
                            if f.startswith("ckpt_") and f.endswith(".pt")
                            and f != "ckpt_best.pt")
        if not ckpt_files:
            sys.exit("--resume, aber kein Checkpoint in run-dir gefunden.")
        last = os.path.join(args.run_dir, ckpt_files[-1])
        ckpt = load_ckpt(last, device)
        manifest = ckpt["manifest"]
        start_step = ckpt["step"]
        epoch = ckpt["epoch"]
        best_val = ckpt["best_val_loss"]
        train_seqs = ckpt["train_seq_ids"]
        val_seqs = ckpt["val_seq_ids"]
        schedule_total = args.schedule_total or ckpt["schedule_total"]
        # Beim Resume zählt NUR die im Checkpoint fixierte Architektur.
        arch = manifest["architecture"]
        log_line(args.run_dir,
                 f"Resume von {os.path.basename(last)} (step {start_step}, "
                 f"epoch {epoch}, best_val {best_val:.4f}, schedule_total {schedule_total})")
    else:
        arch = override_config(arch, args)
        games = load_games(args.data)
        train_games, val_games = split_s0(games, args.val_frac, args.seed)
        train_seqs = tokenize_games(train_games, tokenizer)
        val_seqs = tokenize_games(val_games, tokenizer)
        manifest = {
            "data": os.path.basename(args.data),
            "val_frac": args.val_frac,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "warmup_steps": args.warmup_steps,
            "architecture": arch,
            "vocab_size": tokenizer.d_vocab,
            "n_train_games": len(train_games),
            "n_val_games": len(val_games),
            "train_game_ids": sorted(g["game_id"] for g in train_games),
            "val_game_ids": sorted(g["game_id"] for g in val_games),
        }
        schedule_total = args.schedule_total or args.max_steps
        start_step, epoch, best_val = 0, 0, float("inf")
        log_line(args.run_dir, f"Daten: {len(games)} Spiele -> train {len(train_games)}, "
                               f"val {len(val_games)} (S0-Split via data.splits.split_s0, "
                               f"seed {args.seed})")

    # ------------------------- Modell / Optimizer -------------------------
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = HookedTransformerConfig(**arch)
    model = HookedTransformer(cfg).to(device)
    model.tokenizer = None
    if resume:
        model.load_state_dict(ckpt["model"])

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if amp else None
    if resume:
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        # WICHTIG: RNG erst NACH der Modellerstellung (die selbst RNG konsumiert)
        # wiederherstellen, sonst verschiebt sich der Zustand gegenüber dem
        # ununterbrochenen Lauf.
        restore_rng(ckpt, device)
        log_line(args.run_dir,
                 "RNG-Zustand aus Checkpoint wiederhergestellt "
                 "(torch+cuda best-effort / python)")

    n_params = sum(p.numel() for p in model.parameters())
    log_line(args.run_dir, f"Modell-Parameter: {n_params:,} | d_vocab={model.cfg.d_vocab} "
                           f"n_ctx={model.cfg.n_ctx}")

    val_batches = make_batches(val_seqs, args.batch_size, arch["n_ctx"], tokenizer.pad_id, device)

    def cosine_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, schedule_total - args.warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    def evaluate() -> float:
        model.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for b in val_batches:
                tot += compute_loss(model, b, tokenizer.pad_id).item()
                cnt += 1
        model.train()
        return tot / max(1, cnt)

    train_batches = make_batches(train_seqs, args.batch_size, arch["n_ctx"], tokenizer.pad_id, device)

    # ------------------------- Trainings-Loop -------------------------
    log_line(args.run_dir, "=== Training startet ===")
    global_step = start_step
    t0 = time.time()
    tokens_per_step = args.batch_size * (arch["n_ctx"] - 1)

    def train_step(batch):
        nonlocal global_step
        for g in optimizer.param_groups:
            g["lr"] = cosine_lr(global_step)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda", enabled=amp):
                loss = compute_loss(model, batch, tokenizer.pad_id)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = compute_loss(model, batch, tokenizer.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        global_step += 1
        return loss

    while global_step < args.max_steps:
        model.train()
        for batch in train_batches:
            if global_step >= args.max_steps:
                break
            loss = train_step(batch)
            now = time.time()
            dt = now - t0
            lr_now = optimizer.param_groups[0]["lr"]
            if global_step % args.log_every == 0:
                tok_s = (tokens_per_step * global_step) / max(1e-9, dt)
                s_per_step = dt / max(1, global_step)
                eta = s_per_step * (args.max_steps - global_step) / 60.0
                log_line(args.run_dir,
                         f"step {global_step}/{args.max_steps} | loss {loss.item():.4f} | "
                         f"lr {lr_now:.2e} | {tok_s:,.0f} tok/s | {s_per_step:.2f}s/step | "
                         f"ETA {eta:.1f}min")
            if global_step % args.val_every == 0:
                v = evaluate()
                improved = v < best_val
                if improved:
                    best_val = v
                    save_ckpt(os.path.join(args.run_dir, "ckpt_best.pt"), args.run_dir,
                              model=model, optimizer=optimizer, scaler=scaler,
                              step=global_step, epoch=epoch, best_val=best_val,
                              manifest=manifest, train_seqs=train_seqs, val_seqs=val_seqs,
                              schedule_total=schedule_total)
                log_line(args.run_dir, f"  -> VAL loss {v:.4f} (best {best_val:.4f})"
                         + ("  [BEST gespeichert]" if improved else ""))
            if global_step % args.ckpt_every == 0:
                save_ckpt(os.path.join(args.run_dir, f"ckpt_step{global_step}.pt"), args.run_dir,
                          model=model, optimizer=optimizer, scaler=scaler,
                          step=global_step, epoch=epoch, best_val=best_val,
                          manifest=manifest, train_seqs=train_seqs, val_seqs=val_seqs,
                          schedule_total=schedule_total)
                prune_old_ckpts(args.run_dir, args.keep_ckpts)
        # Epoch-Ende: Trainingssequenzen neu mischen (kein Batch-Compositions-
        # Overfitting), danach Batchliste neu aufbauen.
        random.shuffle(train_seqs)
        train_batches = make_batches(train_seqs, args.batch_size, arch["n_ctx"],
                                     tokenizer.pad_id, device)
        epoch += 1
        log_line(args.run_dir, f"Epoch {epoch} abgeschlossen (train reshuffled)")

    # finaler Checkpoint + Manifest
    save_ckpt(os.path.join(args.run_dir, f"ckpt_step{global_step}.pt"), args.run_dir,
              model=model, optimizer=optimizer, scaler=scaler,
              step=global_step, epoch=epoch, best_val=best_val,
              manifest=manifest, train_seqs=train_seqs, val_seqs=val_seqs,
              schedule_total=schedule_total)
    prune_old_ckpts(args.run_dir, args.keep_ckpts)
    manifest["final_step"] = global_step
    manifest["final_val_loss"] = best_val
    manifest["schedule_total"] = schedule_total
    with open(os.path.join(args.run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log_line(args.run_dir, f"=== Training beendet (step {global_step}, best_val {best_val:.4f}) ===")


if __name__ == "__main__":
    main()
