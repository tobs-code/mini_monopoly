"""
H1-Position-Sanity-Probe (§6 Schritt 6, kostenloser Sanity-Check zuerst).

Ziel: Verifizieren, dass das Modell die Position deterministisch aus der
Event-Historie rekonstruieren kann (position = cumsum(rolls) mod board_size).
Erst wenn ein linearer Probe hier ~99% erreicht, ist die Pipeline
(Tokenisierung/Architektur/Training) sauber — sonst zuerst die Pipeline
verdächtigen, NICHT ein "mechanistisches Phänomen" (§8-Kernprinzip).

Methode:
- Lädt einen Checkpoint (Default: runs/run1/ckpt_best.pt) + dessen Architektur
  und Val-Split (game_ids aus dem Manifest, damit NUR gehaltene Spiele — kein
  Memorisierungs-Confound).
- Führt die Val-Spiele mit run_with_cache aus; Alignment Token->State über
  tokenize_events_aligned (Miet-Event = 5 Tokens, gleicher State).
- Pro Layer ein Probe (d_model -> 12 Positionen) JE Spieler, trainiert auf 80% der
  SPIELE (per-game-Split, Trajectory-Integrity §3: ganze Spiele in Train/Test,
  kein pro-Sample-Mischen -> kein Leakage-Confound), ausgewertet auf den held-out
  Spielen (verhindert Probe-Overfitting auf spiel-id-abhängige Nebenmerkmale).
- Ausgabe: Tabelle Layer x Spieler -> Accuracy, plus Zufalls-/Majoritäts-Baseline.

Nutzung:  (Conda-Env iprop, CUDA)
    python probes/probe_position.py --ckpt runs/run1/ckpt_best.pt
"""

from __future__ import annotations
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformer_lens import HookedTransformer, HookedTransformerConfig
from data.tokenizer import MiniTokenizer
from data.splits import load_games


def build_model(manifest: dict, ckpt_state: dict, device) -> HookedTransformer:
    arch = dict(manifest["architecture"])
    cfg = HookedTransformerConfig(**arch)
    model = HookedTransformer(cfg).to(device)
    model.tokenizer = None
    model.load_state_dict(ckpt_state["model"])
    model.eval()
    return model


def knowable_mask(events: list[str]) -> list[bool]:
    """True, falls die Position an Token i aus events[:i+1] ableitbar ist.

    states[i] ist der Snapshot NACH allen Events bis inkl. i (game.py _append).
    Die Wurf-Gruppe [P, ROLL, dice] teilt EINEN post-move-Snapshot: an den
    führenden P/ROLL-Tokens ist die Würfelzahl noch nicht sichtbar, die
    Post-Move-Position dort also NICHT knowable -> diese Tokens aus der
    Positions-Probe ausschließen (sonst lernen Probes zukünftige Info).
    """
    n = len(events)
    mask = [True] * n
    for i in range(n - 2):
        if events[i].startswith("P") and events[i][1:].isdigit() \
                and events[i + 1] == "ROLL":
            mask[i] = False
            mask[i + 1] = False
    return mask


def collect_samples(model, tokenizer, games, n_ctx, device, batch_size=8):
    """Pro Layer Feature-Matrix + Labels (Position je Spieler) aller Val-Tokens.

    Zusätzlich Kontroll-Labels für Sanity-Probes:
      - token_ids: ID des aktuellen Tokens (muss aus resid_post ~100% dekodierbar sein,
        sonst ist Alignment/Probe kaputt, NICHT das Modell).
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_players = games[0]["num_players"]
    feats = {L: [] for L in range(n_layers)}
    labels = {L: [[] for _ in range(n_players)] for L in range(n_layers)}
    token_labels = {L: [] for L in range(n_layers)}
    turn_labels = {L: [] for L in range(n_layers)}
    game_sizes = []  # Sampleanzahl pro Spiel (in Verarbeitungsreihenfolge), fuer pro-Spiel-Split

    for start in range(0, len(games), batch_size):
        chunk = games[start:start + batch_size]
        B = len(chunk)
        ids_list, raw_list, states_list = [], [], []
        for g in chunk:
            ids, raw = tokenizer.tokenize_events_aligned(g["events"])
            ids_list.append(ids)
            raw_list.append(raw)
            states_list.append(g["states"])
        batch = torch.full((B, n_ctx), tokenizer.pad_id, dtype=torch.long)
        raw_mat = torch.full((B, n_ctx), -1, dtype=torch.long)
        for b in range(B):
            L = min(len(ids_list[b]), n_ctx)
            batch[b, :L] = torch.tensor(ids_list[b][:L], dtype=torch.long)
            raw_mat[b, :L] = torch.tensor(raw_list[b][:L], dtype=torch.long)
        batch = batch.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(batch)

        for b in range(B):
            st = states_list[b]
            mask = knowable_mask(chunk[b]["events"])
            for pos in range(n_ctx):
                ri = raw_mat[b, pos].item()
                if ri < 0:
                    continue
                if not mask[ri]:
                    continue
                state = st[ri]
                for L in range(n_layers):
                    feats[L].append(cache["resid_post", L][b, pos].float().cpu())
                    token_labels[L].append(batch[b, pos].item())
                    turn_labels[L].append(state["current_player"])
                    for p in range(n_players):
                        labels[L][p].append(state["position"][p])
            game_sizes.append(len(feats[0]))
        del cache
    return feats, labels, token_labels, turn_labels, n_layers, d_model, n_players, game_sizes


def fit_probe(Xtr, ytr, Xte, yte, device, steps=500, lr=5e-3, nonlinear=False):
    """Probe (d_model -> n_cls) auf bereits aufgeteiltem Train/Test (pro-Spiel-Split).
    Gibt (acc_tr, acc_te). MLP zeigt, ob das Merkmal ÜBERHAUPT (nicht-linear)
    aus dem Residual dekodierbar ist."""
    Xtr = Xtr.float().to(device)
    Xte = Xte.float().to(device)
    ytr = ytr.long().to(device)
    yte = yte.long().to(device)
    n_cls = int(ytr.max().item()) + 1
    if nonlinear:
        lin = torch.nn.Sequential(
            torch.nn.Linear(Xtr.shape[1], 128), torch.nn.ReLU(),
            torch.nn.Linear(128, n_cls),
        ).to(device)
    else:
        lin = torch.nn.Linear(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(lin(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_tr = (lin(Xtr).argmax(1) == ytr).float().mean().item()
        acc_te = (lin(Xte).argmax(1) == yte).float().mean().item()
    return acc_tr, acc_te


def parse_log_losses(ckpt_path: str) -> dict[int, tuple[float, float]]:
    """Parst train.log neben dem Checkpoint -> {step: (train_loss, val_loss)}.

    Format: Train-Zeilen enthalten 'step N/... | loss X'; VAL-Zeilen ('-> VAL loss Y')
    stehen direkt nach einer Train-Zeile -> dem letzten gesehenen Step zugeordnet
    (Val wird im Training ohnehin nur alle 50 Steps protokolliert)."""
    log = os.path.join(os.path.dirname(ckpt_path), "train.log")
    if not os.path.exists(log):
        return {}
    import re
    train_by_step: dict[int, float] = {}
    val_by_step: dict[int, float] = {}
    last_step = -1
    with open(log, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"step (\d+)/\d+ \| loss ([\d.]+)", line)
            if m:
                last_step = int(m.group(1))
                train_by_step[last_step] = float(m.group(2))
            m = re.search(r"VAL loss ([\d.]+)", line)
            if m and last_step >= 0:
                val_by_step[last_step] = float(m.group(1))
    out: dict[int, tuple[float, float]] = {}
    for step, tl in sorted(train_by_step.items()):
        vl = None
        for vs, vv in sorted(val_by_step.items()):
            if vs <= step:
                vl = vv
        out[step] = (tl, vl)
    return out


def analyze_checkpoint(ckpt_path: str, probe_games, tokenizer, device, nonlinear: bool):
    """Führt das Probesetup für EINEN Checkpoint aus. Gibt Dict mit Ergebnissen."""
    import random
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    model = build_model(manifest, ckpt, device)
    n_ctx = model.cfg.n_ctx
    feats, labels, token_labels, turn_labels, n_layers, d_model, n_players, game_sizes = collect_samples(
        model, tokenizer, probe_games, n_ctx, device)

    # Pro-Spiel-Split (Trajectory-Integrity, §3): erste 80% der SPIELE -> Train,
    # letzte 20% -> Test. game_sizes[k] = Sampleanzahl kumuliert durch Spiel k
    # (0-indiziert); trenne nach Spiel n_tr_games-1.
    n_games = len(game_sizes)
    n_tr_games = max(1, int(0.8 * n_games))
    tr_end = game_sizes[n_tr_games - 1]

    def split(y_list):
        yt = torch.tensor(y_list)
        return yt[:tr_end], yt[tr_end:]

    # Sanity 1 (Token-ID) muss ~100% sein
    sanity = []
    for L in range(n_layers):
        X = torch.stack(feats[L])
        ytr, yte = split(token_labels[L])
        sanity.append(fit_probe(X[:tr_end], ytr, X[tr_end:], yte, device)[1])
    # Sanity 2 (Current-Player)
    turn = []
    for L in range(n_layers):
        X = torch.stack(feats[L])
        ytr, yte = split(turn_labels[L])
        turn.append(fit_probe(X[:tr_end], ytr, X[tr_end:], yte, device)[1])
    # Position je Layer (Train- und Test-Acc, Layer-Mean über Spieler)
    pos_te = []
    pos_tr = []
    for L in range(n_layers):
        X = torch.stack(feats[L])
        row_te, row_tr = [], []
        for p in range(n_players):
            ytr, yte = split(labels[L][p])
            a_tr, a_te = fit_probe(X[:tr_end], ytr, X[tr_end:], yte,
                                   device, nonlinear=nonlinear)
            row_te.append(a_te)
            row_tr.append(a_tr)
        pos_te.append(sum(row_te) / len(row_te))
        pos_tr.append(sum(row_tr) / len(row_tr))
    best_layer = max(range(n_layers), key=lambda L: pos_te[L])
    return {
        "step": ckpt["step"],
        "best_val_loss": ckpt["best_val_loss"],
        "best_pos_test": pos_te[best_layer],
        "best_pos_train": pos_tr[best_layer],
        "pos_per_layer": pos_te,
        "pos_train_per_layer": pos_tr,
        "turn_test": max(turn),
        "sanity_token": max(sanity),
        "n_layers": n_layers,
        "n_games": n_games,
        "n_tr_games": n_tr_games,
    }


def make_plot(rows: list[dict], losses: dict[dict], out_png: str, n_layers: int) -> None:
    """Plottet Position-Probe-Accuracy gegen Train- und Val-Loss pro Checkpoint.

    Rows werden nach Step aufsteigend sortiert, damit die Kurven monoton verlaufen
    (unabhängig von der --ckpt-Eingabereihenfolge; ckpt_best.pt liegt i.d.R. mitten
    im Intervall und wäre sonst ein Rückläufer im Plot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=lambda r: r["step"])
    steps = [r["step"] for r in rows]
    pos = [r["best_pos_test"] for r in rows]
    turn = [r["turn_test"] for r in rows]
    tr = [losses[s][0] if s in losses else float("nan") for s in steps]
    va = [losses[s][1] if s in losses else float("nan") for s in steps]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, pos, "o-", label="Position-Probe (best layer, test)")
    ax1.plot(steps, turn, "s--", label="Current-Player-Probe (test)")
    ax1.axhline(1 / 12, color="gray", linestyle=":", label="Zufall (1/12)")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Probe-Accuracy")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(steps, tr, "x-", color="tab:red", label="Train-Loss")
    ax2.plot(steps, va, "x--", color="tab:purple", label="Val-Loss")
    ax2.set_ylabel("CE-Loss")
    ax2.legend(loc="upper right")

    ax1.set_title("State-Tracking über Checkpoints: Probe-Acc vs. Loss")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"Plot gespeichert: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=[os.path.join("runs", "run1", "ckpt_best.pt"),
                             os.path.join("runs", "run1", "ckpt_step1000.pt"),
                             os.path.join("runs", "run1", "ckpt_step1250.pt"),
                             os.path.join("runs", "run1", "ckpt_step1500.pt")])
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-frac-probe", type=float, default=0.2, help="Anteil Val-Spiele, die probe-seitig (nicht fürs Modell) genutzt werden")
    ap.add_argument("--nonlinear", action="store_true", help="MLP-Probe statt linear")
    ap.add_argument("--plot", default=os.path.join("runs", "run1", "probe_checkpoints.png"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    # Shared Probe-Spiele aus dem Manifest des ERSTEN Checkpoints (Val-IDs sind
    # über Checkpoints desselben Runs identisch); seed-fix für Vergleichbarkeit.
    first = torch.load(args.ckpt[0], map_location=device, weights_only=False)
    manifest = first["manifest"]
    tokenizer = MiniTokenizer()
    val_ids = set(manifest["val_game_ids"])
    all_games = load_games(args.data)
    val_games = [g for g in all_games if g["game_id"] in val_ids]
    import random
    random.seed(0)
    probe_games = random.sample(val_games, max(1, int(len(val_games) * args.val_frac_probe)))
    print(f"Probe-Konfiguration: {len(val_games)} Val-Spiele, davon {len(probe_games)} fürs Proben "
          f"(seed 0); Checkpoints: {len(args.ckpt)}")

    losses = parse_log_losses(args.ckpt[0])
    rows = []
    for path in args.ckpt:
        r = analyze_checkpoint(path, probe_games, tokenizer, device, args.nonlinear)
        rows.append(r)
        tl, vl = losses.get(r["step"], (float("nan"), float("nan")))
        print(f"\n--- Checkpoint step {r['step']} (best_val {r['best_val_loss']:.4f}, "
              f"train {tl:.4f}, val {vl:.4f}) ---")
        print(f"  pro-Spiel-Split: {r['n_tr_games']}/{r['n_games']} Spiele (train/test)")
        print(f"  Position-Probe test je Layer: {[f'{a:.1%}' for a in r['pos_per_layer']]}")
        print(f"  Position-Probe train je Layer: {[f'{a:.1%}' for a in r['pos_train_per_layer']]}")
        print(f"  best Position test: {r['best_pos_test']:.1%} (train {r['best_pos_train']:.1%}) | "
              f"Current-Player max: {r['turn_test']:.1%} | Sanity-Token max: {r['sanity_token']:.1%}")

    print(f"\n{'='*70}")
    print(f"ZUSAMMENFASSUNG (linear={not args.nonlinear}, per-Game-Split, shared {len(probe_games)} Spiele):")
    print(f"{'Step':>6} | {'train':>7} | {'val':>7} | {'Pos te':>7} | {'Pos tr':>7} | {'Player':>6} | {'Tok':>6}")
    print("-" * 70)
    for r in rows:
        tl, vl = losses.get(r["step"], (float("nan"), float("nan")))
        tls = f"{tl:.3f}" if tl == tl else "  -  "
        vls = f"{vl:.3f}" if vl == vl else "  -  "
        print(f"{r['step']:>6} | {tls:>7} | {vls:>7} | {r['best_pos_test']:>6.1%} | "
              f"{r['best_pos_train']:>6.1%} | {r['turn_test']:>5.1%} | {r['sanity_token']:>5.1%}")
    make_plot(rows, losses, args.plot, rows[0]["n_layers"])


if __name__ == "__main__":
    main()
