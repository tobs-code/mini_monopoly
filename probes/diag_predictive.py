"""
Diagnose B: Woher kommt der Loss-Vorsprung des Modells gegenüber Bigramm?

Frage (User-Planung): Ist die gute Val-Loss (0.56 vs Bigramm 0.97) auf
State-Tracking zurückzuführen, oder auf reine Grammatik-Level-Muster
("nach BUY kommt fast immer ROLL"), die OHNE Position/Cash/Ownership
vorhersagbar sind?

Methode: Argmax-Next-Token-Genauigkeit des Modells auf Val-Spielen,
aufgeschlüsselt nach echtem nächsten Token-TYP, verglichen mit einer
Bigramm-Baseline (empirische bedingte Verteilung, Trainingshälfte).
Kern-Trigger: die Event-Entscheidung direkt NACH dem Würfel ('D1..D6') —
dort entscheidet das gelandete Feld, ob BUY/PASS/PAY_RENT kommt. Wenn das
Modell diese Wahl deutlich über der Randverteilung trifft, NUTZT es
Positions-/Ownership-Info; wenn nur auf Randniveau, ist der Loss vor allem
Würfel-Grammatik.

Nutzung:
    python probes/diag_predictive.py [--ckpt runs/run1/ckpt_best.pt]
"""

from __future__ import annotations
import argparse
import os
import sys
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformer_lens import HookedTransformer, HookedTransformerConfig
from data.tokenizer import MiniTokenizer
from data.splits import load_games

_EVENT = {"START", "ROLL", "BUY", "PASS", "PAY_RENT"}


def build_model(manifest: dict, ckpt_state: dict, device) -> HookedTransformer:
    cfg = HookedTransformerConfig(**dict(manifest["architecture"]))
    model = HookedTransformer(cfg).to(device)
    model.tokenizer = None
    model.load_state_dict(ckpt_state["model"])
    model.eval()
    return model


def token_type(tok_ids: list[int], tokenizer: MiniTokenizer) -> list[str]:
    """Klassifiziert jede Token-ID in einen Typ (P/ROLL/Dice/BUY/PASS/PAY_RENT/digit/...)."""
    itos = tokenizer.itos
    types = []
    for t in tok_ids:
        s = itos[t]
        if s.startswith("P") and s[1:].isdigit():
            types.append("P")
        elif s in _EVENT:
            types.append(s)
        elif s.startswith("D") and s[1:].isdigit():
            types.append("Dice")
        else:
            types.append("digit")  # MONEY/TH/H/T/O-Stelle
    return types


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "run1", "ckpt_best.pt"))
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    tokenizer = MiniTokenizer()
    model = build_model(manifest, ckpt, device)

    all_games = load_games(args.data)
    val_ids = set(manifest["val_game_ids"])
    val_games = [g for g in all_games if g["game_id"] in val_ids]
    train_games = [g for g in all_games if g["game_id"] not in val_ids]

    n_ctx = model.cfg.n_ctx
    seqs = []
    for g in val_games:
        seqs.append(tokenizer.tokenize_events(g["events"]))

    # Modell-Argmax-Predictions auf Val
    all_ids, all_types = [], []
    for s in seqs:
        all_ids.extend(s)
        all_types.extend(token_type(s, tokenizer))
    N = len(all_ids)

    model_hits = Counter()          # (typ_korr, typ_next) -> hits
    model_tot = Counter()
    bigram_hits = Counter()
    bigram_tot = Counter()

    # Bigramm-Modell aus Trainingshälfte der VAL-Spiele aufbauen? Nein: Bigramm
    # braucht Trainingsdaten. Wir nutzen die Train-Spiele (held-out fürs Bigramm).
    tr_ids = []
    for g in train_games:
        tr_ids.extend(tokenizer.tokenize_events(g["events"]))
    tr_pairs = Counter(zip(tr_ids, tr_ids[1:]))
    tr_first = Counter(tr_ids)
    pred_bigram = {}
    for a in set(tr_ids):
        best = None
        for (x, b), c in tr_pairs.items():
            if x == a and (best is None or c >= best[1]):
                best = (b, c)
        pred_bigram[a] = best[0] if best else a

    # Batch-Inferenz fürs Modell
    BATCH = 8
    model_pred = []
    for start in range(0, len(seqs), BATCH):
        chunk = seqs[start:start + BATCH]
        B = len(chunk)
        batch = torch.full((B, n_ctx), tokenizer.pad_id, dtype=torch.long)
        for b in range(B):
            L = min(len(chunk[b]), n_ctx)
            batch[b, :L] = torch.tensor(chunk[b][:L], dtype=torch.long)
        batch = batch.to(device)
        with torch.no_grad():
            logits = model(batch)
        preds = logits.argmax(-1).cpu().tolist()
        for b in range(B):
            L = min(len(chunk[b]), n_ctx)
            model_pred.extend(preds[b][:L])

    # Nur Paare (i, i+1), wo beide real vorkommen (ohne pad)
    for i in range(N - 1):
        cur, nxt = all_ids[i], all_ids[i + 1]
        model_tot[(all_types[i], all_types[i + 1])] += 1
        if model_pred[i] == nxt:
            model_hits[(all_types[i], all_types[i + 1])] += 1
        bigram_tot[(all_types[i], all_types[i + 1])] += 1
        if pred_bigram[cur] == nxt:
            bigram_hits[(all_types[i], all_types[i + 1])] += 1

    print(f"Modell: step {ckpt['step']}, best_val {ckpt['best_val_loss']:.4f}"
          f" | Val-Tokens: {N:,}")
    print(f"\nNext-Token-Argmax-Genauigkeit je Kontext->nächster Typ (Modell vs Bigramm, n=n):")
    types_order = ["P", "ROLL", "Dice", "BUY", "PASS", "PAY_RENT"]
    seen = Counter()
    print(f"{'ctx':>7}->{'nxt':>8} | {'Modell':>6} | {'Bigramm':>7} | {'n':>6}")
    print("-" * 50)
    for (c, n), tot in model_tot.items():
        if tot < 50 or (c, n) in seen:
            continue
        seen[(c, n)] = 1
        mh = model_hits[(c, n)] / tot if tot else 0
        bh = bigram_hits[(c, n)] / tot if tot else 0
        print(f"{c:>7}->{n:>8} | {mh:>5.1%} | {bh:>6.1%} | {tot:>6,}")

    # Der entscheidende Block: Event-Entscheidung nach DICE
    print("\n=== Event-Entscheidung NACH dem Würfel (BUY/PASS/PAY_RENT/nichts/lokal?) ===")
    print("Dice->nächster Typ (fällt mit dem Event-Typ zusammen):")
    tot_all = sum(v for (c, n), v in model_tot.items() if c == "Dice")
    mh_all = sum(model_hits[(c, n)] for (c, n) in model_tot if c == "Dice")
    bh_all = sum(bigram_hits[(c, n)] for (c, n) in bigram_tot if c == "Dice")
    print(f"  Modell gesamt: {mh_all/tot_all:6.1%}  Bigramm gesamt: {bh_all/tot_all:6.1%}  (n={tot_all:,})")
    for (c, n), tot in sorted(model_tot.items()):
        if c != "Dice":
            continue
        print(f"  {'':>6}-> {n:>8}: Modell {model_hits[(c,n)]/tot:5.1%} | "
              f"Bigramm {bigram_hits[(c,n)]/tot:5.1%} | n={tot:,}")


if __name__ == "__main__":
    main()