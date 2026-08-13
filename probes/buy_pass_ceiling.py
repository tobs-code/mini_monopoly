"""
BUY_PASS-Policy-Ceiling-Check (User-Punkt 1 & 2 vor 10x-Entscheidung).

Frage 1: Ist die 69.6%-Clean-Accuracy von BUY_PASS durch die Kauf-Policy-
         Zufälligkeit gedeckelt (theoretische Obergrenze < 100%), oder ein
         Lern-/Coverage-Gap?
These: Die KLASSE 'BUY_PASS' ist deterministisch aus dem Board-Zustand bestimmt
       ("unbesetztes Preisfeld"), weil BUY und PASS in der Klasse verschmolzen
       sind (logsumexp). Policy-Zufall wirkt nur auf TOKEN-Ebene (BUY vs PASS),
       nicht auf Klassen-Ebene -> Obergrenze = 100%. Wir zeigen das empirisch:
       (a) BUY-vs-PASS-Verteilung nach Policy (random ~ 50/50, always_buy ~ 100% BUY),
       (b) BUY_PASS-Klassen-Accuracy nach Policy (muss auch bei deterministischer
           always_buy-Policy unter 100% bleiben, wenn's ein Lern-Gap ist).

Frage 2: Ist die Do-over-Nicht-Flächigkeit über Würfelwerte (1,6 niedrig, 3,4,5
         hoch) bedeutsam? Wir binen die Do-over-Accuracy nach der Trainings-
         Haeufigkeit des kontrafaktisch gelandeten Feldes.

Nutzung:
    python probes/buy_pass_ceiling.py --ckpt runs/run2_wd0.5_s1200/ckpt_best.pt \
        [--max-decisions N]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformer_lens import HookedTransformer, HookedTransformerConfig
from data.tokenizer import MiniTokenizer
from data.splits import load_games

CLS_ORDER = ["P", "BUY_PASS", "PAY_RENT"]


def build_model(manifest, ckpt_state, device):
    cfg = HookedTransformerConfig(**dict(manifest["architecture"]))
    model = HookedTransformer(cfg).to(device)
    model.tokenizer = None
    model.load_state_dict(ckpt_state["model"])
    model.eval()
    return model


def counter_field(true_roll, query_roll, player, state, board):
    """Kontrafaktisches Landefeld, wenn `player` mit query_roll gewuerfelt haette."""
    post = state["position"][player]
    pre = (post - true_roll) % board
    return (pre + query_roll) % board


def class_of_field(field, player, prices, owner):
    if prices[field] > 0 and owner[field] is None:
        return "BUY_PASS"
    if owner[field] is not None and owner[field] != player:
        return "PAY_RENT"
    return "P"


def collect_decisions(games):
    """Erweitert causal_roll-Decision um policy, next-token, kontrafaktische Felder."""
    dec = []
    for g in games:
        events = g["events"]
        prices = g["prices"]
        board = g["board_size"]
        policy = g["policy_name"]
        ids, raw = tokenizer.tokenize_events_aligned(events)
        n = len(events)
        for j in range(3, n):
            if not (events[j].startswith("P") and events[j][1:].isdigit()):
                continue
            if not (events[j - 2] == "ROLL" and events[j - 1].isdigit()):
                continue
            roller = int(events[j - 3][1:])
            pos = raw.index(j)
            if pos + 1 >= len(ids):
                continue
            roll = int(events[j - 1])
            nxt = tokenizer.itos[ids[pos + 1]]
            if nxt in ("BUY", "PASS"):
                true_cls = "BUY_PASS"
            elif nxt == "PAY_RENT":
                true_cls = "PAY_RENT"
            elif nxt == "ROLL":
                true_cls = "P"
            else:
                continue
            state = g["states"][j - 1]
            cf = [counter_field(roll, d2, roller, state, board) for d2 in range(1, 7)]
            dec.append({
                "seq": ids[:pos + 1], "pos": pos, "roller": roller, "roll": roll,
                "true_cls": true_cls, "next_tok": nxt, "policy": policy,
                "board": board, "prices": prices,
                "owner": state["owner"], "cf_fields": cf,
            })
    return dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "run2_wd0.5_s1200", "ckpt_best.pt"))
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--max-decisions", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    global tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    tokenizer = MiniTokenizer()
    model = build_model(manifest, ckpt, device)

    all_games = load_games(args.data)
    val_ids = set(manifest["val_game_ids"])
    val_games = [g for g in all_games if g["game_id"] in val_ids]

    decs = collect_decisions(val_games)
    if args.max_decisions:
        decs = decs[:args.max_decisions]
    print(f"Entscheidungen: {len(decs):,} (val)")

    # ---- (a) BUY vs PASS Token-Ebene nach Policy (bei BUY_PASS-Entscheidungen) ----
    print("\n=== (a) BUY-vs-PASS (Token-Ebene) bei BUY_PASS-Entscheidungen, nach Policy ===")
    bp_by_policy = {}
    for d in decs:
        if d["true_cls"] != "BUY_PASS":
            continue
        k = d["policy"]
        c = bp_by_policy.setdefault(k, Counter())
        c[d["next_tok"]] += 1
    for pol, c in bp_by_policy.items():
        tot = sum(c.values())
        print(f"  {pol:12s}: BUY {c.get('BUY',0)/tot:5.1%}  PASS {c.get('PASS',0)/tot:5.1%}  (n={tot})")

    # ---- Forwards: clean (1x) + do-over (6x) ----
    def make_batch(seqs, pos_list, replace=None):
        B = len(seqs); L = max(len(s) for s in seqs)
        batch = torch.full((B, L), tokenizer.pad_id, dtype=torch.long)
        for b, s in enumerate(seqs):
            batch[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            if replace is not None:
                batch[b, pos_list[b] - 1] = replace[b]
        return batch.to(device)

    def gather(logits, pos_list):
        B = logits.shape[0]
        idx = torch.arange(B)
        l = logits[idx, torch.tensor(pos_list, device=logits.device)]
        st = tokenizer.stoi
        p = l[:, st["ROLL"]]
        bp = l[:, [st["BUY"], st["PASS"]]].logsumexp(1)
        r = l[:, st["PAY_RENT"]]
        return torch.softmax(torch.stack([p, bp, r], dim=1), dim=1)

    B = args.batch_size
    clean = {c: {p: [0, 0] for p in ("always_buy", "conservative", "random")} for c in CLS_ORDER}
    clean_all = {c: [0, 0] for c in CLS_ORDER}
    dover = {d2: [0, 0] for d2 in range(1, 7)}
    dover_field = {d2: {} for d2 in range(1, 7)}   # d' -> (landed_field -> [hit,total])

    for start in range(0, len(decs), B):
        chunk = decs[start:start + B]
        seqs = [d["seq"] for d in chunk]
        pos_c = [d["pos"] for d in chunk]
        with torch.no_grad():
            inp = make_batch(seqs, pos_c)
            logits = model(inp, return_type="logits")
            pc = gather(logits, pos_c)
            for b, d in enumerate(chunk):
                tc = d["true_cls"]
                pol = d["policy"]
                clean_all[tc][1] += 1
                clean[tc][pol][1] += 1
                if pc[b].argmax().item() == CLS_ORDER.index(tc):
                    clean_all[tc][0] += 1
                    clean[tc][pol][0] += 1
            for d2 in range(1, 7):
                repl = [tokenizer.stoi[f"D{d2}"] for _ in chunk]
                inpm = make_batch(seqs, pos_c, replace=repl)
                lm = model(inpm, return_type="logits")
                pm = gather(lm, pos_c)
                for b, d in enumerate(chunk):
                    ccl = class_of_field(d["cf_fields"][d2 - 1], d["roller"], d["prices"], d["owner"])
                    dover[d2][1] += 1
                    landed = d["cf_fields"][d2 - 1]
                    df = dover_field[d2].setdefault(landed, [0, 0])
                    df[1] += 1
                    if pm[b].argmax().item() == CLS_ORDER.index(ccl):
                        dover[d2][0] += 1
                        df[0] += 1
        if (start // B) % 20 == 0:
            print(f"  ...{min(start+B, len(decs)):,}/{len(decs):,}")

    print("\n=== (b) Clean-Klassen-Accuracy nach Policy ===")
    for c in CLS_ORDER:
        parts = "  ".join(f"{p}:{clean[c][p][0]}/{clean[c][p][1]}"
                          for p in ("always_buy", "conservative", "random"))
        print(f"  {c:9s}: {parts}")
    print("\n=== (c) Clean gesamt (alle Policies) ===")
    for c in CLS_ORDER:
        h, t = clean_all[c]
        print(f"  {c:9s}: {h/max(t,1):6.1%}  ({h:,}/{t:,})")

    print("\n=== (d) Do-over je Wuerfelwert (gesamt) ===")
    for d2 in range(1, 7):
        h, t = dover[d2]
        print(f"  d'={d2}: {h/max(t,1):6.1%}  ({h:,}/{t:,})")

    print("\n=== (e) Do-over-Accuracy nach kontrafaktischem Landefeld (und Trans.-Coverage) ===")
    all_fields = sorted({f for d2 in dover_field for f in dover_field[d2]})
    hdr = "  f".join("" for _ in [0])
    print("  Landefeld:    " + "  ".join(f"{f:>3d}" for f in all_fields))
    first_board = val_games[0]["prices"]
    print("  Preis:        " + "  ".join(f"{first_board[f]:>4d}" for f in all_fields))
    for d2 in range(1, 7):
        row = []
        for f in all_fields:
            if f in dover_field[d2]:
                h, t = dover_field[d2][f]
                row.append(f"{h/max(t,1)*100:3.0f}")
            else:
                row.append("  -")
        print(f"  d'={d2}: " + "  ".join(row))


if __name__ == "__main__":
    main()
