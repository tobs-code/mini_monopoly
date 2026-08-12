"""
Billige Vor-Checks (User-Feedback, vor der 10x-Daten-Entscheidung):

1. Layer/Hook-Breakdown der Position-Probe bei Step 550: resid_pre, attn_out,
   mlp_out, resid_post JE Layer linear proben. Wenn Position in KEINEM Hook/
   keiner Layer lesbar ist, ist die "settop-limitiert"-Diagnose abgesichert
   (nicht "falsche Analyse-Ebene").

2. Naive Current-Player-Baseline: "nimm den letzten P-Token" — trivial, ohne
   Modell. Zeigt, ob die triviale Heuristik 100% erreicht (die Modell-Decode-
   Werte 85->77% dann klar unter der sturen Zählheurstik liegen).

Nutzung:
    python probes/checks_12.py
"""

from __future__ import annotations
import os
import sys
import random

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from transformer_lens import HookedTransformer, HookedTransformerConfig
from data.tokenizer import MiniTokenizer
from data.splits import load_games

CKPT = os.path.join("runs", "run1", "ckpt_best.pt")
DATA = os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl")
HOOKS = ["resid_pre", "attn_out", "mlp_out", "resid_post"]


def knowable_mask(events):
    n = len(events)
    mask = [True] * n
    for i in range(n - 2):
        if events[i].startswith("P") and events[i][1:].isdigit() and events[i + 1] == "ROLL":
            mask[i] = mask[i + 1] = False
    return mask


def fit_linear(X, y, device, steps=400, lr=5e-3):
    X = X.float().to(device)
    y = y.long().to(device)
    n_tr = int(0.8 * X.shape[0])
    Xtr, Xte, ytr, yte = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    lin = torch.nn.Linear(X.shape[1], int(y.max().item()) + 1).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        lossf(lin(Xtr), ytr).backward()
        opt.step()
    with torch.no_grad():
        return (lin(Xte).argmax(1) == yte).float().mean().item()


def main():
    device = torch.device("cpu")
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    tokenizer = MiniTokenizer()
    cfg = HookedTransformerConfig(**dict(manifest["architecture"]))
    model = HookedTransformer(cfg).to(device)
    model.tokenizer = None
    model.load_state_dict(ckpt["model"])
    model.eval()

    all_games = load_games(DATA)
    val_ids = set(manifest["val_game_ids"])
    val_games = [g for g in all_games if g["game_id"] in val_ids]
    random.seed(0)
    probe_games = random.sample(val_games, max(1, int(len(val_games) * 0.2)))
    n_players = probe_games[0]["num_players"]
    n_ctx = model.cfg.n_ctx

    feats = {h: {L: [] for L in range(model.cfg.n_layers)} for h in HOOKS}
    labels = {h: {L: [[] for _ in range(n_players)] for L in range(model.cfg.n_layers)} for h in HOOKS}

    for start in range(0, len(probe_games), 8):
        chunk = probe_games[start:start + 8]
        B = len(chunk)
        ids_list, raw_list, states_list, masks = [], [], [], []
        for g in chunk:
            ids, raw = tokenizer.tokenize_events_aligned(g["events"])
            ids_list.append(ids); raw_list.append(raw)
            states_list.append(g["states"]); masks.append(knowable_mask(g["events"]))
        batch = torch.full((B, n_ctx), tokenizer.pad_id, dtype=torch.long)
        raw_mat = torch.full((B, n_ctx), -1, dtype=torch.long)
        for b in range(B):
            L = min(len(ids_list[b]), n_ctx)
            batch[b, :L] = torch.tensor(ids_list[b][:L], dtype=torch.long)
            raw_mat[b, :L] = torch.tensor(raw_list[b][:L], dtype=torch.long)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch, names_filter=None)
        for b in range(B):
            st = states_list[b]; mask = masks[b]
            for pos in range(n_ctx):
                ri = raw_mat[b, pos].item()
                if ri < 0 or not mask[ri]:
                    continue
                state = st[ri]
                for h in HOOKS:
                    for L in range(model.cfg.n_layers):
                        feats[h][L].append(cache[h, L][b, pos].float().cpu())
                        for p in range(n_players):
                            labels[h][L][p].append(state["position"][p])

    print(f"Step {ckpt['step']} | Probe-Spiele {len(probe_games)} | Samples {len(feats['resid_post'][0])}")
    print("Linear-Position-Probe je (Hook, Layer) — Test-Acc:")
    print(f"{'Hook':>10} | " + " | ".join(f"L{L}" for L in range(model.cfg.n_layers)))
    for h in HOOKS:
        row = []
        for L in range(model.cfg.n_layers):
            X = torch.stack(feats[h][L])
            accs = [fit_linear(X, torch.tensor(labels[h][L][p]), device) for p in range(n_players)]
            row.append(sum(accs) / len(accs))
        print(f"{h:>10} | " + " | ".join(f"{a:.1%}" for a in row))

    # 2) Naive Current-Player-Baseline: letzter P-Token
    tot = ok = 0
    for g in all_games:
        cp = 0
        for i, tok in enumerate(g["events"]):
            if i > 0 and tok.startswith("P") and tok[1:].isdigit():
                cp = int(tok[1:])
            tot += 1
            if g["states"][i]["current_player"] == cp:
                ok += 1
    print(f"\nNaive Current-Player-Baseline (letzter P-Token): {ok/tot:.1%} ({ok:,}/{tot:,})")


if __name__ == "__main__":
    main()