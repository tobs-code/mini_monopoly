"""
H2-Affordability-Probe (Vortest b, §6b) — läuft auf run2 (trainiert auf cap=1000,
99.5% affordable) gegen den separaten Low-Cap-Diagnose-Datensatz (cap=400), der
echte Unaffordability enthält (28.1%).

Was hier getestet wird (S1-held-out für Affordability):
  `can_afford = cash >= price` an Kauf-Entscheidungspunkten (BUY/PASS), mit
  S1-held-out-Paaren. Ziel ist die Komposition: Einzelwerte (cash, price) sind
  im Train sichtbar, aber die spezifischen (cash, price)-Kombinationen sind held
  out. Wenn der Probe auf den held-out Paaren generalisiert (> Majority), ist
  die Komposition `cash >= price` aus linear zugänglichen Teilen gelernt.

Pre-registriertes S1-Design (2x2):
  held-out cash-Werte {100, 300} x held-out price-Werte {150, 200}
  = (100,150) unaff grenznah (diff=-50), (100,200) unaff,
    (300,150) afford, (300,200) afford
  -> im Test balanciert 2 afford + 2 unafford (Komposition ohne Memo-Shortcut).
  Konfund-ok (alle 3 Policies gut durchmischt); gewaehlt via Design-Search
  (tools/s1_design_search.py, docs/06 7d-2) als best-effort S1-Test unter
  struktureller Simulator-Kopplung (Einzelwert-Coverage niedrig, min ~178).
  Single Source of Truth: data/s1_design.py (identisch zu data/merge_corpora.py).

Metriken:
  - Acc auf den held-out-Paar-Punkten (nur Testregion) — der S1-Kern-Test.
  - Majority-Baseline INNERHALB der held-out-Paare (immer häufigste Klasse).
  - Per-Paar-Accuracy (zeigt, ob die Komposition je Paar stimmt).
  - Zum Kontext: Acc auf dem vollen Test (alle Paare) + Train.
  - Sanity: Token-ID, Current-Player.

Methodik:
  - Linearer Probe (je Layer) + optional MLP (--nonlinear).
  - pro-Spiel-Split (Trajectory-Integrity §3): Partien, die ein held-out-Paar
    enthalten, wandern komplett in den Test; die übrigen in Train.
  - Probe wird NUR auf Train-Paaren trainiert, NUR auf held-out-Paaren evaluiert.

Hintergrund (siehe EXPERIMENT.md §5.7 / §6): run2 wurde auf cap=1000 trainiert
und hat kaum Unaffordability gesehen (99.5% affordable). Ein POSITIVES Ergebnis
hier (emergente Komposition ohne Trainingsdruck) wäre eindeutig; ein negatives
ist zweideutig (nie gelernt vs. OOD-Artefakt) und billig auszuschließen. Das ist
der Vortest vor einem möglichen Zusatztraining auf Daten MIT Affordability-
Varianz (§6b Option a).

Nutzung:  (Conda-Env iprop, CUDA)
    python probes/probe_afford.py --ckpt runs/run2_wd0.5_s1200/ckpt_best.pt \
        --data data/games_so9001_n800_always_buy-conservative-random_sc400.jsonl
"""

from __future__ import annotations
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.tokenizer import MiniTokenizer
from data.splits import load_games, decision_points
from data.s1_design import HELD_OUT_PAIRS, CASH_VALUES, PRICE_VALUES
from probes.probe_position import build_model, fit_probe


def collect_decision_feats(model, tokenizer, games, n_ctx, device, batch_size=8):
    """Pro Layer Feature-Matrix + Labels NUR an den Kauf-Entscheidungspunkten.

    Liefert pro Spiel die Entscheidungspunkte (cash, price, afford) mit ihrem
    Aktivierungsvektor (resid_post) an der BUY/PASS-Token-Position, für jeden
    Layer separat. Zusätzlich Sanity-Labels (token_id, current_player) und
    triviale Konfund-Features (policy, player, position, price, token_idx,
    decision) für die Baseline-Kontrolle (siehe §6b/Check 2).

    game_sizes ist KUMULATIV (Sampleanzahl durch Spiel k).

    Rückgabe: feats{L}[i], meta[i]={cash,price,afford,policy,player,position,
    decision,token_idx}, token_ids{L}[i], turn{L}[i], game_sizes, n_layers,
    d_model, points(L ist agnostisch).
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    feats = {L: [] for L in range(n_layers)}
    token_ids = {L: [] for L in range(n_layers)}
    turn = {L: [] for L in range(n_layers)}
    # Je Entscheidungspunkt: (cash, price, afford) + triviale Konfund-Features.
    meta = []  # list of dict per decision point, layer-agnostic
    game_sizes = []  # kumulativ

    # Pre-tokenize + pre-extract decision points je Spiel, sortiert nach token_idx
    prep = []
    for gi, g in enumerate(games):
        ids, raw = tokenizer.tokenize_events_aligned(g["events"])
        pts = decision_points(g)
        policy = g.get("policy_name", "?")
        # Merke Output-Token-Position der BUY/PASS-Events (raw == token_idx)
        # BUY/PASS sind atomar -> genau ein Output-Token mit raw == token_idx.
        dec_pos = {}
        for tpos, ridx in enumerate(raw):
            if ridx in dec_pos:
                pass
            else:
                dec_pos[ridx] = tpos
        prep.append({"ids": ids, "raw": raw, "pts": pts, "dec_pos": dec_pos,
                     "events": g["events"], "policy": policy})

    for start in range(0, len(games), batch_size):
        chunk = prep[start:start + batch_size]
        B = len(chunk)
        batch = torch.full((B, n_ctx), tokenizer.pad_id, dtype=torch.long)
        for b, item in enumerate(chunk):
            L = min(len(item["ids"]), n_ctx)
            batch[b, :L] = torch.tensor(item["ids"][:L], dtype=torch.long)
        batch = batch.to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(batch)
        for b, item in enumerate(chunk):
            st = item["pts"]
            # Reihenfolge nach token_idx sortieren (entscheidend für konsistentes game_sizes)
            st = sorted(st, key=lambda p: p["token_idx"])
            for p in st:
                ridx = p["token_idx"]
                tpos = item["dec_pos"].get(ridx)
                if tpos is None or tpos >= n_ctx:
                    continue
                state = None  # Wir brauchen current_player: aus decision point 'player'
                for L in range(n_layers):
                    feats[L].append(cache["resid_post", L][b, tpos].float().cpu())
                    token_ids[L].append(batch[b, tpos].item())
                    turn[L].append(p["player"])
                meta.append({
                    "cash": p["cash"], "price": p["price"],
                    "afford": int(p["cash"] >= p["price"]),
                    "policy": item["policy"],
                    "player": p["player"],
                    "position": p["position"],
                    "decision": p["decision"],
                    "token_idx": p["token_idx"],
                })
            game_sizes.append(len(feats[0]))
        del cache
    return feats, token_ids, turn, meta, game_sizes, n_layers, d_model


def split_s1_indices(meta, game_sizes):
    """S1-Split: Partien MIT einem held-out-Paar -> Test, sonst Train.

    Gibt (tr_mask, te_mask) als Bool-Listen über die Sample-Reihenfolge zurück,
    sowie die Indizes der Test-Samples, die zu einem held-out-Paar gehören
    (die eigentliche S1-Testregion).

    game_sizes ist KUMULATIV: Sampleanzahl durch Spiel k. Ein Spiel k liegt im
    Test, falls eines seiner Samples (Index-Bereich) ein held-out-Paar hat.
    """
    n_games = len(game_sizes)
    n = len(meta)
    # Sample -> Spiel-Index
    game_of = []
    prev = 0
    g_idx = 0
    for s in range(n):
        while g_idx < n_games and game_sizes[g_idx] <= s:
            g_idx += 1
        game_of.append(g_idx)
    # Spiel ist Test, wenn es ein held-out-Paar-Sample hat
    game_is_test = [False] * n_games
    for s, m in enumerate(meta):
        if (m["cash"], m["price"]) in HELD_OUT_PAIRS:
            game_is_test[game_of[s]] = True
    tr_mask = [not game_is_test[game_of[s]] for s in range(n)]
    te_mask = [game_is_test[game_of[s]] for s in range(n)]
    held_te_idx = [s for s in range(n)
                   if game_is_test[game_of[s]] and (meta[s]["cash"], meta[s]["price"]) in HELD_OUT_PAIRS]
    return tr_mask, te_mask, held_te_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=[os.path.join("runs", "run2_wd0.5_s1200", "ckpt_best.pt")])
    ap.add_argument("--data", default="data/games_so9001_n800_always_buy-conservative-random_sc400.jsonl")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--nonlinear", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

    tokenizer = MiniTokenizer()
    games = load_games(args.data)
    print(f"Low-Cap-Korpus: {len(games)} Spiele | {args.data}")

    for path in args.ckpt:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        manifest = ckpt["manifest"]
        model = build_model(manifest, ckpt, device)
        n_ctx = model.cfg.n_ctx
        feats, token_ids, turn, meta, game_sizes, n_layers, d_model = collect_decision_feats(
            model, tokenizer, games, n_ctx, device)

        tr_mask, te_mask, held_te_idx = split_s1_indices(meta, game_sizes)
        n_tr = sum(tr_mask)
        n_te = sum(te_mask)
        n_held = len(held_te_idx)
        print(f"\n--- Checkpoint {os.path.basename(path)} (step {ckpt['step']}) ---")
        print(f"S1-Design: cash{CASH_VALUES} x price{PRICE_VALUES} | held-out {len(HELD_OUT_PAIRS)} Paare")
        print(f"Entscheidungspunkte gesamt: {len(meta)} | Train {n_tr} | Test {n_te} | held-out-test {n_held}")

        # Majority-Baseline in der held-out-Testregion (S1-Kern)
        held_meta = [meta[i] for i in held_te_idx]
        held_labels = torch.tensor([m["afford"] for m in held_meta])
        maj_held = (held_labels == held_labels.mode().values).float().mean().item()
        n_aff = int((held_labels == 1).sum())
        print(f"S1 held-out Testregion: {n_held} Pts, afford={n_aff}/{n_held}, "
              f"Majority-Baseline={maj_held:.1%}")

        # --- CHECK 1: pro-Paar Majority-Baseline + Konfund-Schieflage ---
        print("\n[Check 1] Pro-Paar-Majority + Konfund-Schieflage (Policy/Player/Position):")
        from collections import Counter
        for hp in sorted(HELD_OUT_PAIRS):
            idxs = [i for i in held_te_idx if (meta[i]["cash"], meta[i]["price"]) == hp]
            if not idxs:
                print(f"  {hp}: KEINE Samples in Testregion")
                continue
            labs = [meta[i]["afford"] for i in idxs]
            n_unaff = sum(1 for l in labs if l == 0)
            pair_maj = max(n_unaff, len(idxs) - n_unaff) / len(idxs)
            policies = Counter(meta[i]["policy"] for i in idxs)
            players = Counter(meta[i]["player"] for i in idxs)
            positions = Counter(meta[i]["position"] for i in idxs)
            print(f"  {hp} afford={hp[0]>=hp[1]} | n={len(idxs)} unaff={n_unaff} "
                  f"| pro-Paar-Majority={pair_maj:.1%}")
            print(f"      policies={dict(policies)} | players={dict(players)} | "
                  f"top-positions={dict(positions.most_common(3))}")

        # --- CHECK 2: Trivial-Feature-Kontrollproben (Konfund 'low-cap dataset smell') ---
        print("\n[Check 2] Trivial-Feature-Kontrollproben (gleicher S1-Split, eval auf held-out):")
        print("  Ziel: erreichen triviale, direkt verfuegbare Features schon ~74%, ist das"
              "Modell-Signal weniger wert.")
        # Feature-Encoding als kategoriale One-Hot/Embedding-freie Spalten je Merkmal.
        import torch.nn.functional as F
        def onehot(vals, n):
            t = torch.tensor(vals, dtype=torch.long)
            return F.one_hot(t, num_classes=n).float()
        def catenc(key, vs):
            """Kategoriales String-Feature -> int-Encoding (deterministisch sortiert)."""
            vals = sorted({m[key] for m in vs}) or [None]
            return onehot([vals.index(m[key]) for m in vs], len(vals))
        tr_i = [i for i in range(len(meta)) if tr_mask[i]]
        n_price = max(m["price"] for m in meta) + 1
        n_cash = max(m["cash"] for m in meta) + 1
        # triviale Merkmal-Matrizen (train + held-out-test)
        feat_sets = {
            "price": lambda vs: onehot([m["price"] for m in vs], n_price),
            "position": lambda vs: onehot([m["position"] for m in vs], 12),
            "player": lambda vs: onehot([m["player"] for m in vs], 3),
            "policy": lambda vs: catenc("policy", vs),
            "decision": lambda vs: catenc("decision", vs),
            "price+position": lambda vs: torch.cat([onehot([m["price"] for m in vs], n_price),
                                                    onehot([m["position"] for m in vs], 12)], 1),
            "price+decision": lambda vs: torch.cat([onehot([m["price"] for m in vs], n_price),
                                                    catenc("decision", vs)], 1),
            "price+cash": lambda vs: torch.cat([onehot([m["price"] for m in vs], n_price),
                                                onehot([m["cash"] for m in vs], n_cash)], 1),
        }
        tr_meta = [meta[i] for i in tr_i]
        ytr_all = torch.tensor([meta[i]["afford"] for i in tr_i])
        for name, fn in feat_sets.items():
            Xtr_t = fn(tr_meta).to(device)
            Xte_t = fn(held_meta).to(device)
            acc_tr, acc_te = fit_probe(Xtr_t, ytr_all, Xte_t, held_labels, device,
                                       nonlinear=args.nonlinear)
            print(f"    {name:>16}: TrAcc {acc_tr:.1%} | HeldAcc {acc_te:.1%}")

        # Beste Residual-Probe als Referenz (best Layer über held-out)
        held_accs_ref = []
        for L in range(n_layers):
            X = torch.stack(feats[L])
            Xtr = X[tr_i]
            ytr = torch.tensor([meta[i]["afford"] for i in tr_i])
            _, a = fit_probe(Xtr, ytr, X[held_te_idx], held_labels, device,
                             nonlinear=args.nonlinear)
            held_accs_ref.append(a)
        print(f"    Referenz (resid, best Layer): {max(held_accs_ref):.1%}")
        print()

        # Pro Layer: Probe auf Train-Paaren, eval auf held-out-Paaren
        print(f"{'Layer':>5} | {'TrAcc':>7} | {'HeldAcc':>8} | {'Tok':>5} | {'Player':>6}")
        held_accs = []
        for L in range(n_layers):
            X = torch.stack(feats[L])
            Xtr = X[[i for i in range(len(meta)) if tr_mask[i]]]
            ytr = torch.tensor([meta[i]["afford"] for i in range(len(meta)) if tr_mask[i]])
            Xte_held = X[held_te_idx]
            yte_held = held_labels
            acc_tr, acc_te_held = fit_probe(Xtr, ytr, Xte_held, yte_held, device, nonlinear=args.nonlinear)
            held_accs.append(acc_te_held)

            # Sanity token/player (auf demselben Split, nur Train/Test-Region)
            tok_tr = torch.tensor([token_ids[L][i] for i in range(len(meta)) if tr_mask[i]])
            tok_te = torch.tensor([token_ids[L][i] for i in held_te_idx])
            turn_tr = torch.tensor([turn[L][i] for i in range(len(meta)) if tr_mask[i]])
            turn_te = torch.tensor([turn[L][i] for i in held_te_idx])
            _, tok_acc = fit_probe(Xtr, tok_tr, Xte_held, tok_te, device)
            _, turn_acc = fit_probe(Xtr, turn_tr, Xte_held, turn_te, device)

            print(f"{L:>5} | {acc_tr:>7.1%} | {acc_te_held:>8.1%} | {tok_acc:>5.1%} | {turn_acc:>6.1%}")

        # Per-Paar-Breakdown auf der held-out-Testregion (best Layer = argmax HeldAcc)
        best_L = int(torch.tensor(held_accs).argmax())
        Xbest = torch.stack(feats[best_L])
        Xtr = Xbest[[i for i in range(len(meta)) if tr_mask[i]]]
        ytr = torch.tensor([meta[i]["afford"] for i in range(len(meta)) if tr_mask[i]])
        print(f"\nPer-held-out-Paar (linear, best Layer L={best_L}, HeldAcc={held_accs[best_L]:.1%}):")
        for hp in sorted(HELD_OUT_PAIRS):
            idxs = [i for i in held_te_idx if (meta[i]["cash"], meta[i]["price"]) == hp]
            if not idxs:
                continue
            Xte_p = Xbest[idxs]
            yte_p = torch.tensor([meta[i]["afford"] for i in idxs])
            acc_tr, acc_te = fit_probe(Xtr, ytr, Xte_p, yte_p, device, nonlinear=args.nonlinear)
            n_unaff = int((yte_p == 0).sum())
            print(f"  (cash={hp[0]:>3},price={hp[1]:>3}) afford={hp[0]>=hp[1]} | "
                  f"test n={len(idxs)} unaff={n_unaff} | HeldAcc={acc_te:.1%} (train {acc_tr:.1%})")


if __name__ == "__main__":
    main()
