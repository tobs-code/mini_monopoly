"""
H1-Cash/Ownership-Proben (§5.7 Fortsetzung) — von Anfang an mit korrektem
pro-Spiel-Split (Trajectory-Integrity §3), direkt anschließend an den
Positions-Befund (linear train 42% / test 18%, MLP 100% / 19%, "Bags of
Heuristics").

Warum diese Proben vor der 10x-Daten-Frage: Die 10x-Entscheidung ist nicht
belastbar, solange nur EIN Zustands-Befund (Position) vorliegt. Wenn Cash und
Ownership dasselbe Muster zeigen (verhaltenswirksam, aber nicht als
game-invariante, dekodierbare Variable), ist das ein stärkeres Argument gegen
reine Skalierung als Position allein — Position könnte sonst ein Sonderfall
sein (Board-Zyklizität, wenig Trainingsdruck). Erst wenn H1 (Position+Cash+
Ownership) komplett durchgetestet ist, wird der 10x-Strang neu bewertet.

Methodik-Lektion explizit befolgt: Der frühere Positions-Probe splittete
per-Sample über Spiele gemischt (Leakage -> Train 100%/Test 18%, falsch
interpretiert). Hier wird ausschließlich der korrekte pro-Spiel-Split
verwendet: game_sizes ist KUMULATIV (Sampleanzahl durch Spiel k), und der
Train/Test-Schnitt liegt bei tr_end = game_sizes[n_tr_games-1]. KEIN
Re-Akkumulieren (das war der frühere Bug -> tr_end 76947 >> total).

Knowability-Unterschied zu Position: Cash und Ownership sind an ALLEN
Token-Positionen aus der bisherigen Historie eindeutig bestimmt (kein
Würfel-Bezug wie bei Position) -> mask = all True. states[i] ist der
Snapshot nach Events bis inkl. i; cash/owner dort sind stets knowable.

Metriken:
- Cash: Regressions-Probe (linear + MLP), Metrik Pearson R je Layer/Spieler,
  gemittelt über Spieler. Ein game-invarianter Tracker sollte R -> ~1 geben;
  "Bags of Heuristics" ergibt niedriges R auf held-out Spielen.
- Ownership: Klassifikations-Probe je Preisfeld (4 Klassen: None/0/1/2),
  Metrik Accuracy je Layer, gemittelt über Preisfelder. Zusätzlich Accuracy
  nur auf Fällen, in denen das Feld BESETZT ist (Isoliert "wer besitzt es"
  von "ist es überhaupt besetzt"); Majority-Baseline (Anteil None) als Kontext.

Nutzung:  (Conda-Env iprop, CUDA)
    python probes/probe_state.py --ckpt runs/run2_wd0.5_s1200/ckpt_best.pt
"""

from __future__ import annotations
import argparse
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.tokenizer import MiniTokenizer
from data.splits import load_games
from probes.probe_position import build_model


def prop_fields(prices: list[int]) -> list[int]:
    """Indizes der Preisfelder (price > 0). Feld 0 ist immer preislos (Start)."""
    return [i for i, p in enumerate(prices) if p > 0]


def collect_state(model, tokenizer, games, n_ctx, device, batch_size=8):
    """Pro Layer Feature-Matrix + Labels (Cash je Spieler, Owner je Preisfeld).

    Zusätzlich Kontroll-Labels für Sanity-Probes:
      - token_ids: ID des aktuellen Tokens (muss ~100% dekodierbar sein).
      - current_player: muss als naives State-Tracking erkennbar sein.
    Cash/Ownership sind an ALLEN Tokens knowable -> kein knowable_mask-Filter
    nötig (anders als Position, siehe Modul-Docstring).

    Rückgabe: feats, cash_labels, own_labels, token_labels, turn_labels,
    n_layers, d_model, n_players, prop_fields, game_sizes (KUMULATIV).
    """
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    n_players = games[0]["num_players"]
    pfields = prop_fields(games[0]["prices"])
    n_prop = len(pfields)
    feats = {L: [] for L in range(n_layers)}
    cash_labels = {L: [[] for _ in range(n_players)] for L in range(n_layers)}
    own_labels = {L: [[] for _ in range(n_prop)] for L in range(n_layers)}
    token_labels = {L: [] for L in range(n_layers)}
    turn_labels = {L: [] for L in range(n_layers)}
    progress = []  # pro Token: Anzahl gesehener ROLL-Events (= "wie spät im Spiel"), layer-unabhängig
    game_sizes = []  # KUMULATIVE Sampleanzahl durch Spiel k (0-indiziert)

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
            ev = chunk[b]["events"]
            for pos in range(n_ctx):
                ri = raw_mat[b, pos].item()
                if ri < 0:
                    continue
                state = st[ri]
                progress.append(sum(1 for e in ev[:ri + 1] if e == "ROLL"))
                for L in range(n_layers):
                    feats[L].append(cache["resid_post", L][b, pos].float().cpu())
                    token_labels[L].append(batch[b, pos].item())
                    turn_labels[L].append(state["current_player"])
                    for p in range(n_players):
                        cash_labels[L][p].append(state["cash"][p])
                    for j, f in enumerate(pfields):
                        o = state["owner"][f]
                        own_labels[L][j].append(0 if o is None else o + 1)
            game_sizes.append(len(feats[0]))  # kumulativ: Sampleanzahl durch Spiel b
        del cache
    return (feats, cash_labels, own_labels, token_labels, turn_labels, progress,
            n_layers, d_model, n_players, pfields, game_sizes)


def fit_class(Xtr, ytr, Xte, yte, device, steps=500, lr=5e-3, nonlinear=False):
    """Klassifikations-Probe (d_model -> n_cls) auf bereits aufgeteiltem Train/Test."""
    Xtr = Xtr.float().to(device)
    Xte = Xte.float().to(device)
    ytr = ytr.long().to(device)
    yte = yte.long().to(device)
    n_cls = int(ytr.max().item()) + 1
    if nonlinear:
        lin = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                            nn.Linear(128, n_cls)).to(device)
    else:
        lin = nn.Linear(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(lin(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_tr = (lin(Xtr).argmax(1) == ytr).float().mean().item()
        pred_te = lin(Xte).argmax(1)
        acc_te = (pred_te == yte).float().mean().item()
        owned_mask = yte != 0  # Feld besetzt (owner != None)
        if owned_mask.any():
            acc_owned = (pred_te[owned_mask] == yte[owned_mask]).float().mean().item()
        else:
            acc_owned = float("nan")
    return acc_tr, acc_te, acc_owned


def fit_regression(Xtr, ytr, Xte, yte, device, steps=500, lr=5e-3, nonlinear=False):
    """Regressions-Probe (d_model -> 1), Metrik Pearson R auf Train/Test.

    Target wird z-standardsiert auf den TRAIN-Mittelwert (R ist affine-invariant,
    Standardisierung stabilisiert nur das Training).
    """
    Xtr = Xtr.float().to(device)
    Xte = Xte.float().to(device)
    ytr = ytr.float().to(device).view(-1, 1)
    yte = yte.float().to(device).view(-1, 1)
    ymean = ytr.mean()
    ystd = ytr.std() + 1e-6
    ytr_s = (ytr - ymean) / ystd
    yte_s = (yte - ymean) / ystd
    if nonlinear:
        model = nn.Sequential(nn.Linear(Xtr.shape[1], 128), nn.ReLU(),
                              nn.Linear(128, 1)).to(device)
    else:
        model = nn.Linear(Xtr.shape[1], 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr_s)
        loss.backward()
        opt.step()

    def pearson(pred, tgt):
        pc = pred - pred.mean()
        tc = tgt - tgt.mean()
        d = pc.norm() * tc.norm()
        if d == 0:
            return float("nan")
        return (pc * tc).sum() / d

    with torch.no_grad():
        r_tr = pearson(model(Xtr), ytr_s)
        r_te = pearson(model(Xte), yte_s)
    return r_tr.item(), r_te.item()


def _pearson(pred, tgt):
    pc = pred - pred.mean()
    tc = tgt - tgt.mean()
    d = pc.norm() * tc.norm()
    return float("nan") if d == 0 else (pc * tc).sum() / d


def _linproj_coef(A, p):
    """Fit p -> A auf einer Stichprobe (lstsq), liefert Koeffizienten+Intercept (dims x (1+d))."""
    p1 = torch.cat([p, torch.ones_like(p)], dim=1)  # n x 2
    return torch.linalg.lstsq(p1, A).solution  # 2 x d


def _apply_proj(A, p, W):
    p1 = torch.cat([p, torch.ones_like(p)], dim=1)
    return A - p1 @ W


def cash_confound(Xtr, cash_tr, prog_tr, Xte, cash_te, prog_te, device):
    """Kontrolliert den Trivial-Konfund 'Cash driftet mit Spielfortschritt'.

    Liefert (R_base, R_model, R_partial):
      - R_base:   Progress(ROLL-Zähler) -> Cash, linear (was ein reines
                  'wie spät im Spiel'-Modell ohne Cash-Tracking erreichen würde).
      - R_model:  Residual -> Cash, linear (der rohe Probe).
      - R_partial: Residual -> Cash, wobei Residual UND Cash gegen Progress
                  orthogonalisiert sind (alle Koeffizienten nur auf Train gefittet).
    Der Signal-Anteil ÜBER die Turn-Drift-Baseline hinaus ist R_partial.
    """
    Xtr = Xtr.float().to(device)
    Xte = Xte.float().to(device)
    ctr = cash_tr.float().to(device).view(-1, 1)
    cte = cash_te.float().to(device).view(-1, 1)
    ptr = prog_tr.float().to(device).view(-1, 1)
    pte = prog_te.float().to(device).view(-1, 1)

    def linpred(A, b, Ar):
        # Fit A -> b (mit Intercept) auf Train, vorhersagen auf Ar
        A1 = torch.cat([A, torch.ones(A.shape[0], 1, device=A.device)], dim=1)
        Ar1 = torch.cat([Ar, torch.ones(Ar.shape[0], 1, device=Ar.device)], dim=1)
        W = torch.linalg.lstsq(A1, b).solution
        return Ar1 @ W

    with torch.no_grad():
        # (a) Trivial-Baseline: Progress -> Cash
        R_base = _pearson(linpred(ptr, ctr, pte), cte)
        # (b) roher Probe: Residual -> Cash
        R_model = _pearson(linpred(Xtr, ctr, Xte), cte)
        # (c) partiell: beides gegen Progress orthogonalisiert (Train-Koeffizienten)
        Wc = _linproj_coef(ctr, ptr)   # ptr -> ctr  (2 x 1)
        Wx = _linproj_coef(Xtr, ptr)   # ptr -> Xtr  (2 x d)
        ctr_r = ctr - torch.cat([ptr, torch.ones_like(ptr)], dim=1) @ Wc
        cte_r = cte - torch.cat([pte, torch.ones_like(pte)], dim=1) @ Wc
        Xtr_r = Xtr - torch.cat([ptr, torch.ones_like(ptr)], dim=1) @ Wx
        Xte_r = Xte - torch.cat([pte, torch.ones_like(pte)], dim=1) @ Wx
        R_partial = _pearson(linpred(Xtr_r, ctr_r, Xte_r), cte_r)
    return float(R_base), float(R_model), float(R_partial)


def parse_log_losses(ckpt_path: str) -> dict[int, tuple[float, float]]:
    """Parst train.log neben dem Checkpoint -> {step: (train_loss, val_loss)}."""
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    model = build_model(manifest, ckpt, device)
    n_ctx = model.cfg.n_ctx
    (feats, cash_labels, own_labels, token_labels, turn_labels, progress,
     n_layers, d_model, n_players, pfields, game_sizes) = collect_state(
        model, tokenizer, probe_games, n_ctx, device)

    # Korrekter pro-Spiel-Split (Trajectory-Integrity §3): erste 80% der SPIELE
    # -> Train, letzte 20% -> Test. game_sizes ist KUMULATIV; trenne nach Spiel
    # n_tr_games-1. NICHT re-akkumulieren.
    n_games = len(game_sizes)
    n_tr_games = max(1, int(0.8 * n_games))
    tr_end = game_sizes[n_tr_games - 1]

    def split(y_list):
        yt = torch.tensor(y_list)
        return yt[:tr_end], yt[tr_end:]

    # Sanity: Token-ID muss ~100%, Current-Player als naives Tracking erkennbar
    sanity = [fit_class(torch.stack(feats[L])[:tr_end], split(token_labels[L])[0],
                        torch.stack(feats[L])[tr_end:], split(token_labels[L])[1],
                        device)[1]
              for L in range(n_layers)]
    turn = [fit_class(torch.stack(feats[L])[:tr_end], split(turn_labels[L])[0],
                      torch.stack(feats[L])[tr_end:], split(turn_labels[L])[1],
                      device)[1]
            for L in range(n_layers)]

    # Cash: Pearson R je Layer (Mittel über Spieler), Train + Test
    cash_te = []
    cash_tr = []
    for L in range(n_layers):
        X = torch.stack(feats[L])
        Xtr, Xte = X[:tr_end], X[tr_end:]
        rs_tr, rs_te = [], []
        for p in range(n_players):
            ytr, yte = split(cash_labels[L][p])
            r_tr, r_te = fit_regression(Xtr, ytr, Xte, yte, device, nonlinear=nonlinear)
            rs_tr.append(r_tr)
            rs_te.append(r_te)
        cash_tr.append(sum(rs_tr) / len(rs_tr))
        cash_te.append(sum(rs_te) / len(rs_te))

    # Cash-Konfund-Kontrolle (Turn-Drift): R_base / R_model / R_partial je Layer.
    # Nur linear sinnvoll interpretierbar (partielle Korrelation); bei nonlinear
    # wird die Analyse trotzdem mit linearem Readout gerechnet (R_model unten
    # ist dann der lineare, nicht der MLP-Wert).
    prog = torch.tensor(progress, dtype=torch.float)
    cash_base = []
    cash_model_lin = []
    cash_partial = []
    for L in range(n_layers):
        X = torch.stack(feats[L])
        Xtr, Xte = X[:tr_end], X[tr_end:]
        rb, rm, rp = [], [], []
        for p in range(n_players):
            ytr, yte = split(cash_labels[L][p])
            b, m, part = cash_confound(Xtr, ytr, prog[:tr_end], Xte, yte, prog[tr_end:], device)
            rb.append(b)
            rm.append(m)
            rp.append(part)
        cash_base.append(sum(rb) / len(rb))
        cash_model_lin.append(sum(rm) / len(rm))
        cash_partial.append(sum(rp) / len(rp))

    # Ownership: Accuracy je Layer (Mittel über Preisfelder) + Acc nur auf
    # besetzten Feldern + Majority-Baseline (Anteil None über alle Felder)
    own_te = []
    own_owned_te = []
    own_tr = []
    for L in range(n_layers):
        X = torch.stack(feats[L])
        Xtr, Xte = X[:tr_end], X[tr_end:]
        row_te, row_owned_te, row_tr = [], [], []
        for j in range(len(pfields)):
            ytr, yte = split(own_labels[L][j])
            acc_tr, acc_te, acc_owned = fit_class(Xtr, ytr, Xte, yte, device, nonlinear=nonlinear)
            row_te.append(acc_te)
            row_owned_te.append(acc_owned)
            row_tr.append(acc_tr)
        own_te.append(sum(row_te) / len(row_te))
        own_owned_te.append(sum(x for x in row_owned_te if x == x) / max(1, sum(x == x for x in row_owned_te)))
        own_tr.append(sum(row_tr) / len(row_tr))

    # Majority-Baseline Ownership (Anteil None über alle Preisfelder, Test)
    yte_all = torch.tensor([v for j in range(len(pfields)) for v in split(own_labels[0][j])[1]])
    own_maj = (yte_all == 0).float().mean().item()
    # Owner-Majority-Baseline UNTER den besetzten Feldern (Test): häufigste
    # Owner-Klasse -> würde ein "rate immer häufigsten Besitzer"-Modell erreichen.
    owned_yte = yte_all[yte_all != 0]
    if owned_yte.numel() == 0:
        owner_maj = float("nan")
    else:
        owner_maj = (owned_yte == owned_yte.mode().values).float().mean().item()

    best_layer_cash = max(range(n_layers), key=lambda L: cash_te[L])
    best_layer_own = max(range(n_layers), key=lambda L: own_te[L])
    return {
        "step": ckpt["step"],
        "best_val_loss": ckpt["best_val_loss"],
        "cash_test": cash_te,
        "cash_train": cash_tr,
        "cash_base": cash_base,
        "cash_model_lin": cash_model_lin,
        "cash_partial": cash_partial,
        "best_cash_test": cash_te[best_layer_cash],
        "best_cash_train": cash_tr[best_layer_cash],
        "own_test": own_te,
        "own_owned_test": own_owned_te,
        "own_train": own_tr,
        "best_own_test": own_te[best_layer_own],
        "best_own_train": own_tr[best_layer_own],
        "own_majority": own_maj,
        "owner_majority_owned": owner_maj,
        "turn_test": max(turn),
        "sanity_token": max(sanity),
        "n_layers": n_layers,
        "n_games": n_games,
        "n_tr_games": n_tr_games,
        "n_prop_fields": len(pfields),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+",
                    default=[os.path.join("runs", "run1", "ckpt_best.pt"),
                             os.path.join("runs", "run1", "ckpt_step1000.pt"),
                             os.path.join("runs", "run1", "ckpt_step1250.pt"),
                             os.path.join("runs", "run1", "ckpt_step1500.pt")])
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val-frac-probe", type=float, default=0.2)
    ap.add_argument("--nonlinear", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

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
        print(f"  Cash-Probe R  test je Layer: {[f'{a:.3f}' for a in r['cash_test']]}")
        print(f"  Cash-Probe R train je Layer: {[f'{a:.3f}' for a in r['cash_train']]}")
        print(f"  Cash-Konfund (progress): Base {[f'{a:.3f}' for a in r['cash_base']]} | "
              f"Model(lins) {[f'{a:.3f}' for a in r['cash_model_lin']]} | "
              f"Partial {[f'{a:.3f}' for a in r['cash_partial']]}")
        print(f"  Own-Probe Acc test je Layer: {[f'{a:.1%}' for a in r['own_test']]}")
        print(f"  best Cash R: {r['best_cash_test']:.3f} (train {r['best_cash_train']:.3f}) | "
              f"best Own Acc: {r['best_own_test']:.1%} (train {r['best_own_train']:.1%}) | "
              f"Owned-Acc: {r['own_owned_test'][0]:.1%} | Majority(None): {r['own_majority']:.1%} | "
              f"OwnerMaj(besetzt): {r['owner_majority_owned']:.1%} | "
              f"Player: {r['turn_test']:.1%} | Tok: {r['sanity_token']:.1%}")

    print(f"\n{'='*78}")
    print(f"ZUSAMMENFASSUNG (nonlinear={args.nonlinear}, per-Game-Split, shared {len(probe_games)} Spiele):")
    print(f"{'Step':>6} | {'train':>7} | {'val':>7} | {'CashR te':>8} | {'CashR tr':>8} | "
          f"{'OwnAcc te':>9} | {'OwnAcc tr':>9} | {'Player':>6} | {'Tok':>5}")
    print("-" * 78)
    for r in rows:
        tl, vl = losses.get(r["step"], (float("nan"), float("nan")))
        tls = f"{tl:.3f}" if tl == tl else "  -  "
        vls = f"{vl:.3f}" if vl == vl else "  -  "
        print(f"{r['step']:>6} | {tls:>7} | {vls:>7} | {r['best_cash_test']:>8.3f} | "
              f"{r['best_cash_train']:>8.3f} | {r['best_own_test']:>8.1%} | "
              f"{r['best_own_train']:>8.1%} | {r['turn_test']:>5.1%} | {r['sanity_token']:>5.1%}")
    print(f"\nOwnership Majority-Baseline (Anteil None): {rows[0]['own_majority']:.1%} "
          f"über {rows[0]['n_prop_fields']} Preisfelder; Owner-Majority unter besetzten "
          f"Feldern: {rows[0]['owner_majority_owned']:.1%}.")


if __name__ == "__main__":
    main()
