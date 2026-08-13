"""
H3-light kausaler Mechanismus-Test: Verschieben Aktivierungen an der
Entscheidungsposition die BUY/PASS-Vorhersage in Richtung der Affordability-
Klasse? (siehe docs/06, H3-light).

Was hier getestet wird (bewusst vorsichtig formuliert):
  "Aktivierungen an der Entscheidungsposition verschieben BUY/PASS kausal in
   Richtung der Affordability-Klasse" — NICHT "die isolierte Affordability-
   Variable ist kausal" (Activation Patching ersetzt den ganzen Residual-Zustand).

Methode:
  Kontrastive Paare aus dem sc400-Korpus mit gleichem Preis/Position/Player/
  Policy und |token_idx-Differenz| <= 10, aber unterschiedlichem cash:
    (100,150) unaff vs (300,150) afford   [Preis 150 -> Felder 3/6/8]
    (100,200) unaff vs (300,200) afford   [Preis 200 -> Feld 11]
  Pro Paar wird resid_post[L, site] der einen Klasse in die andere eingespeist
  (fwd_hook). Zielgroesse ist die Entscheidungsmarge am pre_decision-Token:
    Δ = logit(BUY) - logit(PASS)
  gemessen an der Position direkt VOR dem BUY/PASS-Token (t-1, das P-Token
  nach ROLL D*), wo das Modell das naechste Token (=BUY/PASS) vorhersagt.

  Sites (--site):
    pre_decision  : Patch auf Position t-1 (primär; Logits dort erzeugen BUY/PASS)
    decision_token: Patch auf Position t (das BUY/PASS-Token selbst). Kann die
                    t-1-Logits NICHT bewegen (causale Maske) -> erwartet ~0.
                    Diese ~0 ist der präregistrierte Check: die Entscheidung
                    entsteht am pre_decision-Token, nicht am (post-hoc im
                    Kontext sichtbaren) BUY/PASS-Token.

Kontrollen (alle präregistriert, norm-/positionskompatibel):
  C1 random      : resid eines zufälligen anderen Punkts gleicher Position
  C2 wrong_pos   : Partner-resid an Position site-1 statt site
  C3 player_dir  : lineare Probe-Richtung für Player, norm-matched
  C4 shuffled    : lineare Probe-Richtung für Afford auf SHUFFLED Labels
  C5 same_class  : resid eines anderen Punkts GLEICHER Klasse (Null-Erwartung)

Erfolgskriterien (docs/06, H3-light):
  K1 Richtung       : >=70% der Paare in erwarteter Richtung (unaff->afford Δ↑,
                      afford->unaff Δ↓)
  K2 Kontroll-Dom.  : |Haupteffekt| >= 2x größte Kontroll-Effektgröße (je Layer)
  K3 Layer-Lokal.   : Effekt-Maximum in/nähe einem Layer mit hoher H2-Probe-Acc
                      (wird je Layer als lineare Afford-Probe auf resid_post
                      des pre_decision-Tokens gemessen)
  K4 Symmetrie      : beide Patch-Richtungen zeigen entgegengesetzte, betrags-
                      ähnliche Effekte

Nutzung:  (Conda-Env iprop, CUDA)
    python probes/causal_afford.py \
        --ckpt runs/h2_7e_neighbor/ckpt_best.pt \
        --data data/games_so7003_n4000_always_buy-conservative-random_sc400.jsonl \
        --site pre_decision
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.tokenizer import MiniTokenizer
from data.splits import load_games, decision_points
from probes.probe_position import build_model, fit_probe

# Kontrast-Universe: cash {100,300} x price {150,200}
CASH_UNAFF, CASH_AFF = 100, 300
PRICES = (150, 200)


def collect_records(games, tokenizer, n_ctx):
    """Sammelt alle Kauf-Entscheidungspunkte im Kontrast-Universe.

    Rückgabe: Liste von dicts (pro Entscheidungspunkt) mit Token-Positionen
    (pos_pre = t-1, pos_buy = t) und Meta-Feldern.
    """
    records = []
    for gi, g in enumerate(games):
        ids, raw = tokenizer.tokenize_events_aligned(g["events"])
        policy = g.get("policy_name", "?")
        pts = decision_points(g)
        # Event-Index -> erste Token-Position (BUY/PASS sind atomar)
        pos_of = {}
        for tpos, ridx in enumerate(raw):
            if ridx not in pos_of:
                pos_of[ridx] = tpos
        for p in pts:
            if p["cash"] not in (CASH_UNAFF, CASH_AFF):
                continue
            if p["price"] not in PRICES:
                continue
            pos_buy = pos_of.get(p["token_idx"])
            if pos_buy is None or pos_buy < 1 or pos_buy >= n_ctx:
                continue
            records.append({
                "game_idx": gi,
                "seq": ids,
                "pos_buy": pos_buy,
                "pos_pre": pos_buy - 1,
                "cash": p["cash"],
                "price": p["price"],
                "position": p["position"],
                "player": p["player"],
                "policy": policy,
                "decision": p["decision"],
                "afford": int(p["cash"] >= p["price"]),
                "token_idx": p["token_idx"],
            })
    return records


def greedy_pairs(records):
    """Matcht unaff (cash=100) mit afford (cash=300) innerhalb gleicher
    (price, position, player, policy)-Gruppen; token_idx-Naehe <= 10.

    Greedy-Nearest: beide Seiten nach token_idx sortiert, zweiseitige Zeiger.
    Gibt Liste von (idx_unaff, idx_aff, key).
    """
    from collections import defaultdict
    groups = defaultdict(lambda: {"unaff": [], "aff": []})
    for i, r in enumerate(records):
        key = (r["price"], r["position"], r["player"], r["policy"])
        (groups[key]["unaff"] if r["cash"] == CASH_UNAFF else groups[key]["aff"]).append(i)
    pairs = []
    for key, grp in groups.items():
        unaff = sorted(grp["unaff"], key=lambda i: records[i]["token_idx"])
        aff = sorted(grp["aff"], key=lambda i: records[i]["token_idx"])
        used_aff = set()
        for a in unaff:
            ta = records[a]["token_idx"]
            best, best_d = None, None
            for b in aff:
                if b in used_aff:
                    continue
                d = abs(records[b]["token_idx"] - ta)
                if d > 10:
                    continue
                if best is None or d < best_d:
                    best, best_d = b, d
            if best is not None:
                used_aff.add(best)
                pairs.append((a, best, key))
    return pairs


def pad_batch(records, idx_list, tokenizer, device):
    """Baut gepaddete Batch-Tensoren für eine Auswahl von Records."""
    B = len(idx_list)
    L = max(len(records[i]["seq"]) for i in idx_list)
    batch = torch.full((B, L), tokenizer.pad_id, dtype=torch.long)
    for row, i in enumerate(idx_list):
        s = records[i]["seq"]
        batch[row, :len(s)] = torch.tensor(s, dtype=torch.long)
    return batch.to(device)


def model_device(model):
    return next(model.parameters()).device


def pos_arr(records, idx_list, key):
    return torch.tensor([records[i][key] for i in idx_list], dtype=torch.long)


def decision_margin(logits, pos_list, tokenizer):
    """Δ = logit(BUY) - logit(PASS) an Positionen pos_list. [B]"""
    B = logits.shape[0]
    idx = torch.arange(B, device=logits.device)
    l = logits[idx, pos_list.to(logits.device)]
    return l[:, tokenizer.stoi["BUY"]] - l[:, tokenizer.stoi["PASS"]]


def fit_weights(Xtr, ytr, device, steps=500, lr=5e-3, n_classes=None):
    """Lineare Probe (d -> n_cls), gibt (W [n_cls,d], acc_tr) zurück."""
    Xtr = Xtr.float().to(device)
    ytr = ytr.long().to(device)
    n_cls = int(ytr.max().item()) + 1 if n_classes is None else n_classes
    lin = nn.Linear(Xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss = lossf(lin(Xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (lin(Xtr).argmax(1) == ytr).float().mean().item()
    return lin.weight.detach().clone(), acc


def norm_dir(W, cls):
    w = W[cls]
    n = w.norm().clamp_min(1e-12)
    return (w / n).float()


def run_patches(model, records, idx_list, site_positions, replacements, tokenizer,
                column_offset=0, batch_size=64):
    """Forward mit resid_post[L, site+offset] <- replacements[L] je Layer.

    Liefert je Layer die Δ-Marge (logit(BUY)-logit(PASS)) am pre_decision-Token.
    Gepaddet je Layer-Sub-Batch (Gedächtnis-schonend).
    """
    md = model_device(model)
    out = {}
    for L in replacements:
        margs = []
        for start in range(0, len(idx_list), batch_size):
            chunk = idx_list[start:start + batch_size]
            batch = pad_batch(records, chunk, tokenizer, md)
            pos_pre = pos_arr(records, chunk, "pos_pre").to(md)
            reps = replacements[L][start:start + batch_size].to(md)
            positions = site_positions[start:start + batch_size].to(md) + column_offset
            def hook(act, hook, _reps=reps, _pos=positions):
                act = act.clone()
                r = torch.arange(act.shape[0], device=act.device)
                act[r, _pos] = _reps
                return act
            with torch.no_grad():
                logits = model.run_with_hooks(
                    batch, fwd_hooks=[(f"blocks.{L}.hook_resid_post", hook)])
                margs.append(decision_margin(logits, pos_pre, tokenizer).cpu())
            del batch, logits
            torch.cuda.empty_cache()
        out[L] = torch.cat(margs)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "h2_7e_neighbor", "ckpt_best.pt"))
    ap.add_argument("--data", default="data/games_so7003_n4000_always_buy-conservative-random_sc400.jsonl")
    ap.add_argument("--site", choices=["pre_decision", "decision_token"], default="pre_decision")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = alle Paare")
    ap.add_argument("--layers", type=str, default="all",
                    help="Layer-Auswahl, z.B. '0,1,2,3' oder 'all'")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    model = build_model(manifest, ckpt, device)
    n_layers = model.cfg.n_layers
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    tokenizer = MiniTokenizer()
    games = load_games(args.data)
    records = collect_records(games, tokenizer, model.cfg.n_ctx)
    pairs = greedy_pairs(records)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    print(f"Korpus: {len(games)} Spiele | Kontrast-Records: {len(records)} | Paare: {len(pairs)}")

    if not pairs:
        print("Keine kontrastiven Paare gefunden.")
        return

    # ---------------- Meta / Pair-Statistiken ----------------
    from collections import Counter
    pool = sorted({i for p in pairs for i in (p[0], p[1])})
    pool_map = {i: k for k, i in enumerate(pool)}
    unaff_dec = Counter(records[p[0]]["decision"] for p in pairs)
    aff_dec = Counter(records[p[1]]["decision"] for p in pairs)
    aff_pass_by_pol = Counter(records[p[1]]["policy"] for p in pairs
                              if records[p[1]]["decision"] == "PASS")
    print("  Pair-Verteilung unaff (cash=100):", dict(unaff_dec))
    print("  Pair-Verteilung afford (cash=300):", dict(aff_dec))
    print("  afford+PASS je Policy:", dict(aff_pass_by_pol))
    by_pol = Counter(records[p[0]]["policy"] for p in pairs)
    print("  Policies (unaff-Seite):", dict(by_pol))

    # ---------------- Clean-Forward + resid-Cache für Pool ----------------
    site_key = "pos_pre" if args.site == "pre_decision" else "pos_buy"
    resid_cache = {L: torch.zeros(len(pool), model.cfg.d_model) for L in layers}
    clean_margin = torch.zeros(len(pool))
    for start in range(0, len(pool), args.batch_size):
        chunk = pool[start:start + args.batch_size]
        batch = pad_batch(records, chunk, tokenizer, device)
        with torch.no_grad():
            logits, cache = model.run_with_cache(batch)
        for row, i in enumerate(chunk):
            pk = pool_map[i]
            pos = records[i][site_key]
            dm = decision_margin(
                logits[row:row + 1],
                torch.tensor([records[i]["pos_pre"]], dtype=torch.long).to(device), tokenizer)
            clean_margin[pk] = dm.item()
            for L in layers:
                resid_cache[L][pk] = cache["resid_post", L][row, pos].float().cpu()
        del cache
        torch.cuda.empty_cache()

    # ---------------- Kontroll-Direktionen (C3/C4) ----------------
    # per-game Split für die Proben: erste 80% der Spiele = Probe-Train
    n_games = len(games)
    tr_games = set(range(int(0.8 * n_games)))
    def pool_split():
        tr, te = [], []
        for i in pool:
            (tr if records[i]["game_idx"] in tr_games else te).append(i)
        return tr, te
    tr_ids, te_ids = pool_split()

    # H2-Referenz je Layer (K3): lineare Afford-Probe auf pre_decision-resid
    afford_te = {}
    y_aff = {i: records[i]["afford"] for i in pool}
    for L in layers:
        Xtr = torch.stack([resid_cache[L][pool_map[i]] for i in tr_ids])
        ytr = torch.tensor([y_aff[i] for i in tr_ids])
        Xte = torch.stack([resid_cache[L][pool_map[i]] for i in te_ids])
        yte = torch.tensor([y_aff[i] for i in te_ids])
        _, a_te = fit_probe(Xtr, ytr, Xte, yte, device)
        afford_te[L] = a_te
    print("  H2-Afford-Probe (held-out Spiele, je Layer):",
          {L: f"{afford_te[L]:.1%}" for L in layers})

    # C3: Player-Richtung (meistbelegte Klasse), C4: Shuffle-Richtung
    dir_cache = {}
    Xtr_all = torch.stack([resid_cache[layers[0]][pool_map[i]] for i in tr_ids])
    y_pl = torch.tensor([records[i]["player"] for i in tr_ids])
    W_pl, _ = fit_weights(Xtr_all, y_pl, device)
    from collections import Counter as _C
    maj_cls = _C(y_pl.tolist()).most_common(1)[0][0]
    rng = torch.Generator().manual_seed(0)
    y_aff_tr = torch.tensor([y_aff[i] for i in tr_ids])
    y_sh = y_aff_tr.clone()[torch.randperm(len(y_aff_tr), generator=rng)]
    W_sh, _ = fit_weights(Xtr_all, y_sh, device)
    dir_cache["player"] = {L: norm_dir(W_pl, maj_cls).cpu() for L in layers}
    dir_cache["shuffled"] = {L: norm_dir(W_sh, 1).cpu() for L in layers}

    # ---------------- Patch-Replacements vorbereiten ----------------
    # site-Position je Pool-Element (Tensor) + Zufalls-/Same-Class-Quellen
    site_pos = torch.tensor([records[i][site_key] for i in pool], dtype=torch.long)
    rnd_src = torch.zeros(len(pool), dtype=torch.long)
    same_src = torch.zeros(len(pool), dtype=torch.long)
    by_pos = {}
    for k, i in enumerate(pool):
        by_pos.setdefault(records[i]["position"], []).append(k)
    import random as _random
    _random.seed(0)
    for k, i in enumerate(pool):
        cands = [c for c in by_pos[records[i]["position"]] if c != k]
        rnd_src[k] = _random.choice(cands)
        same = [c for c in cands if records[pool[c]]["afford"] == records[i]["afford"]]
        same_src[k] = _random.choice(same) if same else rnd_src[k]
    # Norm-Target (C3/C4): mittlere |resid_B - resid_A| je Layer über Paare
    norm_target = {}
    for L in layers:
        ds = []
        for a, b, _ in pairs:
            ds.append((resid_cache[L][pool_map[b]] - resid_cache[L][pool_map[a]]).norm().item())
        norm_target[L] = (sum(ds) / len(ds)) if ds else 0.0

    # ---------------- Effekt-Metriken ----------------
    result = {"config": {"ckpt": args.ckpt, "data": args.data, "site": args.site,
                         "n_pairs": len(pairs), "n_pool": len(pool)},
              "pair_stats": {"unaff": dict(unaff_dec), "afford": dict(aff_dec),
                             "afford_pass_by_policy": dict(aff_pass_by_pol),
                             "policies": dict(by_pol)},
              "layers": {}, "k_metrics": {}}

    # Haupt-Kontrast je Richtung: A-Seite = Klasse, Quelle = Partner-resid
    for direction in ("unaff_to_afford", "afford_to_unaff"):
        if direction == "unaff_to_afford":
            a_idx = [p[0] for p in pairs]        # A=unaff, patch afford-resid rein
            b_idx = [p[1] for p in pairs]
        else:
            a_idx = [p[1] for p in pairs]        # A=afford, patch unaff-resid rein
            b_idx = [p[0] for p in pairs]
        pk_a = [pool_map[i] for i in a_idx]

        def make_repl(src, off=0):
            repl = {}
            for L in layers:
                v = torch.stack([resid_cache[L][pool_map[b]] for b in src])
                repl[L] = v
            return repl

        # Main + Kontrollen als Batches (ein Forward je Layer je Kontrolle)
        def patch_effects(repl, offset):
            out = run_patches(model, records, a_idx, site_pos[pk_a], repl,
                              tokenizer, column_offset=offset, batch_size=args.batch_size)
            res = {}
            for L in out:
                res[L] = (out[L] - clean_margin[pk_a])
            return res

        main_eff = patch_effects(make_repl(b_idx), 0)
        c1_eff = patch_effects(make_repl([pool[rnd_src[k].item()] for k in pk_a]), 0)
        c2_eff = patch_effects(make_repl(b_idx), -1)
        c3_eff = patch_effects(
            {L: resid_cache[L][pk_a] + dir_cache["player"][L] * norm_target[L] for L in layers}, 0)
        c4_eff = patch_effects(
            {L: resid_cache[L][pk_a] + dir_cache["shuffled"][L] * norm_target[L] for L in layers}, 0)
        c5_eff = patch_effects(make_repl([pool[same_src[k].item()] for k in pk_a]), 0)

        # Subgruppen-Masken über die Paar-Reihenfolge (beide Richtungen nutzen
        # dieselben Paare): opposite-decision = afford-Seite entscheidet BUY;
        # je Policy = afford- und unaff-Seite teilen die Policy (Matching-Bedingung).
        aff_dec = [records[p[1]]["decision"] for p in pairs]
        mask_opp = torch.tensor([d == "BUY" for d in aff_dec], dtype=torch.bool)
        masks = {"all": torch.ones(len(pairs), dtype=torch.bool),
                 "opposite_afford_BUY": mask_opp}
        for pol in sorted(by_pol):
            masks[f"policy_{pol}"] = torch.tensor(
                [records[p[0]]["policy"] == pol for p in pairs], dtype=torch.bool)

        def metrics(me):
            return {
                "mean": me.mean().item(),
                "mean_abs": me.abs().mean().item(),
                "median": me.median().item(),
                "frac_correct": (me > 0).float().mean().item() if direction == "unaff_to_afford"
                                else (me < 0).float().mean().item(),
                "n": int(me.numel()),
            }

        for L in layers:
            me = main_eff[L]
            ctrls = {"random": c1_eff[L].abs().mean().item(),
                     "wrong_pos": c2_eff[L].abs().mean().item(),
                     "player_dir": c3_eff[L].abs().mean().item(),
                     "shuffled": c4_eff[L].abs().mean().item(),
                     "same_class": c5_eff[L].abs().mean().item()}
            entry = {
                "mean": me.mean().item(),
                "mean_abs": me.abs().mean().item(),
                "median": me.median().item(),
                "frac_correct": (me > 0).float().mean().item() if direction == "unaff_to_afford"
                                else (me < 0).float().mean().item(),
                "controls": ctrls,
                "subgroups": {name: metrics(me[mask]) for name, mask in masks.items()},
            }
            result["layers"].setdefault(str(L), {})[direction] = entry

    # ---------------- Zusammenfassung + K-Metriken ----------------
    print(f"\n=== H3-light (site={args.site}) ===")
    print(f"Δ = logit(BUY)-logit(PASS) am pre_decision-Token; {len(pairs)} Paare")
    n_opp = int(mask_opp.sum())
    print(f"  Opposite-decision (afford=BUY): {n_opp} Paare")
    for L in layers:
        un = result["layers"][str(L)]["unaff_to_afford"]
        af = result["layers"][str(L)]["afford_to_unaff"]
        uo = un["subgroups"]["opposite_afford_BUY"]
        ao = af["subgroups"]["opposite_afford_BUY"]
        print(f"  Layer {L}: unaff->afford {un['mean']:+.4f} ({un['frac_correct']:.0%}) | "
              f"opp {uo['mean']:+.4f} ({uo['frac_correct']:.0%} von {uo['n']}) | "
              f"afford->unaff {af['mean']:+.4f} ({af['frac_correct']:.0%}) | "
              f"opp {ao['mean']:+.4f} ({ao['frac_correct']:.0%}) | H2 {afford_te[L]:.0%}")
    # Policy-Breakdown (best Layer)
    best_L_all = max(layers, key=lambda L: min(
        result["layers"][str(L)]["unaff_to_afford"]["frac_correct"],
        result["layers"][str(L)]["afford_to_unaff"]["frac_correct"]))
    print(f"\n  Policy-Breakdown unaff->afford (Layer {best_L_all}):")
    for pol in sorted(by_pol):
        sg = result["layers"][str(best_L_all)]["unaff_to_afford"]["subgroups"][f"policy_{pol}"]
        print(f"    {pol:12s}: mean {sg['mean']:+.4f} | richtig {sg['frac_correct']:.0%} (n={sg['n']})")

    # K1: beste Layer über beide Richtungen, jeweils auf opposite-decision-Subgruppe
    best_L = max(layers, key=lambda L: min(
        result["layers"][str(L)]["unaff_to_afford"]["subgroups"]["opposite_afford_BUY"]["frac_correct"],
        result["layers"][str(L)]["afford_to_unaff"]["subgroups"]["opposite_afford_BUY"]["frac_correct"]))
    def sg(L, d):
        return result["layers"][str(L)][d]["subgroups"]["opposite_afford_BUY"]
    k1 = min(sg(best_L, "unaff_to_afford")["frac_correct"],
             sg(best_L, "afford_to_unaff")["frac_correct"])
    # K2: |main| vs max|control| je Layer (gesamte Paarmenge, konservativ)
    k2_ratios = {}
    for L in layers:
        main_abs = max(result["layers"][str(L)]["unaff_to_afford"]["mean_abs"],
                       result["layers"][str(L)]["afford_to_unaff"]["mean_abs"])
        max_ctrl = max(result["layers"][str(L)]["unaff_to_afford"]["controls"].values())
        k2_ratios[str(L)] = (main_abs / max_ctrl) if max_ctrl > 0 else float("inf")
    # K3: Effekt-Maximum-Layer vs H2-Probe
    eff_layer = max(layers, key=lambda L: max(
        abs(result["layers"][str(L)]["unaff_to_afford"]["mean"]),
        abs(result["layers"][str(L)]["afford_to_unaff"]["mean"])))
    # K4: Symmetrie (beste Layer, opposite-decision)
    sym = result["layers"][str(best_L)]
    k4 = abs(sg(best_L, "unaff_to_afford")["mean"] + sg(best_L, "afford_to_unaff")["mean"]) / max(
        abs(sg(best_L, "unaff_to_afford")["mean"]), abs(sg(best_L, "afford_to_unaff")["mean"]), 1e-9)

    result["k_metrics"] = {
        "n_pairs_opposite": n_opp,
        "K1_frac_correct_bestL_opp": k1,
        "K1_best_layer": best_L,
        "K1_all_pairs_bestL": min(
            result["layers"][str(best_L)]["unaff_to_afford"]["frac_correct"],
            result["layers"][str(best_L)]["afford_to_unaff"]["frac_correct"]),
        "K2_ratio_per_layer": {k: round(v, 2) for k, v in k2_ratios.items()},
        "K2_min_ratio": min(k2_ratios.values()),
        "K3_eff_max_layer": eff_layer,
        "K3_h2_probe_at_eff_layer": afford_te[eff_layer],
        "K3_h2_probe_max_layer": max(afford_te.values()),
        "K4_asymmetry": k4,
        "afford_probe_per_layer": {str(L): afford_te[L] for L in layers},
    }
    print(f"\n  K1 Richtung (opp, best Layer {best_L}): {k1:.0%} (Ziel >=70%)")
    print(f"  K2 min-Kontroll-Ratio: {min(k2_ratios.values()):.2f} (Ziel >=2.0)")
    print(f"  K3 Effekt-Layer {eff_layer} vs H2-Probe dort {afford_te[eff_layer]:.0%} "
          f"(max Probe {max(afford_te.values()):.0%})")
    print(f"  K4 Asymmetrie: {k4:.2f} (Ziel ~0)")

    # ---------------- JSON-Ausgabe ----------------
    if args.out_json is None:
        run_dir = os.path.dirname(os.path.abspath(args.ckpt))
        args.out_json = os.path.join(run_dir, f"causal_afford_{args.site}.json")
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {args.out_json}")


if __name__ == "__main__":
    main()