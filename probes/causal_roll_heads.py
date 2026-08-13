"""
H3-position: Position-Kausalpfad — Mechanismus-Lokalisierung Roll->Event
(präregistriert in docs/06; Plan committet, dann gemessen).

Frage: Über welche Layer/Heads fließt die Roll->Position->Event-Information
kausal in die BUY/PASS/PAY_RENT-Entscheidung, und ist die Nutzung lokal/
rollbezogen (same-player-Heuristik) statt über eine stabile dekodierbare
Zustandsvariable (H1-Befund: Position ist linear/nichtlinear NICHT
game-invariant dekodierbar)?

Ausgangslage (causal_roll.py): kausale Nutzung auf Layer-Ebene belegt
(F_L~1). Dieses Skript löst auf Head-Ebene auf:

  T1 Head-Attribution   : activation patching je Head (L,h) über
                          blocks.{L}.attn.hook_z am Entscheidungs-Token,
                          Recovery F_{L,h} = |patch-clean| / |mod-clean|.
  T2 Head-Ablation      : z=0 je Head am Entscheidungs-Token auf dem
                          Original-Würfel -> direkter Effekt auf die
                          Event-Klassen-Accuracy (notwendige Heads).
  T3 Same-player-attn   : Attention-Patterns vom Entscheidungs-Token auf den
                          Roller-Token (P{roller} VOR [ROLL,d]); Korrelation
                          Head-Kausalität (T1) <-> Roller-Attention.
  T4 Remote-roll        : Würfel-Substitution d->d' an einer FREMDEN
                          D*-Position im Kontext statt an pos-1 ->
                          Spezifitäts-Kontrolle (K4).

Setup wie causal_roll.py: Entscheidungspunkt = P-Token nach [ROLL, d];
Klassen 'P' (->ROLL), 'BUY_PASS', 'PAY_RENT'; Kontrafakt-Klassen je
d' in 1..6 über counter_class.

Nutzung:  (Conda-Env iprop, CUDA)
    python probes/causal_roll_heads.py \
        --ckpt runs/run2_wd0.5_s1200/ckpt_best.pt \
        --data data/games_so0_n2000_always_buy-conservative-random.jsonl
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
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


def counter_class(true_roll: int, query_roll: int, player: int, state: dict,
                  prices: list[int], board: int) -> str:
    post = state["position"][player]
    pre = (post - true_roll) % board
    landed = (pre + query_roll) % board
    owner = state["owner"][landed]
    if prices[landed] > 0 and owner is None:
        return "BUY_PASS"
    if owner is not None and owner != player:
        return "PAY_RENT"
    return "P"


def collect_decisions(games, tokenizer, n_ctx, max_decisions=0):
    """Wie causal_roll: Entscheidungspunkte P-nach-dice.

    Zusätzlich: roller_pos (Token-Position des Roller-P-Tokens vor [ROLL,d])
    und dice_positions (alle D*-Token-Positionen im Kontext bis inkl. pos).
    """
    decisions = []
    for g in games:
        events = g["events"]
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
            counters = []
            for d2 in range(1, 7):
                counters.append(counter_class(roll, d2, roller, g["states"][j - 1],
                                              g["prices"], g["board_size"]))
            roller_pos = raw.index(j - 3)
            dice_positions = [t for t in range(pos)
                              if tokenizer.itos[ids[t]] in
                              (f"D{x}" for x in range(1, 7))]
            decisions.append((ids[:pos + 1], pos, roller_pos, roll,
                              true_cls, counters, dice_positions))
    if max_decisions:
        decisions = decisions[:max_decisions]
    return decisions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "run2_wd0.5_s1200", "ckpt_best.pt"))
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-decisions", type=int, default=0, help="0 = alle")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    tokenizer = MiniTokenizer()
    model = build_model(manifest, ckpt, device)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    all_games = load_games(args.data)
    val_ids = set(manifest["val_game_ids"])
    val_games = [g for g in all_games if g["game_id"] in val_ids]
    decisions = collect_decisions(val_games, tokenizer, model.cfg.n_ctx, args.max_decisions)
    print(f"Entscheidungen (Dice->Event): {len(decisions):,} | "
          f"Modell {n_layers}L/{n_heads}H")

    def make_batch(seqs, pos_list, replace=None, remote=None):
        """replace=(pos, token_id) ersetzt die Würfel-Position; remote=Liste von
        (pos, token_id)-Paaren für entfernte D*-Substitutionen."""
        B = len(seqs)
        L = max(len(s) for s in seqs)
        batch = torch.full((B, L), tokenizer.pad_id, dtype=torch.long)
        for b, s in enumerate(seqs):
            batch[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            if replace is not None:
                batch[b, pos_list[b] - 1] = replace[b]
            if remote is not None:
                for rp, rt in remote[b]:
                    batch[b, rp] = rt
        return batch.to(device)

    def gather_cls_probs(logits, pos_list):
        B = logits.shape[0]
        idx = torch.arange(B)
        l = logits[idx, torch.tensor(pos_list, device=logits.device)]
        st = tokenizer.stoi
        p_tok = l[:, st["ROLL"]]
        bp_tok = l[:, [st["BUY"], st["PASS"]]].logsumexp(1)
        pr_tok = l[:, st["PAY_RENT"]]
        return torch.softmax(torch.stack([p_tok, bp_tok, pr_tok], dim=1), dim=1)

    pos_list = [d[1] for d in decisions]
    roller_pos_list = [d[2] for d in decisions]
    seqs = [d[0] for d in decisions]

    # Ergebnisse
    F = {(L, h): [] for L in range(n_layers) for h in range(n_heads)}       # T1
    abl = {(L, h): [] for L in range(n_layers) for h in range(n_heads)}     # T2
    abl_correct = {(L, h): [] for L in range(n_layers) for h in range(n_heads)}
    roller_attn = {(L, h): [] for L in range(n_layers) for h in range(n_heads)}  # T3
    med_attn = {(L, h): [] for L in range(n_layers) for h in range(n_heads)}
    remote_eff = []                                                          # T4
    local_eff = []
    n_remote_ok = 0
    acc_true_clean = {c: [0, 0] for c in CLS_ORDER}

    B = args.batch_size
    n_proc = 0

    for start in range(0, len(decisions), B):
        chunk = decisions[start:start + B]
        seqs_c = [d[0] for d in chunk]
        pos_c = [d[1] for d in chunk]
        roller_pos_c = [d[2] for d in chunk]
        true_cls = [d[4] for d in chunk]
        dice_pos_c = [d[6] for d in chunk]
        idxs = torch.arange(len(chunk))
        pos_t = torch.tensor(pos_c, device=device)

        with torch.no_grad():
            # clean
            inp = make_batch(seqs_c, pos_c)
            logits_c, cache_c = model.run_with_cache(
                inp, names_filter=lambda n: ("resid_post" in n or "hook_z" in n
                                             or "hook_pattern" in n),
                return_cache_object=False)
            pc = gather_cls_probs(logits_c, pos_c)
            for b, tc in enumerate(true_cls):
                acc_true_clean[tc][1] += 1
                if pc[b].argmax().item() == CLS_ORDER.index(tc):
                    acc_true_clean[tc][0] += 1
            # T3: Attention vom Entscheidungs-Token auf Roller-Token
            rp_t = torch.tensor(roller_pos_c, device=device)
            for L in range(n_layers):
                pat = cache_c[f"blocks.{L}.attn.hook_pattern"]  # [B, H, q, k]
                for h in range(n_heads):
                    rv = pat[torch.arange(len(chunk)), h, pos_t, rp_t].cpu()
                    roller_attn[(L, h)].append(rv)
                    mv = pat[torch.arange(len(chunk)), h, pos_t].mean(-1).cpu()
                    med_attn[(L, h)].append(mv)

            # T4: remote-roll (fremde D*-Position, seed-fix)
            rng = torch.Generator().manual_seed(0)
            remote_repl = [[] for _ in range(len(chunk))]
            remote_local_eff = torch.zeros(len(chunk))
            for b in range(len(chunk)):
                lpos = pos_c[b] - 1
                others = [p for p in dice_pos_c[b] if p != lpos]
                if others:
                    ri = int(torch.randint(len(others), (1,), generator=rng).item())
                    rp = others[ri]
                    d_cur = int(tokenizer.itos[seqs_c[b][rp]][1:])
                    d_new = 1 + (d_cur + 1) % 6
                    remote_repl[b].append((rp, tokenizer.stoi[f"D{d_new}"]))
                    remote_local_eff[b] = 1.0
                else:
                    remote_repl[b].append((lpos, tokenizer.stoi["D1"]))
                    remote_local_eff[b] = 0.0
            inp_remote = make_batch(seqs_c, pos_c, remote=remote_repl)
            logits_r, _ = model.run_with_cache(inp_remote, names_filter=lambda n: "resid_post" in n,
                                               return_cache_object=False)
            pr = gather_cls_probs(logits_r, pos_c).cpu()

            # lokaler Do-over-Effekt (Mittel über d'), + pm/probs für T1-F
            pm_all = {}
            z_wrong = {}  # (L,h) -> [B, d_head] gemittelt über d'
            for L in range(n_layers):
                z_wrong[L] = {}
            for d2 in range(1, 7):
                repl = [tokenizer.stoi[f"D{d2}"] for _ in chunk]
                inp_m = make_batch(seqs_c, pos_c, replace=repl)
                logits_m, cache_m = model.run_with_cache(
                    inp_m, names_filter=lambda n: "hook_z" in n, return_cache_object=False)
                pm_all[d2] = gather_cls_probs(logits_m, pos_c).cpu()
                for L in range(n_layers):
                    zw = cache_m[f"blocks.{L}.attn.hook_z"][idxs, pos_t]  # [B,H,d]
                    for h in range(n_heads):
                        z_wrong[L].setdefault(h, torch.zeros_like(zw[:, 0]))
                        z_wrong[L][h] += zw[:, h]
                del cache_m
            for L in range(n_layers):
                for h in range(n_heads):
                    z_wrong[L][h] /= 6.0

            # T4: lokale Referenz (Mittel über d')
            local_eff_chunk = torch.stack(
                [ (pm_all[d2] - pc.cpu()).abs().sum(1) for d2 in range(1, 7) ]).mean(0)
            local_eff.append(local_eff_chunk.cpu().numpy())
            remote_eff.append((pr - pc.cpu()).abs().sum(1).numpy())
            n_remote_ok += int(remote_local_eff.sum().item())

            # T1: Head-Patching — ein Forward je (L,h) auf ORIGINAL-Input
            for L in range(n_layers):
                for h in range(n_heads):
                    rep = z_wrong[L][h].to(device)  # [B, d_head]
                    def hook(act, hook, _rep=rep, _h=h):
                        act = act.clone()
                        act[torch.arange(act.shape[0], device=act.device),
                            pos_t, _h] = _rep
                        return act
                    logits_p = model.run_with_hooks(
                        inp, fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook)])
                    pp = gather_cls_probs(logits_p, pos_c).cpu()
                    for d2 in range(1, 7):
                        dc = (pm_all[d2] - pc.cpu()).abs().sum(1)
                        dp = (pp - pc.cpu()).abs().sum(1)
                        mask = dc > 0.01  # nur Samples mit messbarem mod-Effekt
                        if mask.any():
                            F[(L, h)].append((dp[mask] / dc[mask]).numpy())

            # T2: Ablation (z=0) je Head auf Original-Würfel
            for L in range(n_layers):
                for h in range(n_heads):
                    def hook_ab(act, hook, _h=h):
                        act = act.clone()
                        act[torch.arange(act.shape[0], device=act.device), pos_t, _h] = 0.0
                        return act
                    logits_a = model.run_with_hooks(
                        inp, fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook_ab)])
                    pa = gather_cls_probs(logits_a, pos_c).cpu()
                    abl[(L, h)].append((pa - pc.cpu()).abs().sum(1).numpy())
                    abl_correct[(L, h)].append(
                        (pa.argmax(1) == torch.tensor(
                            [CLS_ORDER.index(c) for c in true_cls])).numpy())
        del cache_c
        torch.cuda.empty_cache()
        n_proc += len(chunk)
        print(f"  ...{n_proc:,}/{len(decisions):,}", flush=True)

    # ---------------- Ergebnisse ----------------
    def flat(d):
        if isinstance(d, dict):
            arrs = [a for v in d.values() for a in v]
        else:
            arrs = list(d)
        arrs = [a for a in arrs if len(a) > 0]
        if not arrs:
            return np.array([0.0])
        return np.concatenate(arrs)

    print(f"\n=== Clean-Event-Accuracy (Original d=d') ===")
    for c in CLS_ORDER:
        hh, tt = acc_true_clean[c]
        print(f"  {c:9s}: {hh/max(tt,1):6.1%}  ({hh:,}/{tt:,})")

    print(f"\n=== T1 Head-Attribution F_{{L,h}} (|patch-clean|/|mod-clean|) ===")
    print("  Layer x Head -> mean | median | >0.3-Fraktion")
    for L in range(n_layers):
        row = []
        for h in range(n_heads):
            fs = flat(F[(L, h)])
            row.append(f"h{h}: {fs.mean():.2f}/{np.median(fs):.2f}/{(fs > 0.3).mean():.0%}")
        print(f"  L{L}: " + "  ".join(row))
    all_f = flat(F)
    print(f"  Gesamt: mean {all_f.mean():.2f} | median {np.median(all_f):.2f} | "
          f">0.3 {(all_f > 0.3).mean():.1%}")
    # K1/K2
    by_head = {key: float(flat(v).mean()) for key, v in F.items()}
    order = sorted(by_head.items(), key=lambda kv: -kv[1])
    total = sum(by_head.values())
    top1 = order[0]
    top4 = sum(v for _, v in order[:4])
    print(f"  K1: Top-Head {top1[0]} F={top1[1]:.2f} (Ziel >=0.3)")
    print(f"  K2: Top-Head-Anteil {top1[1]/max(total,1e-9):.1%} | "
          f"Top-25%-Anteil {top4/max(total,1e-9):.1%} (Ziel >=50%)")

    print(f"\n=== T2 Head-Ablation (z=0) — Effekt auf Event-Entscheidung ===")
    print("  Layer x Head -> mittl. |abl-clean| | Accuracy nach Ablation")
    for L in range(n_layers):
        row = []
        for h in range(n_heads):
            ea = float(np.concatenate(abl[(L, h)]).mean())
            ac = float(np.concatenate(abl_correct[(L, h)]).mean())
            row.append(f"h{h}: {ea:.2f}/{ac:.0%}")
        print(f"  L{L}: " + "  ".join(row))

    print(f"\n=== T3 Same-player roll attention (Roller-Token-Anteil) ===")
    attn_means = {}
    for L in range(n_layers):
        row = []
        for h in range(n_heads):
            rv = np.concatenate(roller_attn[(L, h)])
            mv = np.concatenate(med_attn[(L, h)])
            ratio = rv.mean() / max(mv.mean(), 1e-9)
            attn_means[(L, h)] = ratio
            row.append(f"h{h}: {ratio:.2f}")
        print(f"  L{L}: " + "  ".join(row))
    med_ratio = float(np.median(list(attn_means.values())))
    top_heads = [k for k, _ in order[:4]]
    top_ratio = float(np.mean([attn_means[k] for k in top_heads])) if top_heads else 0.0
    print(f"  K3 (interpretierend): Median-Ratio {med_ratio:.2f} | "
          f"Top-Heads-Ratio {top_ratio:.2f} (Referenz >=2x Median)")

    print(f"\n=== T4 Remote-roll Spezifität ===")
    lr = float(np.concatenate(local_eff).mean())
    rr = float(np.concatenate(remote_eff).mean())
    print(f"  lokaler Do-over-Effekt: {lr:.3f} | remote-roll Effekt: {rr:.3f} | "
          f"Ratio remote/local: {rr/max(lr,1e-9):.2f} (K4-Ziel <=0.3) | "
          f"remote-fähige Samples: {n_remote_ok}")

    # ---------------- JSON ----------------
    result = {
        "config": {"ckpt": args.ckpt, "data": args.data,
                   "n_decisions": len(decisions),
                   "n_layers": n_layers, "n_heads": n_heads},
        "clean_acc": {c: acc_true_clean[c] for c in CLS_ORDER},
        "F": {f"{L}_{h}": {"mean": float(np.concatenate(F[(L, h)]).mean()),
                           "median": float(np.median(flat(F[(L, h)]))),
                           "frac_gt_03": float((flat(F[(L, h)]) > 0.3).mean())}
              for L in range(n_layers) for h in range(n_heads)},
        "K1_top_head": {"head": f"{top1[0][0]}_{top1[0][1]}", "F": float(top1[1])},
        "K2": {"top1_share": float(top1[1] / max(total, 1e-9)),
               "top4_share": float(top4 / max(total, 1e-9))},
        "T2_ablation": {f"{L}_{h}": {"mean_effect": float(np.concatenate(abl[(L, h)]).mean()),
                                     "acc_after": float(np.concatenate(abl_correct[(L, h)]).mean())}
                        for L in range(n_layers) for h in range(n_heads)},
        "T3_attention": {f"{L}_{h}": float(attn_means[(L, h)])
                         for L in range(n_layers) for h in range(n_heads)},
        "K3": {"median_ratio": float(med_ratio), "top_heads_ratio": float(top_ratio)},
        "T4": {"local_eff": float(lr), "remote_eff": float(rr),
               "ratio": float(rr / max(lr, 1e-9)), "n_remote_ok": int(n_remote_ok)},
    }
    if args.out_json is None:
        run_dir = os.path.dirname(os.path.abspath(args.ckpt))
        args.out_json = os.path.join(run_dir, "causal_roll_heads.json")
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {args.out_json}")


if __name__ == "__main__":
    main()