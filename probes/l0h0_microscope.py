"""
H3-position-2: L0h0-Microscope — Was trägt der kausale Top-Head wirklich?
(präregistriert in docs/06; Plan committet, dann gemessen).

Frage: Ist L0h0 (kausaler Top-Head des Roll->Event-Kanals, F~0.45) ein
Positions-/History-Integrator, ein Event-class-Shortcut-Head, oder ein
Player/Policy/Timing-Head, der nur mit Position korreliert?

Teiltests (Diagnose-Schema in docs/06, H3-position-2):

  T1 Token-Spezifität der Ablation (z=0 von L0h0 an verschiedenen Positionen):
       pre_decision, current_dice, current_player, prev_same_player_dice,
       prev_other_player_dice, prev_events -> wo im Kontext wirkt L0h0?
  T2 Remote-Roll-Sensitivität (systematisch): alte Würfel ändern, getrennt nach
       same/other-player und recent/distant; mit Kontrast-Ablation von L0h0
       am pos -> läuft die Remote-Wirkung über L0h0?
  T3 Head-Output-Proben auf hook_result von L0h0 (residual space) an pos
       (vergleichend pos-1): current player, Position mod 12, Feldtyp/Preis/
       Owner-Status, nächste Event-Klasse, Token-ID (Sanity).
       PRO-SPIEL-SPLIT (game-invariance, wie H1).
  T4 Unembedding-Richtungen von L0h0: head_result @ W_U -> Logit-Beiträge zu
       BUY/PASS/PAY_RENT/ROLL, aufgeschlüsselt nach wahrer Event-Klasse.

Nutzung:  (Conda-Env iprop, CUDA)
    python probes/l0h0_microscope.py \
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
from probes.probe_position import fit_probe

CLS_ORDER = ["P", "BUY_PASS", "PAY_RENT"]
EVENT_TOKENS = {"BUY", "PASS", "PAY_RENT", "ROLL"}


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


def collect_records(games, tokenizer, max_records=0):
    """Sammelt Entscheidungspunkte P-nach-dice inkl. Positions-Metadaten.

    Rückgabe: Liste von dicts mit
      seq (Token-IDs bis inkl. pos), pos (pre_decision), pos_dice (pos-1),
      roller_pos, roll, true_cls, counters, game_idx,
      prev_same_dice, prev_other_dice (letzter Wurf davor),
      first_same_dice, first_other_dice (ältester Wurf im Kontext),
      prev_event (letztes Event-Token vor pos),
      landed, price, owner_status, current_player.
    max_records trunkiert auf VOLLE Spiele (Split-Granularität bleibt gültig).
    """
    records = []
    for gi, g in enumerate(games):
        if max_records and len(records) >= max_records:
            break
        events = g["events"]
        ids, raw = tokenizer.tokenize_events_aligned(events)
        n = len(events)
        # Wurf-Historie: dice_event_idx -> (dice_pos, roller)
        roll_history = []
        for j in range(3, n):
            if events[j - 2] == "ROLL" and events[j - 1].isdigit():
                roller_ev = j - 3
                if events[roller_ev].startswith("P") and events[roller_ev][1:].isdigit():
                    roll_history.append((j - 1, raw.index(j - 1),
                                         int(events[roller_ev][1:])))
        # Event-Token-Positionen (BUY/PASS/PAY_RENT/ROLL)
        event_positions = [raw.index(j) for j, ev in enumerate(events)
                           if ev in EVENT_TOKENS]
        game_records = []
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
            counters = [counter_class(roll, d2, roller, g["states"][j - 1],
                                      g["prices"], g["board_size"])
                        for d2 in range(1, 7)]
            roller_pos = raw.index(j - 3)
            prev_same = prev_other = None
            first_same = first_other = None
            for (dice_ev, dpos, droller) in roll_history:
                if dice_ev >= j - 1:
                    continue
                if droller == roller:
                    if prev_same is None:
                        prev_same = dpos
                    first_same = dpos
                else:
                    if prev_other is None:
                        prev_other = dpos
                    first_other = dpos
            prev_event = max([p for p in event_positions if p < pos], default=None)
            state = g["states"][j - 1]
            landed = state["position"][roller]
            price = g["prices"][landed]
            owner = state["owner"][landed]
            if price == 0:
                owner_status = "no_field"
            elif owner is None:
                owner_status = "unowned"
            elif owner == roller:
                owner_status = "own"
            else:
                owner_status = "other"
            game_records.append({
                "seq": ids[:pos + 1], "pos": pos,
                "pos_dice": pos - 1, "roller_pos": roller_pos,
                "roll": roll, "true_cls": true_cls, "counters": counters,
                "game_idx": gi,
                "prev_same_dice": prev_same, "prev_other_dice": prev_other,
                "first_same_dice": first_same, "first_other_dice": first_other,
                "prev_event": prev_event,
                "landed": int(landed), "price": int(price),
                "owner_status": owner_status, "current_player": roller,
            })
        records.extend(game_records)
    return records


def make_batch(records, idx_list, tokenizer, device, extra_replace=None):
    """Gepaddete Batch; extra_replace = {(row, pos): token_id}."""
    B = len(idx_list)
    L = max(len(records[i]["seq"]) for i in idx_list)
    batch = torch.full((B, L), tokenizer.pad_id, dtype=torch.long)
    for row, i in enumerate(idx_list):
        s = records[i]["seq"]
        batch[row, :len(s)] = torch.tensor(s, dtype=torch.long)
    if extra_replace:
        for (row, pos), tid in extra_replace.items():
            batch[row, pos] = tid
    return batch.to(device)


def gather_cls_probs(logits, pos_list, tokenizer, device):
    B = logits.shape[0]
    idx = torch.arange(B)
    l = logits[idx, torch.tensor(pos_list, dtype=torch.long, device=device)]
    st = tokenizer.stoi
    p_tok = l[:, st["ROLL"]]
    bp_tok = l[:, [st["BUY"], st["PASS"]]].logsumexp(1)
    pr_tok = l[:, st["PAY_RENT"]]
    return torch.softmax(torch.stack([p_tok, bp_tok, pr_tok], dim=1), dim=1)


def run_batch_probs(model, records, chunk, tokenizer, device,
                    extra_replace=None, abl_pos=None):
    """Forward und (P|BUY_PASS|PAY_RENT)-Probs an pos pro Sample."""
    batch = make_batch(records, chunk, tokenizer, device,
                       extra_replace=extra_replace)
    pos_pre = [records[i]["pos"] for i in chunk]
    hooks = []
    if abl_pos is not None:
        abl = torch.tensor(abl_pos, dtype=torch.long, device=device)
        def hook(act, hook, _abl=abl, _head=0):
            act = act.clone()
            r = torch.arange(act.shape[0], device=act.device)
            valid = _abl >= 0
            if valid.any():
                act[r[valid], _abl[valid], _head] = 0.0
            return act
        hooks.append((f"blocks.0.attn.hook_z", hook))
    with torch.no_grad():
        if hooks:
            logits = model.run_with_hooks(batch, fwd_hooks=hooks)
        else:
            logits = model(batch)
    return gather_cls_probs(logits, pos_pre, tokenizer, device).cpu()


def dice_value(token_id, tokenizer):
    """D-Token-ID -> Würfelwert 1..6 (None falls kein Dice-Token)."""
    d1 = tokenizer.stoi["D1"]
    if d1 <= token_id < d1 + 6:
        return token_id - d1 + 1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "run2_wd0.5_s1200", "ckpt_best.pt"))
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--max-records", type=int, default=0, help="0 = alle")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    tokenizer = MiniTokenizer()
    model = build_model(manifest, ckpt, device)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    H = (0, 0)  # L0h0

    all_games = load_games(args.data)
    val_ids = set(manifest["val_game_ids"])
    val_games = [g for g in all_games if g["game_id"] in val_ids]
    records = collect_records(val_games, tokenizer, args.max_records)
    n = len(records)
    print(f"Entscheidungen (Dice->Event): {n:,} | Modell {n_layers}L/{n_heads}H")
    if n == 0:
        return
    idx_all = list(range(n))
    st = tokenizer.stoi
    d1 = st["D1"]

    # ---------------- T1: Token-Spezifität der L0h0-Ablation ----------------
    cat_pos = {
        "pre_decision": [r["pos"] for r in records],
        "current_dice": [r["pos_dice"] for r in records],
        "current_player": [r["roller_pos"] for r in records],
        "prev_same_player_dice": [r["prev_same_dice"] if r["prev_same_dice"] is not None else -1
                                  for r in records],
        "prev_other_player_dice": [r["prev_other_dice"] if r["prev_other_dice"] is not None else -1
                                   for r in records],
        "prev_events": [r["prev_event"] if r["prev_event"] is not None else -1
                        for r in records],
    }
    clean_acc = {c: [0, 0] for c in CLS_ORDER}
    t1 = {}
    for cat, positions in cat_pos.items():
        eff_abs = []
        acc_correct = []
        used = 0
        for start in range(0, n, args.batch_size):
            chunk = idx_all[start:start + args.batch_size]
            pc = run_batch_probs(model, records, chunk, tokenizer, device)
            for row, i in enumerate(chunk):
                c = records[i]["true_cls"]
                clean_acc[c][1] += 1
                if pc[row].argmax().item() == CLS_ORDER.index(c):
                    clean_acc[c][0] += 1
            pa = run_batch_probs(model, records, chunk, tokenizer, device,
                                 abl_pos=[positions[i] for i in chunk])
            for row, i in enumerate(chunk):
                if positions[i] < 0:
                    continue
                used += 1
                eff_abs.append((pa[row] - pc[row]).abs().sum().item())
                acc_correct.append(1 if pa[row].argmax().item()
                                   == CLS_ORDER.index(records[i]["true_cls"]) else 0)
        t1[cat] = {"n_used": used,
                   "mean_abs_effect": float(np.mean(eff_abs)) if eff_abs else 0.0,
                   "acc_after_abl": float(np.mean(acc_correct)) if acc_correct else 0.0}
        print(f"  T1 {cat:22s}: n={used:6d} | mean|Δ|={t1[cat]['mean_abs_effect']:.4f} | "
              f"acc={t1[cat]['acc_after_abl']:.0%}")

    # ---------------- T2: Remote-Roll-Sensitivität ----------------
    t2_cats = {
        "same_recent": "prev_same_dice",
        "same_distant": "first_same_dice",
        "other_recent": "prev_other_dice",
        "other_distant": "first_other_dice",
    }
    t2 = {}
    # lokale Referenz (Do-over am aktuellen Dice) über alle Samples
    local_eff = []
    for start in range(0, n, args.batch_size):
        chunk = idx_all[start:start + args.batch_size]
        pc = run_batch_probs(model, records, chunk, tokenizer, device)
        extra = {}
        for row, i in enumerate(chunk):
            dv = dice_value(records[i]["seq"][records[i]["pos_dice"]], tokenizer)
            if dv is None:
                continue
            extra[(row, records[i]["pos_dice"])] = st[f"D{1 + dv % 6}"]
        pl = run_batch_probs(model, records, chunk, tokenizer, device, extra_replace=extra)
        for row, i in enumerate(chunk):
            if dice_value(records[i]["seq"][records[i]["pos_dice"]], tokenizer) is None:
                continue
            local_eff.append((pl[row] - pc[row]).abs().sum().item())
    mean_local = float(np.mean(local_eff)) if local_eff else 0.0
    print(f"  T2 lokale Do-over-Referenz (aktueller Dice): mean|Δ|={mean_local:.4f} (n={len(local_eff)})")

    for name, key in t2_cats.items():
        eff_abs = []
        used = 0
        for start in range(0, n, args.batch_size):
            chunk = idx_all[start:start + args.batch_size]
            pc = run_batch_probs(model, records, chunk, tokenizer, device)
            extra = {}
            for row, i in enumerate(chunk):
                tp = records[i][key]
                if tp is None or tp < 0:
                    continue
                dv = dice_value(records[i]["seq"][tp], tokenizer)
                if dv is None:
                    continue
                extra[(row, tp)] = st[f"D{1 + dv % 6}"]
            pm = run_batch_probs(model, records, chunk, tokenizer, device,
                                 extra_replace=extra)
            for row, i in enumerate(chunk):
                tp = records[i][key]
                if tp is None or tp < 0:
                    continue
                if dice_value(records[i]["seq"][tp], tokenizer) is None:
                    continue
                used += 1
                eff_abs.append((pm[row] - pc[row]).abs().sum().item())
        mean_remote = float(np.mean(eff_abs)) if eff_abs else 0.0
        t2[name] = {"n_used": used, "mean_remote": mean_remote,
                    "mean_local_ref": mean_local,
                    "ratio_remote_local": mean_remote / max(mean_local, 1e-9)}
        print(f"  T2 {name:16s}: n={used:6d} | remote={mean_remote:.4f} | "
              f"local={mean_local:.4f} | ratio={t2[name]['ratio_remote_local']:.2f}")

    # ---------------- T3: Head-Output-Proben (hook_result L0h0) ----------------
    # Pro-Spiel-Split (80/20 Spiele) über die GESAMMELTEN Spiele
    games_used = sorted({records[i]["game_idx"] for i in range(n)})
    split_at = int(0.8 * len(games_used))
    tr_games = set(games_used[:split_at])
    tr_idx = [i for i in range(n) if records[i]["game_idx"] in tr_games]
    te_idx = [i for i in range(n) if records[i]["game_idx"] not in tr_games]
    print(f"  T3 Split nach Spielen: train={len(tr_idx)} test={len(te_idx)} "
          f"({len(tr_games)}/{len(games_used)} Spiele)")

    # ---------------- T3+T4: Ein Pass, hook_result von L0h0 ----------------
    # head_result ist bereits im residual space: (B, pos, head, d_model).
    # Ein einziger Capture-Pass liefert T3-Features (an pos und pos_dice) und
    # T4-Unembedding-Contributions (am pos).
    W_O = model.blocks[H[0]].attn.W_O[H[1]]   # (d_head, d_model)
    W_U = model.unembed.W_U                    # (d_model, n_vocab)
    tok_ids = {"ROLL": st["ROLL"], "BUY": st["BUY"], "PASS": st["PASS"],
               "PAY_RENT": st["PAY_RENT"]}
    feats_pre = []
    feats_dice = []
    labels = {"player": [], "landed": [], "price": [], "owner": [],
              "event": [], "token": []}
    contrib_by_cls = {c: {name: [] for name in tok_ids} for c in CLS_ORDER}
    hook_name = f"blocks.{H[0]}.attn.hook_result"

    def capture(act, hook):
        cap.append(act)
        return act

    for start in range(0, n, args.batch_size):
        chunk = idx_all[start:start + args.batch_size]
        batch = make_batch(records, chunk, tokenizer, device)
        cap = []
        with torch.no_grad():
            model.run_with_hooks(batch, fwd_hooks=[(hook_name, capture)])
        hr = cap[0]  # (B, pos, head, d_model)
        pos_pre_t = torch.tensor([records[i]["pos"] for i in chunk], dtype=torch.long)
        pos_dice_t = torch.tensor([records[i]["pos_dice"] for i in chunk], dtype=torch.long)
        for row, i in enumerate(chunk):
            feats_pre.append(hr[row, pos_pre_t[row], H[1]].float().cpu())
            feats_dice.append(hr[row, pos_dice_t[row], H[1]].float().cpu())
            r = records[i]
            labels["player"].append(r["current_player"])
            labels["landed"].append(r["landed"])
            labels["price"].append(r["price"])
            labels["owner"].append(["no_field", "unowned", "own", "other"].index(r["owner_status"]))
            labels["event"].append(CLS_ORDER.index(r["true_cls"]))
            labels["token"].append(r["seq"][r["pos"]])
        hr_at = hr[torch.arange(len(chunk)), pos_pre_t, H[1]]  # (B, d_model)
        contrib = hr_at @ W_U                                  # (B, n_vocab)
        for row, i in enumerate(chunk):
            c = records[i]["true_cls"]
            for name, tid in tok_ids.items():
                contrib_by_cls[c][name].append(contrib[row, tid].item())
        del cap, hr
        torch.cuda.empty_cache()

    # T3: Probes, pro-Spiel-Split, pro Position
    t3 = {}
    for feats, pos_name in ((feats_pre, "pre_decision"), (feats_dice, "current_dice")):
        Xall = torch.stack(feats)
        Xtr = torch.stack([Xall[i] for i in tr_idx])
        Xte = torch.stack([Xall[i] for i in te_idx])
        probe_res = {}
        for lab_name, lab_list in labels.items():
            yl = torch.tensor(lab_list)
            ytr = torch.stack([yl[i] for i in tr_idx])
            yte = torch.stack([yl[i] for i in te_idx])
            _, acc_te = fit_probe(Xtr, ytr, Xte, yte, device)
            probe_res[lab_name] = acc_te
        t3[pos_name] = probe_res
        print(f"  T3 {pos_name:14s} (L0h0 hook_result, pro-Spiel-Split):")
        for k, v in probe_res.items():
            print(f"      {k:8s}: {v:.1%}")

    # T4: Unembedding-Beitrag von L0h0 je wahrer Klasse
    t4 = {c: {name: float(np.mean(v)) for name, v in d.items()}
          for c, d in contrib_by_cls.items()}
    print("\n  T4 Unembedding-Beitrag von L0h0 (head_result @ W_U) je wahrer Klasse:")
    print("      " + " ".join(f"{name:>9}" for name in tok_ids))
    for c in CLS_ORDER:
        print(f"      {c:9s}: " + " ".join(f"{t4[c][name]:+.3f}" for name in tok_ids))

    # ---------------- JSON ----------------
    result = {
        "config": {"ckpt": args.ckpt, "data": args.data,
                   "n_records": n, "n_layers": n_layers, "n_heads": n_heads,
                   "head": f"{H[0]}_{H[1]}"},
        "clean_acc": clean_acc,
        "T1_token_ablation": t1,
        "T2_remote_roll": t2,
        "T3_probes": t3,
        "T4_unembed_contrib": t4,
    }
    if args.out_json is None:
        run_dir = os.path.dirname(os.path.abspath(args.ckpt))
        args.out_json = os.path.join(run_dir, "l0h0_microscope.json")
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {args.out_json}")


if __name__ == "__main__":
    main()