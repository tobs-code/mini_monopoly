"""
Kausaler Position-Test (User-Vorab-Check vor 10x-Daten) auf Step 550.

Frage: Braucht das Modell die (absolute) Position kausal für die
BUY/PASS/PAY_RENT-Entscheidung nach dem Würfel? Mehr Daten hilft nur, wenn
Position kausal genutzt wird, aber unsauber gelernt ist. Wird sie strukturell
nicht gebraucht (Lösung über andere Signale), reproduziert 10x-Daten denselben
flachen Probe-Wert.

Methode (zwei verschränkte Tests, beide im SELBEN Kontext -> Position isoliert):

1) Do-over (Input-Substitution): Am Würfel-Token t (Rohwert d) wird der Token
   auf D{d'} (d' != d) geändert -> einzige Änderung ist das gelandete Feld.
   Unter dem substituierten Input misst P_mod die Event-Vorhersage. Wenn das
   Modell dabei hin zum KONTRAFAKTISCH-korrekten Event (für d') wandert, nutzt
   es Roll->Position->Event kausal. Wenn die Vorhersage invariant bleibt, wird
   Position an dieser Stelle nicht gebraucht.

2) Residual-Patch (Layer-Attribution): resid_post[L, t] aus dem Wrong-Roll-Lauf
   in einen Forward mit ORIGINAL-Würfel einspeisen. Recovery-Fraktion F_L =
   ||patch - clean|| / ||mod - clean|| zeigt, wie viel vom vollen Effekt das
   Residual von Layer L an Position t kausal trägt. F_L ~ 1 -> Layer-L-Residual
   trägt Position; F_L ~ 0 in allen Layern -> Position nicht im Residual nutzbar.

Klassen für die Event-Entscheidung:
   'P'          = kein weiteres Event (eigenes/preisloses Feld)
   'BUY_PASS'   = unbesetztes Preisfeld
   'PAY_RENT'   = fremdes Feld

Nutzung:
    python probes/causal_roll.py [--ckpt runs/run1/ckpt_best.pt]
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

CLS_TOK = {"P0": "P", "P1": "P", "P2": "P", "BUY": "BUY_PASS", "PASS": "BUY_PASS",
           "PAY_RENT": "PAY_RENT"}
CLS_ORDER = ["P", "BUY_PASS", "PAY_RENT"]


def build_model(manifest, ckpt_state, device):
    cfg = HookedTransformerConfig(**dict(manifest["architecture"]))
    model = HookedTransformer(cfg).to(device)
    model.tokenizer = None
    model.load_state_dict(ckpt_state["model"])
    model.eval()
    return model


def cls_of(token_str: str) -> str | None:
    return CLS_TOK.get(token_str)


def counter_class(true_roll: int, query_roll: int, player: int, state: dict,
                  prices: list[int], board: int) -> str:
    """Event-Klasse, wenn Spieler `player` mit `query_roll` gewürfelt HÄTTE.

    `state` ist der post-Zug-Snapshot des ORIGINAL-Wurfs (d_true). Prä-Zug-
    Position = (post - d_true) % board; Gegenwurf-Feld = (pre + query_roll) % board.
    Für query_roll == d_true ergibt sich konsistenterweise genau der post-Zustand."""
    post = state["position"][player]
    pre = (post - true_roll) % board
    landed = (pre + query_roll) % board
    owner = state["owner"][landed]
    if prices[landed] > 0 and owner is None:
        return "BUY_PASS"
    if owner is not None and owner != player:
        return "PAY_RENT"
    return "P"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("runs", "run1", "ckpt_best.pt"))
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-decisions", type=int, default=0, help="0 = alle")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    manifest = ckpt["manifest"]
    tokenizer = MiniTokenizer()
    model = build_model(manifest, ckpt, device)
    n_layers = model.cfg.n_layers

    all_games = load_games(args.data)
    val_ids = set(manifest["val_game_ids"])
    val_games = [g for g in all_games if g["game_id"] in val_ids]

    # ---------------- Entscheidungen sammeln ----------------
    # Entscheidungspunkt: Der P-Token NACH dem Würfel (raw: [.., ROLL, d, P, <event>]).
    # Am P-Token sagt das Modell den nächsten Token voraus = Event (BUY/PASS/PAY_RENT)
    # oder nächster Spieler-P (kein Event). Der Würfelwert (-> Position) ist an dieser
    # Stelle bereits im Kontext sichtbar.
    decisions = []  # (seq ids bis inkl. P-nach-dice, pos, player, roll, true_cls, [counter per d'])
    for g in val_games:
        events = g["events"]
        ids, raw = tokenizer.tokenize_events_aligned(events)
        n = len(events)
        for j in range(1, n):
            if not (events[j].startswith("P") and events[j][1:].isdigit()):
                continue
            if not (j >= 2 and events[j - 2] == "ROLL" and events[j - 1].isdigit()):
                continue  # nur P-NACH-dice
            pos = raw.index(j)
            if pos + 1 >= len(ids):
                continue  # kein Ziel-Token
            roll = int(events[j - 1])
            player = int(events[j][1:])
            true_tok = tokenizer.itos[ids[pos + 1]]
            true_cls = cls_of(true_tok)
            if true_cls is None:
                continue
            counters = []
            for d2 in range(1, 7):
                counters.append(counter_class(roll, d2, player, g["states"][j - 1],
                                              g["prices"], g["board_size"]))
            decisions.append((ids[:pos + 1], pos, player, roll, true_cls, counters))

    if args.max_decisions:
        decisions = decisions[:args.max_decisions]
    print(f"Entscheidungen (Dice->Event): {len(decisions):,} aus {len(val_games)} Val-Spielen")
    # Verteilung der Klassen im Original
    from collections import Counter
    orig = Counter(d[4] for d in decisions)
    print("True-Verteilung:", dict(orig))

    # ---------------- Batches & Forward-Passes ----------------
    def make_batch(seqs, pos_list, replace=None):
        """pos_list = P-nach-dice-Positionen; bei replace wird der WÜRFEL (pos-1)
        ersetzt (P-Token bleibt unverändert -> Position ändert sich, Kontext sonst nicht)."""
        B = len(seqs)
        L = max(len(s) for s in seqs)
        batch = torch.full((B, L), tokenizer.pad_id, dtype=torch.long)
        for b, s in enumerate(seqs):
            batch[b, :len(s)] = torch.tensor(s, dtype=torch.long)
            if replace is not None:
                batch[b, pos_list[b] - 1] = replace[b]
        return batch.to(device)

    def gather_cls_probs(logits, pos_list):
        """Logits [B, L, V] an pos je Zeile -> Klassenwahrscheinlichkeiten [B, 3]."""
        B = logits.shape[0]
        idx = torch.arange(B)
        l = logits[idx, torch.tensor(pos_list, device=logits.device)]
        ids = torch.tensor([tokenizer.stoi[t] for t in CLS_TOK], device=l.device)
        p_tok = l[:, ids[:3]].logsumexp(1)
        bp_tok = l[:, ids[3:5]].logsumexp(1)
        pr_tok = l[:, ids[5]]
        return torch.softmax(torch.stack([p_tok, bp_tok, pr_tok], dim=1), dim=1)

    pos_list = [d[1] for d in decisions]
    seqs = [d[0] for d in decisions]

    B = args.batch_size
    acc_true_mod = {d2: [0, 0] for d2 in range(1, 7)}      # do-over accuracy: Modell sagt kontrafaktische Klasse?
    acc_true_clean = [0, 0]                                  # clean: Modell sagt wahre Klasse?
    F = {L: [] for L in range(n_layers)}                     # recovery pro Layer
    n_eff = 0

    for start in range(0, len(decisions), B):
        chunk = decisions[start:start + B]
        seqs_c = [d[0] for d in chunk]
        pos_c = [d[1] for d in chunk]
        rolls = [d[3] for d in chunk]
        true_cls = [d[4] for d in chunk]
        counters = [d[5] for d in chunk]

        with torch.no_grad():
            nf = lambda name: "resid_post" in name  # nur resid_post cachen
            def resid_dict(cache_dict):
                """String-Keys -> {Layer: Tensor}."""
                return {int(name.split(".")[1]): cache_dict[name]
                        for name in cache_dict if "resid_post" in name}
            # clean
            inp = make_batch(seqs_c, pos_c)
            logits_c, cache_c = model.run_with_cache(inp, names_filter=nf, return_cache_object=False)
            rc = resid_dict(cache_c)
            pc = gather_cls_probs(logits_c, pos_c)
            idxs = torch.arange(len(chunk))
            pos_t = torch.tensor(pos_c)
            res_c = {L: rc[L][idxs, pos_t].cpu() for L in range(n_layers)}
            # clean accuracy
            for b, tc in enumerate(true_cls):
                acc_true_clean[1] += 1
                if pc[b].argmax().item() == CLS_ORDER.index(tc):
                    acc_true_clean[0] += 1

            for d2 in range(1, 7):
                repl = [tokenizer.stoi[f"D{d2}"] for _ in chunk]
                inp_m = make_batch(seqs_c, pos_c, replace=repl)
                logits_m, cache_m = model.run_with_cache(inp_m, names_filter=nf, return_cache_object=False)
                rm = resid_dict(cache_m)
                pm = gather_cls_probs(logits_m, pos_c)
                res_m = {L: rm[L][idxs, pos_t].cpu() for L in range(n_layers)}
                # do-over accuracy vs kontrafaktische Klasse
                for b in range(len(chunk)):
                    ccl = CLS_ORDER.index(counters[b][d2 - 1])
                    acc_true_mod[d2][1] += 1
                    if pm[b].argmax().item() == ccl:
                        acc_true_mod[d2][0] += 1
                # Patch pro Layer (auf ORIGINAL-Input, resid_post[L,pos] = Wrong-Roll-Residual)
                for L in range(n_layers):
                    rep = res_m[L].to(device)
                    def hook(act, hook, _L=L, _rep=rep):
                        act = act.clone()
                        act[torch.arange(len(chunk)), torch.tensor(pos_c, device=act.device)] = _rep
                        return act
                    logits_p = model.run_with_hooks(inp, fwd_hooks=[(f"blocks.{L}.hook_resid_post", hook)])
                    pp = gather_cls_probs(logits_p, pos_c).cpu()
                    # Recovery je (b, d2): |pp - pc| / |pm - pc| (L1 über Klassen)
                    for b in range(len(chunk)):
                        dclean = (pm[b].cpu() - pc[b].cpu()).abs().sum().clamp_min(1e-9)
                        dpatch = (pp[b] - pc[b].cpu()).abs().sum()
                        F[L].append((dpatch / dclean).item())
                del cache_m
            del cache_c
        torch.cuda.empty_cache()
        n_eff += len(chunk)
        if n_eff % (8 * B) == 0:
            print(f"  ...{n_eff:,}/{len(decisions):,}")

    # ---------------- Ergebnisse ----------------
    print(f"\n=== 1) Do-over (Würfel-Substitution im selben Kontext) ===")
    print("Modell-Vorhersage unter Roll=d' stimmt mit KONTRAFAKTISCH-korrektem Event "
          "(d') überein:")
    for d2 in range(1, 7):
        h, t = acc_true_mod[d2]
        print(f"  Würfel d'={d2}: {h/t:6.1%}  ({h:,}/{t:,})")
    h, t = acc_true_clean
    print(f"  [Kontrolle] Original d=d': {h/t:6.1%} (Modell sagt TRUE-Event, "
          f"sollte hoch sein)")
    print("  Zufall-Baseline je Klasse ~33% (3 Klassen) / je nach Randverteilung.")

    print(f"\n=== 2) Residual-Patch-Recovery (resid_post[L, Würfel-Position]) ===")
    print("F_L = |patch-clean| / |mod-clean| (0 = kein Effekt, 1 = voller Effekt):")
    for L in range(n_layers):
        fs = F[L]
        import numpy as np
        fs = np.array(fs)
        print(f"  Layer {L}: mean {fs.mean():.2f} | median {np.median(fs):.2f} | "
              f">0.5: {(fs > 0.5).mean():.1%} | p50/p90: {np.percentile(fs,50):.2f}/{np.percentile(fs,90):.2f}")


if __name__ == "__main__":
    main()