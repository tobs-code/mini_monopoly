"""
S1-Design-Search / Coverage-Scanner (7d-1, §6 Schritt 7d-1).

Problem (7d-Befund): Das bisherige S1-2x2-Design `cash in {150,250} x price in
{150,300}` ist mit dem Simulator nicht sauber identifizierbar: Der
held-out-Ausschluss (Spiele MIT einem held-out-Paar fliegen aus dem Training)
dezimiert die getesteten Einzelwerte (cash=150/250), weil diese fast immer mit
den held-out-Preisen (150/300) auftreten -> die Einzelwert-Basis bricht weg.

Dieses Skript rankt Kandidaten-2x2-Designs (cash-Paar x price-Paar) nach
Coverage-Güte. Für jedes Design:
  [A] held-out-Kombinationen im Trainingskorpus: exakt 0 (per Konstruktion)
      => Spiele, die ein Paar aus dem Design enthalten, werden ausgeschlossen.
  [B] Einzelwert-Coverage NACH Ausschluss (Kern des 7d-Problems):
      min over {cash1, cash2} und {price1, price2} der verbleibenden
      Trainingspunkte mit genau diesem Wert. Je hoeher, desto besser
      (das Modell muss die Einzelwerte im Training reichlich gesehen haben,
      sonst ist der Kompositionstest unfair).
  [C] Testregion: n Punkte der exakten held-out-Paare in der Eval-Region.
  [D] beide Klassen (afford/unafford) in der Testregion vorhanden.
  [E] mindestens ein grenznahes Paar (|cash-price| <= 50) unter den held-out.
  [F] Konfund: Policy/Player/Position-Verteilung in der Testregion (Info).

Kriterien (User-Vorgabe, Pre-Registration):
  held-out = 0 im Training, Einzelwerte N > X, Test-N > Y, beide Klassen,
  grenznahes Paar, Konfund nicht schief.

CLI:
    python tools/s1_design_search.py \
        --train data/games_so0_n2000_always_buy-conservative-random.jsonl \
               data/games_so9001_n800_always_buy-conservative-random_sc400.jsonl \
               data/games_so7001_n800_always_buy-conservative-random_sc300.jsonl \
               data/games_so7002_n1600_always_buy-conservative-random_sc300.jsonl \
        --eval  data/games_so9001_n800_always_buy-conservative-random_sc400.jsonl \
        --min-einzel 150 --min-test 150 --top 25

WICHTIG: Der Scanner ist ein reiner Datenqualitaets-Check (kein Training). Er
wiederholt exakt die Ausschluss-Logik von data/merge_corpora.py
(--exclude-held-out) und die Testregion-Logik von probes/probe_afford.py
(split_s1_indices).
"""

from __future__ import annotations
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.splits import load_games, decision_points


def load_point_sets(path: str) -> tuple[list[dict], list[set]]:
    """Lädt Spiele und pro Spiel das Set der (cash, price)-Tupel (nur BUY/PASS)."""
    games = load_games(path)
    per_game = []
    for g in games:
        pts = decision_points(g)
        s = set()
        for p in pts:
            s.add((p["cash"], p["price"]))
        per_game.append((g, pts, s))
    return games, per_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True,
                    help="Trainingskorpi (Quellen des Mixes)")
    ap.add_argument("--eval", required=True,
                    help="Eval-Korpus, auf dem die S1-Testregion definiert wird")
    ap.add_argument("--min-einzel", type=int, default=150,
                    help="min. Trainingspunkte je Einzelwert nach Ausschluss (X)")
    ap.add_argument("--min-test", type=int, default=150,
                    help="min. Testpunkte in der held-out-Region (Y)")
    ap.add_argument("--top", type=int, default=25, help="wie viele Designs zeigen")
    ap.add_argument("--max-cash-kandidaten", type=int, default=40,
                    help="so viele haeufigste cash-Werte als Kandidaten nutzen")
    ap.add_argument("--min-pair", type=int, default=100,
                    help="min. Testpunkte je held-out-Paar (Balance, sonst 1x2)")
    ap.add_argument("--min-unaff-share", type=float, default=0.15,
                    help="min. Anteil unaffordable in der Testregion")
    args = ap.parse_args()

    print("Lade Trainingskorpi ...")
    train_pts = []
    train_sets = []
    train_games = 0
    train_policy_of = {}
    for p in args.train:
        games, per_game = load_point_sets(p)
        train_games += len(games)
        for g, pts, s in per_game:
            train_pts.append((g, pts, s))
            train_sets.append(s)
            train_policy_of[id(g)] = g.get("policy_name", "?")
    print(f"  {len(args.train)} Korpi, {train_games} Spiele, "
          f"{sum(len(pts) for _, pts, _ in train_pts)} Entscheidungspunkte")

    print(f"Lade Eval-Korpus {os.path.basename(args.eval)} ...")
    _, eval_per_game = load_point_sets(args.eval)
    eval_games, eval_pts, eval_sets = zip(*eval_per_game)
    print(f"  {len(eval_games)} Spiele, {sum(len(p) for p in eval_pts)} Entscheidungspunkte")

    # Kandidaten: price-Werte aus der Eval-Preisverteilung, cash-Werte aus der
    # EVAL-Verteilung (nur Werte, die im Eval-Korpus tatsaechlich vorkommen,
    # sonst degenerieren Paare: cash=850 aus cap1000 existiert im sc400-Eval
    # praktisch nicht -> effektiv 1x2 statt 2x2).
    eval_cash = Counter(p["cash"] for _, pts, _ in eval_per_game for p in pts)
    eval_price = Counter(p["price"] for _, pts, _ in eval_per_game for p in pts)
    cash_pool = [v for v, _ in eval_cash.most_common(args.max_cash_kandidaten)]
    price_pool = sorted(eval_price.keys())
    print(f"\nKandidaten-Pool (aus Eval-Verteilung): cash=({len(cash_pool)} Werte) "
          f"price={price_pool}")

    # Precompute: Trainingspunkt-Indexe je cash-Wert und je price-Wert, pro Spiel.
    # Wir brauchen pro Design die Punkte der NICHT-exkludierten Spiele.
    # Exklusion pro Design: Spiel mit irgendeinem (c,p) in Design-Paaren fliegt raus.
    results = []
    from itertools import combinations
    for c1, c2 in combinations(cash_pool, 2):
        if c1 > c2:
            c1, c2 = c2, c1
        for p1, p2 in combinations(price_pool, 2):
            pairs = [(c1, p1), (c1, p2), (c2, p1), (c2, p2)]
            pair_set = set(pairs)

            # [A] Spiele mit held-out-Paar ausschliessen (merge_corpora-Logik)
            keep = [i for i, s in enumerate(train_sets) if not (s & pair_set)]
            if not keep:
                continue

            # [B] Einzelwert-Coverage nach Ausschluss
            cnt_c1 = cnt_c2 = cnt_p1 = cnt_p2 = 0
            for i in keep:
                for p in train_pts[i][1]:
                    c, pr = p["cash"], p["price"]
                    if c == c1:
                        cnt_c1 += 1
                    elif c == c2:
                        cnt_c2 += 1
                    if pr == p1:
                        cnt_p1 += 1
                    elif pr == p2:
                        cnt_p2 += 1
            min_cash = min(cnt_c1, cnt_c2)
            min_price = min(cnt_p1, cnt_p2)
            min_einzel = min(min_cash, min_price)

            # [C] Testregion in der Eval-Region (probe_afford split_s1-Logik):
            #     Spiele MIT held-out-Paar -> Test; die exakten Paare = Testregion.
            #     Per-Paar-Zaehlung (damit das 2x2 nicht zu 1x2 degeneriert).
            test_game_idx = [i for i, s in enumerate(eval_sets) if s & pair_set]
            per_pair = {pr: 0 for pr in pairs}
            n_held = 0
            aff = 0
            unaff = 0
            for i in test_game_idx:
                for p in eval_pts[i]:
                    if (p["cash"], p["price"]) in pair_set:
                        per_pair[(p["cash"], p["price"])] += 1
                        n_held += 1
                        if p["cash"] >= p["price"]:
                            aff += 1
                        else:
                            unaff += 1
            if n_held < args.min_test:
                continue
            # [D] beide Klassen vorhanden?
            if aff == 0 or unaff == 0:
                continue
            # [D2] jedes der 4 Paare in der Testregion vertreten (sonst degeneriert 2x2)
            if any(n == 0 for n in per_pair.values()):
                continue
            # [D3] Balance: jedes Paar mind. min_pair Testpunkte
            if min(per_pair.values()) < args.min_pair:
                continue
            # [D4] unaffordable-Anteil nicht trivial klein
            if unaff / n_held < args.min_unaff_share:
                continue
            # [E] grenznahes Paar (|diff|<=50) unter den held-out-Paaren?
            near = any(abs(c - p) <= 50 for c, p in pairs)

            results.append({
                "design": (c1, c2, p1, p2), "pairs": pairs,
                "per_pair": per_pair,
                "n_excluded_games": len(train_sets) - len(keep),
                "n_train_games": len(keep),
                "min_cash": min_cash, "min_price": min_price,
                "min_einzel": min_einzel,
                "min_pair_test": min(per_pair.values()),
                "n_held_test": n_held, "aff": aff, "unaff": unaff,
                "near": near,
            })

    # [F] Konfund fuer die besten Designs (Policy/Player/Position in Testregion)
    def confound_of(res):
        c1, c2, p1, p2 = res["design"]
        pair_set = set(res["pairs"])
        pol = Counter()
        ply = Counter()
        pos = Counter()
        for i, s in enumerate(eval_sets):
            if s & pair_set:
                g = eval_games[i]
                for p in eval_pts[i]:
                    if (p["cash"], p["price"]) in pair_set:
                        pol[g.get("policy_name", "?")] += 1
                        ply[p["player"]] += 1
                        pos[p["position"]] += 1
        n = sum(pol.values())
        # Schiefe: max-Policy-Anteil (1.0 = komplett eine Policy)
        pol_skew = max(pol.values()) / n if n else 0.0
        pos_top = pos.most_common(3)
        return pol, dict(sorted(ply.items())), pos_top, pol_skew

    # Ranken: Filter (beide Kriterien erfüllbar) + Sortierung nach min_einzel,
    # dann n_held_test, dann grenznah bevorzugt.
    # Konfund-Filter: alle 3 Policies in der Testregion vertreten, keine
    # einzelne Policy dominiert (<=70%), Positionen nicht extrem schief (<=50%).
    def konfund_ok(res):
        pol, ply, pos_top, skew = confound_of(res)
        n_total = sum(n for _, n in pos_top)
        n_pol = len(pol)
        max_pos_share = max(n for _, n in pos_top) / n_total if n_total else 0.0
        return (n_pol >= 3 and skew <= 0.70 and max_pos_share <= 0.50,
                n_pol, skew, max_pos_share)

    eligible = [r for r in results if r["min_einzel"] >= args.min_einzel]
    ok, _, _, _ = zip(*[konfund_ok(r) for r in eligible]) if eligible else ([],)
    eligible_ok = [r for r, k in zip(eligible, ok) if k]
    eligible.sort(key=lambda r: (-r["min_einzel"], -r["n_held_test"], -r["near"]))

    print(f"\n===== Kandidaten-Designs (X={args.min_einzel}, Y={args.min_test}, "
          f"min_pair={args.min_pair}, unaff>={args.min_unaff_share:.0%}) =====")
    print(f"{len(results)} Designs nach Testregion>=Y, beiden Klassen, 4 Paaren, "
          f"Balance & unaff-Anteil; "
          f"davon {len(eligible)} mit min_einzel>=X, "
          f"{len(eligible_ok)} davon konfund-ok\n")
    if not eligible_ok:
        print("KEINE Designs erfüllen Einzelwert-Schwelle X + Konfund-Kriterium. Beste Kandidaten:")
        for r in eligible[:args.top]:
            pol, ply, pos_top, skew = confound_of(r)
            print(f"  {r['design']}: min_einzel={r['min_einzel']} "
                  f"(cash {r['min_cash']}/{r['min_price']}), "
                  f"test={r['n_held_test']} (aff {r['aff']}/unaff {r['unaff']}), "
                  f"min_pair={r['min_pair_test']}, near={r['near']}, "
                  f"policy={len(pol)} pol, skew={skew:.2f}")
        sys.exit(1)

    for r in eligible_ok[:args.top]:
        pol, ply, pos_top, skew = confound_of(r)
        print(f"--- {r['design']}  (cash {r['min_cash']} / price {r['min_price']}, "
              f"min_einzel={r['min_einzel']})")
        print(f"    held-out: n_test={r['n_held_test']} (aff {r['aff']} / unaff {r['unaff']}), "
              f"min_pair={r['min_pair_test']}, grenznah={r['near']}")
        print(f"    per-pair: {r['per_pair']}")
        print(f"    training: {r['n_train_games']} Spiele behalten, "
              f"{r['n_excluded_games']} exkludiert")
        print(f"    konfund: policy={dict(pol)} player={ply} pos_top={pos_top} "
              f"(max_policy_share={skew:.2f})")


if __name__ == "__main__":
    main()
