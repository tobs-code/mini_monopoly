"""
S1–S2 Held-out-Split-Logik für Mini-Monopoly (siehe EXPERIMENT.md §3, §3a).

Prinzipien:
  - Split auf Ground-Truth-Zuständen/-Transitionen, nicht auf der Tokenfolge.
  - Jede Partie gehört vollständig zu genau einem Split (Trajectory Integrity).
  - Analyse-Einheit für S1/S2 sind die Kauf-Entscheidungspunkte (BUY/PASS).
    Die Affordability-Relation eines Entscheidungspunkts ist `cash - price`
    im Zustand VOR der Entscheidung (states[i-1]): Position bereits aktualisiert,
    Cash noch unverändert.
  - S3 entfällt in V0 (§3a).

Splits:
  S0 — IID-Sanity-Baseline: zufällige held-out Partien aus derselben Verteilung.
  S1 — Novel State Composition: Einzelwerte bekannt, bestimmte (cash, price)-
        Kombinationen held out.
  S2 — Novel Relation/Transition: bestimmte Werte der Relation (cash - price)
        held out, mit Mindest-N-Schwelle pro Relation.
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_games(jsonl_path: str) -> list[dict]:
    """Lädt eine generierte JSONL-Datei (ein GameLog-dict pro Zeile)."""
    games = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


def decision_points(log: dict) -> list[dict]:
    """Extrahiert die Kauf-Entscheidungspunkte (BUY/PASS) eines Spiels.

    WICHTIG (empirisch verifiziert): Der Pre-Buy-Zustand liegt NICHT bei
    states[i-1], sondern beim Roll-Wert-Token. Grund: `apply_buy`/`apply_pass`
    geben ["P{player}", "BUY|PASS"] zurück, und BEIDE Tokens bekommen den
    bereits veränderten Post-Buy-State (bei BUY: Cash schon um den Preis
    reduziert). Der letzte State, in dem Position feststeht und Cash noch
    unverändert ist, ist der des Roll-Wert-Tokens direkt vor dem
    Entscheidungsblock — also states[i-2] (Struktur: ... P ROLL w P BUY|PASS).
    Für PASS ist states[i-1] zufällig gleich (Cash ändert sich nicht), aber
    für BUY wäre states[i-1] bereits der reduzierte Cash -> systematisch falsch.

    Robustheit: Wir gehen nicht hart von i-2 aus, sondern suchen rückwärts vom
    Entscheidungs-Token das erste Token, das ein Roll-Wert ist (Würfel-Zahl),
    dessen State den Pre-Buy-Zustand trägt.

    Rückgabe: Liste von dicts mit game_id, token_idx, decision ('BUY'|'PASS'),
    player, position, price, cash, relation.
    """
    events = log["events"]
    states = log["states"]
    prices = log["prices"]
    points = []
    for i, tok in enumerate(events):
        if tok in ("BUY", "PASS"):
            pre = _pre_decision_state(events, states, i)
            player = pre["current_player"]
            cash = pre["cash"][player]
            pos = pre["position"][player]
            price = prices[pos]
            points.append({
                "game_id": log["game_id"],
                "token_idx": i,
                "decision": tok,
                "player": player,
                "position": pos,
                "price": price,
                "cash": cash,
                "relation": cash - price,
            })
    return points


def _pre_decision_state(events: list, states: list, idx: int) -> dict:
    """Findet den Pre-Entscheidungs-Zustand für ein BUY/PASS-Event an Index idx.

    Scannt rückwärts vom Entscheidungs-Token und liefert den State des ersten
    Roll-Wert-Tokens (nicht-Event, nicht-Player-Token, eine Würfel-Zahl).
    Fallback auf states[idx-1] als letzte Sicherung (sollte nie eintreten).
    """
    for j in range(idx - 1, -1, -1):
        e = events[j]
        if e.isdigit():           # Würfel-Wert (1..6)
            return states[j]
        if e in ("BUY", "PASS"):  # niemals zwei Entscheidungen hintereinander
            break
    return states[idx - 1]


def coverage_report(games: list[dict]) -> dict:
    """Coverage-Statistiken über den Datensatz (siehe §3 'Coverage-Statistiken')."""
    all_cash = []
    all_points = []
    pair_counter = Counter()
    for log in games:
        all_cash.extend(c for s in log["states"] for c in s["cash"])
        for p in decision_points(log):
            all_points.append(p)
            pair_counter[(p["cash"], p["price"])] += 1

    cash_sorted = sorted(all_cash)
    n = len(cash_sorted)

    def quantile(q):
        idx = round((n - 1) * q)
        return cash_sorted[idx]

    relations = Counter(p["relation"] for p in all_points)
    report = {
        "n_games": len(games),
        "n_states": len(all_cash),
        "n_decision_points": len(all_points),
        "n_buy": sum(1 for p in all_points if p["decision"] == "BUY"),
        "n_pass": sum(1 for p in all_points if p["decision"] == "PASS"),
        "cash_min": cash_sorted[0],
        "cash_max": cash_sorted[-1],
        "cash_q10": quantile(0.1),
        "cash_q25": quantile(0.25),
        "cash_med": quantile(0.5),
        "cash_q75": quantile(0.75),
        "cash_q90": quantile(0.9),
        "cash_q99": quantile(0.99),
        "n_distinct_relations": len(relations),
        "n_distinct_pairs": len(pair_counter),
        "relation_counts": {str(k): v for k, v in relations.most_common()},
    }
    return report


def _partition_by_condition(games: list[dict], test_condition) -> tuple[list[dict], list[dict]]:
    """Teilt Partien in train/test nach einer Bedingung über Entscheidungspunkte.

    test_condition(point) -> bool. Eine Partie landet im Test-Split, wenn sie
    mindestens einen Entscheidungspunkt mit test_condition == True hat. Dadurch
    bleiben Train-Partien frei von der held-out Bedingung (keine Leakage) und
    jede Partie gehört vollständig zu einem Split.
    """
    train, test = [], []
    for log in games:
        pts = decision_points(log)
        if any(test_condition(p) for p in pts):
            test.append(log)
        else:
            train.append(log)
    return train, test


def split_s0(games: list[dict], test_frac: float = 0.2, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """S0: zufällige held-out Partien (IID-Sanity-Baseline)."""
    rng = random.Random(seed)
    shuffled = list(games)
    rng.shuffle(shuffled)
    n_test = max(1, int(len(shuffled) * test_frac))
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test


def split_s1(games: list[dict], held_out_pairs: set[tuple[int, int]]) -> tuple[list[dict], list[dict]]:
    """S1: held-out (cash, price)-Kombinationen. Einzelwerte bleiben bekannt,
    nur die Kombinationen sind held out."""
    return _partition_by_condition(games, lambda p: (p["cash"], p["price"]) in held_out_pairs)


def split_s2(games: list[dict], held_out_relations: set[int]) -> tuple[list[dict], list[dict]]:
    """S2: held-out Werte der Affordability-Relation (cash - price)."""
    return _partition_by_condition(games, lambda p: p["relation"] in held_out_relations)


def check_s2_viability(games: list[dict], candidate_relation: int, min_n: int = 200) -> tuple[bool, int]:
    """Prüft, ob eine Kandidaten-Relation für S2 genug Vorkommen hat (Mindest-N)."""
    count = sum(1 for p in decision_points_for(games) if p["relation"] == candidate_relation)
    return count >= min_n, count


def s2_game_impact(games: list[dict], relation: int) -> tuple[int, int]:
    """Wie viele Partien wandern bei diesem held-out durch Trajectory-Integrity
    komplett in den Test-Split? Gibt (n_affected_games, n_total_games) zurück.

    Reine Häufigkeit auf Entscheidungspunkt-Ebene reicht als S2-Auswahlkriterium
    nicht: Eine häufige Relation (z.B. 700) tritt in einem Großteil aller Partien
    mindestens einmal auf und würde per Trajectory-Integrity das Trainingsset
    fast vollständig in den Test ziehen. Deshalb zusätzlich den Partien-Impact
    prüfen, bevor eine Relation als S2-Testrelation festgelegt wird.
    """
    affected = sum(
        1 for log in games
        if any(p["relation"] == relation for p in decision_points(log))
    )
    return affected, len(games)


def decision_points_for(games: list[dict]) -> list[dict]:
    out = []
    for log in games:
        out.extend(decision_points(log))
    return out


def save_split(train: list[dict], test: list[dict], output_dir: str, tag: str,
               config: dict | None = None) -> tuple[str, str]:
    """Schreibt train/test als JSONL + Manifest und gibt die Pfade zurück.

    Manifest (Kernprinzip §8): jeder Split muss eindeutig seiner erzeugenden
    Config zuordenbar sein (welche Paare/Relationen/min_n/Seed/test_frac).
    """
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, f"train_{tag}.jsonl")
    test_path = os.path.join(output_dir, f"test_{tag}.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for log in train:
            f.write(json.dumps(log) + "\n")
    with open(test_path, "w", encoding="utf-8") as f:
        for log in test:
            f.write(json.dumps(log) + "\n")

    if config is not None:
        manifest = dict(config)
        manifest["train_file"] = os.path.basename(train_path)
        manifest["test_file"] = os.path.basename(test_path)
        manifest["n_train_games"] = len(train)
        manifest["n_test_games"] = len(test)
        manifest_path = os.path.join(output_dir, f"{tag}.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return train_path, test_path


def _print_coverage(report: dict) -> None:
    print("=== Coverage-Bericht ===")
    for k in ("n_games", "n_states", "n_decision_points", "n_buy", "n_pass"):
        print(f"  {k}: {report[k]}")
    print(f"  cash: min={report['cash_min']} max={report['cash_max']} "
          f"q10={report['cash_q10']} q25={report['cash_q25']} med={report['cash_med']} "
          f"q75={report['cash_q75']} q90={report['cash_q90']} q99={report['cash_q99']}")
    print(f"  distinct relations: {report['n_distinct_relations']} "
          f"(cash-price), distinct pairs: {report['n_distinct_pairs']}")
    print("  Top-15 Relationen (cash - price):")
    for rel, cnt in list(report["relation_counts"].items())[:15]:
        print(f"    {rel}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-Monopoly Held-out-Split-Logik")
    parser.add_argument("--input", required=True, help="Pfad zur generierten JSONL-Datei")
    parser.add_argument("--output-dir", default="data/splits")
    parser.add_argument("--coverage", action="store_true", help="Nur Coverage-Bericht ausgeben")
    parser.add_argument("--build", action="store_true", help="Splits bauen (siehe --s1-pairs/--s2-relations)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--s1-pairs", nargs="*", default=None,
                        help="Held-out (cash,price)-Paare, Format '500,150' '400,200'")
    parser.add_argument("--s2-relations", nargs="*", type=int, default=None,
                        help="Held-out Relationen (cash-price), z.B. 325 175")
    parser.add_argument("--s2-min-n", type=int, default=200,
                        help="Mindest-N pro S2-Relation (siehe §3a)")
    args = parser.parse_args()

    games = load_games(args.input)

    if args.coverage or not args.build:
        report = coverage_report(games)
        _print_coverage(report)
        if not args.build:
            return

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    # S0
    train0, test0 = split_s0(games, args.test_frac, args.seed)
    t0p, s0p = save_split(train0, test0, args.output_dir, f"{base}_S0",
                          config={"split": "S0", "type": "iid_baseline",
                                  "test_frac": args.test_frac, "seed": args.seed})
    print(f"\nS0: {len(train0)} train / {len(test0)} test -> {s0p}")

    # S1
    if args.s1_pairs:
        pairs = set()
        for item in args.s1_pairs:
            c, p = item.split(",")
            pairs.add((int(c), int(p)))
        train1, test1 = split_s1(games, pairs)
        t1p, s1p = save_split(train1, test1, args.output_dir, f"{base}_S1",
                              config={"split": "S1", "type": "novel_state_composition",
                                      "held_out_pairs": sorted(pairs, key=lambda x: (x[0], x[1]))})
        print(f"S1 (pairs={sorted(pairs)}): {len(train1)} train / {len(test1)} test -> {s1p}")
        held_in_test = sum(1 for log in test1
                           for p in decision_points(log) if (p["cash"], p["price"]) in pairs)
        print(f"    held-out pairs in test: {held_in_test} Vorkommen")

    # S2
    if args.s2_relations:
        ok_rels, rejected = set(), set()
        for r in args.s2_relations:
            viable, cnt = check_s2_viability(games, r, args.s2_min_n)
            if not viable:
                rejected.add(r)
                print(f"S2-Kandidat relation={r}: {cnt} Vorkommen (min_n={args.s2_min_n}) "
                      f"-> UNTERBESETZT, wird NICHT als S2-Split verwendet (§3a)")
                continue
            affected, total = s2_game_impact(games, r)
            frac = affected / total if total else 0.0
            print(f"S2-Kandidat relation={r}: {cnt} Vorkommen (min_n={args.s2_min_n}) -> OK | "
                  f"Trajectory-Impact: {affected}/{total} Partien ({frac:.1%})")
            if frac > 0.35:
                print(f"    -> WARNUNG: >35% der Partien betroffen; "
                      f"Trainingsset würde stark schrumpfen (siehe §3a), eher seltenere Relation wählen")
            else:
                ok_rels.add(r)
        if rejected and not ok_rels:
            print(f"Keine S2-Relation erreicht min_n={args.s2_min_n}. "
                  f"Konsequenz (§3a): mehr Spiele generieren, nicht Schwelle aufweichen.")
        if ok_rels:
            train2, test2 = split_s2(games, ok_rels)
            t2p, s2p = save_split(train2, test2, args.output_dir, f"{base}_S2",
                                  config={"split": "S2", "type": "novel_relation_composition",
                                          "held_out_relations": sorted(ok_rels),
                                          "min_n": args.s2_min_n})
            print(f"S2 (rels={sorted(ok_rels)}): {len(train2)} train / {len(test2)} test -> {s2p}")


if __name__ == "__main__":
    main()
