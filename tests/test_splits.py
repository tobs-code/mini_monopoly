"""
Tests für die Held-out-Split-Logik (data/splits.py).

Verifiziert die Split-Regeln aus EXPERIMENT.md §3:
  1. Trajectory Integrity: keine überlappenden game_ids zwischen train/test.
  2. Kein Leakage: held-out Bedingung (S1-Paare / S2-Relation) erscheint
     NICHT im Trainings-Split.
  3. S2-Mindest-N: unterbesetzte Relationen werden abgelehnt, nicht als Split
     verwendet (§3a).
  4. S0: IID-Sanity-Split erzeugt train+test mit korrekter Aufteilung.
"""

import sys
import os
import glob
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.splits import (
    load_games,
    decision_points,
    split_s0,
    split_s1,
    split_s2,
    coverage_report,
    check_s2_viability,
    s2_game_impact,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "data",
                       "games_so0_n200_always_buy-conservative-random.jsonl")


def _game_ids(games):
    return {g["game_id"] for g in games}


def test_no_overlap_train_test():
    """Trajectory Integrity: eine Partie gehört zu genau einem Split."""
    games = load_games(FIXTURE)
    train, test = split_s0(games)
    assert not (_game_ids(train) & _game_ids(test)), "game_id-Überlappung in S0"

    pairs = {(500, 150), (600, 200)}
    train1, test1 = split_s1(games, pairs)
    assert not (_game_ids(train1) & _game_ids(test1)), "game_id-Überlappung in S1"

    train2, test2 = split_s2(games, {700})
    assert not (_game_ids(train2) & _game_ids(test2)), "game_id-Überlappung in S2"


def test_s1_no_leakage():
    """Held-out (cash, price)-Paare dürfen nicht im Trainings-Split stehen."""
    games = load_games(FIXTURE)
    pairs = {(500, 150), (600, 200)}
    train, _ = split_s1(games, pairs)
    leaks = [p for g in train for p in decision_points(g)
             if (p["cash"], p["price"]) in pairs]
    assert not leaks, f"S1-Leakage: {len(leaks)} held-out Paare im Training"


def test_s2_no_leakage():
    """Held-out Relation darf nicht im Trainings-Split stehen."""
    games = load_games(FIXTURE)
    from collections import Counter
    pts = [p for g in games for p in decision_points(g)]
    rel = Counter(p["relation"] for p in pts).most_common(1)[0][0]
    train, test = split_s2(games, {rel})
    leaks = [p for g in train for p in decision_points(g) if p["relation"] == rel]
    assert not leaks, f"S2-Leakage: {len(leaks)} relation={rel} im Training"
    assert any(p["relation"] == rel for g in test for p in decision_points(g)), \
        "S2-Test enthält die held-out Relation nicht"


def test_s2_viability():
    """Mindest-N (siehe §3a): check_s2_viability und s2_game_impact konsistent.

    Nutzt dynamisch Relationen aus der Fixtur statt hart an konkrete
    Zahlen gebunden zu sein, die sich mit Datensatzgröße verschieben."""
    games = load_games(FIXTURE)
    from collections import Counter
    pts = [p for g in games for p in decision_points(g)]
    relations = sorted({p["relation"] for p in pts})
    assert relations, "Fixtur sollte Entscheidungspunkte enthalten"

    # Sehr hohe Schwelle: keine Relation erfüllt sie -> nicht viable
    for r in relations:
        viable, _ = check_s2_viability(games, r, min_n=10 ** 9)
        assert not viable, f"relation={r} darf bei min_n=1e9 nicht viable sein"

    # Häufigste Relation: bei niedrigerer Schwelle viable, s2_game_impact konsistent
    top_rel, top_cnt = Counter(p["relation"] for p in pts).most_common(1)[0]
    viable, cnt = check_s2_viability(games, top_rel, min_n=top_cnt - 1)
    assert viable and cnt == top_cnt, "Top-Relation sollte bei niedriger Schwelle viable sein"

    affected, total = s2_game_impact(games, top_rel)
    assert affected >= 1 and total == len(games)
    assert affected <= total


def test_decision_point_pre_state():
    """Pre-Buy-Zustand: bei BUY muss states[i-2] (Roll-Wert-Token) den Cash
    VOR der Kaufreduktion tragen. Verifiziert den empirisch gefundenen Fix
    (states[i-1] wäre bereits Post-Buy und systematisch falsch)."""
    games = load_games(FIXTURE)
    for log in games:
        events, states, prices = log["events"], log["states"], log["prices"]
        for i, tok in enumerate(events):
            if i < 2 or tok != "BUY":
                continue
            pre = states[i - 2]
            post = states[i]
            player = pre["current_player"]
            pos = pre["position"][player]
            price = prices[pos]
            assert post["cash"][player] == pre["cash"][player] - price, (
                f"BUY-Pre-State verletzt game={log['game_id']} idx={i}: "
                f"pre={pre['cash'][player]} post={post['cash'][player]} price={price}")


def test_s0_partition():
    """S0 erzeugt beide Splits; Test-Anteil nahe test_frac."""
    games = load_games(FIXTURE)
    train, test = split_s0(games, test_frac=0.2, seed=42)
    total = len(train) + len(test)
    assert total == len(games), "S0 verliert/dupliziert Partien"
    assert abs(len(test) / total - 0.2) < 0.1


def test_coverage_report_runs():
    """Coverage-Bericht läuft und liefert konsistente Kennzahlen."""
    games = load_games(FIXTURE)
    rep = coverage_report(games)
    assert rep["n_games"] == len(games)
    assert rep["n_decision_points"] == rep["n_buy"] + rep["n_pass"]
    assert rep["cash_min"] >= 0
    assert rep["n_distinct_relations"] > 0


if __name__ == "__main__":
    test_no_overlap_train_test()
    test_s1_no_leakage()
    test_s2_no_leakage()
    test_s2_viability()
    test_decision_point_pre_state()
    test_s0_partition()
    test_coverage_report_runs()
    print("Alle Split-Invariant-Tests bestanden.")
