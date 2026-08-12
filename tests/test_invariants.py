"""
Invariant- und Konsistenztests für den Simulator.

Diese Tests laufen VOR jeder Datengenerierung im großen Stil. Ziel:
Simulator-Bugs hier fangen, nicht drei Wochen später als vermeintlich
spannendes mechanistisches Ergebnis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.game import play_game
from simulator.policies import POLICIES


def total_money(state_dict: dict) -> int:
    return sum(state_dict["cash"])


def test_money_conserved_across_full_games():
    """Cash-Summe darf sich NUR durch Kauf (Geld fließt aus dem System an
    'die Bank') oder Miete (Geld fließt zwischen Spielern, Summe bleibt gleich)
    ändern. Insgesamt darf Cash + investiertes Kapital (Summe der Kaufpreise
    bereits verkaufter Felder) konstant bleiben."""
    for policy_name in POLICIES:
        log = play_game(game_id=0, seed=42, policy_name=policy_name)

        for i, state in enumerate(log.states):
            spent_on_properties = sum(
                log.prices[field_idx]
                for field_idx, owner in enumerate(state["owner"])
                if owner is not None
            )
            invariant_total = total_money(state) + spent_on_properties
            initial_total = state["current_player"] * 0 + 1000 * log.num_players  # Startkapital
            assert invariant_total == initial_total, (
                f"[{policy_name}] Geld-Invariante verletzt bei Token {i}: "
                f"cash_sum={total_money(state)} + spent={spent_on_properties} "
                f"!= initial={initial_total}"
            )


def test_no_negative_cash():
    for policy_name in POLICIES:
        log = play_game(game_id=1, seed=7, policy_name=policy_name)
        for state in log.states:
            assert all(c >= 0 for c in state["cash"]), f"[{policy_name}] Negativer Cash gefunden"


def test_ownership_valid_range():
    for policy_name in POLICIES:
        log = play_game(game_id=2, seed=99, policy_name=policy_name)
        for state in log.states:
            for owner in state["owner"]:
                assert owner is None or 0 <= owner < log.num_players


def test_events_states_same_length():
    """Jedes Event-Token muss einen zugeordneten State haben -
    Grundvoraussetzung für spätere Activation-to-State-Zuordnung."""
    for policy_name in POLICIES:
        log = play_game(game_id=3, seed=123, policy_name=policy_name)
        assert len(log.events) == len(log.states)
        assert len(log.events) > 0


def test_position_matches_roll_sum_sanity():
    """Position-Sanity-Check (dein Vorschlag aus der Diskussion): bei
    reinem ROLL/MOVE ohne weitere Position-Effekte muss Position exakt
    aus der Summe der Rolls (mod board_size) berechenbar sein.
    Hier direkt am Simulator geprüft, nicht erst am trainierten Modell."""
    log = play_game(game_id=4, seed=55, policy_name="always_buy", board_size=12, num_players=2)

    running_pos = [0, 0]
    i = 0
    while i < len(log.events):
        tok = log.events[i]
        if tok == "START":
            i += 1
            continue
        if tok.startswith("P") and tok[1:].isdigit():
            player = int(tok[1:])
            if i + 1 < len(log.events) and log.events[i + 1] == "ROLL":
                roll = int(log.events[i + 2])
                running_pos[player] = (running_pos[player] + roll) % log.board_size
                actual_pos = log.states[i + 2]["position"][player]
                assert running_pos[player] == actual_pos, (
                    f"Position-Mismatch für P{player}: erwartet {running_pos[player]}, "
                    f"State sagt {actual_pos}"
                )
                i += 3
                continue
        i += 1


if __name__ == "__main__":
    test_money_conserved_across_full_games()
    test_no_negative_cash()
    test_ownership_valid_range()
    test_events_states_same_length()
    test_position_matches_roll_sum_sanity()
    print("Alle Invariant-Tests bestanden.")
