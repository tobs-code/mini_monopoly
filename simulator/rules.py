"""
Regel-Engine für Mini-Monopoly.

Jede apply_*-Funktion nimmt einen GameState, verändert ihn IN-PLACE
und gibt die Liste der Event-Tokens zurück, die diesen Schritt
repräsentieren.

Zentrales Debugging-Prinzip (siehe Diskussion): pro Event-Typ wird
explizit geprüft, dass NUR die dafür erlaubten Zustandskomponenten
sich verändert haben. Das macht Simulator-Bugs sofort lokalisierbar,
statt sie erst als "komisches Interpretability-Ergebnis" drei Wochen
später wiederzufinden.
"""

from __future__ import annotations
from .state import GameState


class RuleViolation(Exception):
    pass


def _snapshot(state: GameState) -> dict:
    return state.to_dict()


def _assert_only_changed(before: dict, after: dict, allowed_keys: set[str], event_name: str):
    """Prüft, dass sich außerhalb von allowed_keys nichts verändert hat."""
    for key in before:
        if key in allowed_keys:
            continue
        if before[key] != after[key]:
            raise RuleViolation(
                f"[{event_name}] veränderte unerlaubt '{key}': {before[key]} -> {after[key]}"
            )


def apply_roll_and_move(state: GameState, player: int, roll: int) -> list[str]:
    """ROLL + MOVE als ein atomarer Schritt: Position ändert sich, sonst nichts."""
    before = _snapshot(state)

    new_pos = (state.position[player] + roll) % state.board_size
    state.position[player] = new_pos

    after = _snapshot(state)
    _assert_only_changed(before, after, allowed_keys={"position"}, event_name="ROLL_MOVE")
    state.check_invariants()

    return [f"P{player}", "ROLL", str(roll)]


def apply_buy(state: GameState, player: int) -> list[str]:
    """Kauf des Feldes, auf dem der Spieler steht. Nur cash + owner ändern sich.
    Setzt voraus: Feld ist unbesetzt, hat einen Preis > 0, Spieler kann sich's leisten."""
    field_idx = state.position[player]
    price = state.prices[field_idx]

    if price == 0:
        raise RuleViolation(f"Feld {field_idx} hat keinen Preis, kann nicht gekauft werden")
    if state.owner[field_idx] is not None:
        raise RuleViolation(f"Feld {field_idx} gehört bereits Spieler {state.owner[field_idx]}")
    if state.cash[player] < price:
        raise RuleViolation(f"Spieler {player} kann sich Feld {field_idx} nicht leisten")

    before = _snapshot(state)

    state.cash[player] -= price
    state.owner[field_idx] = player

    after = _snapshot(state)
    _assert_only_changed(before, after, allowed_keys={"cash", "owner"}, event_name="BUY")
    state.check_invariants()

    return [f"P{player}", "BUY"]


def apply_pass(state: GameState, player: int) -> list[str]:
    """Spieler kauft das Feld bewusst NICHT. Verändert nichts am State,
    ist aber trotzdem ein eigenes Event-Token (wichtig fürs History-only-Modell:
    sonst kann es 'kein Kauf' nicht von 'Kauf unmöglich, da kein Preisfeld' unterscheiden)."""
    before = _snapshot(state)
    after = _snapshot(state)
    _assert_only_changed(before, after, allowed_keys=set(), event_name="PASS")
    return [f"P{player}", "PASS"]


def apply_rent(state: GameState, player: int) -> list[str]:
    """Miete zahlen für das Feld, auf dem der Spieler steht (falls fremdes Eigentum).
    Nur cash ändert sich (bei beiden Spielern), owner/position bleiben fix."""
    field_idx = state.position[player]
    owner_idx = state.owner[field_idx]

    if owner_idx is None or owner_idx == player:
        raise RuleViolation(f"Kein Mietfall auf Feld {field_idx} für Spieler {player}")

    rent = int(state.prices[field_idx] * state.rent_rate)
    rent = min(rent, state.cash[player])  # keine Schulden in V0 (siehe check_invariants: cash >= 0)

    before = _snapshot(state)

    state.cash[player] -= rent
    state.cash[owner_idx] += rent

    after = _snapshot(state)
    _assert_only_changed(before, after, allowed_keys={"cash"}, event_name="PAY_RENT")
    state.check_invariants()

    return [f"P{player}", "PAY_RENT", str(rent)]


def advance_turn(state: GameState) -> None:
    """Nur Turn-Zähler und current_player weiterschalten. Kein Event-Token,
    das ist reine Buchhaltung außerhalb der eigentlichen Spiellogik."""
    state.turn += 1
    state.current_player = (state.current_player + 1) % state.num_players
