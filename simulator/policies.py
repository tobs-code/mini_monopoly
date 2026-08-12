"""
Kaufentscheidungs-Policies.

Warum mehrere: Wenn der Simulator immer nach derselben Heuristik kauft,
kann eine spätere SAE auf Artefakte dieser einen Heuristik overfitten,
statt die zugrundeliegende Zustandsstruktur zu lernen (siehe Diskussion
zu Trainingsdatenvielfalt). Deshalb mehrere Policies, die per Seed/Config
gemischt werden können.
"""

from __future__ import annotations
import random
from .state import GameState


def policy_always_buy(state: GameState, player: int, rng: random.Random) -> bool:
    """Kauft immer, wenn leistbar."""
    field_idx = state.position[player]
    price = state.prices[field_idx]
    return price > 0 and state.owner[field_idx] is None and state.cash[player] >= price


def policy_conservative(state: GameState, player: int, rng: random.Random) -> bool:
    """Kauft nur, wenn danach noch mind. 300 Cash übrig bleiben."""
    field_idx = state.position[player]
    price = state.prices[field_idx]
    if price == 0 or state.owner[field_idx] is not None:
        return False
    return state.cash[player] - price >= 300


def policy_random(state: GameState, player: int, rng: random.Random) -> bool:
    """Kauft zufällig (wenn möglich), 50/50."""
    field_idx = state.position[player]
    price = state.prices[field_idx]
    if price == 0 or state.owner[field_idx] is not None or state.cash[player] < price:
        return False
    return rng.random() < 0.5


POLICIES = {
    "always_buy": policy_always_buy,
    "conservative": policy_conservative,
    "random": policy_random,
}
