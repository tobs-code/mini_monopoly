"""
Ground-Truth-Zustand des Mini-Monopoly-Spiels.

Design-Prinzip: GameState ist die einzige Quelle der Wahrheit.
Jeder Event-Schritt im Log bekommt eine tiefe Kopie dieses Zustands
zugeordnet, damit spätere Interpretability-Arbeit (Probing, SAEs,
Interventionen) jede Aktivierung exakt auf einen bekannten Zustand
zurückführen kann.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional


@dataclass
class GameState:
    board_size: int
    num_players: int
    prices: list[int]              # prices[field_idx] -> Kaufpreis (0 = kein Grundstück, z.B. Start/Ereignisfeld)
    rent_rate: float = 0.5         # Miete = rent_rate * price
    start_capital: int = 1000      # Startkapital jedes Spielers (Diagnose-Datensatz: niedriger, s. §6b)

    position: list[int] = field(default_factory=list)   # position[player] -> Feldindex
    cash: list[int] = field(default_factory=list)        # cash[player] -> Kontostand
    owner: list[Optional[int]] = field(default_factory=list)  # owner[field_idx] -> player_idx oder None

    turn: int = 0
    current_player: int = 0

    def __post_init__(self):
        if not self.position:
            self.position = [0] * self.num_players
        if not self.cash:
            self.cash = [self.start_capital] * self.num_players
        if not self.owner:
            self.owner = [None] * self.board_size

    def clone(self) -> "GameState":
        return deepcopy(self)

    def to_dict(self) -> dict:
        """Vollständiger Ground-Truth-Snapshot als reines dict (JSON-serialisierbar)."""
        return {
            "turn": self.turn,
            "current_player": self.current_player,
            "position": list(self.position),
            "cash": list(self.cash),
            "owner": list(self.owner),
        }

    def check_invariants(self) -> None:
        """Wirft AssertionError, sobald der Zustand inkonsistent ist.
        Wird nach JEDEM angewendeten Event aufgerufen (siehe rules.py)."""
        assert len(self.position) == self.num_players
        assert len(self.cash) == self.num_players
        assert len(self.owner) == self.board_size

        for c in self.cash:
            assert c >= 0, f"Negativer Kontostand: {self.cash}"

        for p in self.position:
            assert 0 <= p < self.board_size, f"Position außerhalb des Bretts: {p}"

        for owner in self.owner:
            assert owner is None or 0 <= owner < self.num_players, f"Ungültiger Owner: {owner}"

        for field_idx, price in enumerate(self.prices):
            if price == 0:
                assert self.owner[field_idx] is None, f"Preisloses Feld {field_idx} hat einen Owner"
