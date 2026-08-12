"""
Game-Loop: spielt eine deterministische Partie durch und protokolliert
für JEDEN Event-Schritt sowohl die Event-Tokens als auch den vollen
Ground-Truth-State danach.

Determinismus in V0 (bewusste Design-Entscheidung aus der Diskussion):
Würfelwürfe kommen aus einem seed-gesteuerten RNG, sind also reproduzierbar
und für das Modell im Kontext sichtbar (kein Hidden-State-Problem in V0).
Stochastik/Hidden Information ist ein späterer, eigener Experiment-Schritt.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field

from .state import GameState
from . import rules
from .policies import POLICIES


@dataclass
class GameLog:
    game_id: int
    seed: int
    policy_name: str
    board_size: int
    num_players: int
    prices: list[int]

    events: list[str] = field(default_factory=list)          # flache Token-Sequenz
    states: list[dict] = field(default_factory=list)         # states[i] = Ground-Truth NACH events bis inkl. Token i
    # states hat dieselbe Länge wie events: state-Snapshot wird nach jedem
    # einzelnen Token-Anhang wiederholt (auch Zwischentoken wie "P0"/"ROLL"
    # zeigen den zuletzt bekannten Zustand -> vereinfacht spätere Alignment-Logik)

    def to_dict(self) -> dict:
        return {
            "game_id": self.game_id,
            "seed": self.seed,
            "policy_name": self.policy_name,
            "board_size": self.board_size,
            "num_players": self.num_players,
            "prices": self.prices,
            "events": self.events,
            "states": self.states,
        }

    def _append(self, tokens: list[str], state: GameState) -> None:
        snap = state.to_dict()
        for tok in tokens:
            self.events.append(tok)
            self.states.append(snap)


BOARD_SEED = 20260811  # fix, unabhängig vom Spiel-Seed


def make_prices(board_size: int, rng: random.Random | None = None) -> list[int]:
    """Feld 0 (Start) ist immer preislos. Restliche Felder bekommen Preise
    aus einer festen Stufenliste.

    WICHTIG (Bugfix): Preise werden NICHT mehr aus dem Spiel-Seed abgeleitet,
    sondern immer aus BOARD_SEED -- das Board-Layout ist über ALLE Partien
    hinweg identisch. Grund: der Preis eines Feldes wird im Event-Token-Stream
    nirgends explizit offengelegt (BUY zeigt keinen Preis, PAY_RENT erst NACH
    dem ersten Mietfall auf diesem Feld). Bei pro-Spiel-zufälligen Preisen wäre
    der Preis eines noch nie vermieteten Feldes aus den Tokens schlicht nicht
    rekonstruierbar -- ein H2-Fehlschlag wäre dann kein Befund über
    Kompositionalität, sondern ein Artefakt fehlender Information.
    Mit festem Board kann das Modell den Preis pro Feld als gelernte Konstante
    memorieren (analog zu fixen Spielregeln bei Othello-GPT); Cash bleibt die
    tatsächlich dynamische, pro Spiel/Spieler zu verfolgende Variable."""
    board_rng = random.Random(BOARD_SEED)
    price_options = [0, 100, 150, 200, 250, 300]
    prices = [0]  # Startfeld
    for _ in range(board_size - 1):
        prices.append(board_rng.choice(price_options))
    return prices


def play_game(
    game_id: int,
    seed: int,
    policy_name: str,
    board_size: int = 12,
    num_players: int = 3,
    max_turns: int = 40,
) -> GameLog:
    rng = random.Random(seed)
    prices = make_prices(board_size, rng)
    policy_fn = POLICIES[policy_name]

    state = GameState(board_size=board_size, num_players=num_players, prices=prices)
    state.check_invariants()

    log = GameLog(
        game_id=game_id,
        seed=seed,
        policy_name=policy_name,
        board_size=board_size,
        num_players=num_players,
        prices=prices,
    )
    # initialer Zustand (vor jedem Event) als Anker
    log.events.append("START")
    log.states.append(state.to_dict())

    for _ in range(max_turns):
        player = state.current_player
        roll = rng.randint(1, 6)

        tokens = rules.apply_roll_and_move(state, player, roll)
        log._append(tokens, state)

        field_idx = state.position[player]
        owner_idx = state.owner[field_idx]

        if owner_idx is None and state.prices[field_idx] > 0:
            if policy_fn(state, player, rng):
                tokens = rules.apply_buy(state, player)
            else:
                tokens = rules.apply_pass(state, player)
            log._append(tokens, state)
        elif owner_idx is not None and owner_idx != player:
            tokens = rules.apply_rent(state, player)
            log._append(tokens, state)
        # eigenes Feld oder preisloses Feld -> kein weiteres Event

        rules.advance_turn(state)

    return log
