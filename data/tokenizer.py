"""
Tokenizer/Vocab für Mini-Monopoly (TransformerLens).

Bildet den Event-Token-Stream (data/generate.py -> GameLog.events) auf ein
festes, kleines Vokabular von Token-IDs ab, gemäß EXPERIMENT.md §5.1/§5.1a:

- Kategoriale/atomare Event-Tokens: `START`, `P0`..`P2`, `ROLL`, `BUY`,
  `PASS`, `PAY_RENT`, sowie Würfelwerte als eigene atomare Dice-Tokens
  `D1`..`D6` (diskret, KEINE Stellenwert-Semantik).
- Numerische Magnituden (Mietbeträge = Cash-Flüsse) als **Digit-Sequenz**
  mit expliziten Stellenmarkern: `MONEY TH0 H7 T5 O0` (= 750). Die
  Tausenderstelle ist das binäre `TH0`/`TH1` aus §5.1a (max. Cash 1500).

WICHTIG (empirisch verifiziert): Würfel- und Mietwerte sind NICHT allein
über ihren Zahlenwert unterscheidbar, weil die Miete cash-gecappt ist
(`rent = min(0.5*price, cash)`, siehe rules.py) und dadurch einstellige
Werte (0..9) und sogar `0` annehmen kann. Die Unterscheidung Würfel vs.
Miete erfolgt daher KONTEXT-basiert über das vorausgehende Event-Token
(`ROLL` -> Dice-Token, `PAY_RENT` -> Digit-Sequenz). Wird ein numerisches
Token ohne eindeutigen Kontext angetroffen, wird ein ValueError geworfen
(statt still zu raten) — siehe tests/test_tokenizer.py.

Kernprinzip §8: Das Vokabular ist eine feste, deterministische Liste; die
Token-IDs sind stabil über Läufe und Datensätze hinweg.
"""

from __future__ import annotations
from typing import Optional


class TokenizerError(ValueError):
    """Unbekanntes oder nicht disambiguierbares Event-Token."""


# Festes, deterministisches Vokabular (Kernprinzip §8).
# Reihenfolge = Token-ID. `<PAD>` ist ID 0.
# 1 PAD + 1 START + 4 Events + 3 Spieler + 6 Dice + 1 MONEY + 2 TH + 30 Digits = 48
VOCAB = [
    "<PAD>",
    "START",
    "ROLL", "BUY", "PASS", "PAY_RENT",
    "P0", "P1", "P2",
    "D1", "D2", "D3", "D4", "D5", "D6",
    "MONEY",
    "TH0", "TH1",
    "H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9",
    "T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9",
    "O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9",
]

_EVENT_TOKENS = {"START", "ROLL", "BUY", "PASS", "PAY_RENT"}
_MAX_SUPPORTED = 1999  # TH0/TH1 deckt 0..1999 ab (max. Cash 1500, s. §0.1/§5.1a)


def encode_int(value: int) -> list[str]:
    """Cash/Miet-Magnitude -> Digit-Sequenz mit Stellenmarkern.

    750 -> ['MONEY', 'TH0', 'H7', 'T5', 'O0']
    1500 -> ['MONEY', 'TH1', 'H5', 'T0', 'O0']
    """
    if not 0 <= value <= _MAX_SUPPORTED:
        raise TokenizerError(
            f"Wert {value} ausserhalb des TH0/TH1-Bereichs (0..{_MAX_SUPPORTED}). "
            f"Sind Cash-Werte >= 2000 aufgetreten, muss TH erweitert werden (§5.1a)."
        )
    th = value // 1000
    h = (value // 100) % 10
    t = (value // 10) % 10
    o = value % 10
    return ["MONEY", f"TH{th}", f"H{h}", f"T{t}", f"O{o}"]


class MiniTokenizer:
    def __init__(self, num_players: int = 3):
        self.num_players = num_players
        self.itos = list(VOCAB)
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]

    @property
    def d_vocab(self) -> int:
        return len(self.itos)

    def _iter_tokens(self, events: list[str]):
        """Yield (token_string, raw_event_index) — pro Output-Token der Index des
        Roh-Event-Tokens, aus dem es stammt (wichtig fürs State-Alignment beim
        Probing: ein Miet-Event expandiert auf 5 Tokens, die alle denselben
        Ground-Truth-State teilen)."""
        n = len(events)
        for i, tok in enumerate(events):
            if tok in _EVENT_TOKENS:
                yield tok, i
            elif tok.startswith("P") and tok[1:].isdigit():
                yield tok, i
            elif tok.isdigit():
                v = int(tok)
                prev = events[i - 1] if i >= 1 else None
                if prev == "ROLL":
                    if not 1 <= v <= 6:
                        raise TokenizerError(f"Ungültiger Wuerfelwert {v} an Index {i}")
                    yield f"D{v}", i
                elif prev == "PAY_RENT":
                    for t in encode_int(v):
                        yield t, i
                else:
                    raise TokenizerError(
                        f"Numerisches Token '{tok}' an Index {i} ohne eindeutigen Kontext "
                        f"(vorausgehendes Event: {prev!r}). Erwartet nach ROLL oder PAY_RENT."
                    )
            else:
                raise TokenizerError(f"Unbekanntes Event-Token: {tok!r} an Index {i}")

    def tokenize_events(self, events: list[str]) -> list[int]:
        """Konvertiert den Event-Token-Stream eines Spiels in eine Token-ID-Sequenz.

        Position/Kontext-basiert: numerische Tokens werden über das VORANGEHENDE
        Event disambiguiert (ROLL -> Dice, PAY_RENT -> Miet-Magnitude).
        """
        return [self.stoi[t] for t, _ in self._iter_tokens(events)]

    def tokenize_events_aligned(self, events: list[str]) -> tuple[list[int], list[int]]:
        """Wie tokenize_events, liefert zusätzlich pro Output-Token den Index des
        zugehörigen Roh-Event-Tokens (für die State-Zuordnung beim Probing)."""
        ids, raw = [], []
        for t, i in self._iter_tokens(events):
            ids.append(self.stoi[t])
            raw.append(i)
        return ids, raw

    def decode_ids(self, ids: list[int]) -> list[str]:
        """Token-IDs zurück zu Strings (für Debug/Inspektion)."""
        out = []
        for i in ids:
            if not 0 <= i < self.d_vocab:
                raise TokenizerError(f"Token-ID {i} ausserhalb des Vokabulars")
            out.append(self.itos[i])
        return out
