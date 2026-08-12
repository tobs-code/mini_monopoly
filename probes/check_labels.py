"""
§2-H1-Rekonstruktions-Sanity-Check (NUR Datenqualität, kein Modell).

Prüft, dass die Ground-Truth-Positionen in den GameLogs deterministisch aus
den Würfel-Tokens rekonstruierbar sind: position[p] = cumsum(dice_p) mod
board_size. Wenn das nicht ~100% übereinstimmt, ist die Daten/Alignment-
Pipeline kaputt (Label-Extraktions-Bug o.Ä.) und KEIN Positions-Probe-Befund
ist verwertbar.

State-Semantik (game.py _append): events[i]/states[i] = Snapshot NACH allen
Events bis inkl. Token i. Ein Wurf-Gruppe [P, ROLL, dice] teilt EINEN
post-move-Snapshot => alle drei Tokens vergleichen wir mit der Position
NACH dem Zug.

Zusätzlich: Positionsverteilung + Majority-Class-Baseline, um eine niedrige
Probe-Accuracy korrekt einzuordnen (ist die Baseline evtl. schon hoch?).

Nutzung:
    python probes/check_labels.py [--data data/games_....jsonl]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.splits import load_games


def check_game(game: dict) -> tuple[int, int, int]:
    """Rekonstruiert Positionen aus Dice-Tokens, vergleicht mit states[].
    Rückgabe: (tokens_ok, tokens_total, n_players)."""
    events = game["events"]
    states = game["states"]
    n = len(events)
    if len(states) != n:
        return 0, -abs(len(states) - n), game["num_players"]

    P = game["num_players"]
    board = game["board_size"]
    pos = [0] * P
    ok = 0

    def check(i: int) -> None:
        nonlocal ok
        if pos == states[i]["position"]:
            ok += 1

    i = 0
    while i < n:
        tok = events[i]
        if tok == "START":
            check(i); i += 1; continue
        if tok.startswith("P") and tok[1:].isdigit() and i + 2 < n \
                and events[i + 1] == "ROLL" and events[i + 2].isdigit():
            player = int(tok[1:])
            roll = int(events[i + 2])
            if not 1 <= roll <= 6:
                return 0, n, P
            pos[player] = (pos[player] + roll) % board
            check(i); check(i + 1); check(i + 2)
            i += 3
            continue
        check(i); i += 1
    return ok, n, P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "games_so0_n2000_always_buy-conservative-random.jsonl"))
    args = ap.parse_args()

    games = load_games(args.data)
    total_ok = total_n = 0
    bad = 0
    for g in games:
        ok, n, P = check_game(g)
        if n < 0:
            print(f"!!! game {g['game_id']}: len(events)={len(g['events'])} != len(states)={len(g['states'])}")
            bad += 1
            continue
        total_ok += ok
        total_n += n
        if ok != n:
            bad += 1
            # erste Mismatch-Stelle melden
            from data.tokenizer import MiniTokenizer
            t = MiniTokenizer()
            ev, st = g["events"], g["states"]
            pos = [0] * g["num_players"]
            i = 0
            while i < len(ev):
                tok = ev[i]
                if tok.startswith("P") and tok[1:].isdigit() and i + 2 < len(ev) \
                        and ev[i + 1] == "ROLL" and ev[i + 2].isdigit():
                    p = int(tok[1:]); pos[p] = (pos[p] + int(ev[i + 2])) % g["board_size"]
                    if pos != st[i]["position"]:
                        print(f"  game {g['game_id']} Mismatch @ Token {i} '{ev[i]}': "
                              f"rekonstruiert={pos} ground_truth={st[i]['position']}")
                        break
                    i += 3; continue
                if pos != st[i]["position"]:
                    print(f"  game {g['game_id']} Mismatch @ Token {i} '{ev[i]}': "
                          f"rekonstruiert={pos} ground_truth={st[i]['position']}")
                    break
                i += 1
    acc = total_ok / total_n if total_n else 0.0
    print(f"\nSpiele gesamt: {len(games)} | betroffen: {bad}")
    print(f"Token-Rekonstruktion: {total_ok:,}/{total_n:,} = {acc:.4%}")

    # Positionsverteilung + Majority-Class-Baseline (nur Tokens MIT Zustand)
    dist = Counter()
    for g in games:
        for s in g["states"]:
            for p in s["position"]:
                dist[p] += 1
    total = sum(dist.values())
    most = dist.most_common(1)[0][1]
    print(f"\nPositionsverteilung (Feld: Anteil) über {total:,} Samples:")
    for f, c in sorted(dist.items()):
        print(f"  Feld {f:2d}: {c / total:6.1%}")
    print(f"Majority-Class-Baseline (immer häufigstes Feld): {most / total:.4%}")
    if acc >= 0.999:
        print("\nErgebnis: Labels KORREKT (~100%) -> Positions-Probe-Befunde verwertbar.")
    else:
        print("\nErgebnis: LABEL-BUG -> zuerst Pipeline fixen, Probe-Befunde derzeit nicht verwertbar.")


if __name__ == "__main__":
    main()
