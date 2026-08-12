"""
Datengenerierung für Mini-Monopoly.

Erzeugt viele Partien über gemischte Seeds + gemischte Policies und
exportiert sie als JSONL. Jede Zeile ist ein vollständiger `GameLog`
(siehe `GameLog.to_dict()`): game_id, seed, policy_name, board_size,
num_players, prices, events, states.

Determinismus: Seed-Bereiche werden explizit konfiguriert, damit jeder
Lauf reproduzierbar ist (Kernprinzip §8: Config-Hash im Dateinamen).
"""

from __future__ import annotations
import argparse
import json
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.game import play_game
from simulator.policies import POLICIES


def make_run_id(seed_offset: int, num_games: int, policies: list[str]) -> str:
    """Config-Hash für Dateinamen (Kernprinzip §8: jeder Datensatz muss
    eindeutig seiner Erzeugungs-Config zuordenbar sein)."""
    policy_tag = "-".join(sorted(policies))
    return f"so{seed_offset}_n{num_games}_{policy_tag}"


def generate(
    output_dir: str,
    num_games: int,
    seed_offset: int,
    board_size: int = 12,
    num_players: int = 3,
    max_turns: int = 40,
    policies: list[str] | None = None,
    game_id_start: int = 0,
) -> str:
    """Generiert num_games Partien und schreibt sie als JSONL + Config-Manifest.

    game_id_start: Startwert für game_id, wichtig um Kollisionen zu vermeiden,
    wenn mehrere Generierungsläufe (z.B. unterschiedliche seed_offset/Policy-
    Mischungen) später zu einem gemeinsamen Trainingsdatensatz zusammengeführt
    werden. Beispiel: Lauf 1 mit game_id_start=0, Lauf 2 mit
    game_id_start=100000 -> keine Überschneidung möglich.

    Rückgabe: Pfad zur erzeugten JSONL-Datei.
    """
    if policies is None:
        policies = list(POLICIES.keys())

    run_id = make_run_id(seed_offset, num_games, policies)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"games_{run_id}.jsonl")
    manifest_path = os.path.join(output_dir, f"games_{run_id}.manifest.json")

    rng = random.Random(seed_offset)

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(num_games):
            game_id = game_id_start + i
            seed = rng.randint(0, 2**31 - 1)
            policy = rng.choice(policies)
            log = play_game(
                game_id=game_id,
                seed=seed,
                policy_name=policy,
                board_size=board_size,
                num_players=num_players,
                max_turns=max_turns,
            )
            f.write(json.dumps(log.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "run_id": run_id,
        "num_games": num_games,
        "seed_offset": seed_offset,
        "game_id_start": game_id_start,
        "game_id_end": game_id_start + num_games - 1,
        "board_size": board_size,
        "num_players": num_players,
        "max_turns": max_turns,
        "policies": policies,
        "output_file": os.path.basename(output_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generiert: {num_games} Partien -> {output_path}")
    print(f"Manifest -> {manifest_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-Monopoly Datengenerierung")
    parser.add_argument("--output-dir", default="data", help="Zielverzeichnis für JSONL + Manifest")
    parser.add_argument("--num-games", type=int, default=2000, help="Anzahl Partien")
    parser.add_argument("--seed-offset", type=int, default=0, help="Seed-Offset für Reproduzierbarkeit")
    parser.add_argument("--game-id-start", type=int, default=0, help="Start-game_id (Kollisionsvermeidung bei mehreren Läufen)")
    parser.add_argument("--board-size", type=int, default=12)
    parser.add_argument("--num-players", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Zu mischende Policies (Default: alle drei)",
    )
    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        num_games=args.num_games,
        seed_offset=args.seed_offset,
        board_size=args.board_size,
        num_players=args.num_players,
        max_turns=args.max_turns,
        policies=args.policies,
        game_id_start=args.game_id_start,
    )


if __name__ == "__main__":
    main()
