"""
Zusammenführen mehrerer generierter Korpus-JSONLs zu EINEM Trainingskorpus.

Grund (§6b/7c): H2-Zusatztraining auf einem gemischten Korpus MIT echter
Affordability-Varianz — bestehender cap=1000-Korpus (alte Dynamik, 99.5%
affordable) + Low-Cap-Korpus (cap=400, 28.1% unaffordable). Nur so erlebt das
Modell die `cash>=price`-Entscheidung im Training tatsächlich, ohne die alte
Spieldynamik zu verlieren.

WICHTIG (game_id-Kollisionsfreiheit): Die zu mischenden Korpi müssen disjunkte
`game_id`-Bereiche haben (cap=1000: 0-1999; cap=400: 200000-200799). Das Skript
prüft das und lehnt überlappende game_ids ab — sonst wäre der entstehende
Datensatz nicht eindeutig seiner Quelle zuordenbar (§8).

Kernprinzip §8:
- Deterministischer Mix-Seed (im Manifest festgehalten).
- Manifest dokumentiert Quellen, n_games je Quelle, Mix-Seed, resultierende
  Datei. Der gemischte Korpus ist damit eindeutig reproduzierbar.
- Das S0-Train/Val-Split geschieht anschließend in training/train.py (Single
  Source of Truth, data.splits.split_s0); hier wird NUR zusammengeführt.

CLI:
    python data/merge_corpora.py \
        --inputs data/games_so0_n2000_always_buy-conservative-random.jsonl \
                 data/games_so9001_n800_always_buy-conservative-random_sc400.jsonl \
        --output data/games_mixed_h2.jsonl \
        --mix-seed 7
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def merge(inputs: list[str], output: str, mix_seed: int) -> dict:
    """Lädt die Korpi, validiert disjunkte game_ids, mischt und schreibt.

    Rückgabe: Manifest-Dict (Kernprinzip §8) für die entstandene Datei.
    """
    all_games = []
    sources = []
    seen_ids = {}
    for path in inputs:
        games = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
        # game_id-Kollisionscheck über die gesamte Quellmenge
        for g in games:
            gid = g["game_id"]
            if gid in seen_ids:
                raise ValueError(
                    f"game_id-Kollision: {gid} in {path} und {seen_ids[gid]}. "
                    f"Korpi müssen disjunkte game_id-Bereiche haben (§8)."
                )
            seen_ids[gid] = path
        all_games.extend(games)
        sources.append({"file": os.path.basename(path), "n_games": len(games)})

    rng = random.Random(mix_seed)
    rng.shuffle(all_games)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for g in all_games:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    # start_capital pro Quelle sauber aus dem ersten Cash der Start-States ableiten
    # (GameLog-dicts tragen kein explizites start_capital-Feld; bei fehlendem
    # States[0]-Cash-Bezug wird der dokumentierte Default 1000 angenommen).
    def infer_start_capital(log):
        st = log.get("states") or []
        if st and isinstance(st[0], dict) and st[0].get("cash"):
            return st[0]["cash"][0]
        return log.get("start_capital", 1000)

    manifest = {
        "merge_type": "mixed_h2_affordability",
        "mix_seed": mix_seed,
        "sources": sources,
        "n_total_games": len(all_games),
        "game_id_min": min(seen_ids.keys()),
        "game_id_max": max(seen_ids.keys()),
        "output_file": os.path.basename(output),
        "n_policies": len({p for g in all_games for p in [g.get("policy_name", "?")]}),
        "start_capitals_present": sorted({infer_start_capital(g) for g in all_games}),
    }
    manifest_path = os.path.splitext(output)[0] + ".manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Merge: {len(all_games)} Spiele -> {output}")
    print(f"  Quellen: {sources}")
    print(f"  start_capitals: {manifest['start_capitals_present']}")
    print(f"  Manifest -> {manifest_path}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Zu mischende JSONL-Korpi (disjunkte game_id-Bereiche)")
    ap.add_argument("--output", default="data/games_mixed_h2.jsonl")
    ap.add_argument("--mix-seed", type=int, default=7)
    args = ap.parse_args()
    merge(args.inputs, args.output, args.mix_seed)


if __name__ == "__main__":
    main()