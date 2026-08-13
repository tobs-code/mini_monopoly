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

    python data/merge_corpora.py \
        --inputs ... \
        --output data/games_mixed_h2d_boundary.jsonl \
        --exclude-held-out   # 7d: Spiele mit S1-held-out-Paaren aus dem Trainingsmix ausschliessen
"""

from __future__ import annotations
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.splits import decision_points
from data.s1_design import HELD_OUT_PAIRS


# S1-held-out-Paare (Single Source of Truth: data/s1_design.py, identisch zu
# probes/probe_afford.py). Aktuelles Design (7d-2): cash{100,300} x price{150,200}.


def merge(inputs: list[str], output: str, mix_seed: int,
          exclude_held_out: bool = False, merge_type: str | None = None) -> dict:
    """Lädt die Korpi, validiert disjunkte game_ids, mischt und schreibt.

    exclude_held_out=True (7d): Spiele, die ein S1-held-out-Paar enthalten,
    werden aus dem Trainingsmix ausgeschlossen — sonst sieht das Modell die
    exakten (cash, price)-Kombinationen im Training und der S1-Kompositionstest
    (probe_afford.py) wäre durch Memorierung verfälscht (Leakage-Befund 7c/8L).

    Rückgabe: Manifest-Dict (Kernprinzip §8) für die entstandene Datei.
    """
    all_games = []
    sources = []
    seen_ids = {}
    n_excluded = 0
    for path in inputs:
        games = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
        for g in games:
            gid = g["game_id"]
            if gid in seen_ids:
                raise ValueError(
                    f"game_id-Kollision: {gid} in {path} und {seen_ids[gid]}. "
                    f"Korpi müssen disjunkte game_id-Bereiche haben (§8)."
                )
            seen_ids[gid] = path
        if exclude_held_out:
            kept = []
            for g in games:
                if any((p["cash"], p["price"]) in HELD_OUT_PAIRS
                       for p in decision_points(g)):
                    n_excluded += 1
                    continue
                kept.append(g)
            games = kept
        all_games.extend(games)
        sources.append({"file": os.path.basename(path),
                        "n_games": len(games) + (0 if not exclude_held_out else n_excluded)})
    if exclude_held_out:
        # n_games je Quelle korrekt dokumentieren (nach Filter)
        sources = []
        seen_ids = {}
        for path in inputs:
            with open(path, encoding="utf-8") as f:
                raw = [json.loads(l) for l in f if l.strip()]
            n_kept = sum(
                1 for g in raw
                if not any((p["cash"], p["price"]) in HELD_OUT_PAIRS
                           for p in decision_points(g))
            )
            for g in raw:
                seen_ids[g["game_id"]] = path
            sources.append({"file": os.path.basename(path), "n_games": n_kept})

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
        "merge_type": merge_type or ("mixed_h2d_boundary" if exclude_held_out
                                     else "mixed_h2_affordability"),
        "mix_seed": mix_seed,
        "exclude_held_out": exclude_held_out,
        "held_out_pairs": sorted(HELD_OUT_PAIRS),
        "n_excluded_held_out_games": n_excluded if exclude_held_out else 0,
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
    ap.add_argument("--exclude-held-out", action="store_true",
                    help="7d: Spiele mit S1-held-out-Paaren aus dem Trainingsmix ausschliessen "
                         "(Leakage-Fix fuer probe_afford.py)")
    ap.add_argument("--merge-type", default=None,
                    help="Manifest-Label fuer den Mix (z.B. mixed_h2e_neighbor)")
    args = ap.parse_args()
    merge(args.inputs, args.output, args.mix_seed, args.exclude_held_out, args.merge_type)


if __name__ == "__main__":
    main()