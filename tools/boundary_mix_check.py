"""Datenqualitäts-Checks für den 7d-Boundary-Trainingsmix (§6/7d).

Verifiziert die Kernprinzip-Bedingungen, die der User für 7d präregistriert hat:
  1. Kein held-out-Paar (150,150),(150,300),(250,150),(250,300) im Trainingsmix.
  2. Einzelwerte cash=150/250 und price=150/300 weiterhin gut im Training vertreten
     (sonst fehlt dem Kompositionstest die Basis).
  3. Genug NAHE Grenzrelationen (nicht exakt die held-out-Paare).
  4. Policy/Player/Position nicht schief (Konfund-Kontrolle, analog Check 1 in
     probe_afford.py, aber über den Trainingsmix statt über die Eval-Region).

Nutzung:  python tools/boundary_mix_check.py --data data/games_mixed_h2d_boundary.jsonl
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.splits import load_games, decision_points

HELD = {(150, 150), (250, 150), (150, 300), (250, 300)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/games_mixed_h2d_boundary.jsonl")
    args = ap.parse_args()

    games = load_games(args.data)
    pts_with_policy = []
    for g in games:
        for p in decision_points(g):
            p = dict(p)
            p["policy"] = g.get("policy_name", "?")
            pts_with_policy.append(p)
    pts = pts_with_policy
    n = len(pts)

    print(f"Korpus: {args.data}")
    print(f"Spiele: {len(games)} | Entscheidungspunkte: {n}\n")

    # 1) held-out-Paare MÜSSEN komplett fehlen
    held_hits = [(p["cash"], p["price"]) for p in pts if (p["cash"], p["price"]) in HELD]
    print("[1] held-out-Paare im Trainingsmix:")
    print(f"    {held_hits if held_hits else 'KEINE (OK)'}\n")

    # 2) Einzelwerte gut vertreten?
    for v in (150, 250):
        k = sum(1 for p in pts if p["cash"] == v)
        print(f"[2] cash={v}: {k:6d}  ({100 * k / n:.2f}%)")
    for v in (150, 300):
        k = sum(1 for p in pts if p["price"] == v)
        print(f"    price={v}: {k:6d}  ({100 * k / n:.2f}%)")
    # held-out-Einzelwerte als Nicht-held-out-Kombination (z.B. cash=150 mit price!=150/300)
    print("    Einzelwerte in NICHT-held-out-Kombinationen:")
    for v in (150, 250):
        k = sum(1 for p in pts if p["cash"] == v
                and (p["cash"], p["price"]) not in HELD)
        print(f"      cash={v} (non-held-out): {k}")
    print()

    # 3) Nahe Grenzrelationen (|diff| <= 50), aber nicht exakt held-out
    near = [p for p in pts if abs(p["cash"] - p["price"]) <= 50
            and (p["cash"], p["price"]) not in HELD]
    print(f"[3] |cash-price|<=50, nicht held-out: {len(near)} ({100 * len(near) / n:.2f}%)")
    unaff_near = sum(1 for p in near if p["cash"] < p["price"])
    print(f"    davon knapp-unaffordable: {unaff_near} | knapp-affordable: {len(near) - unaff_near}\n")

    # 4) Policy / Player / Position Verteilung (Konfund-Check)
    pol = Counter(g.get("policy_name", "?") for g in games)
    players = Counter(p["player"] for p in pts)
    pos = Counter(p["position"] for p in pts)
    print("[4] Policy:", dict(pol))
    print("    Player:", dict(sorted(players.items())))
    print("    Position:", dict(sorted(pos.items())))

    # policy je Entscheidungspunkt aus dem zugehörigen Spiel ableiten
    policy_of = {}
    for g in games:
        for p in decision_points(g):
            policy_of[id(p)] = g.get("policy_name", "?")

    # Konfund-Schieflage je Nahbereich: ist ein bestimmtes (cash,price)-Nahpaar
    # stark an eine Policy/Player/Position gekoppelt?
    print("\n    Nah-Klassen (|diff|<=50) je Policy/Player:")
    for lbl, sel in [("knapp-unaff", lambda p: p["cash"] < p["price"]),
                     ("knapp-aff", lambda p: p["cash"] >= p["price"])]:
        sub = [p for p in near if sel(p)]
        if not sub:
            continue
        pl = Counter(policy_of[id(p)] for p in sub)
        ply = Counter(p["player"] for p in sub)
        po = Counter(p["position"] for p in sub)
        print(f"    {lbl}: n={len(sub)} | policy={dict(pl)} | player={dict(sorted(ply.items()))} "
              f"| top-pos={dict(po.most_common(3))}")


if __name__ == "__main__":
    main()