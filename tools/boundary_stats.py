"""Grenzfall-Verteilung an Entscheidungspunkten messen (H2-7d Voranalyse).

Gibt pro Korpus aus: wie viele Entscheidungspunkte liegen NAH an der
Affordability-Grenze (|cash-price| klein) — der Bereich, den das Modell
beim 7d-Boundary-Mix gezielt mehr sehen soll.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.splits import load_games, decision_points


def stats(path: str, label: str) -> None:
    games = load_games(path)
    pts = [p for g in games for p in decision_points(g)]
    n = len(pts)
    print(f"--- {label} | {len(games)} Spiele | {n} decision points ---")
    checks = [
        ("unaff (cash<price)", lambda p: p["cash"] < p["price"]),
        ("diff<=25", lambda p: abs(p["cash"] - p["price"]) <= 25),
        ("diff<=50", lambda p: abs(p["cash"] - p["price"]) <= 50),
        ("diff<=100", lambda p: abs(p["cash"] - p["price"]) <= 100),
        ("cash 200..350", lambda p: 200 <= p["cash"] <= 350),
        ("cash 150..300", lambda p: 150 <= p["cash"] <= 300),
        ("price 250/300", lambda p: p["price"] in (250, 300)),
        ("cash 150..300 & price 250/300", lambda p: 150 <= p["cash"] <= 300 and p["price"] in (250, 300)),
    ]
    for name, fn in checks:
        k = sum(1 for p in pts if fn(p))
        print(f"  {name:32s} {k:6d}  {100 * k / n:6.1f}%")
    # Verteilung der cash-price-Differenz (Kernbereich)
    from collections import Counter
    buckets = Counter()
    for p in pts:
        d = p["cash"] - p["price"]
        if d < -200:
            buckets["<-200"] += 1
        elif d < -100:
            buckets["-200..-101"] += 1
        elif d < -50:
            buckets["-100..-51"] += 1
        elif d <= 0:
            buckets["-50..0"] += 1
        elif d <= 50:
            buckets["1..50"] += 1
        elif d <= 100:
            buckets["51..100"] += 1
        elif d <= 200:
            buckets["101..200"] += 1
        else:
            buckets[">200"] += 1
    print("  cash-price-Verteilung:", dict(buckets))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="*", default=[
        "data/games_so0_n2000_always_buy-conservative-random.jsonl",
        "data/games_so9001_n800_always_buy-conservative-random_sc400.jsonl",
    ])
    args = ap.parse_args()
    for p in args.data:
        if os.path.exists(p):
            stats(p, os.path.basename(p))
    if os.path.exists("data/games_mixed_h2.jsonl"):
        stats("data/games_mixed_h2.jsonl", "mixed_h2 (7c)")