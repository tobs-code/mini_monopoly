"""Parst runs/<run>/train.log -> Loss-Kurve + Overfit-Diagnose.

Zeigt je Logging-Schnitt: step, train loss, val loss (best), sowie einen
einfachen Overfit-Indikator: Ist der Train-Loss über die letzten 200 Steps
gefallen, während der best_val sich verschlechtert oder der letzte val-Wert
über dem best_val liegt?

Nutzung:  python tools/monitor_train.py --run runs/h2_7c_mixed
"""
from __future__ import annotations
import argparse
import os
import re


def parse(log_path: str):
    train = {}
    val = {}
    last_step = -1
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"step (\d+)/\d+ \| loss ([\d.]+)", line)
            if m:
                last_step = int(m.group(1))
                train[last_step] = float(m.group(2))
            m = re.search(r"VAL loss ([\d.]+) \(best ([\d.]+)\)", line)
            if m and last_step >= 0:
                val[last_step] = (float(m.group(1)), float(m.group(2)))
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/h2_7c_mixed")
    args = ap.parse_args()
    log = os.path.join(args.run, "train.log")
    if not os.path.exists(log):
        raise SystemExit(f"Kein train.log unter {log}")

    train, val = parse(log)
    if not train:
        raise SystemExit("Noch keine Trainingszeilen.")
    steps = sorted(train)
    print(f"Run: {args.run} | aktuelle Steps: {steps[-1]}")
    print(f"{'step':>6} | {'train':>7} | {'val':>7} | {'best_val':>8} | {'Overfit?':>8}")
    last_vals = [v for s, v in sorted(val.items())]
    for s in steps:
        t = train[s]
        v_info = ""
        if s in val:
            v, b = val[s]
            flag = ""
            # Overfit: Train fällt unter den letzten val-Wert & val über best
            if v > b + 1e-9:
                flag = "VAL>BEST"
            v_info = f"{v:>7.4f} | {b:>8.4f} | {flag:>8}"
        else:
            v_info = f"{'-':>7} | {'-':>8} | {'':>8}"
        print(f"{s:>6} | {t:>7.4f} | {v_info}")

    # Overfit-Indikator (letzte 200 Steps): train deutlich gefallen, val gestiegen
    if steps[-1] >= 200:
        t_now = train[steps[-1]]
        t_200 = train[steps[-1] - 200]
        drop = t_now - t_200
        val_steps = sorted(val)
        recent_val = val[val_steps[-1]][0] if val_steps else None
        best_val = min(v[1] for v in val.values()) if val else None
        print("\n--- Diagnose (letzte 200 Steps) ---")
        print(f"train {t_200:.4f} -> {t_now:.4f} (Delta {drop:+.4f})")
        if recent_val is not None and best_val is not None:
            print(f"letzter val {recent_val:.4f} vs best_val {best_val:.4f}")
            if drop < -0.01 and recent_val > best_val + 0.005:
                print(">>> OVERFIT-VERDACHT: Train fällt, val über best.")
            elif recent_val <= best_val + 0.005:
                print(">>> val nahe best: noch ok, weiter beobachten.")
            else:
                print(">>> val unter best, aber Train nicht mehr gefallen.")


if __name__ == "__main__":
    main()