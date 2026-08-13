"""
S1-Design-Konfiguration (Single Source of Truth).

Nach dem 7d-Leakage-Befund und der Design-Search (tools/s1_design_search.py)
wurde das S1-2x2-Design auf `cash in {100,300} x price in {150,200}` gewechselt
(dokumentiert in docs/06 Schritt 7d-2). Dieses Design ist konfund-ok (alle 3
Policies gut durchmischt) und der best-effort-Kompromiss unter der strukturellen
Simulator-Kopplung (Einzelwert-Coverage niedrig, min ~178 / ~0.2%).

sowohl data/merge_corpora.py (held-out-Ausschluss aus dem Trainingsmix) als auch
probes/probe_afford.py (S1-Split + Testregion) muessen dieselbe Paarmenge nutzen
— diese Datei ist die einzige Quelle dafuer.
"""

from __future__ import annotations

# Pre-registriertes S1-2x2-Design: held-out cash-Werte x held-out price-Werte.
# Held-out-Paare: (100,150) unaff grenznah diff=-50, (100,200) unaff,
# (300,150) afford, (300,200) afford  -> balanciert 2 afford + 2 unafford.
CASH_VALUES = (100, 300)
PRICE_VALUES = (150, 200)

HELD_OUT_PAIRS = {(c, p) for c in CASH_VALUES for p in PRICE_VALUES}


def design_label() -> str:
    """Kompakte Beschriftung, z.B. 'cash100,300 x price150,200'."""
    return f"cash{','.join(map(str, CASH_VALUES))} x price{','.join(map(str, PRICE_VALUES))}"