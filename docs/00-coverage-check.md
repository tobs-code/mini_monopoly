# 0.1 Empirischer Coverage-Check (2000 Spiele, gemischte Policies) — Befunde

Vor dem Tokenizer-Bau wurde `data/generate.py` mit 2000 Spielen (gemischte Policies) laufen gelassen und die resultierende Cash-/Relations-Verteilung ausgewertet. Vier Befunde mit direkten Konsequenzen für §5.1 und §3:

1. **Tokenisierungs-Lücke:** Maximaler beobachteter Cash-Wert ist **1500**. Das ursprünglich in §5.1 festgelegte H/T/O-Schema (`H0..H9 T0..T9 O0..O9`) deckt nur bis 999 ab. → Tausenderstelle fehlt, siehe Fix in §5.1a.
2. **Cash ist nicht durchgängig 50er-diskret:** Die Annahme in §5.1/§3 (S2-Beispiel `±100/±200` vs. `±50`) galt implizit nur für Kauf-Abflüsse. `rent = 0.5 × price` erzeugt bei Preisen wie 150/250 aber Renten von 75/125 und damit "krumme" Cash-Werte (z.B. 325, 175, 625, 1225). Digit-Zerlegung bleibt möglich, aber die S2-Abstandsrelationen sind **nicht rund** — siehe angepasste Beispielliste in §3.
3. **S3 (Struktur-erhaltende Extrapolation) ist mit aktuellen Simulator-Parametern (`max_turns=40`, Startkapital 1000, Preisliste bis 300) praktisch nicht durchführbar:** Von ca. 1,06 Mio. beobachteten States liegen nur 7345 über 1000 und nur 760 über 1200 (q99 = 1000). Das ist kein tragfähiger natürlicher OOD-Bereich. **Entscheidung gefällt in §3a: S3 als Option (b) — nicht durchführbar in V0, V1-Erweiterung vorgemerkt.**
4. **S2-Tragfähigkeit ist durch starke Schieflage gefährdet:** 2000 Spiele → ~20.000 Kauf-Entscheidungspunkte (14.689 BUY, 5.072 PASS), 57 distinkte Cash-Preis-Relationen. Verteilung stark ungleich: Relation `700` kommt 2216×, `400` 1972× vor, während viele andere Relationen nur ~100× auftreten. Ein held-out einer seltenen Relation ist damit unterbesetzt, ein held-out einer häufigen Relation verzerrt durch den Verteilungsunterschied zur Baseline. → Mindest-Samplezahl pro S2-Relation nötig, siehe §3a.
   **> Überholt (Pre-Fix-Datenstand):** Die konkreten Vorkommenszahlen hier (700: 2216, 400: 1972) stammen aus dem Datenstand VOR dem `states[i-1]`-Bugfix (4.3.1) und sind damit systematisch um den Preis verschoben. **Maßgeblich sind die korrigierten Werte in §4.3.2** (voller Datensatz: Top-Relation 850: 2840, 700: 2811; 22 taugliche S2-Kandidaten). Die *qualitative* Aussage — starke Schieflage, Mindest-Samplezahl nötig — bleibt unverändert gültig.

**Hardware-Constraint:** RTX 2080, 8 GB VRAM. Kein Tage-langes Training möglich. Modellgröße ist entsprechend bewusst klein zu halten (siehe unten).

---

