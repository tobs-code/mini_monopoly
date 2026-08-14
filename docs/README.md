# Mini-Monopoly Interpretability-Experiment

**Status:** V0-Simulator + Datengenerierung (`data/generate.py`) fertig und getestet. Preis-Bug gefixt (siehe 4.1). Tokenisierungsentscheidung getroffen (siehe 5.1), **um Tausenderstelle erweitert** (siehe 5.1a). Coverage-Check für S2/S3 durchgeführt (siehe 3a); **S3-Entscheidung (b):** nicht durchführbar in V0, als V1-Erweiterung via größerem Preisraum vorgemerkt. **Bucketing-Trigger präzisiert** (siehe 5.1). **Framework-Spike abgeschlossen** (siehe 5.5 + `spikes/spike_tl.py`): TransformerLens gewählt. **S1–S2-Filterlogik implementiert und getestet** (siehe 4.3 + `data/splits.py` + `tests/test_splits.py`), inkl. **Bugfix des Pre-Buy-Zustands** (4.3.1) und korrigierter S2-Kandidatenliste (4.3.2). **Tokenizer/Vocab gebaut** (siehe 5.6 + `data/tokenizer.py` + `tests/test_tokenizer.py`): 48-Token-Vokabular, `n_ctx=288` (§5.2); **Embedding-Init-Check grün**. **GPT-Training gelaufen** (siehe 5.7): run1 (abgebrochen, best_val 0.5638), run2_wd0.5_s1200 (best_val 0.5627). **H1 abgeschlossen** (§6a): Position wird **kausal** genutzt (`causal_roll.py`, F_L≈1), ist aber weder linear noch nichtlinear **game-invariant** dekodierbar (~17–19%); Cash-Probe R_partial ≈ 0.46; **10x-Daten-Frage abschließend beantwortet** (equal-epoch-Scaling: Deckel ist strukturell, nicht Coverage). **H2 abgeschlossen — POSITIV** (§7e+): generalisierbares Affordability-Signal (S1-HeldAcc 89.7%, grenznah 78.4%), das frühere Scheitern war das alte S1-Design; aber H2-Probe am tatsächlichen Entscheidungspunkt (`pre_decision`) nur 65–69% vs. ~90% an der BUY/PASS-Token-Position. **H3-light abgeschlossen — NICHT bestätigt** (K1–K4 verfehlt, `causal_afford.py`): resid-Patch am pre_decision-Token verschiebt die BUY/PASS-Marge nicht kausal in Richtung der Affordability-Klasse (run2-Gegencheck konsistent). **H3-position abgeschlossen** (`causal_roll_heads.py`): Roll→Event-Kanal head-lokalisiert in **Layer 0** (K1/K2 ✓, Top-Head **L0h0** F≈0.45), aber K3 schwach (Attention ist kein verlässlicher Mechanismus-Proxy) und **K4 ✗** (Remote-Roll-Ratio 0.85, nicht lokal — das Modell rekonstruiert Position/Ownership über die **gesamte** Wurf-Historie). **Aktuell: H3-position-2 (L0h0-Microscope)** — Plan präregistriert (Commit `e64bbf9`), Implementierung von `probes/l0h0_microscope.py` in Arbeit: Diagnose, ob L0h0 Positions-/History-Integrator, Event-class-Shortcut oder Player/Policy/Timing-Head ist (Token-Spezifität, Remote-Roll-Kontrast, Head-Output-Proben, Unembedding-Richtungen). Gesamtstory schärft sich zu: **kausal wirksame, verteilte/rechenhafte Nutzung der Historie ohne stabile, game-invariant linear dekodierbare Position**.


## Dokumente

- [00 Coverage-Check](00-coverage-check.md)
- [01 Forschungsfrage](01-forschungsfrage.md)
- [02 Hypothesen](02-hypothesen.md)
- [03 Held-out-Splits](03-held-out-splits.md)
- [04 Simulator und Daten](04-simulator-und-daten.md)
- [05 Modell, Tokenisierung und Analyse](05-modell-tokenisierung-analyse.md)
- [06 Naechste Schritte](06-naechste-schritte.md)
- [07 Zurueckgestellte Erweiterungen](07-zurueckgestellte-erweiterungen.md)
- [08 Kernprinzipien](08-kernprinzipien.md)

## Hinweis

Diese Datei ist der Einstiegspunkt. Die ausfuehrliche Experiment-Dokumentation liegt in den verlinkten Dateien.
