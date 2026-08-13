# Mini-Monopoly Interpretability-Experiment

**Status:** V0-Simulator + Datengenerierung (`data/generate.py`) fertig und getestet. Preis-Bug gefixt (siehe 4.1). Tokenisierungsentscheidung getroffen (siehe 5.1), **um Tausenderstelle erweitert** (siehe 5.1a). Coverage-Check für S2/S3 durchgeführt (siehe 3a); **S3-Entscheidung (b):** nicht durchführbar in V0, als V1-Erweiterung via größerem Preisraum vorgemerkt. **Bucketing-Trigger präzisiert** (siehe 5.1). **Framework-Spike abgeschlossen** (siehe 5.5 + `spikes/spike_tl.py`): TransformerLens gewählt. **S1–S2-Filterlogik implementiert und getestet** (siehe 4.3 + `data/splits.py` + `tests/test_splits.py`), inkl. **Bugfix des Pre-Buy-Zustands** (4.3.1, 0 Verletzungen bei 19.946 Entscheidungspunkten, voller Datensatz) und korrigierter S2-Kandidatenliste (4.3.2). **Tokenizer/Vocab gebaut** (siehe 5.6 + `data/tokenizer.py` + `tests/test_tokenizer.py`): 48-Token-Vokabular (TH/H/T/O + kategorial), Kontext-Disambiguierung Würfel/Miete, alle 2000 Spiele tokenisierbar; `n_ctx=288` (§5.2); **Embedding-Init-Check grün** (`spikes/check_embed_init.py`). **GPT-Training gelaufen** (siehe 5.7): run1 (abgebrochen, best_val 0.5638), run2_wd0.5_s1200 (best_val 0.5627). **Erste H1-Vorproben durchgeführt** (siehe 5.7): Labels/Rekonstruktion 100% korrekt, aber **Position-Sanity-Probe flach (~17–19%) statt ~99%** — offene Diskrepanz, nächster Schritt ist der kausale Test (`probes/causal_roll.py`, finaler Lauf steht aus). Repo gepusht auf GitHub (`tobs-code/mini_monopoly`, Commit `4bac662`).


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
