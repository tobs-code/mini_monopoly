# Mini-Monopoly Interpretability-Experiment

**Status:** V0-Simulator + Datengenerierung (`data/generate.py`) fertig und getestet. Preis-Bug gefixt (siehe 4.1). Tokenisierungsentscheidung getroffen (siehe 5.1), **um Tausenderstelle erweitert** (siehe 5.1a). Coverage-Check für S2/S3 durchgeführt (siehe 3a); **S3-Entscheidung (b):** nicht durchführbar in V0, als V1-Erweiterung via größerem Preisraum vorgemerkt. **Bucketing-Trigger präzisiert** (siehe 5.1). **Framework-Spike abgeschlossen** (siehe 5.5 + `spikes/spike_tl.py`): TransformerLens gewählt. **S1–S2-Filterlogik implementiert und getestet** (siehe 4.3 + `data/splits.py` + `tests/test_splits.py`), inkl. **Bugfix des Pre-Buy-Zustands** (4.3.1, 0 Verletzungen bei 19.946 Entscheidungspunkten, voller Datensatz) und korrigierter S2-Kandidatenliste (4.3.2). **Tokenizer/Vocab gebaut** (siehe 5.6 + `data/tokenizer.py` + `tests/test_tokenizer.py`): 48-Token-Vokabular (TH/H/T/O + kategorial), Kontext-Disambiguierung Würfel/Miete, alle 2000 Spiele tokenisierbar; `n_ctx=288` (§5.2); **Embedding-Init-Check grün** (`spikes/check_embed_init.py`). **GPT-Training gelaufen** (siehe 5.7): run1 (abgebrochen, best_val 0.5638), run2_wd0.5_s1200 (best_val 0.5627). **Erste H1-Vorproben durchgeführt** (siehe 5.7): Labels/Rekonstruktion 100% korrekt, aber **Position-Sanity-Probe flach (~17–19%) statt ~99%** — offene Diskrepanz, nächster Schritt ist der kausale Test (`probes/causal_roll.py`, finaler Lauf steht aus). Repo gepusht auf GitHub (`tobs-code/mini_monopoly`, Commit `4bac662`).

### 0.1 Empirischer Coverage-Check (2000 Spiele, gemischte Policies) — Befunde

Vor dem Tokenizer-Bau wurde `data/generate.py` mit 2000 Spielen (gemischte Policies) laufen gelassen und die resultierende Cash-/Relations-Verteilung ausgewertet. Vier Befunde mit direkten Konsequenzen für §5.1 und §3:

1. **Tokenisierungs-Lücke:** Maximaler beobachteter Cash-Wert ist **1500**. Das ursprünglich in §5.1 festgelegte H/T/O-Schema (`H0..H9 T0..T9 O0..O9`) deckt nur bis 999 ab. → Tausenderstelle fehlt, siehe Fix in §5.1a.
2. **Cash ist nicht durchgängig 50er-diskret:** Die Annahme in §5.1/§3 (S2-Beispiel `±100/±200` vs. `±50`) galt implizit nur für Kauf-Abflüsse. `rent = 0.5 × price` erzeugt bei Preisen wie 150/250 aber Renten von 75/125 und damit "krumme" Cash-Werte (z.B. 325, 175, 625, 1225). Digit-Zerlegung bleibt möglich, aber die S2-Abstandsrelationen sind **nicht rund** — siehe angepasste Beispielliste in §3.
3. **S3 (Struktur-erhaltende Extrapolation) ist mit aktuellen Simulator-Parametern (`max_turns=40`, Startkapital 1000, Preisliste bis 300) praktisch nicht durchführbar:** Von ca. 1,06 Mio. beobachteten States liegen nur 7345 über 1000 und nur 760 über 1200 (q99 = 1000). Das ist kein tragfähiger natürlicher OOD-Bereich. **Entscheidung gefällt in §3a: S3 als Option (b) — nicht durchführbar in V0, V1-Erweiterung vorgemerkt.**
4. **S2-Tragfähigkeit ist durch starke Schieflage gefährdet:** 2000 Spiele → ~20.000 Kauf-Entscheidungspunkte (14.689 BUY, 5.072 PASS), 57 distinkte Cash-Preis-Relationen. Verteilung stark ungleich: Relation `700` kommt 2216×, `400` 1972× vor, während viele andere Relationen nur ~100× auftreten. Ein held-out einer seltenen Relation ist damit unterbesetzt, ein held-out einer häufigen Relation verzerrt durch den Verteilungsunterschied zur Baseline. → Mindest-Samplezahl pro S2-Relation nötig, siehe §3a.
   **> Überholt (Pre-Fix-Datenstand):** Die konkreten Vorkommenszahlen hier (700: 2216, 400: 1972) stammen aus dem Datenstand VOR dem `states[i-1]`-Bugfix (4.3.1) und sind damit systematisch um den Preis verschoben. **Maßgeblich sind die korrigierten Werte in §4.3.2** (voller Datensatz: Top-Relation 850: 2840, 700: 2811; 22 taugliche S2-Kandidaten). Die *qualitative* Aussage — starke Schieflage, Mindest-Samplezahl nötig — bleibt unverändert gültig.

**Hardware-Constraint:** RTX 2080, 8 GB VRAM. Kein Tage-langes Training möglich. Modellgröße ist entsprechend bewusst klein zu halten (siehe unten).

---

## 1. Forschungsfrage

Ursprünglich stand die Frage "Haben LLMs ein Weltmodell?" im Raum — das wurde im Verlauf der Diskussion bewusst verworfen, weil zu groß und nicht scharf genug definiert.

**Aktuelle, präzisere Forschungsfrage:**

> Unter welchen Bedingungen entwickeln kleine Transformer, die nur auf Ereignissequenzen (nicht auf explizitem Zustand) trainiert werden, kompositionale interne Repräsentationen eines latenten dynamischen Zustands **und führen Operationen auf diesen Zuständen aus** — und wie hängen Repräsentation und Operation zusammen? Lässt sich das mechanistisch (nicht nur behavioral) nachweisen?

**Wichtige Klarstellung (aus der Diskussion):** "Zustand repräsentieren" und "Operationen über Zustände ausführen" sind **kein striktes entweder/oder**. Ein Modell kann beides haben — eine latente Zustandsdarstellung und Operationen darauf, die kausal in die nächste-Token-Vorhersage einfließen. Das eigentlich Spannende ist daher: **Welche Zustandsvariablen werden intern repräsentiert, welche Operationen werden darauf ausgeführt, und sind diese Strukturen kompositionell und kausal wiederverwendbar?**

Methodischer Leitstern für das ganze Experiment: auf jeder Stufe sauber unterscheiden zwischen
- *"Kann ich die Information aus der Aktivierung lesen?"* (Decodability/Probing) und
- *"Ist die Information kausal an der Berechnung beteiligt?"* (Intervention).

Probing allein beweist keine Kausalität — erst die Intervention.

Das Mini-Monopoly-Spiel ist dabei **nur das kontrollierte Versuchssystem**, kein Selbstzweck. Der eigentliche Wert liegt darin, dass wir (anders als bei echten LLMs auf Naturalsprache) die **vollständige Ground Truth** des latenten Zustands zu jedem Zeitpunkt kennen.

### Bezug zur laufenden Forschung (Kontext, warum das relevant ist)

- **Othello-GPT** (Li et al. 2022/2023): Transformer, nur auf legalen Zugfolgen trainiert, entwickelt nachweisbar internen Board-State. Ursprünglich nur mit *nichtlinearen* Probes nachweisbar.
- **Nanda (2023)**: Zeigte, dass die Repräsentation eigentlich *linear* ist — nur in der falschen Basis gesucht ("schwarz/weiß" statt "eigene Farbe/gegnerische Farbe"). Wichtige Lektion: die richtige Feature-Basis zu finden ist selbst ein Problem (Bezug zu Superposition/SAEs).
- **Gegenposition "Bags of Heuristics"** (Vafa et al., aufgegriffen von Melanie Mitchell): Was wie ein kohärentes Weltmodell aussieht, könnte in Wahrheit ein Bündel vieler lokaler Heuristiken sein, ohne echte kompositionale/generalisierbare Struktur. **Dieses Experiment ist im Kern ein direkter Test gegen diese Hypothese**, weil wir durch Held-out-Splits (s.u.) explizit prüfen können, ob Generalisierung stattfindet oder nur Lookup-artiges Verhalten.
- **Chess-LLM-Nachfolgearbeiten** (Mei et al. 2025 u.a.): Analoge Befunde zu Othello, linear dekodierbare Board-Features, kausale Interventionen wirksam.
- **"Revisiting the Othello World Model Hypothesis"** (2025): Zeigt, dass die Robustheit der Repräsentation vom Trainingsregime abhängt — Ergebnisse sind nicht automatisch trainingsdaten-invariant. Relevanz hier: deshalb bewusst mehrere Kauf-Policies mischen (siehe Abschnitt 4).

### Bezug zu Tobs' eigener laufender Forschung

Dieses Experiment ist konzeptionell verwandt mit Tobs' Hauptprojekt `llm-control-circuits` (mechanistische Interpretability an chinesischen Open-Source-LLMs, SAE-basierte Feature-Analyse, Layer-8–23-Kaskaden bei Zensur-Mechanismen). Der entscheidende Unterschied und Vorteil hier: **volle Ground Truth** des latenten Zustands ist bekannt — bei der Qwen-Arbeit muss Ground Truth erst mühsam rekonstruiert werden. Mini-Monopoly ist als kleines, sauberes Vorstudien-/Laborsystem gedacht, dessen Methodik (Probing, SAE, Interventionen) später wieder auf die Hauptforschung zurückwirken kann.

---

## 2. Hypothesen-Hierarchie (H1–H4)

Zentrale Struktur des gesamten Experiments. Jede Stufe baut auf der vorherigen auf — **nicht überspringen**, da bei einem Fehlschlag in einer späteren Stufe sonst unklar ist, woran es liegt.

### H1 — State Tracking
> Das Modell hält Informationen über den latenten Zustand (Position, Cash, Ownership) verfügbar, obwohl sie nie explizit im Input standen.

**Test:** `history → linear/nichtlinear Probe → state`
Erst linearer Probe versuchen, dann nichtlinearer (Analogie zu Li et al. vs. Nanda bei Othello — die richtige Basis kann entscheidend sein).

**Kostenloser Sanity-Check zuerst:** Position ist deterministisch aus der Roll-Summe berechenbar (`position = cumsum(rolls) mod board_size`). Wenn ein Probe hier nicht ~99% erreicht, ist die Pipeline (Tokenisierung/Architektur/Training) kaputt — **nicht weiter zu Cash/Ownership gehen, bevor das nicht sauber funktioniert.**

### H2 — State-Dependent Computation
> Das Modell verwendet Zustandsinformationen, um neue, nicht direkt beobachtete Größen zu berechnen (z.B. `affordability = cash >= price`).

**Wichtige Unterscheidung (aus der Diskussion):** Ein Output-Probe, der `can_afford(A, X)` korrekt vorhersagt, beweist noch keine Komposition — das Modell könnte es als eigenständiges, memoriertes Feature gelernt haben (`CAN_AFFORD_X` als Shortcut), ohne intern wirklich `cash - price` zu berechnen.

**Deshalb: Held-out-Test zwingend nötig, nicht optional.** Siehe Abschnitt 3 (Splits S0–S3).

### H3 — Causal Internal Mechanism
> Die Zustandsrepräsentationen sind tatsächlich Teil des kausalen Mechanismus, der die Vorhersage erzeugt (nicht nur ein epiphenomenales Korrelat).

**Test:** Aktivierung gezielt manipulieren (z.B. Cash-Repräsentation patchen), prüfen ob sich nachgelagerte Vorhersagen (Affordability, Buy-Entscheidung) entsprechend ändern.

**Wichtige Einschränkung:** Ein Probe, der 95%+ Accuracy erreicht, zeigt nur: *Information ist irgendwo in der Aktivierung vorhanden.* Nicht: *Das Modell nutzt diese Information kausal für seine Vorhersage.* Erst die Intervention zeigt Kausalität.

**Risiko:** Off-Distribution-Artefakte. Wenn man einen Feature-Vektor auf einen Wert patcht, den das Modell in dieser Zustandskombination nie im Training gesehen hat, kann das Modell "kaputt" reagieren — nicht weil der Mechanismus getroffen wurde, sondern weil der Patch die Aktivierung aus der Trainingsverteilung schiebt. Vorsicht bei der Interpretation nötig.

### H4 — Counterfactual Cascade Consistency
> Eine einzelne Intervention propagiert konsistent durch eine ganze Kette abhängiger Zustandsvariablen (nicht nur ein isolierter nachgelagerter Output).

**Beispiel-Kaskade:**
```
cash(A) +200
   → can_afford(X) = TRUE
   → buy(X)
   → ownership(X, A)
   → zukünftige Rent-Konsequenzen
```

Das ist der stärkste Test — er prüft, ob ein **konsistenter interner Simulationszustand** mitgeführt wird, statt vieler separater, lokal korrekter Einzeltricks. Das kommt dem ursprünglichen "Weltmodell"-Konzept am nächsten von allem, was hier getestet wird.

**Wichtig:** H4 **erst angehen, wenn H1–H3 für die einzelnen Kettenglieder separat etabliert sind** (cash→afford, afford→buy, buy→ownership). Sonst ist bei einem H4-Fehlschlag unklar, ob es an einem der Einzelmechanismen liegt oder an der Verkettung selbst. Je länger die Kaskade, desto schwerer die Fehlerzuordnung — deshalb Reihenfolge zwingend einhalten.

**Zwingende Kontrollbedingung für H4:** Bei einer Kette von vier Schritten (cash→afford→buy→ownership→rent) akkumulieren sich Off-Distribution-Artefakte potenziell über die Kette — das Risiko der Fehlinterpretation ist hier am größten (siehe §H3). Deshalb ist eine negative Kontrolle Pflicht: **Dieselbe Intervention wird zusätzlich auf einem irrelevanten/nicht-kausalen Feature gepatcht.** Wenn auch diese Kontroll-Intervention downstream die erwartete Änderung auslöst, ist der Kaskaden-Befund wertlos ("irgendwas kaputt machen → irgendwas ändert sich"). Die Kaskade zählt nur als gezeigt, wenn die kausale Intervention propagiert **und** die nicht-kausale Kontrolle dies nicht tut.

### Wichtige methodische Klarstellung (Repräsentation vs. Verarbeitung)

Held-out-Generalisierung (H2) zeigt nur, dass das **Verhalten** systematisch ist. Das beweist noch nicht, dass es eine einzelne, lokalisierte, linear dekodierbare interne Repräsentation der Relation gibt — die Systematizität könnte auch über mehrere verteilte, redundante Pfade erzeugt werden, die sich erst in Kombination kohärent verhalten ("verteilte Heuristiken, die zusammen systematisch wirken" statt "eine saubere interne Variable").

**Die Evidenzkette, die wirklich überzeugend ist:**

```
Behavior (Held-out-Generalisierung)
   ↓  "Es kann die Relation"
Representation (Probing)
   ↓  "cash und price sind separat dekodierbar"
Intervention (Kausalität)
   ↓  "Wenn ich cash ändere, ändert sich affordance entsprechend"
= Konvergenz aller drei Evidenzarten ist das eigentlich starke Ergebnis.
```

Keine einzelne Methode ist für sich beweiskräftig:
- **Behavior allein** → sagt nichts über Mechanismus
- **Probing allein** → sagt nichts über Kausalität (Probe kann epiphenomenale Info finden)
- **Intervention allein** → kann Off-Distribution-Artefakte erzeugen

---

## 3. Held-out Splits und Filterlogik (S0–S3)

Die Generalisierungstests werden **nicht als zufällige Token-Splits** definiert, sondern anhand der **Ground-Truth-Zustände und -Transitionen** des Simulators. **Ganze Partien/Trajektorien bleiben innerhalb eines Splits**, um Leakage zwischen nahezu identischen Spielverläufen zu vermeiden. S0–S3 testen vier unterschiedliche Fragen — nicht drei Schwierigkeitsstufen desselben Splits.

### S0 — IID-Sanity-Baseline

S0 ist der Referenztest: zufällig held-out Partien aus derselben Zustandsverteilung wie das Training.

S0 beantwortet zunächst, ob Modell bzw. Probe die Aufgabe überhaupt zuverlässig gelernt haben. **Alle späteren S1–S3-Ergebnisse werden relativ zu dieser Baseline interpretiert** — ohne S0 kann ein schlechter S3-Wert nicht von "schlecht trainiert" unterschieden werden.

### S1 — Novel State Composition

S1 hält die einzelnen Wertebereiche der State-Variablen bekannt, hält aber bestimmte **Kombinationen dieser Variablen** aus dem Training zurück.

Beispiel: Die einzelnen Cash-, Position- und Ownership-Werte kommen im Training vor, bestimmte Kombinationen aus **Cash × Position × Ownership** jedoch ausschließlich im Test.

**Hypothese:** Das Modell kann bekannte Zustandskomponenten zu neuen Zuständen kombinieren, anstatt lediglich beobachtete vollständige Zustände zu memorisieren.

**Wichtig:** "Neue Kombinationen" bezieht sich auf Kombinationen von *State-Variablen* (nicht bloß neue vollständige Spielzustände), damit kontrollierbar ist, was genau held out ist. Einzelne Werte (z.B. `500`, `400`) dürfen im Training vorkommen — nur die Kombination `(500, 400)` darf nie zusammen auftreten, sonst könnte das Modell die Werte separat memorieren, ohne die Relation zu abstrahieren.

### S2 — Novel Relation / Transition Composition

S2 hält keine abstrakten frei wählbaren numerischen Operationen zurück, sondern **Relationen, die tatsächlich durch die Monopoly-Events entstehen**.

Cash-Transitionen entstehen insbesondere durch:
- `BUY`: `cash' = cash - price`
- `PAY_RENT`: `cash' = cash - 0.5 × price` (bzw. Zahlungsempfänger-Transition)
- `PASS`: keine relevante Cash-Änderung

Die zentrale relationale Entscheidung ist damit die **Affordability-Relation** `cash - price`. Deshalb werden bestimmte Abstandsrelationen zwischen Cash und Preis im Training beobachtet und andere held out.

Beispiel (**korrigiert nach empirischem Coverage-Check, siehe §0.1 und §3a — Relationen sind nicht durchgängig 50er-rund, da Rent krumme Cash-Werte erzeugt**):
- Training: die häufigen Relationen, u.a. `700` (2216 Vorkommen), `400` (1972 Vorkommen)
- Test: eine seltenere, aber ausreichend besetzte Relation, z.B. aus `{250, 325, 175, ...}` — konkrete Auswahl erst nach Anwendung der Mindest-Samplezahl aus §3a
- **> Überholt (Pre-Fix-Datenstand):** Die Vorkommenszahlen (2216/1972) und die Test-Kandidatenliste `{250, 325, 175}` beruhen auf dem durch den `states[i-1]`-Bug verfälschten Datenstand. **Maßgeblich: korrigierte S2-Kandidatenliste in §4.3.2** (Top-Relationen 850: 2840, 700: 2811; taugliche Kandidaten u.a. `400`/`650`/`500`/`600` mit je ~800 Punkten). `175` hat dort nur 146 Punkte und fällt unter die Mindest-N-Schwelle.

Damit kennt das Modell die beteiligten Cash- und Preiswerte sowie die BUY-Entscheidung, muss die gelernte Relation aber auf einen nicht beobachteten Abstand anwenden.

**Hypothese:** Das Modell generalisiert eine gelernte Zustandsrelation/-transition, statt nur beobachtete Cash-Preis-Konstellationen zu memorisieren.

Die konkreten Abstände werden **erst nach Inspektion der tatsächlich generierten Transitionen** festgelegt (siehe Coverage-Statistiken unten und §3a).

### S3 — Struktur-erhaltende Extrapolation

S3 testet Generalisierung außerhalb des im Training beobachteten Werte-/Zustandsbereichs.

Der Testbereich darf dabei **nicht lediglich deshalb gewählt werden, weil er numerisch "höher" liegt**. Er muss eine **struktur-erhaltende Extrapolation** sein: Die zugrunde liegenden Variablen und Operationen bleiben dieselben, während der konkrete Zustandsbereich außerhalb der Trainingsverteilung liegt.

Vor Konstruktion des Splits wird deshalb geprüft, welche Zustände im tatsächlich generierten Datensatz überhaupt natürlich vorkommen. Insbesondere wegen `max_turns`, der Preisstruktur und der Rentenregel darf S3 nicht auf einen theoretisch möglichen, aber praktisch nicht erreichten Wertebereich gelegt werden.

**Falls kein ausreichend großer natürlicher OOD-Bereich vorhanden ist, wird S3 nicht künstlich erzwungen, sondern als nicht durchführbar dokumentiert.**

### Split-Regeln

1. **Split auf Ground Truth:** Filter werden auf Zuständen und Transitionen des Simulators angewandt, nicht auf der Tokenfolge.
2. **Trajectory integrity:** Eine Partie/Trajektorie gehört vollständig zu genau einem Split (kein Token-Level-Splitting).
3. **Strict held-out probing:** Bei OOD-Probes respektiert auch das Probe-Training den jeweiligen Split. Insbesondere bei S3 wird der Probe ausschließlich auf der Trainingsregion trainiert und ausschließlich auf der held-out Region evaluiert.
4. **Pre-registration of filters:** Die Filterbedingungen werden vor der finalen Modellauswertung festgelegt und anhand des generierten Datensatzes auf Abdeckung und ausreichende Samplezahlen geprüft.
5. **No forced OOD:** Ein Split gilt nicht als erfolgreich, wenn seine Testregion nur durch seltene oder künstlich konstruierte Zustände gefüllt werden kann.

**Reihenfolge:** S1 zuerst (Minimaltest — wenn das schon scheitert, sind S2/S3 akademisch). Bei Erfolg erst dann S2, dann S3.

### Coverage-Statistiken (vor dem Training des GPT)

Bevor das GPT überhaupt trainiert wird, wird der Generator einmal laufen gelassen und für S1–S3 Coverage-Statistiken ausgegeben:

```text
S0: N train / N test
S1: N train / N test
S2: counts per (cash-price) relation
S3: min/max + Quantile + Anzahl erreichbarer Zustände
```

Gerade S2 ist wichtig: Wenn `±50` im Generator viel häufiger vorkommt als `±200`, kann ein vermeintlicher "held-out relation"-Test durch Verteilungsunterschiede verzerrt werden. Bei S3 wird **erst nach diesem Histogramm entschieden, wo die Grenze liegt** — nicht vorher ein Wertebereich wie `1050–1500` festgelegt.

### 3a. Konkrete Konsequenzen aus dem empirischen Coverage-Check (2000 Spiele)

Der in §0.1 beschriebene Coverage-Check wurde durchgeführt. Ergebnis für S2 und S3:

**S2 — Mindest-Samplezahl pro held-out Relation:** Von 57 beobachteten Cash-Preis-Relationen sind nur wenige stark besetzt (`700`: 2216, `400`: 1972 Vorkommen — **> überholt (Pre-Fix-Datenstand), korrigierte Werte in §4.3.2**), viele liegen bei ~100. Als Mindestschwelle für eine als S2-Kandidat zulässige held-out Relation gilt: **N ≥ 200–300 Vorkommen**, um sowohl Trainings- als auch Test-Split des Probes verlässlich zu befüllen. Relationen unterhalb dieser Schwelle werden nicht als alleinstehender S2-Split verwendet. Falls nach Anwendung dieser Schwelle zu wenige Relationen übrigbleiben, ist die Konsequenz **mehr Spiele generieren**, nicht die Schwelle aufweichen. *(Schwellenlogik bleibt gültig; nur die hier zitierten Beispielzahlen sind durch den `states[i-1]`-Bug veraltet — maßgeblich ist §4.3.2.)*

**S3 — ENTSCHIEDEN: nicht durchführbar in V0, V1-Erweiterung vorgemerkt.** Bei `max_turns=40`, Startkapital 1000, Preisliste bis 300 liegen q99 der Cash-Verteilung bei 1000, nur 760 von ~1,06 Mio. States über 1200. Das ist kein tragfähiger natürlicher OOD-Bereich (vgl. Split-Regel 5, "No forced OOD", §3). Zusätzlich spricht **die Kontextlängen-Constraint** (§5.2: ~128–256 Tokens) gegen die naheliegende Gegenmaßnahme: Cash-Extremwerte streuen grob mit √Turns (Random-Walk über Rent-Events), sodass ein tragfähiger OOD-Bereich eine Verdopplung/Verdreifachung von `max_turns` erforderte — das sprengt das Token-Budget pro Partie bzw. verletzt die Trajectory-Integrity-Regel (§3) beim Kappen.

**Entscheidung: Option (b).** S3 wird bewusst als "nicht durchführbar in V0" dokumentiert und aus dem V0-Scope gestrichen. Das ist ein **pre-registrierter Negativbefund**, kein Rückzug — genau das verlangt Split-Regel 5 und das Kernprinzip §8 (klein anfangen). S3 blockiert damit S0/S1/S2 und das GPT-Training nicht.

**V1-Vormerkung für S3 (größerer Preisraum statt mehr Turns):** Falls S3 später als eigener Folgeversuch angegangen wird, dann über ein strukturell anderes Board mit **höheren Preisstufen** (nicht längere Spiele), um den OOD-Bereich ohne Sequenzlängen-Wachstum zu erzeugen. Dabei gelten drei Vorsichtsregeln:
1. **Board bleibt fix (BOARD_SEED)** — sonst wiederholt sich die §4.1-Lernproblematik (Preis steht nie explizit im Token-Stream). Das ist notwendig, aber nicht hinreichend.
2. **Preis-Mindestfrequenz pro Stufe:** Jede neue/höhere Preisstufe ist wieder eine "gelernte Konstante pro Feld", die erst über genug Vorkommen gelernt werden muss. Bei fixer Feldzahl (12) verdünnt mehr Preisstufen die Besuchsfrequenz pro Stufe → für V1 vorab prüfen, dass jede Preisstufe eine Mindestanzahl von Vorkommen im generierten Datensatz hat (analog zur S2-N-Schwelle).
3. **V1-Coverage-Check als gemeinsame Neuvermessung von S2 UND S3:** Da beide an derselben Stellschraube (Preisliste) hängen, verändert der größere Preisraum auch die S2-Relationsverteilung. Der V1-Check darf deshalb nicht nur "reicht der OOD-Bereich" (S3) fragen, sondern muss gleich die S2-Mindest-N-Schwelle (§3a) mit den neuen Preisstufen neu validieren — sonst löst man S3 und importiert sich unbemerkt eine neue S2-Schieflage.

---

## 4. Simulator (V0) — bereits implementiert und getestet

**Design-Prinzip:** V0 ist bewusst "langweilig" gehalten. Keine Häuser, keine Hypotheken, keine Ereigniskarten, kein echtes Monopoly-Regelwerk. Keine Würfel-Stochastik als Unsicherheitsquelle (Würfel sind seed-deterministisch und für das Modell im Kontext sichtbar) — Hidden Information ist ein bewusst nach hinten verschobenes, separates Experiment (siehe Abschnitt 7, "Stretch Goal").

### Projektstruktur

```
mini-monopoly/
├── simulator/
│   ├── __init__.py
│   ├── state.py       (70 Zeilen)  — GameState-Dataclass + Invarianten
│   ├── rules.py        (115 Zeilen) — Event-Anwendung mit Pro-Event-Konsistenzchecks
│   ├── policies.py     (45 Zeilen)  — 3 austauschbare Kauf-Heuristiken
│   └── game.py          (115 Zeilen) — Game-Loop, erzeugt Event-Token + Ground-Truth-State pro Schritt
├── data/
│   └── generate.py      — Datengenerierung: viele Partien, gemischte Seeds/Policies, JSONL + Config-Manifest
├── tests/
│   └── test_invariants.py (103 Zeilen) — 5 Tests, alle grün
└── EXPERIMENT.md         (dieses Dokument)
```

### `state.py` — GameState

```python
@dataclass
class GameState:
    board_size: int
    num_players: int
    prices: list[int]              # prices[field_idx], 0 = kein Grundstück
    rent_rate: float = 0.5         # Miete = rent_rate * price

    position: list[int]            # position[player] -> Feldindex
    cash: list[int]                # cash[player] -> Kontostand
    owner: list[Optional[int]]     # owner[field_idx] -> player_idx oder None

    turn: int
    current_player: int
```

- `.to_dict()` → vollständiger, JSON-serialisierbarer Ground-Truth-Snapshot
- `.check_invariants()` → wirft `AssertionError` bei Inkonsistenz (negativer Cash, ungültige Positionen/Owner, preislose Felder mit Owner). Wird nach **jedem** Event automatisch aufgerufen.

### `rules.py` — Event-Anwendung

Vier Events: `apply_roll_and_move`, `apply_buy`, `apply_pass`, `apply_rent`. Jede Funktion:
1. Nimmt Snapshot vor der Änderung
2. Wendet die Änderung an
3. Nimmt Snapshot danach
4. **Prüft explizit, dass NUR die für diesen Event-Typ erlaubten State-Keys sich verändert haben** (`_assert_only_changed`) — z.B. `ROLL_MOVE` darf nur `position` ändern, `BUY` nur `cash`+`owner`, `PAY_RENT` nur `cash`. Das ist der zentrale Debugging-Mechanismus aus der Diskussion: Simulator-Bugs werden sofort lokalisiert, statt erst als falsches Interpretability-Ergebnis wieder aufzutauchen.
5. Ruft `state.check_invariants()` auf

`apply_pass` verändert bewusst nichts am State, ist aber ein **eigenes Event-Token** — wichtig, damit das spätere History-only-Modell "kein Kauf, weil Feld schon besetzt" von "kein Kauf, obwohl möglich" unterscheiden kann.

`RuleViolation`-Exception bei Regelverstoß (z.B. Kauf ohne ausreichend Cash).

### `policies.py` — Kauf-Heuristiken

Drei Policies: `always_buy`, `conservative` (kauft nur wenn danach ≥300 Cash übrig), `random` (50/50 wenn möglich). **Grund für mehrere Policies:** Wenn Trainingsdaten nur einer Heuristik folgen, kann eine spätere SAE auf Artefakte dieser einen Heuristik overfitten statt die zugrundeliegende Zustandsstruktur zu lernen. Bei der Datengenerierung sollten Policies gemischt werden.

### `game.py` — Game-Loop

`play_game(game_id, seed, policy_name, board_size=12, num_players=3, max_turns=40) → GameLog`

- `make_prices()`: Feld 0 (Start) immer preislos, restliche Felder aus fester Preisstufenliste `[0,100,150,200,250,300]`. **Seit Bugfix (siehe 4.1): Board-Layout ist über ALLE Partien identisch (`BOARD_SEED` fix), nicht mehr abhängig vom Spiel-Seed.**

### 4.1 Bugfix: Preise waren nicht lernbar (gefunden von Tobs' Coding-LLM)

**Problem:** Preise wurden ursprünglich pro Spiel aus dem Spiel-Seed generiert. Der Preis eines Feldes steht aber an **keiner Stelle** explizit im Event-Token-Stream — `apply_buy` zeigt keinen Preis, `apply_rent` zeigt erst den Rent-Betrag (`= price * 0.5`), und zwar erst **nachdem** auf dem Feld schon einmal Miete gezahlt wurde. Für ein Feld, das im bisherigen Spielverlauf noch nie vermietet wurde, ist der Preis aus den Tokens schlicht nicht rekonstruierbar — und da Preise pro Spiel zufällig sind, auch nicht über mehrere Spiele hinweg lernbar.

**Konsequenz, wäre es unentdeckt geblieben:** H2 (Affordability) wäre am ersten Kauf-Entscheidungspunkt eines Feldes strukturell unlösbar gewesen — nicht wegen fehlender Kompositionalität im Modell, sondern wegen fehlender Information im Input. Ein H2-Fehlschlag hätte sich wie ein negatives mechanistisches Ergebnis angefühlt, wäre aber nur ein Artefakt gewesen — exakt das Szenario, vor dem die Invariant-Tests eigentlich schützen sollten (hier aber nicht gegriffen haben, weil es kein Simulator-Bug im engeren Sinne ist, sondern ein Informationsdesign-Problem).

**Fix:** Preise werden jetzt aus einem festen `BOARD_SEED` generiert, unabhängig vom Spiel-Seed — das Board-Layout ist über alle Partien identisch. Das Modell kann den Preis pro Feld als gelernte Konstante memorieren (analog zu fixen Spielregeln bei Othello-GPT, wo die Legalitätsregeln ebenfalls fix sind und nur die Zugfolge variiert). Cash bleibt die tatsächlich dynamische, pro Spiel/Spieler zu verfolgende Variable — die eigentlich interessante Rechenoperation (`cash >= price`) bleibt bestehen, nur der Preis-Teil ist jetzt aus dem Kontext (implizit gelernt über Feld-Position) statt aus dem Token-Stream ableitbar.

**Wichtige Implikation für H1/H2:** Ein Cash-Probe testet jetzt wirklich, ob das Modell die dynamische Variable trackt. Ein Affordability-Test (H2) testet jetzt wirklich Komposition aus "gelernter Konstante (Preis pro Feld)" + "verfolgter Variable (Cash)" — nicht mehr eine unlösbare Aufgabe. Für spätere Erweiterungen mit variablem Board (siehe Abschnitt 7) müsste diese Problematik erneut bedacht werden — z.B. durch einen expliziten `PRICE`-Token beim ersten Betreten eines Feldes.
- Pro Runde: `ROLL_MOVE` → falls unbesetztes Preisfeld: `BUY`/`PASS` gemäß Policy → falls fremdes Eigentum: `PAY_RENT`.
- **`GameLog.events`** und **`GameLog.states`** haben exakt dieselbe Länge — jedes einzelne Token (auch Zwischentoken wie Spieler-ID `"P0"` oder Event-Typ `"ROLL"`) hat einen zugeordneten vollständigen Ground-Truth-Snapshot. Das ist die Grundvoraussetzung für spätere `activation[layer, token] → ground_truth_state[token]`-Zuordnung.
- `GameLog.to_dict()` → JSON-serialisierbar, für den Datenexport.

**Beispiel-Output** (Auszug, seed=42, policy="always_buy", board_size=12, 3 Spieler):
```
  0 START      pos=[0, 0, 0]   cash=[1000, 1000, 1000] owner=[None]*12
  1 P0         pos=[6, 0, 0]   cash=[1000, 1000, 1000] owner=[None]*12
  2 ROLL       pos=[6, 0, 0]   cash=[1000, 1000, 1000] owner=[None]*12
  3 6          pos=[6, 0, 0]   cash=[1000, 1000, 1000] owner=[None]*12
  4 P0         pos=[6, 0, 0]   cash=[900, 1000, 1000]  owner=[...,0,...]  (Kauf-Event: nur cash+owner ändern sich)
  5 BUY        pos=[6, 0, 0]   cash=[900, 1000, 1000]  owner=[...,0,...]
```

### `tests/test_invariants.py` — 5 Tests, alle grün

1. `test_money_conserved_across_full_games` — Cash-Summe + investiertes Kapital (Summe der Kaufpreise besetzter Felder) muss konstant = Startkapital bleiben, über alle drei Policies.
2. `test_no_negative_cash`
3. `test_ownership_valid_range`
4. `test_events_states_same_length`
5. `test_position_matches_roll_sum_sanity` — **Der kostenlose Position-Sanity-Check**: rekonstruiert Position manuell aus Roll-Summe und vergleicht mit dem tatsächlichen State. Wichtig: beim Parsen ist zu beachten, dass Event-Token wie `"PASS"`/`"PAY_RENT"` auch mit `"P"` beginnen — Spieler-Token nur erkennen, wenn der Rest reine Ziffern sind (`tok[1:].isdigit()`), sonst Fehlklassifikation.

Alle Tests laufen aktuell fehlerfrei durch (`python3 tests/test_invariants.py` → `Alle Invariant-Tests bestanden.`).

---

### 4.2 `data/generate.py` — Datengenerierung

`generate(output_dir, num_games, seed_offset, board_size=12, num_players=3, max_turns=40, policies=None, game_id_start=0) → Pfad zur JSONL-Datei`

- Erzeugt `num_games` Partien über gemischte Seeds (aus `seed_offset` deterministisch abgeleitet) und gemischte Policies (Default: alle drei aus `POLICIES`), schreibt sie zeilenweise als JSONL (ein `GameLog.to_dict()` pro Zeile).
- **Config-Manifest:** Neben der JSONL-Datei wird automatisch eine `.manifest.json` mit allen Erzeugungsparametern geschrieben (Kernprinzip §8: Datensatz muss eindeutig seiner Config zuordenbar sein). Dateiname wird automatisch aus der Config gebaut: `games_so<seed_offset>_n<num_games>_<policies>.jsonl`.
- **`game_id_start`-Parameter:** verhindert `game_id`-Kollisionen, wenn mehrere Generierungsläufe (z.B. unterschiedliche Seed-Offsets/Policy-Mischungen) später zu einem gemeinsamen Trainingsdatensatz zusammengeführt werden sollen. Beispiel: Lauf 1 mit `--game-id-start 0`, Lauf 2 mit `--game-id-start 100000`.
- CLI: `python3 data/generate.py --num-games 2000 --seed-offset 0 --output-dir data [--game-id-start N] [--policies always_buy conservative]`
- **Wichtig:** `generate.py` erzeugt nur den Basisdatensatz (entspricht S0/Rohdaten). Die held-out-Filterlogik für S1–S2 setzt als eigener Schritt darauf auf — kein erneutes Sampling nötig. **S3 ist in V0 nicht durchführbar** (siehe §3a) und wird nicht implementiert.

### 4.3 `data/splits.py` — Held-out-Split-Logik (S0/S1/S2)

Implementiert die Split-Regeln aus §3 auf dem Basisdatensatz. Analyse-Einheit für S1/S2 sind die **Kauf-Entscheidungspunkte** (BUY/PASS). Die Affordability-Relation eines Entscheidungspunkts ist `relation = cash - price`.

### 4.3.1 WICHTIGER Bugfix: Pre-Buy-Zustand (empirisch verifiziert)

Der Pre-Buy-Zustand liegt **NICHT** bei `states[i-1]`. Grund: `apply_buy`/`apply_pass` geben `["P{player}", "BUY|PASS"]` zurück, und **beide** Tokens bekommen denselben bereits veränderten Post-Buy-State (bei BUY: Cash schon um den Preis reduziert). Empirisch geprüft: `states[i-1]` ist bei allen 361 BUY-Punkten bereits Post-Buy (Verletzung der ursprünglichen Annahme); der letzte State mit Position fest + Cash unverändert ist das **Roll-Wert-Token** (`states[i-2]`, Struktur `... P ROLL w P BUY|PASS`). Verifiziert: 2066/2066 Entscheidungspunkte konsistent, 0 Verletzungen. Gesichert durch `_pre_decision_state()` (rückwärts suchend, robust gegen hartes `i-2`) und `tests/test_splits.py::test_decision_point_pre_state`. **Ein stiller Fehler hier hätte sich als verfälschtes H2-Ergebnis gezeigt** (Relation um genau den Preis verschoben, nur bei BUY, bei PASS unsichtbar) — analog zur §4.1-Logik, eine Ebene höher.

**Konsequenz für die Verteilung:** Durch den Fix sind die Relationen jetzt aus dem Cash **vor** der Entscheidung abgeleitet. Die frühere Auswahl häufiger Relationen (z.B. `400`) basierte auf verzerrten Werten; die korrigierte Top-Relation ist `850` (2840×), gefolgt von `700` (2811×).

**Voller Verifikationslauf (2000 Spiele):** Pre-Buy-State-Fix auf dem vollen Datensatz bestätigt — **0 Verletzungen bei 19.946 Entscheidungspunkten** (vorher Teil-Check auf 200-Spiele-Fixture mit 2.066 Punkten). Damit ist die Korrektheitsgarantie auf vollem Umfang abgesichert.

**Korrigierte S2-Kandidatenliste (2000 Spiele, beide Schwellen):** Von 22 Relationen erfüllen min_n ≥ 200 UND Game-Impact ≤ 35% (Trajectory-Integrity). Die früher im Chat favorisierten `175`/`325` sind NICHT mehr die besten: `175` hat nur 146 Punkte (< 200). Gut besetzte Kandidaten (Relation → Punkte → Impact):
```
400 → 843 (34%)    650 → 843 (32%)    500 → 797 (33%)
600 → 785 (33%)    625 → 774 (29%)    475 → 567 (25%)
750 → 547 (25%)    350 → 532 (24%)    450 → 454 (20%)
425 → 422 (19%)    575 → 401 (18%)    800 → 397 (17%)
325 → 381 (17%)    250 → 334 (16%)    ... bis 675 → 216 (10%)
```
Sowohl "runde" (400, 500, 600) als auch "krumme" (475, 625, 425, 575) Relationen sind vertreten — gut, da S2 die Relation unabhängig von der Rundheit testet. **Endgültige Auswahl der held-out S2-Relation(en) wird beim eigentlichen Split-Bauen (nach dem Tokenizer) getroffen**, nicht vorher (§3).

### 4.3.2 S2-Auswahlkriterien (Mindest-N + Partien-Impact)

Zwei Schwellen, beide in `main()` geprüft:
- **Mindest-N (§3a):** `check_s2_viability` — Relation braucht ≥ `--s2-min-n` (Default 200) Entscheidungspunkte; sonst abgelehnt.
- **Partien-Impact (Trajectory-Integrity):** `s2_game_impact` — wie viele *Partien* würden komplett in den Test wandern (da eine Partie mit ≥1 Vorkommen der held-out Relation ganz test-seitig wird). Warnung bei >35% der Partien. **Grund:** Reine Entscheidungspunkt-Häufigkeit reicht nicht — z.B. betrifft Relation `700` (296×) laut `s2_game_impact` 168/200 Partien (84%) und würde das Trainingsset damit fast vollständig in den Test ziehen. Seltenere Relationen sind deshalb nicht nur wegen Verteilungs-Bias, sondern wegen Trajectory-Impact vorzuziehen.

Kernfunktionen:
- `coverage_report(games)` → Coverage-Statistiken (§3): Anzahl Spiel/States/Entscheidungspunkte, Cash-Quantile, distinct Relationen/Paare, Häufigkeit pro Relation.
- `split_s0(games, test_frac, seed)` → S0 IID-Baseline: zufällige held-out Partien.
- `split_s1(games, held_out_pairs)` → S1: held-out `(cash, price)`-Kombinationen, Einzelwerte bleiben bekannt.
- `split_s2(games, held_out_relations)` → S2: held-out Werte der Relation `cash - price`.
- `check_s2_viability(...)` → Mindest-N-Prüfung (§3a).
- `s2_game_impact(...)` → Partien-Impact-Prüfung (Trajectory-Integrity).

CLI:
```
python data/splits.py --input data/games_*.jsonl --coverage          # nur Bericht
python data/splits.py --input data/games_*.jsonl --build \
    --s1-pairs 500,150 600,200 \
    --s2-relations 700 850 --s2-min-n 200
```

**Verifizierte Split-Invarianten** (`tests/test_splits.py`, alle grün):
1. **Trajectory Integrity:** keine überlappenden `game_id`s zwischen train/test.
2. **Kein Leakage:** held-out Bedingung (S1-Paare / S2-Relation) erscheint NICHT im Trainings-Split.
3. **S2-Mindest-N (§3a):** unterbesetzte Relationen werden abgelehnt; Konsequenz bei zu wenigen → mehr Spiele generieren, Schwelle nicht aufweichen.
4. **S2-Partien-Impact:** `s2_game_impact` konsistent, >35%-Warnung greift.
5. **Pre-Buy-State:** BUY-Reduktion `post.cash == pre.cash - price` über alle Punkte verifiziert (Bugfix).
6. **S0-Aufteilung:** train+test = Gesamtmenge, Test-Anteil ≈ `test_frac`.
7. **Coverage-Bericht konsistent.**

**Config-Manifest (Kernprinzip §8):** `save_split` schreibt zu jedem Split eine `.manifest.json` mit erzeugender Config (Split-Typ, held-out Paare/Relationen, min_n, seed, test_frac, Dateinamen, Spielanzahl) — analog zum `generate.py`-Manifest.

**Wichtig:** Die tatsächlichen Split-Grenzen (welche Paare/Relationen held out) werden **erst nach dem Coverage-Bericht** festgelegt (§3), mit Mindest-N-Schwelle (§3a) und Partien-Impact-Schwelle (4.3.2).



### 5.1 Tokenisierung von Cash — ENTSCHIEDEN

**Feste Entscheidung (aus der Diskussion mit GPT/Coding-LLM):**

**Haupt-Encoding — strukturierte Digit-Tokens mit expliziten Stellenmarkern:**

```
P0 MONEY H7 T3 O0
```

- Numerische Größen (Cash) werden als **Ziffern mit Stellenmarker** tokenisiert: `H0..H9` (Hunderter), `T0..T9` (Zehner), `O0..O9` (Einer) als **eigene, atomare Tokens**.
- Kategoriale Größen bleiben **atomar**: `P0`, `P1`, ... als Spieler; `PROPERTY_12`, `OWNER_P0` usw. direkt.
- **Keine numerische Initialisierung der Embeddings.** Das Modell bekommt nur diskrete Tokens und muss die numerische/Stellenwert-Struktur selbst lernen. Würde man Embeddings numerisch vorstrukturieren, könnte man sich das Ergebnis (z.B. einen "konstanten +50-Schritt im Aktivierungsraum") selbst einbauen — das Encoding-Confound, das wir vermeiden wollen.

**Warum H/T/O statt Bucketing (die Begründung):**
- **Bucketing** gibt dem Modell bereits starke *diskrete* Semantik vor (`MONEY_700` vs. `MONEY_800` als völlig verschiedene Tokens). Danach kann man nicht mehr unterscheiden, ob das Modell Geld als kontinuierliche Größe oder als Kategorien gelernt hat.
- **H/T/O** zwingt das Modell, die numerische Struktur selbst zu bilden. Das eröffnet eine Hierarchie testbarer Hypothesen:
  - H1: Geld ist intern überhaupt repräsentiert (linearer Probe auf Cash).
  - H2: Die Repräsentation ist kompositionell (Hunderter/Zehner/Einer separat dekodierbar).
  - H3: Numerische Nähe entspricht geometrischer Nähe (`700→750` vs. `700→1200` unterscheiden sich systematisch).
  - H4: Die numerische Repräsentation ist **kausal** (State-Swap-Intervention, s.u.).
  - H5: Dieselbe Intervention funktioniert **out-of-distribution** (analoge Manipulation bei kaum gesehenen Cash-Werten → "gelernte Operation" vs. "memorierte Tabelle").
- **Zwingend für die Splits:** S2 (`cash-price`-Relationen, siehe §3a) und S3 (nie gesehene höhere Werte) sind mit Buckets strukturell unsinnig; H/T/O macht sie möglich.

### 5.1a Tokenisierungs-Fix: Tausenderstelle fehlt — ENTSCHIEDEN nach Coverage-Check

**Problem (siehe §0.1):** Der empirische Coverage-Check auf 2000 Spielen zeigt einen maximalen Cash-Wert von **1500**. Das in §5.1 festgelegte H/T/O-Schema deckt nur `0–999` ab (`H0..H9 T0..T9 O0..O9`) — Cash-Werte ≥ 1000 sind damit nicht tokenisierbar.

**Fix:** Ein zusätzlicher, minimaler Stellenmarker für die Tausenderstelle. Da der beobachtete Bereich nur `0` oder `1` Tausender umfasst (max. 1500), reicht ein **binäres Token** statt einer vollen `K0..K9`-Range:

```
P0 MONEY TH1 H5 T0 O0   (Cash = 1500)
P0 MONEY TH0 H9 T0 O0   (Cash = 900)
```

- `TH0`/`TH1` als eigene, atomare Tokens (analog zu `H`/`T`/`O`).
- Kein volles `K0..K9`-Vokabular, da unnötige Sparsity für ein Token, das nur bei Cash ≥ 1000 überhaupt auftritt.
- Falls spätere Simulator-Änderungen (z.B. die V1-Preisraum-Erweiterung für S3, siehe §3a) den Cash-Bereich über 1999 hinaus verschieben, muss `TH` auf `TH0..TH9` erweitert werden — vor dem nächsten Trainingslauf erneut coverage-prüfen.
- **Auch hier: keine numerische Embedding-Initialisierung** — gleiche Begründung wie bei H/T/O (Encoding-Confound vermeiden).

**Ablation — Bucketing:** Aus dem Hauptmodell **raus**. Als **Negativ-/Kontrollbedingung mit präzisem Trigger** behalten, nicht als vages "später". Konkreter Trigger:

> Das Bucket-Modell wird gebaut, **sobald H2 (Affordability) mit S1-Split erfolgreich ist UND der Digit-Probe (§5.1 H2: Hunderter/Zehner/Einer separat dekodierbar) eine klare, layer-weise ansteigende numerische Struktur zeigt.**

Begründung: Erst dann existiert etwas, wogegen kontrastiert werden kann ("Digit-Encoding zeigt X, Bucket-Encoding zeigt X nicht"). Vorher ist die Kontrolle witzlos, weil unklar ist, *was* man kontrastiert. Als Schritt ist er **zwischen §6-Punkt 7 (H2 testen) und Punkt 9 (H3) eingehängt** — nicht nachgeschoben. Wegen Compute-Constraint (§5.3): es bleibt beim Hauptmodell erst ein sauber analysiertes; das Bucket-Modell ist der erste *zusätzliche* Trainingslauf, sobald der Trigger erfüllt ist.

**Wichtiger methodischer Hinweis:** Der `+50`-Geometrie-Test ist **kein direkter Beweis** einer gelernten Operation — er zeigt nur systematische Struktur im Aktivierungsraum. Für die kausale Interpretation braucht es anschließend die **Intervention** (State-Swap), plus eine Kontroll-/Permutations-Baseline, um Token-/Positionsartefakte auszuschließen.

### 5.2 Modellarchitektur (grobe Eckdaten, für 8GB VRAM realistisch)

- 4–6 Transformer-Layer
- `d_model` ~256–384
- 4–6 Attention-Heads
- Kontextlänge ~128–256 Tokens
- ~5–15M Parameter gesamt
- FP16-Training

**Kontextlänge (empirisch, voller Datensatz, nach Tokenisierung §6-4):** min=180, Median 226, max=284; 66/2000 Spiele (3.3%) überschreiten 256. Da Tokenisieren der Miete als 5-Token-Digit-Sequenz jeden Rent-Event aufbläht, **wird `n_ctx = 288` gewählt** (≥ max. 284), um KEINE Trajectory zu kürzen (Trajectory-Integrity-Regel, §3). Das überschreitet die obige 256-Obergrenze nur geringfügig (~12%) bei vernachlässigbarem Parameterzuwachs und erhält die Integrität aller Partien. Die Zahl ist an die aktuelle Tokenisierung gebunden; ändert sich die Digit-Zerlegung, muss sie neu gemessen werden.

**Grober VRAM-Überschlag für `n_ctx=288` auf der RTX 2080 (8GB):** Bei `d_model=256`, 4 Layer, `d_mlp=1024`, `d_vocab=48` ≈ **3.3M Parameter** (≈7 MB FP16). Trainings-Aktivierungen bei Batch 32: ~32×288×256×4 Layer ≈ 75 MB FP16; Adam-Optimizer + Gradienten addieren ~3× Gewichte. Selbst `run_with_cache` (hält Aktivierungen **aller** Layer gleichzeitig) liegt bei Batch 32 im Bereich ~0.4–2 GB je nach Hook-Umfang. Für 8GB ist damit **komfortabel Luft**; falls OOM droht, zuerst Batch halbieren (16), nicht `n_ctx` senken (Integrität). Der Überschlag ist bewusst konservativ — vor dem ersten echten Trainingslauf einmal mit einem echten Batch laufen lassen und VRAM prüfen.

Das ist eine völlig andere Größenordnung als reguläre LLM-Finetuning-Jobs — realistisch in Stunden, nicht Tagen, auf einer RTX 2080 trainierbar. Vorteil zieht sich auch durch die Analyse-Pipeline: Residual-Stream-Dimension (256–384) ist so klein, dass auch SAE-Training auf den Aktivierungen dieses Modells trivial billig ist (Minuten statt Tage, im Vergleich zur Qwen-Arbeit).

### 5.3 "Ein Modell, viele Analysen" statt viele Trainingsläufe

Wegen der Hardware-Beschränkung: **nicht** sechs separate Modellversionen (V1 nur Position, V2 +Cash, V3 +Ownership, ...) einzeln trainieren. Stattdessen:

- **Ein** History-only-Modell trainieren, das die volle Ereignishistorie sieht (Position, Cash, Ownership sind alle nur implizit über Events erschließbar, nie explizit im Input).
- Danach beliebig viele *Analysen* auf demselben Checkpoint fahren: Position-Probe, Cash-Probe, Ownership-Probe, Cross-Variable-Probes, Interventionen — das sind reine Forward-Passes, kein erneutes Training.
- Das explizite-State-Modell ("Modell A" aus früheren Diskussionsrunden, das den Zustand direkt im Input bekommt, zum Vergleich mit dem History-only-Modell "Modell B") ist **zunächst gestrichen** — kostet einen kompletten zusätzlichen Trainingslauf und ist nur relevant, falls das History-only-Modell überhaupt erstmal etwas Interessantes zeigt. Erst bei Erfolg von H1 wieder aufgreifen.
- Seeds/mehrere unabhängige Trainingsläufe für Reproduzierbarkeit: ebenfalls zurückgestellt, bis ein erstes reproduzierbares mechanistisches Phänomen gefunden wurde ("lohnt sich ein zweiter Seed erst danach, nicht vorher").

### 5.4 Analyse-Reihenfolge: Layer-wise Probing vor SAE

Vor jeder SAE-Analyse: Probing-Accuracy pro Layer messen (nicht nur eine Gesamt-Accuracy), um zu sehen, **wo** im Netzwerk welche Information entsteht:

```
Layer 0   position 30%     cash 10%
Layer 1   position 60%     cash 15%
Layer 2   position 90%     cash 35%
Layer 3   position 99%     cash 80%
Layer 4   position 99%     cash 95%
```

Das ergibt eine grobe "Computational Story" (z.B. Position wird zuerst rekonstruiert, ökonomischer Zustand danach) — mit Vorsicht zu interpretieren, aber ein guter Ausgangspunkt für gezielte SAE-Analyse. **SAE erst einsetzen, wenn klar ist, "hier zwischen Layer X und Y passiert etwas Interessantes"** — nicht explorativ ohne Hypothese ("ich hab ne SAE, mal schauen ob irgendein Feature Monopoly macht").

### 5.5 Framework / Analyse-Pipeline — ENTSCHIEDEN (Spike abgeschlossen)

**Entscheidungsgrundlage:** Das Experiment lebt von **Aktivierungs-Interventionen** (H3: Cash-Repräsentation patchen, H4: Kaskaden-Konsistenz). Das braucht direkten Residual-Stream-Zugriff pro Layer, sauberes Hooking und gezieltes Überschreiben von Aktivierungen an bestimmten Token-Positionen. HuggingFace Transformers ist dafür nicht gebaut — man kämpft gegen die Abstraktion. Für 8GB VRAM und 5–15M Parameter spielt die Frameworkwahl bei reinem Training kaum eine Rolle; der Unterschied entsteht komplett in der **Analyse-Pipeline**.

**Empfehlung (Primär): TransformerLens** (Neel Nanda) für Training + Analyse.
- Gebaut exakt für diesen Workflow: `run_with_cache` (Layer-wise-Probing), `run_with_hooks` (Activation-Patching), native Unterstützung für kleine/custom Modelle. Spart Wochen an Boilerplate für H3/H4, weil Activation Patching bereits eingebaut ist. Dieselbe Library, die bei der zitierten Othello-GPT-Nachfolgearbeit (§1) verwendet wurde.
- **Nachteil:** eigene Lernkurve; beim custom H/T/O/TH-Tokenizer muss das Vokabular/Embedding-Setup manuell an die Library angepasst werden — machbar, nicht Zero-Config.

**Alternative (Fallback):** nanoGPT-artiges rohes PyTorch-Setup + eigene Hooks (`register_forward_hook` auf Residual-Stream-Punkte).
- Maximale Kontrolle über den ungewöhnlichen Tokenizer, mehr Eigenbau, aber kein vorgefertigtes Activation-Patching.

**Entscheidung: kurzer Spike vor der Festlegung.** Vor §6-Punkt 4 (Tokenizer/Vocab bauen) wird getestet, ob sich der custom H/T/O/TH-Tokenizer sauber in TransformerLens einbinden lässt. Wenn ja → TransformerLens (Primär). Wenn der Custom-Tokenizer zu viel Reibung verursacht → nanoGPT + eigene Hooks. Die Framework-Wahl mitbestimmt, **wie** das Custom-Vokabular eingebunden wird, daher **vor** dem Tokenizer-Bau entscheiden.

**Ergebnis des Spikes (`spikes/spike_tl.py`, ausgeführt, alle Tests grün):**

- **A — `run_with_cache`:** Aktivierungen pro Layer × Token lesbar (`resid_post` Form `(seq, d_model)`), für Layer-wise-Probing (H1/H2, §5.4) direkt nutzbar.
- **B — `run_with_hooks`/Patching:** Gezieltes Überschreiben einer einzelnen Token-Position (Position 6, Layer 2) → messbarer kausaler Effekt auf die Vorhersage-Logits an der Folgeposition (Differenz ≈ 1.33). Genau das Muster, das H3/H4 brauchen.
- **C — HookPoint-Zugriff:** `blocks.1.hook_resid_pre` greifbar; direkter Residual-Stream-Zugriff pro Layer funktioniert.
- **Custom-Digit-Vokabular:** 48 Tokens (TH/H/T/O + kategoriale) ohne HF-Tokenizer-Umweg problemlos einbindbar — das ist exakt das §5.1a-Encoding.

**Schlussfolgerung: TransformerLens ist gewählt** (Primär-Option aus der Empfehlung); kein Fallback nötig. Umgebung: Python 3.14, PyTorch 2.13 (CUDA auf RTX 2080 verfügbar), TransformerLens 3.7.0. **Hinweis:** TransformerLens 3.7 warnt, dass `HookedTransformer` in 4.0 zugunsten von `TransformerBridge` ersetzt wird — für die Zielgröße (5–15M, 4–6 Layer) aktuell unkritisch, Fallback bleibt dokumentiert.

**Follow-up-Anmerkungen (beim echten Aufbau zu beachten, NICHT durch den Spike abgedeckt):**

1. **Logit-Differenz 1.33 ist nur ein Mechanik-Nachweis, kein interpretierbares Ergebnis.** Der Spike zeigte "Hook greift, Effekt propagiert" — nicht "der richtige Mechanismus wurde getroffen". Für die echten H3/H4-Experimente ist die **Kontroll-Baseline zwingend** (Zufalls-/irrelevanter Patch), denn "es patcht überhaupt etwas" ≠ "es patcht den richtigen Mechanismus". (Für H4 ist die nicht-kausale Kontrolle bereits in §2/H4 verankert; die gleiche Logik gilt auch für H3.)
2. **Multi-Hook-Patching ist noch nicht getestet.** Der Spike nutzte eine einzelne Patch-Position in einem Layer. H4 (Kaskaden cash→afford→buy→ownership) braucht Patching, das über mehrere abhängige Zustandsvariablen **und** mehrere Token-Positionen konsistent verfolgt wird. Mit `run_with_hooks` sind mehrere Hooks gleichzeitig registrierbar — aber das war nicht Teil des Spikes. **Eigener kleiner Test vor §6-Punkt 10 (H4) einplanen**, nicht erst live beim echten Experiment merken, ob Multi-Hook-Ketten sauber funktionieren.
3. **SAELens erst später ansehen.** TransformerLens hat mit SAELens ein Schwester-Ökosystem vom selben Autorenkreis, das direkt auf HookedTransformer-Aktivierungen aufsetzt — spart später (§5.4/§7, SAE-Analyse nach H1–H3) Integrationsarbeit. **Nicht jetzt einbauen**, erst wenn der SAE-Schritt wirklich ansteht.
4. **Embedding-Init-Check beim echten Aufbau.** Bei 48 Custom-Tokens muss `d_vocab=48` explizit in der `HookedTransformerConfig` gesetzt werden (nicht der GPT-2-Default). Und da keine numerische Embedding-Init gewollt ist (§5.1/§5.1a): verifizieren, dass die Standard-Initialisierung (zufällig, normalverteilt) tatsächlich **keine** numerische/Stellenwert-Struktur in die Embeddings einbaut — genau das Encoding-Confound, das die Tokenisierungsentscheidung vermeiden soll. Kurzer Check beim Tokenizer-Bau (§6-Punkt 4).

### 5.6 Tokenizer/Vocab — ENTSCHIEDEN (implementiert in `data/tokenizer.py`)

**Vokabular (48 Tokens, feste deterministische Reihenfolge, Kernprinzip §8):** `<PAD>` (ID 0), `START`, `ROLL`/`BUY`/`PASS`/`PAY_RENT`, `P0`/`P1`/`P2`, Würfelwerte `D1`..`D6` (atomar), `MONEY`-Marker, `TH0`/`TH1` (Tausenderstelle, §5.1a), und Digit-Tokens `H0..H9`, `T0..T9`, `O0..O9`.

- **Kategoriale/atomare Event-Tokens:** `START`, `P*`, `ROLL`, `BUY`, `PASS`, `PAY_RENT`.
- **Würfelwerte als eigene Dice-Tokens** `D1`..`D6` (nicht in die Digit-Sequenz übernommen) — diskret, ohne Stellenwert-Semantik, damit das Modell einen Würfel nicht mit einem Hunderter-/Einer-Digit verwechselt.
- **Numerische Magnituden (Mietbeträge = Cash-Flüsse):** als `MONEY TH_h H_h T_t O_o`-Digit-Sequenz (5 Tokens).
- **`PROPERTY_*`/`OWNER_*`:** NICHT im Vokabular. §5.3 hat das explizite-State-Modell (Modell A) gestrichen; diese kategorialen Tokens wären nur dort nötig. Einbeziehen würde ungenutzte Embedding-Zeilen erzeugen (mehr Param, nie trainiert). Falls Modell A später reaktiviert wird, sind sie ergänzbar, ohne bestehende IDs zu verschieben.

**Kritische Disambiguierung Würfel vs. Miete (empirisch notwendig, nicht wertbasiert):** `apply_rent` cappt die Miete auf den verfügbaren Cash (`rent = min(0.5*price, cash)`, rules.py) → Mietbeträge können **einstellig/`0`** sein (beobachtet: 0, 25, 50, 75, 100, 125, 150). Eine rein wertbasierte Unterscheidung (`1..6` = Würfel) wäre daher falsch. Der Tokenizer disambiguiert **kontextbasiert** über das vorausgehende Event-Token: nach `ROLL` → Dice-Token, nach `PAY_RENT` → Miet-Digit-Sequenz. Ein numerisches Token ohne eindeutigen Kontext wirft `TokenizerError` statt still zu raten. `tests/test_tokenizer.py::test_cash_capped_single_digit_rent` deckt diesen Fall explizit ab.

**Korrektes Digit-Encoding:** `encode_int(900)` = `MONEY TH0 H9 T0 O0` (der §5.1a-Doc-Beispieltext zeigte fälschlich `TH0 H9 T3 O0` = 930; korrigiert in §5.1a). Bereich `0..1999` (max. Cash 1500, §0.1); `≥ 2000` wirft einen Fehler, der auf die TH-Erweiterung verweist.

**Klärung TH0/TH1 (empirisch, voller Korpus):** `encode_int()` wird in V0 ausschließlich für **Mietbeträge** aufgerufen (`prev == "PAY_RENT"`), und Mieten sind `rent = min(0.5*price, cash) ≤ 150`. Damit ist in V0 **nur `TH0` aktiv; `TH1` wird nie erzeugt** (gemessen: TH0 = TH1-0 = 24208 = MONEY = Anzahl PAY_RENT-Events). Das ist **kein fehlender Encoding-Aufruf**: laut §5.3 ist Cash ein History-only-Modell explizit **nie** im Input — der §5.1a-Fixgrund (Cash bis 1500) und die tatsächliche Verwendungsstelle (Rent ≤ 150) sind zwei verschiedene Dinge. Entscheidung: `TH0`/`TH1` **bleiben** im Vokabular als **bewusst reservierte**, zukunftssichere Stellenwerte (2 Token, vernachlässigbarer Aufwand), damit `encode_int()` den dokumentierten Cash-Bereich 0–1500 vollständig abbilden kann und die S3-Preiserweiterung (§3a/V1) keine Vocab-Änderung braucht. Die Embedding-Init-Statistik für `TH1` ist in V0 deshalb nicht aussagekräftig (nur 2 Tokens, nie trainiert) — das ist erwartbar und dokumentiert, kein Defekt. `tests/test_tokenizer.py::test_roundtrip_and_full_dataset` sichert diese Struktur (TH1==0, MONEY==PAY_RENT) gegen stille Abweichungen ab.

**Verifiziert (`tests/test_tokenizer.py`, alle grün):** d_vocab=48 ohne Duplikate; Digit-Stellenwerte; Kontext-Disambiguierung inkl. einstelliger Miete; Ambiguity/Unbekannt wirft Fehler; **alle 2000 Spiele des vollen Datensatzes sind fehlerfrei tokenisierbar** (kein OOV), tokenisierte Kontextlängen min 180 / Median 226 / max 284 (→ `n_ctx = 288`, §5.2).

**Embedding-Init-Check (§5.5-Follow-up 4, `spikes/check_embed_init.py`, grün):** `d_vocab=48` explizit in der `HookedTransformerConfig` gesetzt (nicht GPT-2-Default); Standard-Zufalls-Init liefert **keine** numerische/Stellenwert-Struktur: Korrelation numerische-Distanz vs. Embedding-Similarität ist ~0 (|corr| < 0.2) für H/T/O, Nachbar-Mean-Cos ≈ Zufalls-Mean. (TH hat nur 2 Tokens → Korrelation nicht aussagekräftig/n.a., erwartbar.) Damit ist sichergestellt, dass das Modell die numerische Struktur selbst lernen muss und nicht von den Embeddings "geschenkt" bekommt — das Encoding-Confound aus §5.1 ist vermieden. *Wichtig: dieser Check gilt für die Init; NACH dem Training soll die Korrelation signifikant negativ werden (nahe Werte = nahe Embeddings) — das ist dann ein positives Befund (H3-Vorstufe).*

**Hinweis Umgebung:** `transformer_lens` ist nicht im Basis-`C:\Python314\python.exe`, sondern im Conda-Env `iprop` (Torch 2.5.1+cu121, CUDA auf RTX 2080 verfügbar). Der Tokenizer selbst (`data/tokenizer.py`) hat **keine** TransformerLens-Abhängigkeit und läuft überall; nur Trainings-/Analyse-Skripte (Init-Check, später Training) brauchen das Env. Bei der Wahl des Python-Interpreters für das Training darauf achten.

---

### 5.7 Trainingsläufe & erste H1-Vorproben (Stand)

**Trainingsläufe** (Architektur §5.2, S0-Split, 2000 Spiele → train 1800 / val 200, seed 42, Daten `games_so0_n2000_always_buy-conservative-random`). Manifest/Checkpoints unter `runs/<name>/` (gitignored, reproduzierbar via `training/train.py`):

| Run | Config | Ergebnis | Bemerkung |
|---|---|---|---|
| `dryrun` | 300 Steps, lr 1e-3, warmup 200, batch 32 | val 0.5655 | Smoke-Test der Pipeline |
| `run1` | geplant 4000 Steps, sonst identisch | **best_val 0.5638** (≈Step 500), danach Overfit bis ~0.78 | **nach ~1650 Steps abgebrochen** (val steigt trotz fallendem Train-Loss weiter) |
| `run2_d0.1_s1200` | dropout 0.1, 1200 Steps | — | Crash beim Datenladen, nur 2 Logzeilen; nie trainiert |
| `run2_wd0.5_s1200` | weight_decay 0.5, 1200 Steps | best_val 0.5627 (≈Step 750), final 0.6021 | **Overfit ab ~Step 750** (siehe `probe_checkpoints.png`); Early-Stopping-Optimum ~650–750. Regularisierung bringt **keine** Verbesserung ggü. run1 |

Parametrierung: 4 L, d_model 256, 4 Heads (d_head 64), d_mlp 1024, n_ctx 288, gelu, causal, **d_vocab=48 explizit**; 3,257,904 Parameter; CUDA (RTX 2080), AMP, ~0.8–1.0 s/step.

**Zentraler Befund:** Der Val-Loss ist mit ~0.56 deutlich besser als die Bigramm-Baseline (~0.97, §5.7/`diag_predictive.py`). Das Modell lernt also mehr als reine Würfel-Grammatik.

**Erste H1-Vorproben** (Skripte unter `probes/`):

1. **`check_labels.py` — Ground-Truth-Rekonstruktion (nur Datenqualität):** Position aus Würfel-Tokens (`cumsum mod board_size`) rekonstruiert → **100% Übereinstimmung** mit den `states[]`. Die Label-/Alignment-Pipeline ist sauber; ein flacher Probe-Befund ist also **kein** Daten-Bug.
2. **`probe_position.py` — Position-Sanity-Probe (linear je Layer/Spieler):** erreicht **~17–19% Test** (flach über alle Layer und Checkpoints) statt der geforderten ~99% (§6 Schritt 6). Damit ist H1 (Position) **vorerst nicht bestätigt** — siehe offene Diagnose unten. `check_labels.py` schließt einen Label-Bug aus; Befund stabil über Checkpoints und beide Läufe. **Methodischer Fix (pro-Spiel-Split):** Der frühere Probe splittete **per-Sample über Spiele gemischt** (nicht per-Spiel) — ein Leakage-Konfund. Mit sauberem **pro-Spiel-Split** (32/40 Spiele, Trajectory-Integrity §3) ergibt sich: **linear Train 42% / Test 18%** (die frühere 100%-Train war per-Sample-Memorisierung von Spiel-/Sample-Identität), **MLP Train 100% / Test 19%** (overfittet die 32 Trainings-Spiele über spielspezifische Confounds, generalisiert aber nicht auf held-out Spiele). Position ist also weder linear noch nichtlinear als konsistente, **game-invariante** Größe aus `resid_post` dekodierbar.
3. **`checks_12.py` — Hook-Scan + triviale Baselines:** Layer/Hook-Breakdown (resid_pre/attn_out/mlp_out/resid_post je Layer) zeigt Position in **keinem** Hook lesbar (~17% überall) → "falsche Analyse-Ebene" ist als Erklärung ausgeschlossen. Die naive Heuristik "letzter `P*`-Token = aktueller Spieler" erreicht **100%** (triviale, reine Zähl-Heuristik); die **Current-Player-Probe** des Modells liegt mit **~87–89%** (run2-Plot, über Checkpoints stagnierend) darunter. Das State-Tracking (Current-Player + Position) reift bereits früh aus (~Step 250) und profitiert nicht mehr von späteren Steps.
4. **`diag_predictive.py` — Diagnose B (woher kommt der Loss-Vorsprung?):** Argmax-Next-Token auf Val, getrennt nach Token-Typ, vs. Bigramm-Baseline. Am **Dice→Event-Trigger** (direkt nach dem Würfel entscheidet das gelandete Feld BUY/PASS/PAY_RENT): Modell **~61%** vs. Bigramm **~33%**. Das Modell nutzt also **konditional-positions-/ownership-abhängige** Information für die Event-Entscheidung — d.h. der Loss-Vorsprung ist **nicht** nur Würfel-Grammatik.
5. **`causal_roll.py` — kausaler Position-Test (Do-over-Input-Substitution + Residual-Patch):** **erledigt.** Position geht **kausal** in die BUY/PASS/PAY_RENT-Entscheidung ein (F_L≈1: der Würfel-zu-Event-Kanal trägt nahezu den vollen Effekt; Do-over-Substitution + Residual-Patch wirksam). Position wird also **verhaltenswirksam kausal genutzt** — siehe den Kontrast zu Befund 2 in der offenen Diagnose. Zusammen entscheidet das, ob mehr Daten/Kapazität (10x) sinnvoll ist.
6. **`probe_state.py` — Cash-Probe (H1, pro-Spiel-Split von Anfang an):** Regressions-Probe (Pearson R) auf Cash je Spieler, gemittelt über Spieler. Rohwert **R_test ≈ 0.64** (linear); MLP Train ~0.94–1.0 / Test ~0.60 (die Nichtlinearität generalisiert nicht dazu — sie memorisiert nur Trainingsspiel-Confounds). **Konfund-Kontrolle (Turn-Drift) ist entscheidend:** Cash startet bei 1000 und driftet mit dem Spielfortschritt tendenziell nach unten; eine reine Progress-Baseline (Anzahl gesehener ROLL-Events → Cash, gleicher pro-Spiel-Split) erreicht bereits **R_base = 0.557**. Nach Orthogonalisierung von Residual UND Cash gegen den Progress (alle Koeffizienten nur auf Train-Spielen gefittet, kein Leakage) bleibt **R_partial ≈ 0.46** (je Layer 0.42–0.47) übrig. → **Cash hat ein reales, aber moderates** Cash-spezifisches Signal über den Spielfortschritt hinaus (R_partial ~0.46), **nicht** den starken Rohwert 0.64 — der war überwiegend Turn-Drift. Das differenziert Cash von Position (kein Signal), ist aber deutlich schwächer als ein game-invarianter Tracker (R→~0.95+ wäre der Idealfall). Für jede künftige Cash-Probe gelten die Base/Model/Partial-Zahlen als Pflicht-Kontext (siehe §8).
7. **`probe_state.py` — Ownership-Probe (H1):** Klassifikation je Preisfeld (None/0/1/2), Accuracy über 8 Preisfelder gemittelt (die Preisfeld-Menge enthält nur Felder mit `price > 0`; `make_prices` kann auch `0` wählen). Test-Acc ~45% vs. Majority(None)-Baseline 32.5% — der Lift kommt fast nur aus korrektem "None"-Raten. **Identitäts-Test (nur besetzte Felder): Owned-Acc ≈ 33.1% vs. Owner-Majority-Baseline 37.1%** — das Modell liegt **leicht UNTER** der trivialsten Baseline (Konstanten-Raten des häufigsten Owners). Ein Wert konsistent leicht unter der Majority-Baseline ist **kein neutrales "≈ Zufall"**: Zufallsrauschen um die Baseline wäre bei "nichts codiert" zu erwarten; ein systematisch leicht darunterliegender Wert deutet auf eine **leichte systematische Verzerrung weg vom häufigsten Owner** hin (möglicherweise eine in Trainingsspielen gelernte Owner-Verteilung, die auf Held-out-Spielen nicht passt — ein weiteres potenzielles Generalisierungs-Artefakt). Ohne weitere Analyse nicht in die eine oder andere Richtung überinterpretieren; die Faktenlage reicht: **Ownership-Identität wird nicht brauchbar game-invariant getrackt.**
8. **Equal-Epoch-Datenmengen-Sensitivitätstest (10x-Frage, `training/train.py` + `probes/probe_state.py`/`probe_position.py`):** Um die teure 10x-Frage empirisch statt extrapoliert zu beantworten, wurde ein Modell auf **500 Spielen** (`data/games_subset500_seed123.jsonl`) trainiert und mit dem Vollmodell (1800 Spiele, run2) verglichen. **Methodik:** Equal-Epoch statt Equal-Step, um den Epochen-Confound zu vermeiden (run2 = 21 Epochs; das 500er-Modell wurde deshalb auf 21 Epochs = 315 Steps trainiert, nicht auf 1200 Steps/80 Epochs). Ein anfänglicher 80-Epochen-Lauf (`runs/scaling_500_seed123`, gleiche 1200 Steps) zeigte identische Ergebnisse und dient als Robustheits-Check. Die Ergebnisse bei **equal epoch** (best-Checkpoints, pro-Spiel-Split):
   - **Cash: komplett flach.** Roh-R ~0.64, Konfund-kontrolliert R_partial ~0.44 — identisch bei 500 und 1800 Spielen. Trotz deutlich schlechterem Gesamtmodell (best_val 0.603 vs. 0.563) keine Änderung → **struktureller Deckel, kein Coverage-/Trainingsartefakt.**
   - **Position: flach/schwach** (~16% bei 500 vs. ~18% bei 1800).
   - **Ownership: reale, aber langsame Sensitivität** (Test ~41% bei 500 vs. ~45% bei 1800; Owned-Acc ~27% vs. ~33%, beides unter der Owner-Majority-Baseline ~37–42%). Dies ist **kein Epochen-Artefakt** (bleibt bei equal epoch bestehen), aber die Verbesserung ist zu langsam für game-invariantes Tracking. **Vorsicht:** Die verbleibende Zwei-Punkte-Betrachtung (500/1800) erlaubt **keine belastbare Extrapolation** auf 4600+ Spiele — eine Trendaussage ("~50% bei 10x") ist nur eine grobe, ausdrücklich unsichere Schätzung, keine Prognose.
   - **Fazit (10x-Frage abschließend):** Reine Skalierung von mehr vom selben Task erzeugt **kein** game-invariantes State-Tracking; der Cash-Befund ist ein wasserdichter Deckel. Empfehlung: **kein pauschales 10x**, stattdessen Architektur-/Task-Druck (siehe §6a).

**Position-Diagnose — AUFGELÖST (pro-Spiel-Split, methodisch abgesichert).** Der Kernbefund ist jetzt belastbar:

- **Die eigentliche Pointe ist der Kontrast:** Das Modell nutzt die Würfelinfo **kausal** (F_L≈1, Kanal existiert und trägt fast den vollen Effekt, Befund 5), baut daraus aber **keine** konsistente, game-invariante, dekodierbare Zustandsvariable auf (Befund 2). Genau dieser Gegensatz — kausale Nutzung ja, saubere Zustandsrepräsentation nein — ist die spannendste Aussage des Kausal-Kapitels und passt zur **"Bags of Heuristics"**-Lesart (verteilte, spielabhängige Heuristiken statt einer sauberen internen Variable).

- **Methodik-Lektion (ausdrücklich festgehalten, nicht nur das Ergebnis):** Der frühere Probe splittete **per-Sample über Spiele gemischt** (Leakage-Konfund) und zeigte **Train 100% / Test ~18%** — die 100%-Train wurden damals als Memorisierung fehlinterpretiert. Der korrekte **pro-Spiel-Split** (32/40 Spiele, Trajectory-Integrity §3) zeigt **Train 42% (linear) / 100% (MLP), Test 18–19% in beiden Fällen**. Wichtig: Die niedrige **Test**-Zahl war schon vorher richtig; aber die **Trainingswerte** waren durch Leakage verzerrt — jemand, der nur die alte 100/18-Zahl im Kopf hat, würde fälschlich schließen, das Modell hätte in-sample eine perfekte lineare Position-Repräsentation gehabt. Das hatte es **nicht** (linear nur 42% Train, d.h. Position ist selbst in den bekannten Spielen nicht sauber linear separierbar). Die früheren Kandidaten (a) falsche Basis / (b) nicht-lineare Verarbeitung / (c) Kapazität sind damit als Erklärung für einen flachen Probe **ausgeschlossen** — weder linear noch nichtlinear ist Position als game-invariante Größe aus `resid_post` dekodierbar.

---

## 6. Nächste konkrete Schritte (in Reihenfolge)

1. ~~**Tokenisierungsentscheidung final treffen**~~ — **erledigt** (siehe 5.1): H/T/O-Digit-Encoding, kein Bucketing im Hauptmodell, keine numerische Embedding-Init, Bucketing nur mit präzisem Trigger (siehe §5.1 + Schritt 8a).
2. ~~`data/generate.py`~~ — **erledigt**, inkl. Config-Manifest und `game_id_start` gegen Kollisionen.
2a. ~~**Framework-Spike**~~ — **erledigt** (siehe §5.5 + `spikes/spike_tl.py`): custom H/T/O/TH-Tokenizer bindet sich sauber in TransformerLens ein, alle kritischen Fähigkeiten (run_with_cache, Patching, HookPoint) grün. **Entscheidung: TransformerLens** (Primär), kein Fallback nötig.
3. ~~**S1–S2-Filterlogik implementieren**~~ — **erledigt** (`data/splits.py`, `tests/test_splits.py`, siehe 4.3). S3 entfällt in V0 (§3a). Split-Regeln verifiziert: Trajectory Integrity (keine game_id-Überlappung), kein Leakage (held-out Bedingung nicht im Training), S2-Mindest-N (§3a).
4. ~~**Tokenizer/Vocab bauen**~~ — **erledigt** (`data/tokenizer.py` + `tests/test_tokenizer.py`, §5.5-Follow-up 4 via `spikes/check_embed_init.py`). Details siehe §5.6. `n_ctx = 288` (siehe §5.2).
5. ~~**Kleines GPT trainieren**~~ — **erledigt** (siehe 5.7): run1 (abgebrochen, best_val 0.5638), run2_wd0.5_s1200 (best_val 0.5627). S0-Split, 3.26M Param, ~0.8–1.0 s/step auf RTX 2080.
6. ~~**H1 testen (erste Vorproben)**~~ — **teilweise erledigt** (siehe 5.7): Labels 100% korrekt; **Position-Sanity-Probe flach** (~18%, Diagnose aufgelöst, §6a); **Cash-Probe R_partial ≈ 0.46** über Turn-Drift-Baseline hinaus; **Ownership-Identität unter Majority-Baseline** (Items 6–7). Cash/Ownership-Probes laufen mit korrektem pro-Spiel-Split von Anfang an (`probe_state.py`).
6a. ~~**Offene Position-Diskrepanz aufklären**~~ — **erledigt** (siehe §5.7, Position-Diagnose): ① `probes/causal_roll.py` final gelaufen → Position wird **kausal** genutzt (F_L≈1). ② Nichtlineare/MLP-Probe + pro-Spiel-Split durchgeführt → weder linear (42/18) noch nichtlinear (100/19) ist Position **game-invariant** dekodierbar. **10x-Daten-Frage — abschließend beantwortet (equal-epoch-Scaling-Test, §5.7 Item 8):** Ein Modell auf 500 Spielen (21 Epochs, identische Hyperparams) liefert praktisch **identische** Probe-Werte wie das Vollmodell auf 1800 Spielen (Cash partial R ~0.44; Position ~16%; Ownership ~41%). Damit ist die Daten-Sensitivität **empirisch flach** (kein Epochen-/Trainingsartefakt, bestätigt durch equal-epoch-Kontrolle) — ein **struktureller Deckel**, kein Coverage-Mangel. Konsequenz: **kein pauschales 10x.** **Präziser Scope (wichtig):** Der Befund ist "**Skalierung ist nicht der Hebel**", **nicht** "diese Variablen sind grundsätzlich unlernbar". Er gilt für **mehr Daten vom selben Task/Setup bei gleichbleibendem Training** und sagt nichts darüber aus, ob eine **Architektur- oder Trainingsziel-Änderung** (z.B. ein Hilfsziel, das explizit State-Vorhersage erzwingt, mehr Kapazität, ein anderes Tokenisierungsschema) die Variablen doch noch game-invariant machen würde. Diese Distinktion ist Grundlage für spätere Entscheidungen (H3/H4, V1).
7. **H2 testen**: Affordability-Probe, aber mit S1-Split (Held-out-Kombinationen) — nicht S0.
8. Bei Erfolg: S2 (siehe Abschnitt 3). **S3 ist in V0 nicht durchführbar und als V1-Erweiterung vorgemerkt** (siehe §3a).
8a. **Bucketing-Kontrollmodell** (nur wenn Trigger erfüllt, siehe §5.1): H2 mit S1 erfolgreich UND Digit-Probe zeigt klare layer-weise numerische Struktur → dann als Negativkontrolle ("Digit zeigt X, Bucket zeigt X nicht") trainieren, **bevor** H3 angegangen wird.
9. **H3 testen**: Interventionen auf Cash-/Price-Repräsentationen, prüfen ob Affordability-Vorhersage sich entsprechend ändert — **mit Kontroll-Baseline** (Zufalls-/irrelevanter Patch), da "es patcht etwas" ≠ "es patcht den richtigen Mechanismus" (siehe §5.5-Follow-up 1).
9a. **Multi-Hook-Vortest**: kleines Testskript, das mehrere gleichzeitig registrierte Hooks über mehrere Token-Positionen verfolgt (Kaskaden-Patching), bevor H4 live läuft (siehe §5.5-Follow-up 2).
10. **H4 testen** (erst nach H1–H3 für alle Einzelglieder): Kaskaden-Konsistenz nach Cash-Intervention über mehrere abhängige Zustände hinweg — **mit nicht-kausaler Kontroll-Intervention** (gleiche Intervention auf irrelevantem Feature, siehe §2/H4), um Off-Distribution-Artefakte auszuschließen.
11. Erst danach: SAE-Analyse auf den layer-wise interessantesten Bereichen.

---

## 7. Zurückgestellte Erweiterungen (bewusst NICHT Teil von V0)

- **Hidden Information / Belief States:** Würfelwurf im Input verstecken, um echte Unsicherheit zu erzwingen. Testet, ob das Modell einen Punktschätzer oder eine echte Verteilung über mögliche Zustände repräsentiert. Methodisch deutlich aufwändiger (braucht Entropie-Kalibrierungs-Analyse oder Ensemble-Probes statt einfacher Punkt-Probes) — explizites Stretch-Goal, nicht Kernexperiment.
- **Explizites-State-Modell zum Vergleich** (Modell A vs. Modell B / Representational Alignment via CKA/SVCCA + Cross-Model-Probing): Interessant, aber erst wenn History-only-Modell (H1) überhaupt erfolgreich ist.
- **Mehrere unabhängige Seeds/Modelle** für Konvergenz-/Universalitäts-Fragen: erst nach einem ersten reproduzierbaren Einzelbefund.
- **Häuser, Hypotheken, Ereigniskarten, echtes Monopoly-Regelwerk:** explizit nicht in V0. Falls V0 gut funktioniert, ggf. spätere Erweiterung um zusätzliche Zustandskomplexität (ordinal: Häuser; binär: Jail-Status; relational: andere Spieler).

---

## 8. Kernprinzipien, die während der gesamten Weiterarbeit gelten sollten

- **Reproduzierbarkeit von Anfang an:** Jeder Trainingslauf muss eindeutig einem Simulator-Seed, einer Policy-Mischung und einem Daten-Split zuordenbar sein (z.B. Config-Hash im Dateinamen). Der größte Zeitfresser bei solchen Projekten ist erfahrungsgemäß nicht "warum ist Probe X bei 70%", sondern "welcher von fünf Datensätzen gehört zu diesem Checkpoint".
- **Erst Behavior, dann Probing, dann Intervention** — nie eine Stufe überspringen und aus einer einzelnen Evidenzart eine starke Schlussfolgerung ziehen.
- **Bei unerwarteten Ergebnissen zuerst Simulator/Tokenisierung/Pipeline verdächtigen, nicht sofort ein "mechanistisches Phänomen" vermuten.** Die Invariant-Tests und der Position-Sanity-Check existieren genau dafür.
- **Klein anfangen, Komplexität nur bei Bedarf hinzufügen** — nicht alle sechs Zustandskomponenten (Position, Cash, Ownership, Häuser, Jail, Hidden Info) gleichzeitig einführen.
- **Jeder Probe-Wert braucht eine explizite, plausible Trivial-Baseline — nicht nur Zufall/Majority — gegen die orthogonalisiert wird, bevor ein Rohwert als "Signal" interpretiert wird.** Diese Regel entstand aus drei wiederkehrenden Konfunden, die je erst beim Gegenhalten einer passenden Baseline sichtbar wurden: (1) per-Sample- statt per-Spiel-Split bei Position (Leakage: Train 100% statt 42%), (2) falsche Zufallsbaseline (1/3 statt tatsächliche Owner-Majority) beim Ownership-Identitätstest, (3) Turn-Drift bei Cash (Progress-Baseline R=0.557 erklärt den Großteil des Rohwerts 0.64; ehrlicher Wert R_partial ≈ 0.46). Der Standard: Rohwert UND (nach Möglichkeit) dessen Konfund-Orthogonalisierung mit denselben Koefizienten nur auf Train-Spielen gefittet — sonst ist der "Signal"-Wert nicht belastbar. Bei H2/H3/H4 (z.B. Cash-Interventionen, Affordability) dieselbe Sorgfalt anwenden, nicht erst nachträglich einfordern.
