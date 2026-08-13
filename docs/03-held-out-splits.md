# 3. Held-out Splits und Filterlogik (S0–S3)

Die Generalisierungstests werden **nicht als zufällige Token-Splits** definiert, sondern anhand der **Ground-Truth-Zustände und -Transitionen** des Simulators. **Ganze Partien/Trajektorien bleiben innerhalb eines Splits**, um Leakage zwischen nahezu identischen Spielverläufen zu vermeiden. S0–S3 testen vier unterschiedliche Fragen — nicht drei Schwierigkeitsstufen desselben Splits.

## S0 — IID-Sanity-Baseline

S0 ist der Referenztest: zufällig held-out Partien aus derselben Zustandsverteilung wie das Training.

S0 beantwortet zunächst, ob Modell bzw. Probe die Aufgabe überhaupt zuverlässig gelernt haben. **Alle späteren S1–S3-Ergebnisse werden relativ zu dieser Baseline interpretiert** — ohne S0 kann ein schlechter S3-Wert nicht von "schlecht trainiert" unterschieden werden.

## S1 — Novel State Composition

S1 hält die einzelnen Wertebereiche der State-Variablen bekannt, hält aber bestimmte **Kombinationen dieser Variablen** aus dem Training zurück.

Beispiel: Die einzelnen Cash-, Position- und Ownership-Werte kommen im Training vor, bestimmte Kombinationen aus **Cash × Position × Ownership** jedoch ausschließlich im Test.

**Hypothese:** Das Modell kann bekannte Zustandskomponenten zu neuen Zuständen kombinieren, anstatt lediglich beobachtete vollständige Zustände zu memorisieren.

**Wichtig:** "Neue Kombinationen" bezieht sich auf Kombinationen von *State-Variablen* (nicht bloß neue vollständige Spielzustände), damit kontrollierbar ist, was genau held out ist. Einzelne Werte (z.B. `500`, `400`) dürfen im Training vorkommen — nur die Kombination `(500, 400)` darf nie zusammen auftreten, sonst könnte das Modell die Werte separat memorieren, ohne die Relation zu abstrahieren.

## S2 — Novel Relation / Transition Composition

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

## S3 — Struktur-erhaltende Extrapolation

S3 testet Generalisierung außerhalb des im Training beobachteten Werte-/Zustandsbereichs.

Der Testbereich darf dabei **nicht lediglich deshalb gewählt werden, weil er numerisch "höher" liegt**. Er muss eine **struktur-erhaltende Extrapolation** sein: Die zugrunde liegenden Variablen und Operationen bleiben dieselben, während der konkrete Zustandsbereich außerhalb der Trainingsverteilung liegt.

Vor Konstruktion des Splits wird deshalb geprüft, welche Zustände im tatsächlich generierten Datensatz überhaupt natürlich vorkommen. Insbesondere wegen `max_turns`, der Preisstruktur und der Rentenregel darf S3 nicht auf einen theoretisch möglichen, aber praktisch nicht erreichten Wertebereich gelegt werden.

**Falls kein ausreichend großer natürlicher OOD-Bereich vorhanden ist, wird S3 nicht künstlich erzwungen, sondern als nicht durchführbar dokumentiert.**

## Split-Regeln

1. **Split auf Ground Truth:** Filter werden auf Zuständen und Transitionen des Simulators angewandt, nicht auf der Tokenfolge.
2. **Trajectory integrity:** Eine Partie/Trajektorie gehört vollständig zu genau einem Split (kein Token-Level-Splitting).
3. **Strict held-out probing:** Bei OOD-Probes respektiert auch das Probe-Training den jeweiligen Split. Insbesondere bei S3 wird der Probe ausschließlich auf der Trainingsregion trainiert und ausschließlich auf der held-out Region evaluiert.
4. **Pre-registration of filters:** Die Filterbedingungen werden vor der finalen Modellauswertung festgelegt und anhand des generierten Datensatzes auf Abdeckung und ausreichende Samplezahlen geprüft.
5. **No forced OOD:** Ein Split gilt nicht als erfolgreich, wenn seine Testregion nur durch seltene oder künstlich konstruierte Zustände gefüllt werden kann.

**Reihenfolge:** S1 zuerst (Minimaltest — wenn das schon scheitert, sind S2/S3 akademisch). Bei Erfolg erst dann S2, dann S3.

## Coverage-Statistiken (vor dem Training des GPT)

Bevor das GPT überhaupt trainiert wird, wird der Generator einmal laufen gelassen und für S1–S3 Coverage-Statistiken ausgegeben:

```text
S0: N train / N test
S1: N train / N test
S2: counts per (cash-price) relation
S3: min/max + Quantile + Anzahl erreichbarer Zustände
```

Gerade S2 ist wichtig: Wenn `±50` im Generator viel häufiger vorkommt als `±200`, kann ein vermeintlicher "held-out relation"-Test durch Verteilungsunterschiede verzerrt werden. Bei S3 wird **erst nach diesem Histogramm entschieden, wo die Grenze liegt** — nicht vorher ein Wertebereich wie `1050–1500` festgelegt.

## 3a. Konkrete Konsequenzen aus dem empirischen Coverage-Check (2000 Spiele)

Der in §0.1 beschriebene Coverage-Check wurde durchgeführt. Ergebnis für S2 und S3:

**S2 — Mindest-Samplezahl pro held-out Relation:** Von 57 beobachteten Cash-Preis-Relationen sind nur wenige stark besetzt (`700`: 2216, `400`: 1972 Vorkommen — **> überholt (Pre-Fix-Datenstand), korrigierte Werte in §4.3.2**), viele liegen bei ~100. Als Mindestschwelle für eine als S2-Kandidat zulässige held-out Relation gilt: **N ≥ 200–300 Vorkommen**, um sowohl Trainings- als auch Test-Split des Probes verlässlich zu befüllen. Relationen unterhalb dieser Schwelle werden nicht als alleinstehender S2-Split verwendet. Falls nach Anwendung dieser Schwelle zu wenige Relationen übrigbleiben, ist die Konsequenz **mehr Spiele generieren**, nicht die Schwelle aufweichen. *(Schwellenlogik bleibt gültig; nur die hier zitierten Beispielzahlen sind durch den `states[i-1]`-Bug veraltet — maßgeblich ist §4.3.2.)*

**S3 — ENTSCHIEDEN: nicht durchführbar in V0, V1-Erweiterung vorgemerkt.** Bei `max_turns=40`, Startkapital 1000, Preisliste bis 300 liegen q99 der Cash-Verteilung bei 1000, nur 760 von ~1,06 Mio. States über 1200. Das ist kein tragfähiger natürlicher OOD-Bereich (vgl. Split-Regel 5, "No forced OOD", §3). Zusätzlich spricht **die Kontextlängen-Constraint** (§5.2: ~128–256 Tokens) gegen die naheliegende Gegenmaßnahme: Cash-Extremwerte streuen grob mit √Turns (Random-Walk über Rent-Events), sodass ein tragfähiger OOD-Bereich eine Verdopplung/Verdreifachung von `max_turns` erforderte — das sprengt das Token-Budget pro Partie bzw. verletzt die Trajectory-Integrity-Regel (§3) beim Kappen.

**Entscheidung: Option (b).** S3 wird bewusst als "nicht durchführbar in V0" dokumentiert und aus dem V0-Scope gestrichen. Das ist ein **pre-registrierter Negativbefund**, kein Rückzug — genau das verlangt Split-Regel 5 und das Kernprinzip §8 (klein anfangen). S3 blockiert damit S0/S1/S2 und das GPT-Training nicht.

**V1-Vormerkung für S3 (größerer Preisraum statt mehr Turns):** Falls S3 später als eigener Folgeversuch angegangen wird, dann über ein strukturell anderes Board mit **höheren Preisstufen** (nicht längere Spiele), um den OOD-Bereich ohne Sequenzlängen-Wachstum zu erzeugen. Dabei gelten drei Vorsichtsregeln:
1. **Board bleibt fix (BOARD_SEED)** — sonst wiederholt sich die §4.1-Lernproblematik (Preis steht nie explizit im Token-Stream). Das ist notwendig, aber nicht hinreichend.
2. **Preis-Mindestfrequenz pro Stufe:** Jede neue/höhere Preisstufe ist wieder eine "gelernte Konstante pro Feld", die erst über genug Vorkommen gelernt werden muss. Bei fixer Feldzahl (12) verdünnt mehr Preisstufen die Besuchsfrequenz pro Stufe → für V1 vorab prüfen, dass jede Preisstufe eine Mindestanzahl von Vorkommen im generierten Datensatz hat (analog zur S2-N-Schwelle).
3. **V1-Coverage-Check als gemeinsame Neuvermessung von S2 UND S3:** Da beide an derselben Stellschraube (Preisliste) hängen, verändert der größere Preisraum auch die S2-Relationsverteilung. Der V1-Check darf deshalb nicht nur "reicht der OOD-Bereich" (S3) fragen, sondern muss gleich die S2-Mindest-N-Schwelle (§3a) mit den neuen Preisstufen neu validieren — sonst löst man S3 und importiert sich unbemerkt eine neue S2-Schieflage.

---

