# 2. Hypothesen-Hierarchie (H1–H4)

Zentrale Struktur des gesamten Experiments. Jede Stufe baut auf der vorherigen auf — **nicht überspringen**, da bei einem Fehlschlag in einer späteren Stufe sonst unklar ist, woran es liegt.

## H1 — State Tracking
> Das Modell hält Informationen über den latenten Zustand (Position, Cash, Ownership) verfügbar, obwohl sie nie explizit im Input standen.

**Test:** `history → linear/nichtlinear Probe → state`
Erst linearer Probe versuchen, dann nichtlinearer (Analogie zu Li et al. vs. Nanda bei Othello — die richtige Basis kann entscheidend sein).

**Kostenloser Sanity-Check zuerst:** Position ist deterministisch aus der Roll-Summe berechenbar (`position = cumsum(rolls) mod board_size`). Wenn ein Probe hier nicht ~99% erreicht, ist die Pipeline (Tokenisierung/Architektur/Training) kaputt — **nicht weiter zu Cash/Ownership gehen, bevor das nicht sauber funktioniert.**

## H2 — State-Dependent Computation
> Das Modell verwendet Zustandsinformationen, um neue, nicht direkt beobachtete Größen zu berechnen (z.B. `affordability = cash >= price`).

**Wichtige Unterscheidung (aus der Diskussion):** Ein Output-Probe, der `can_afford(A, X)` korrekt vorhersagt, beweist noch keine Komposition — das Modell könnte es als eigenständiges, memoriertes Feature gelernt haben (`CAN_AFFORD_X` als Shortcut), ohne intern wirklich `cash - price` zu berechnen.

**Deshalb: Held-out-Test zwingend nötig, nicht optional.** Siehe Abschnitt 3 (Splits S0–S3).

## H3 — Causal Internal Mechanism
> Die Zustandsrepräsentationen sind tatsächlich Teil des kausalen Mechanismus, der die Vorhersage erzeugt (nicht nur ein epiphenomenales Korrelat).

**Test:** Aktivierung gezielt manipulieren (z.B. Cash-Repräsentation patchen), prüfen ob sich nachgelagerte Vorhersagen (Affordability, Buy-Entscheidung) entsprechend ändern.

**Wichtige Einschränkung:** Ein Probe, der 95%+ Accuracy erreicht, zeigt nur: *Information ist irgendwo in der Aktivierung vorhanden.* Nicht: *Das Modell nutzt diese Information kausal für seine Vorhersage.* Erst die Intervention zeigt Kausalität.

**Risiko:** Off-Distribution-Artefakte. Wenn man einen Feature-Vektor auf einen Wert patcht, den das Modell in dieser Zustandskombination nie im Training gesehen hat, kann das Modell "kaputt" reagieren — nicht weil der Mechanismus getroffen wurde, sondern weil der Patch die Aktivierung aus der Trainingsverteilung schiebt. Vorsicht bei der Interpretation nötig.

## H4 — Counterfactual Cascade Consistency
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

## Wichtige methodische Klarstellung (Repräsentation vs. Verarbeitung)

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

