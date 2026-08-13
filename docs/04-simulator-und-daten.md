# 4. Simulator (V0) — bereits implementiert und getestet

**Design-Prinzip:** V0 ist bewusst "langweilig" gehalten. Keine Häuser, keine Hypotheken, keine Ereigniskarten, kein echtes Monopoly-Regelwerk. Keine Würfel-Stochastik als Unsicherheitsquelle (Würfel sind seed-deterministisch und für das Modell im Kontext sichtbar) — Hidden Information ist ein bewusst nach hinten verschobenes, separates Experiment (siehe Abschnitt 7, "Stretch Goal").

## Projektstruktur

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

## `state.py` — GameState

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

## `rules.py` — Event-Anwendung

Vier Events: `apply_roll_and_move`, `apply_buy`, `apply_pass`, `apply_rent`. Jede Funktion:
1. Nimmt Snapshot vor der Änderung
2. Wendet die Änderung an
3. Nimmt Snapshot danach
4. **Prüft explizit, dass NUR die für diesen Event-Typ erlaubten State-Keys sich verändert haben** (`_assert_only_changed`) — z.B. `ROLL_MOVE` darf nur `position` ändern, `BUY` nur `cash`+`owner`, `PAY_RENT` nur `cash`. Das ist der zentrale Debugging-Mechanismus aus der Diskussion: Simulator-Bugs werden sofort lokalisiert, statt erst als falsches Interpretability-Ergebnis wieder aufzutauchen.
5. Ruft `state.check_invariants()` auf

`apply_pass` verändert bewusst nichts am State, ist aber ein **eigenes Event-Token** — wichtig, damit das spätere History-only-Modell "kein Kauf, weil Feld schon besetzt" von "kein Kauf, obwohl möglich" unterscheiden kann.

`RuleViolation`-Exception bei Regelverstoß (z.B. Kauf ohne ausreichend Cash).

## `policies.py` — Kauf-Heuristiken

Drei Policies: `always_buy`, `conservative` (kauft nur wenn danach ≥300 Cash übrig), `random` (50/50 wenn möglich). **Grund für mehrere Policies:** Wenn Trainingsdaten nur einer Heuristik folgen, kann eine spätere SAE auf Artefakte dieser einen Heuristik overfitten statt die zugrundeliegende Zustandsstruktur zu lernen. Bei der Datengenerierung sollten Policies gemischt werden.

## `game.py` — Game-Loop

`play_game(game_id, seed, policy_name, board_size=12, num_players=3, max_turns=40) → GameLog`

- `make_prices()`: Feld 0 (Start) immer preislos, restliche Felder aus fester Preisstufenliste `[0,100,150,200,250,300]`. **Seit Bugfix (siehe 4.1): Board-Layout ist über ALLE Partien identisch (`BOARD_SEED` fix), nicht mehr abhängig vom Spiel-Seed.**

## 4.1 Bugfix: Preise waren nicht lernbar (gefunden von Tobs' Coding-LLM)

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

## `tests/test_invariants.py` — 5 Tests, alle grün

1. `test_money_conserved_across_full_games` — Cash-Summe + investiertes Kapital (Summe der Kaufpreise besetzter Felder) muss konstant = Startkapital bleiben, über alle drei Policies.
2. `test_no_negative_cash`
3. `test_ownership_valid_range`
4. `test_events_states_same_length`
5. `test_position_matches_roll_sum_sanity` — **Der kostenlose Position-Sanity-Check**: rekonstruiert Position manuell aus Roll-Summe und vergleicht mit dem tatsächlichen State. Wichtig: beim Parsen ist zu beachten, dass Event-Token wie `"PASS"`/`"PAY_RENT"` auch mit `"P"` beginnen — Spieler-Token nur erkennen, wenn der Rest reine Ziffern sind (`tok[1:].isdigit()`), sonst Fehlklassifikation.

Alle Tests laufen aktuell fehlerfrei durch (`python3 tests/test_invariants.py` → `Alle Invariant-Tests bestanden.`).

---

## 4.2 `data/generate.py` — Datengenerierung

`generate(output_dir, num_games, seed_offset, board_size=12, num_players=3, max_turns=40, policies=None, game_id_start=0) → Pfad zur JSONL-Datei`

- Erzeugt `num_games` Partien über gemischte Seeds (aus `seed_offset` deterministisch abgeleitet) und gemischte Policies (Default: alle drei aus `POLICIES`), schreibt sie zeilenweise als JSONL (ein `GameLog.to_dict()` pro Zeile).
- **Config-Manifest:** Neben der JSONL-Datei wird automatisch eine `.manifest.json` mit allen Erzeugungsparametern geschrieben (Kernprinzip §8: Datensatz muss eindeutig seiner Config zuordenbar sein). Dateiname wird automatisch aus der Config gebaut: `games_so<seed_offset>_n<num_games>_<policies>.jsonl`.
- **`game_id_start`-Parameter:** verhindert `game_id`-Kollisionen, wenn mehrere Generierungsläufe (z.B. unterschiedliche Seed-Offsets/Policy-Mischungen) später zu einem gemeinsamen Trainingsdatensatz zusammengeführt werden sollen. Beispiel: Lauf 1 mit `--game-id-start 0`, Lauf 2 mit `--game-id-start 100000`.
- CLI: `python3 data/generate.py --num-games 2000 --seed-offset 0 --output-dir data [--game-id-start N] [--policies always_buy conservative] [--start-capital 400]`

### 4.2.1 `start_capital`-Parameter (additiv, für H2-Diagnose) — implementiert

**Auslöser (§6b):** H2 (Affordability `cash >= price`) war im Hauptkorpus uninformativ: `can_afford` ist dort zu **99.5% TRUE** (19850/19946, nur 96 FALSE), weil Cash bei 1000 startet und Preise nur bis 300 gehen — das Modell hat im Training praktisch **keinen Druck**, `cash >= price` zu berechnen. Ein binärer Probe wäre ein 99.5%-Trivial-Klassifikator gewesen.

**Lösung (Spiegelbild des §3a-S3-Fixes):** ein **separater, markierter Diagnose-Datensatz** mit niedrigerem Startkapital, bei **fixem Board/Preisliste** (Lehre §4.1: den Preis pro Feld nicht antasten). Dafür `start_capital` additiv durch den Stack gezogen:

- `GameState(... start_capital=1000)` → `__post_init__` setzt `cash = [start_capital]*num_players` (statt hartkodiert 1000).
- `play_game(..., start_capital=1000)` reicht es durch.
- `generate()` + CLI `--start-capital`, im `make_run_id` (`_sc<cap>`-Tag) und im Manifest dokumentiert.

**Klassenbalance-Messung (800 Spiele, `seed_offset=9001`, cap=400):** Unaffordability ist mit niedrigem Startkapital reichlich vorhanden:

| Startkapital | 800 Spiele | Entscheidungspunkte | unaffordabel | Anteil |
|---|---|---|---|---|
| 150 | 18332 | 12869 | 70.2% |
| 200 | 17641 | 10898 | 61.8% |
| 250 | 16933 | 9731 | 57.5% |
| 300 | 15987 | 5785 | 36.2% |
| **400** | 14158 | 3919 | **27.7%** |

`cap=400` hat die gesündeste Balance (~72% affordable / ~28% unaffordable), ist weniger degeneriert als 150–250 und lässt die `conservative`-Policy (<300 Rest) Policy-PASS von echter Unaffordability unterscheiden. **Datensatz:** `data/games_so9001_n800_always_buy-conservative-random_sc400.jsonl` (800 Spiele, `game_id` 200000–200799, eigener seed_offset → **nicht** in den dokumentierten Hauptkorpus gemischt). Coverage: 14384 Entscheidungspunkte (3065 BUY / 11319 PASS), 28.1% unaffordable.
- **Wichtig:** `generate.py` erzeugt nur den Basisdatensatz (entspricht S0/Rohdaten). Die held-out-Filterlogik für S1–S2 setzt als eigener Schritt darauf auf — kein erneutes Sampling nötig. **S3 ist in V0 nicht durchführbar** (siehe §3a) und wird nicht implementiert.

## 4.3 `data/splits.py` — Held-out-Split-Logik (S0/S1/S2)

Implementiert die Split-Regeln aus §3 auf dem Basisdatensatz. Analyse-Einheit für S1/S2 sind die **Kauf-Entscheidungspunkte** (BUY/PASS). Die Affordability-Relation eines Entscheidungspunkts ist `relation = cash - price`.

## 4.3.1 WICHTIGER Bugfix: Pre-Buy-Zustand (empirisch verifiziert)

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

## 4.3.2 S2-Auswahlkriterien (Mindest-N + Partien-Impact)

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



