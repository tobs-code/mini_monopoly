# 5. Modell, Tokenisierung und Analyse

## 5.1 Tokenisierung von Cash — ENTSCHIEDEN

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

## 5.1a Tokenisierungs-Fix: Tausenderstelle fehlt — ENTSCHIEDEN nach Coverage-Check

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

## 5.2 Modellarchitektur (grobe Eckdaten, für 8GB VRAM realistisch)

- 4–6 Transformer-Layer
- `d_model` ~256–384
- 4–6 Attention-Heads
- Kontextlänge ~128–256 Tokens
- ~5–15M Parameter gesamt
- FP16-Training

**Kontextlänge (empirisch, voller Datensatz, nach Tokenisierung §6-4):** min=180, Median 226, max=284; 66/2000 Spiele (3.3%) überschreiten 256. Da Tokenisieren der Miete als 5-Token-Digit-Sequenz jeden Rent-Event aufbläht, **wird `n_ctx = 288` gewählt** (≥ max. 284), um KEINE Trajectory zu kürzen (Trajectory-Integrity-Regel, §3). Das überschreitet die obige 256-Obergrenze nur geringfügig (~12%) bei vernachlässigbarem Parameterzuwachs und erhält die Integrität aller Partien. Die Zahl ist an die aktuelle Tokenisierung gebunden; ändert sich die Digit-Zerlegung, muss sie neu gemessen werden.

**Grober VRAM-Überschlag für `n_ctx=288` auf der RTX 2080 (8GB):** Bei `d_model=256`, 4 Layer, `d_mlp=1024`, `d_vocab=48` ≈ **3.3M Parameter** (≈7 MB FP16). Trainings-Aktivierungen bei Batch 32: ~32×288×256×4 Layer ≈ 75 MB FP16; Adam-Optimizer + Gradienten addieren ~3× Gewichte. Selbst `run_with_cache` (hält Aktivierungen **aller** Layer gleichzeitig) liegt bei Batch 32 im Bereich ~0.4–2 GB je nach Hook-Umfang. Für 8GB ist damit **komfortabel Luft**; falls OOM droht, zuerst Batch halbieren (16), nicht `n_ctx` senken (Integrität). Der Überschlag ist bewusst konservativ — vor dem ersten echten Trainingslauf einmal mit einem echten Batch laufen lassen und VRAM prüfen.

Das ist eine völlig andere Größenordnung als reguläre LLM-Finetuning-Jobs — realistisch in Stunden, nicht Tagen, auf einer RTX 2080 trainierbar. Vorteil zieht sich auch durch die Analyse-Pipeline: Residual-Stream-Dimension (256–384) ist so klein, dass auch SAE-Training auf den Aktivierungen dieses Modells trivial billig ist (Minuten statt Tage, im Vergleich zur Qwen-Arbeit).

## 5.3 "Ein Modell, viele Analysen" statt viele Trainingsläufe

Wegen der Hardware-Beschränkung: **nicht** sechs separate Modellversionen (V1 nur Position, V2 +Cash, V3 +Ownership, ...) einzeln trainieren. Stattdessen:

- **Ein** History-only-Modell trainieren, das die volle Ereignishistorie sieht (Position, Cash, Ownership sind alle nur implizit über Events erschließbar, nie explizit im Input).
- Danach beliebig viele *Analysen* auf demselben Checkpoint fahren: Position-Probe, Cash-Probe, Ownership-Probe, Cross-Variable-Probes, Interventionen — das sind reine Forward-Passes, kein erneutes Training.
- Das explizite-State-Modell ("Modell A" aus früheren Diskussionsrunden, das den Zustand direkt im Input bekommt, zum Vergleich mit dem History-only-Modell "Modell B") ist **zunächst gestrichen** — kostet einen kompletten zusätzlichen Trainingslauf und ist nur relevant, falls das History-only-Modell überhaupt erstmal etwas Interessantes zeigt. Erst bei Erfolg von H1 wieder aufgreifen.
- Seeds/mehrere unabhängige Trainingsläufe für Reproduzierbarkeit: ebenfalls zurückgestellt, bis ein erstes reproduzierbares mechanistisches Phänomen gefunden wurde ("lohnt sich ein zweiter Seed erst danach, nicht vorher").

## 5.4 Analyse-Reihenfolge: Layer-wise Probing vor SAE

Vor jeder SAE-Analyse: Probing-Accuracy pro Layer messen (nicht nur eine Gesamt-Accuracy), um zu sehen, **wo** im Netzwerk welche Information entsteht:

```
Layer 0   position 30%     cash 10%
Layer 1   position 60%     cash 15%
Layer 2   position 90%     cash 35%
Layer 3   position 99%     cash 80%
Layer 4   position 99%     cash 95%
```

Das ergibt eine grobe "Computational Story" (z.B. Position wird zuerst rekonstruiert, ökonomischer Zustand danach) — mit Vorsicht zu interpretieren, aber ein guter Ausgangspunkt für gezielte SAE-Analyse. **SAE erst einsetzen, wenn klar ist, "hier zwischen Layer X und Y passiert etwas Interessantes"** — nicht explorativ ohne Hypothese ("ich hab ne SAE, mal schauen ob irgendein Feature Monopoly macht").

## 5.5 Framework / Analyse-Pipeline — ENTSCHIEDEN (Spike abgeschlossen)

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

## 5.6 Tokenizer/Vocab — ENTSCHIEDEN (implementiert in `data/tokenizer.py`)

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

## 5.7 Trainingsläufe & erste H1-Vorproben (Stand)

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

**H2-Vortest (b) — Affordability auf dem Low-Cap-Korpus (`probes/probe_afford.py`, laufend auf run2).** Der unter §6b geplante billige Vortest: run2 (trainiert auf cap=1000, nie mit Affordability-Varianz) wird auf den `cap=400`-Spielen geprobt. Ein **positives** Ergebnis wäre eindeutig (emergente Komposition aus linear zugänglichen Teilen ohne Trainingsdruck); ein negatives zweideutig und billig auszuschließen.

- **Pre-registriertes S1-Design (2x2, balanciert):** held-out `(cash, price)`-Paare `cash∈{150,250} × price∈{150,300}` = 2 afford `(150,150),(250,150)` + 2 unafford `(150,300),(250,300)`. Verifiziert: diese 4 Paare erscheinen **nicht** im Trainings-Split (Trajectory-Integrity, keine Leakage); Einzelwerte `cash=150/250` und `price=150/300` sind im Train sichtbar → sauberer Kompositionstest.
- **Ergebnis (S1-held-out-Testregion, 1110 Punkte, 441 afford / 669 unaff):**

| Checkpoint | HeldAcc (best Layer, linear) | Majority-Baseline |
|---|---|---|
| step 250 | 74.5% | 60.3% |
| step 500 | 72.5% | 60.3% |
| **step 650 (best)** | **73.9%** | **60.3%** |
| step 750 | 74.1% | 60.3% |

  Stabil über alle Checkpoints. Der nichtlineare Probe ändert nichts (HeldAcc ~73.6%) — die Info ist nicht stärker über linear hinaus verfügbar.
- **Per-Paar-Breakdown (best, linear):** `(150,150)` afford 80%, `(150,300)` unafford 85%, `(250,150)` afford 90%, aber **`(250,300)` unafford nur ~53–56%** (über Checkpoints konsistent der schwächste, nahe Zufall/Majority). Das ist das Paar, dessen `cash` und `price` **nah an der Entscheidungsgrenze** liegen (`250` vs. `300`, Differenz 50).
- **Interpretation:** HeldAcc ~74% vs. 60.3% Majority-Baseline ist ein **reales, positives** Signalt für Affordability auf held-out `(cash, price)`-Kombinationen — das Modell hat die Komposition **ohne** expliziten Trainingsdruck teilweise gelernt (emergente Komposition aus linear zugänglichen Teilen). **Aber:** Die Grenznahe `(250,300)`-Kombination bleibt nahe Zufall — das Muster ist eher ein **preis-/cash-nähe-basiertes, imperfektes Heuristik-Signal** als eine saubere, game-invariante `cash>=price`-Variable. Der Befund ist damit **moderat positiv**, nicht stark: die Komposition existiert ansatzweise, generalisiert aber nicht sauber auf die hart-grenznahe Konstellation.
- **Gate-Checks vor dem Zusatztraining (7c) — beide bestanden:**
  - **Check 1 (pro-Paar-Majority + Konfund-Schieflage):** Alle 4 held-out-Paare sind je rein in einer Klasse (pro-Paar-Majority 100%, da `afford` deterministisch ist). Innerhalb jedes Paars sind **Policy, Player und Position gut durchmischt** (alle Policies vertreten, Player ~balanced, Positionen verteilt) — **kein** "Paar kommt fast nur bei P2 auf Feld X"-Konfund.
  - **Check 2 (trivial-Feature-Kontrollproben, gleicher S1-Split, eval auf held-out):** `price` 39.7%, `position` 39.7%, `player` 39.7%, `policy` 50.6%, `decision` 39.7%, `price+position` 39.7%, `price+decision` 39.7%, `price+cash` 48.6% — **alle deutlich UNTER** der Majority-Baseline (60.3%) und dem Residual-Signal (74%). Besonders wichtig: trotz des 2x2-Designs (alle afford-Paare `price=150`, alle unafford-Paare `price=300`) schafft **`price` allein nur 39.7%** — ein price-Shortcut hätte die held-out-Paare trivial lösen können, tut es aber nicht. Auch `price+cash` (die beiden "echten" Variablen direkt) bleibt bei 48.6%. **Das 74%-Signal ist also kein "low-cap dataset smell", sondern stammt echt aus dem Modell-Residual.**
- **Konsequenz (b → a, §6b):** Der Vortest ist **positiv** (nicht eindeutig negativ), also lohnt der gezielte **Zusatztrainingslauf (a)**: ein neues Modell **from scratch** auf einem Korpus MIT Affordability-Varianz (Low-Cap + bestehender Korpus gemischt, nicht run2 feintunen — §6b), um zu testen, ob ein Modell, das die Entscheidung im Training **wirklich erlebt**, die Komposition sauber game-invariant lernt. Nur so ist ein sauberer Vergleichspunkt zu den flachen H1-Befunden gegeben. **Zielformulierung für 7c (Gate aus der Diskussion): nicht nur bessere Gesamt-Accuracy, sondern Generalisierung auf grenznahe held-out Relationen — `(250,300)` deutlich über Zufall.**

**H2-Zusatztraining (7c) — ERGEBNIS: teilweise positiv, Grenzfall nicht bestanden.** Neues Modell **from scratch** auf einem gemischten Korpus `data/games_mixed_h2.jsonl` (2800 Spiele = 2000 × cap=1000 + 800 × cap=400, deterministisch gemixt, `data/merge_corpora.py`, Mix-Seed 7; 34330 Entscheidungspunkte, davon 4132 unaffordable ≈ **12.0%** — Affordability-Varianz, aber cap=1000-Dynamik dominant). Lauf `runs/h2_7c_mixed/` (`training/train.py`): identische Hyperparams zu run2 (lr 1e-3, wd 0.5, warmup 200, batch 32, seed 42), 1650 Steps ≈ 21 Epochen (Epochen-zahl an das größere Corpus angeglichen, val_frac 0.1 → train 2520/val 280).

- **Verlauf:** best_val **0.5708 bei step 1100** (= `ckpt_best.pt`), praktisch identisch zu run2 (0.5627) trotz schwierigerem Korpus. Danach **klassisches Overfitting**: Train fällt weiter bis ~0.50, Val steigt auf 0.5911 (final) — `(250,300)` wird dabei **nicht** besser. Die Über-Checkpoint-Stabilität (User-Gegenprobe) bestätigt: step 750 → HeldAcc 74.1% / `(250,300)` 57.3%; step 1000 → 76.3% / 62.1%; **step 1100 → 77.0% / 61.5%**; step 1250 → 76.3% / 61.9%; step 1500 → 76.4% / 60.4%; step 1650 → 76.5% / 60.6%. (HeldAcc-Werte des 7c-Runs: linear 77.0–77.6% je nach Auswertungslauf; MLP best Layer 73.9–75.5%, also **nicht** besser als linear.)
- **Vergleich Vortest (b, run2) → 7c (linear, best):** `(150,150)` 80.5% → 79.6–80.5%; `(150,300)` 84.9% → 91.0–91.5%; `(250,150)` 90.2% → 87.5–87.8%; **`(250,300)` ~53–56% → ~61–62%**; Gesamt-HeldAcc ~74% → ~77%.
- **Befund (nüchtern):** Der Trainingsdruck durch Affordability-Varianz hat **klar geholfen**: `(250,300)` von ~53–56% auf ~61–62%, Gesamt-HeldAcc von ~74% auf ~77%. Kontrollfeatures bleiben weit darunter (beste Kontrolle `policy` 50.6%, `price+cash` 48.6%) → das Signal ist **echt** aus dem Modell-Residual, kein trivialer Shortcut. **Linear > MLP** (77.0% vs. 75.5%) — das Signal ist linear zugänglich, nicht nur MLP-Zauberei. **Aber:** `(250,300)` bleibt der Schwachpunkt und ist **weit unter der 70%-Hürde**. Vorsicht mit "über Majority-Baseline" für dieses einzelne Paar: global knapp über 60%, aber dieses Paar ist komplett unaffordable und das Modell verwechselt dort weiterhin viele Fälle als afford. **Die Grenze ist nicht repariert, nur etwas weniger kaputt.**
- **Kein längeres Training als Hebel:** Nach step 1100 wurde Val schlechter, ohne `(250,300)` zu verbessern — mehr Schritte bringen nichts. Falls weitere Optimierung: **stärkerer Low-Cap-Anteil / besser balancierter Trainingskorpus / gezielt mehr Grenzfälle** in der Nähe der Entscheidungsgrenze, nicht mehr Training auf demselben Mix.
- **7c gilt als teilweise bestanden:** Der Trainingsdruck wirkt (das Modell nutzt die trainierte Affordability-Erfahrung für eine bessere, aber weiterhin imperfekte Näherung), **H2 ist damit aber nicht sauber gelöst** — die grenznahe Relation generalisiert weiterhin nicht sauber game-invariant.

- **Die eigentliche Pointe ist der Kontrast:** Das Modell nutzt die Würfelinfo **kausal** (F_L≈1, Kanal existiert und trägt fast den vollen Effekt, Befund 5), baut daraus aber **keine** konsistente, game-invariante, dekodierbare Zustandsvariable auf (Befund 2). Genau dieser Gegensatz — kausale Nutzung ja, saubere Zustandsrepräsentation nein — ist die spannendste Aussage des Kausal-Kapitels und passt zur **"Bags of Heuristics"**-Lesart (verteilte, spielabhängige Heuristiken statt einer sauberen internen Variable).

- **Methodik-Lektion (ausdrücklich festgehalten, nicht nur das Ergebnis):** Der frühere Probe splittete **per-Sample über Spiele gemischt** (Leakage-Konfund) und zeigte **Train 100% / Test ~18%** — die 100%-Train wurden damals als Memorisierung fehlinterpretiert. Der korrekte **pro-Spiel-Split** (32/40 Spiele, Trajectory-Integrity §3) zeigt **Train 42% (linear) / 100% (MLP), Test 18–19% in beiden Fällen**. Wichtig: Die niedrige **Test**-Zahl war schon vorher richtig; aber die **Trainingswerte** waren durch Leakage verzerrt — jemand, der nur die alte 100/18-Zahl im Kopf hat, würde fälschlich schließen, das Modell hätte in-sample eine perfekte lineare Position-Repräsentation gehabt. Das hatte es **nicht** (linear nur 42% Train, d.h. Position ist selbst in den bekannten Spielen nicht sauber linear separierbar). Die früheren Kandidaten (a) falsche Basis / (b) nicht-lineare Verarbeitung / (c) Kapazität sind damit als Erklärung für einen flachen Probe **ausgeschlossen** — weder linear noch nichtlinear ist Position als game-invariante Größe aus `resid_post` dekodierbar.

---

