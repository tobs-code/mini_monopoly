# 1. Forschungsfrage

Ursprünglich stand die Frage "Haben LLMs ein Weltmodell?" im Raum — das wurde im Verlauf der Diskussion bewusst verworfen, weil zu groß und nicht scharf genug definiert.

**Aktuelle, präzisere Forschungsfrage:**

> Unter welchen Bedingungen entwickeln kleine Transformer, die nur auf Ereignissequenzen (nicht auf explizitem Zustand) trainiert werden, kompositionale interne Repräsentationen eines latenten dynamischen Zustands **und führen Operationen auf diesen Zuständen aus** — und wie hängen Repräsentation und Operation zusammen? Lässt sich das mechanistisch (nicht nur behavioral) nachweisen?

**Wichtige Klarstellung (aus der Diskussion):** "Zustand repräsentieren" und "Operationen über Zustände ausführen" sind **kein striktes entweder/oder**. Ein Modell kann beides haben — eine latente Zustandsdarstellung und Operationen darauf, die kausal in die nächste-Token-Vorhersage einfließen. Das eigentlich Spannende ist daher: **Welche Zustandsvariablen werden intern repräsentiert, welche Operationen werden darauf ausgeführt, und sind diese Strukturen kompositionell und kausal wiederverwendbar?**

Methodischer Leitstern für das ganze Experiment: auf jeder Stufe sauber unterscheiden zwischen
- *"Kann ich die Information aus der Aktivierung lesen?"* (Decodability/Probing) und
- *"Ist die Information kausal an der Berechnung beteiligt?"* (Intervention).

Probing allein beweist keine Kausalität — erst die Intervention.

Das Mini-Monopoly-Spiel ist dabei **nur das kontrollierte Versuchssystem**, kein Selbstzweck. Der eigentliche Wert liegt darin, dass wir (anders als bei echten LLMs auf Naturalsprache) die **vollständige Ground Truth** des latenten Zustands zu jedem Zeitpunkt kennen.

## Bezug zur laufenden Forschung (Kontext, warum das relevant ist)

- **Othello-GPT** (Li et al. 2022/2023): Transformer, nur auf legalen Zugfolgen trainiert, entwickelt nachweisbar internen Board-State. Ursprünglich nur mit *nichtlinearen* Probes nachweisbar.
- **Nanda (2023)**: Zeigte, dass die Repräsentation eigentlich *linear* ist — nur in der falschen Basis gesucht ("schwarz/weiß" statt "eigene Farbe/gegnerische Farbe"). Wichtige Lektion: die richtige Feature-Basis zu finden ist selbst ein Problem (Bezug zu Superposition/SAEs).
- **Gegenposition "Bags of Heuristics"** (Vafa et al., aufgegriffen von Melanie Mitchell): Was wie ein kohärentes Weltmodell aussieht, könnte in Wahrheit ein Bündel vieler lokaler Heuristiken sein, ohne echte kompositionale/generalisierbare Struktur. **Dieses Experiment ist im Kern ein direkter Test gegen diese Hypothese**, weil wir durch Held-out-Splits (s.u.) explizit prüfen können, ob Generalisierung stattfindet oder nur Lookup-artiges Verhalten.
- **Chess-LLM-Nachfolgearbeiten** (Mei et al. 2025 u.a.): Analoge Befunde zu Othello, linear dekodierbare Board-Features, kausale Interventionen wirksam.
- **"Revisiting the Othello World Model Hypothesis"** (2025): Zeigt, dass die Robustheit der Repräsentation vom Trainingsregime abhängt — Ergebnisse sind nicht automatisch trainingsdaten-invariant. Relevanz hier: deshalb bewusst mehrere Kauf-Policies mischen (siehe Abschnitt 4).

## Bezug zu Tobs' eigener laufender Forschung

Dieses Experiment ist konzeptionell verwandt mit Tobs' Hauptprojekt `llm-control-circuits` (mechanistische Interpretability an chinesischen Open-Source-LLMs, SAE-basierte Feature-Analyse, Layer-8–23-Kaskaden bei Zensur-Mechanismen). Der entscheidende Unterschied und Vorteil hier: **volle Ground Truth** des latenten Zustands ist bekannt — bei der Qwen-Arbeit muss Ground Truth erst mühsam rekonstruiert werden. Mini-Monopoly ist als kleines, sauberes Vorstudien-/Laborsystem gedacht, dessen Methodik (Probing, SAE, Interventionen) später wieder auf die Hauptforschung zurückwirken kann.

---

