Zahlenwerte und Baselines sind unverändert. Zwei Befunde geringer Schwere betreffen Dokumentation und Testabsicherung.

1. **Übernommene Zahlenwerte und Herkunft: Keine Befunde.**  
   Alle fünf Archivdateien entsprechen bytegenau `06c73bb`. Übernommene Werte, umbenannte Felder und berechnete Anzahlen stimmen. Archivpfade, Commit, Dateigrößen und SHA256-Prüfsummen sind korrekt.

2. **Auslassungen und Verbraucher: P3, irreführende Linkbeschriftung.**  
   **Ort:** [PERFORMANCE.md:427](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/master-studies-review-wt/PERFORMANCE.md:427).  
   **Beleg:** „Compact timeline and profile summary“ verweist auf eine Zusammenfassung ohne `timeline`. Dieser komplette Abschnitt existiert nur noch im verlinkten Archiv.  
   **Korrektur:** Den Link „Compact profile summary“ nennen und die Timeline ausdrücklich dem Archiv zuordnen. Alternativ die Timeline-Zusammenfassung übernehmen.  
   Weitere irreführende Auslassungen oder betroffene ausführbare Leser habe ich nicht gefunden.

3. **Tests, Archivlinks und 1.1-Baselines: P3, unvollständige Herkunftsprüfung.**  
   **Ort:** [tests/test_evidence_summaries.py:121](/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/master-studies-review-wt/tests/test_evidence_summaries.py:121).  
   **Beleg:** Der Test vergleicht lediglich zwei deklarierte Prüfsummen und prüft das URL-Ende. `original(name)` liest unabhängig davon den Manifest-Commit. Ein falscher `raw_evidence.commit` mit entsprechend angepasster URL würde deshalb weiterhin akzeptiert. Die tatsächlichen Archivbytes werden nicht gegen SHA256 geprüft.  
   **Korrektur:** Commit und Pfad an Manifest beziehungsweise Dateiname binden. Die vollständige URL prüfen und SHA256 aus den gelesenen Archivbytes berechnen.  
   Die aktuellen Archivlinks sind korrekt. Alle acht 1.1-Baselines sind bytegleich zwischen Ausgangscommit, Zielcommit und Archiv. Der Fixture-Generator blieb unverändert.

Erledigt: Statischer Review abgeschlossen, keine Dateien geändert.  
Offen: Zwei Befunde. Tests, Builds und Modellläufe wurden auftragsgemäß nicht ausgeführt.  
Benötigt von dir: nichts.