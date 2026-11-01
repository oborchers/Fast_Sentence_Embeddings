Fünf Befunde sind im Code belegt. Ein tatsächlicher Schadensfall ist damit nicht nachgewiesen.

**1. Repositoryumfang, Konfliktschutz und Parallelität**

**Fehler: Parallele Publisher können gegenseitig ihre Belege überschreiben.**  
[publish_hub_now.py:64](/Users/oliverborchers/fse-dr157/publication-review/check-out/benchmarks/baseline/results/dr157-20261101/publication-preparation/scripts/publish_hub_now.py:64)

`LEDGER.exists()` reserviert den Lauf nicht atomar. Zwei Prozesse können diese Prüfung und die Namensprüfung gleichzeitig passieren. Beide schreiben anschließend dieselben Belegdateien über `persist()`. Auch der Fehlerpfad schreibt seinen eigenen Zustand dorthin. Dadurch können bereits gespeicherte Uploadrevisionen verloren gehen. `exist_ok=False` verhindert diesen lokalen Belegverlust nicht.

**Korrektur:** Vor der Ledgerprüfung eine exklusive Prozesssperre erwerben und bis zum Prozessende halten. Ein abgewiesener Prozess darf keine gemeinsamen Belege schreiben.

**2. Lokale Dateiprüfung, Modellidentitäten, Manifest und Karten**

Keine Befunde

Prüfgrenze: Modellbytes, ursprüngliches Manifest und Implementierung von `check_files` liegen außerhalb des erlaubten Lesebereichs. Ihre vollständige Übereinstimmung beziehungsweise Prüfstrenge bleibt deshalb unbestätigt. Die archivierten Receipts ersetzen diese unabhängige Prüfung nicht.

**3. Öffentlicher Readback, dauerhafte Belege, Abschlussprüfung und Bereinigung**

**Fehler: Die Abschlussprüfung stoppt bei Verifikationsfehlern nicht.**  
[finish_publication_now.py:69](/Users/oliverborchers/fse-dr157/publication-review/check-out/benchmarks/baseline/results/dr157-20261101/publication-preparation/scripts/finish_publication_now.py:69), ergänzend `147a4bf:benchmarks/verify_published_hub.py:228`.

Der gestartete Verifier speichert fehlgeschlagene Ergebnisse, setzt seine Schleife aber mit dem nächsten Repository fort. Erst nach allen Einträgen liefert er einen Fehlerstatus. Das widerspricht der Vorgabe, bei Verifikationsfehlern anzuhalten.

**Korrektur:** Nach dem dauerhaften Speichern des ersten fehlgeschlagenen Ergebnisses abbrechen. Weitere Repositories dürfen anschließend nicht mehr geprüft werden.

**Fehler: Ein erneuter Monitorstart überschreibt vorhandene Abschlussinformationen.**  
[finish_publication_now.py:98](/Users/oliverborchers/fse-dr157/publication-review/check-out/benchmarks/baseline/results/dr157-20261101/publication-preparation/scripts/finish_publication_now.py:98)

Die Existenzprüfung in Zeile 48 verweigert ausdrücklich das Überschreiben vorhandener Belege. Ihr `FileExistsError` landet jedoch im äußeren Fehlerhandler. Dieser speichert den neu initialisierten Zustand nach `completion-monitor.json`. Vorhandene Abschlussinformationen, einschließlich `evidence_sha256`, gehen dabei verloren. Die eigentliche Readback-Datei bleibt bestehen.

**Korrektur:** Existenzkonflikte vor dem schreibenden Fehlerhandler behandeln. Nur ein Lauf, der seine Zustandsdatei selbst angelegt hat, darf sie aktualisieren.

**4. Betriebsfehler und Offenlegung**

**Fehler: Benachrichtigungsfehler verhindern die verpflichtende Abschlussprüfung.**  
[finish_publication_now.py:42](/Users/oliverborchers/fse-dr157/publication-review/check-out/benchmarks/baseline/results/dr157-20261101/publication-preparation/scripts/finish_publication_now.py:42)

`notify()` wirft bei Herdr-Fehlern oder Zeitüberschreitungen eine Ausnahme. Diese beendet über `progress()` die gesamte Überwachung. Während des Uploads entfällt dadurch die spätere Abschlussprüfung. Während der Abschlussprüfung läuft der gestartete Kindprozess ohne weitere Überwachung weiter.

**Korrektur:** Benachrichtigungsfehler separat protokollieren und erneut zustellen. Prüfung und Prozessüberwachung müssen unabhängig davon weiterlaufen.

**Warnung: Fehlgeschlagene Prüfungen erscheinen in Erfolgsmeldungen.**  
[finish_publication_now.py:78](/Users/oliverborchers/fse-dr157/publication-review/check-out/benchmarks/baseline/results/dr157-20261101/publication-preparation/scripts/finish_publication_now.py:78)

`len(evidence['repositories'])` zählt erfolgreiche und fehlgeschlagene Prüfungen. `progress()` meldet diese Anzahl anschließend als „erfolgreich geprüft“. Beispielsweise können vier bestandene und eine fehlgeschlagene Prüfung als fünf Erfolge gemeldet werden.

**Korrektur:** Für Erfolgsmeldungen ausschließlich Einträge mit `status == 'passed'` zählen. Fehlgeschlagene Einträge müssen den Abbruch auslösen.

Prüfgrenze zur Offenlegung: Die Implementierung von `safe_error()` im ausgeschlossenen Credential-Helfer ist unbekannt. Eine zuverlässige Bereinigung seiner Fehlerausgaben ist daher weder bestätigt noch widerlegt.