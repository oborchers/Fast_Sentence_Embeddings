Ein **P2-Befund zur Testabdeckung**. Die übrigen Kategorien zeigen bei statischer Prüfung keine Befunde.

**P2: Die echte Laufzeitprüfung wird in keinem neuen Test ausgeführt.**  
Fundstelle: [tests/test_hub_release_cards.py:56](/Users/oliverborchers/fse-dr157/review-4d0bc04/tests/test_hub_release_cards.py:56).

Die gemeinsame Fixture ersetzt `check_runtime` vollständig. Auch der Fehlertest ersetzt diese Funktion durch eine direkt ausgelöste Ausnahme, siehe Zeile 166. Dadurch bleiben Modellladen, Metadatenvergleich und beide Probe-Bedingungen ungetestet. Eine versehentliche Entfernung dieser Prüfungen würde die vorhandenen Tests nicht brechen.

**Konkrete Korrektur:** `check_runtime` mit kleinen lokalen Modellen tatsächlich ausführen. Metadatenabweichung, Toleranzverletzung und Bitabweichung jeweils getrennt prüfen. Die FastText-Fixture braucht echte Subword-Parameter und N-Gramm-Daten. Aktuell verwendet sie gewöhnliche Wortvektoren unter FastText-Repositorynamen. Produktionsmodelle sind dafür nicht nötig.

| Kategorie | Ergebnis und Evidenz |
|---|---|
| Exakt sechs FastText-Holds | **Keine Befunde.** Der strukturierte Manifestvergleich bestätigt ausschließlich sechs Änderungen von `publication_hold` auf `null`. Nach deren Normalisierung sind beide Manifeste vollständig gleich. [publication-manifest.json:7678](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/baseline/dr157/publication-manifest.json:7678) |
| Neue SHA256-Bindung | **Keine Befunde.** Nachgerechnet: `ac85cae41b05788bc278d68fd3231489de1615c7c15b8b79229844c3d0dabe69`, 175.786 Bytes. Beides stimmt mit der Bindungsdatei überein. Der Runner prüft den übergebenen Hash vor Releaseprüfung und Ausgabe. [hub_release_cards.py:153](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/hub_release_cards.py:153) |
| Release-, Tag- und Laufzeitbedingungen | **Keine Befunde im Prüfcode.** Beide Versionsangaben müssen exakt `2.0.0` sein. Der Remote-Tag ist erforderlich; annotierte Tags werden aufgelöst. Metadaten, Toleranz und Bitgleichheit werden vor Ausgabe geprüft. [hub_release_cards.py:40](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/hub_release_cards.py:40), [hub_release_cards.py:80](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/hub_release_cards.py:80) |
| Exklusive lokale Hardlink-Verzeichnisse | **Keine Befunde.** Bestehende Ausgaben werden abgelehnt. Verzeichnisse entstehen exklusiv, Nicht-README-Dateien ausschließlich durch `os.link`, Karten durch exklusives Öffnen. [hub_release_cards.py:130](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/hub_release_cards.py:130) |
| Karteninhalt und Writer-Provenienz | **Keine Befunde.** Alle 30 eingecheckten Kartenvorschauen stimmen nach Größe und SHA256 mit dem Manifest überein. Beide Ersetzungsmuster kommen jeweils genau einmal vor. Der historische Writer `fse 1.1.1.dev917+g81e64c0` bleibt erhalten, ebenso Lizenztext, Messwerte und Fit-Provenienz. [hub_release_cards.py:55](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/hub_release_cards.py:55) |
| Alte und neue Modell-SHA256 | **Keine Befunde.** Alle 135 bisherigen Dateispezifikationen bleiben unverändert. Der neue Ausgabeprüfpfad übernimmt sämtliche Nicht-README-Spezifikationen und verifiziert deren tatsächliche Hashes erneut. [hub_release_cards.py:138](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/hub_release_cards.py:138) |
| Manifest, Übergangsbeleg und Fehlerverhalten | **Keine Befunde.** Der Übergangsbeleg bindet Basis- und Ausgabemanifest sowie Tag-Commit und Modellhashes. Ausnahmen brechen ab. Unvollständige Ausgaben werden weder automatisch entfernt noch wiederverwendet. Upload-Aufrufe fehlen. [hub_release_cards.py:150](/Users/oliverborchers/fse-dr157/review-4d0bc04/benchmarks/hub_release_cards.py:150) |
| Aussagekräftige Tests aller 30 Varianten | **P2, oben beschrieben.** Hardlinks, Dateihashes und README-Wechsel sind für alle 30 Namen abgedeckt. Die tatsächliche Laufzeitkompatibilität und FastText-Struktur sind es nicht. |

Keine Tests, Netzwerkzugriffe oder Tensorlesevorgänge ausgeführt. Keine Dateien geändert. Die ausstehende Produktionsausführung nach Release ist ausdrücklich kein Befund.

Erledigt: Statische Prüfung von `802b0a3..4d0bc04` im angefragten Umfang.  
Offen: P2-Testlücke schließen; Produktionsprüfung planmäßig nach Release.  
Von dir benötigt: nichts.