**Ein Befund, P2: Das Ausgabeverzeichnis wird nicht mehr exklusiv angelegt.**

[verify_published_hub.py:225](/Users/oliverborchers/fse-dr157/review-2f4c1a8/benchmarks/verify_published_hub.py:225), [hub_readback_evidence.py:29](/Users/oliverborchers/fse-dr157/review-2f4c1a8/benchmarks/hub_readback_evidence.py:29)

Zwei Prozesse können beide die Existenzprüfung in Zeile 216 passieren. Erstellt einer danach das Verzeichnis, akzeptiert `mkdir_durable` dieses beim zweiten Prozess stillschweigend. Das bisherige `mkdir(parents=True)` hätte den zweiten Prozess abgewiesen.

Beide Prozesse können dadurch dieselben Logs und Ergebnisdateien überschreiben. Vorhandene Belege können verändert oder unterschiedlichen Durchläufen zugeordnet werden.

**Konkreter Fix:** Für das Ausgabeverzeichnis eine exklusive Erstellung verlangen. Elternverzeichnisse weiterhin dauerhaft anlegen, das Ziel mit `exist_ok=False` erstellen und anschließend dessen Elternverzeichnis synchronisieren. Für Scratch bestehende Verzeichnisse zulassen. Einen Regressionstest für die Erstellung zwischen Existenzprüfung und `mkdir_durable` ergänzen.

Für Verzeichnissynchronisierung, Messwerterhalt bei Timeout beziehungsweise Prozessfehler und atomare Veröffentlichung vor Cache-Löschung sonst keine Befunde.

Erledigt: Statische Prüfung von `73176a9..2f4c1a8`. Keine Änderungen, Tests oder Netzwerkzugriffe.  
Offen: Ein P2-Befund. Tatsächliche Hub-Prüfungen warten auf ausdrückliche Freigabe und Upload-Commit-Belege.  
Von dir benötigt: nichts für dieses Review.