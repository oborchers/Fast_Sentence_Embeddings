# Finaler Tempo-Lauf bereit

DR-151 (Tempo, Qualität und Kosten gegen die Alternativen vergleichen). Der Koordinator startet, überwacht, sichert und löscht die Maschine. Diese Übergabe ersetzt den fehlgeschlagenen ersten Start. Dieser Auftrag startet keine Maschine selbst. Kein Merge und kein automatischer Wiederholungsversuch.

## Festgeschriebener Stand

Quell-Pin: `0c639113f80383fda461f3e776db38eee0efee71`. Alle sieben folgenden Dateien sind in diesem gepushten Commit enthalten. Der Übergabe-Commit darf später sein, solange diese Dateien bytegleich bleiben.

| Datei unter benchmarks/ | Bytes | SHA256 |
| --- | ---: | --- |
| `speed-run-2026-11.py` | 19382 | `d3d9215f279aa1926a1c29d9246db333513cd60c1ae7542938b80c162c769ee1` |
| `_speed_cells.py` | 18529 | `dc2fb1704b070cdc8c8598a8a31b48af3f69d81403ad70d38cf76bf6bf6b9bf9` |
| `_speed_pipeline.py` | 19431 | `84de9036028065814211d929284f073f0476a2f4269617f92662ec6e81e8a54f` |
| `speed-input-2026-11.json` | 10108 | `7a446913d180d4aeece6ac3c17eaac983dec7bc1f6c19407c71640cf50d8c7eb` |
| `speed-main.lock.txt` | 173667 | `e7db4ba8b6afdf36d9eb0cd41499cb10ff01d3a4a3bacbd7a413d3e2c964f875` |
| `speed-reference.lock.txt` | 98420 | `218d9855038b4ef6f3253c04c1f031350f4c52135fe16cbff35bf625e348f73d` |
| `speed-rust.lock` | 26727 | `b64150bf3e47180247cf6d220848c4ac1191dfc6dfc6b62ac86cc178dbaebe0f` |

Genau ein ausführbarer Python-Einstieg und zwei private Messmodule. Die drei Locks binden Python-Pakete einschließlich Artefakt-Hashes sowie die Rust-Abhängigkeiten. Das native fse wird separat aus `624251df2269e3099f5c61987be63e718893b58c` gebaut. Die Rust-Bibliothek bleibt auf `19f58bef6ce145f2bc08de7016498f28d521feea`, Compiler 1.98.0.

## Exakter Start

- Image: `python:3.13@sha256:f23cf6c01ace0da9aa4d04fd078dba0fcce0c23cf3320329a38a2b6890761f9c`.
- Plattform: Linux amd64, Python 3.13.18.
- Maschinentyp: cpu3c, 32 vCPU, 64 GB RAM, **150 GB Container-Disk**.
- Entrypoint: `/bin/bash`.
- Argumente: `-lc` und der vollständige folgende Befehlsblock als ein Argument.
- `startSsh=true`, Ports `["22/tcp"]`. Die Plattform injiziert `PUBLIC_KEY` aus dem registrierten Kontoschlüssel.
- Zusätzliches Env: `SPEED_ALLOCATED_AT`, tatsächliche Allokationszeit als Unix-Sekunden. Nicht erst den späteren Messbeginn einsetzen.
- Weitere Env-Werte stehen ausdrücklich im Befehlsblock. Keinen externen Timeout ergänzen, der eine begonnene Zelle abschneidet.

Der Shell-Block richtet zuerst SSH ein. Danach installiert der eine Python-Einstieg die gesperrten Umgebungen. Er beginnt anschließend denselben vollständigen Messlauf. Der Supervisor startet nichts erneut und wartet nach jedem terminalen Abschluss mit `sleep infinity`, bis der Koordinator die Maschine löscht. Bei einem frühen Fehler erzeugt er einen Abschluss ohne erfundene Messwerte.

```bash
set -euo pipefail
umask 077
mkdir -p /speed-logs
exec > >(tee -a /speed-logs/bootstrap.log) 2>&1
finish() {
  export SPEED_EXIT_STATUS="$?"
  set +e
  trap - EXIT
  /usr/local/bin/python - <<'FAILURE_RECORD'
import hashlib,json,os,pathlib,time
p=pathlib.Path('/speed-results')
if not (p/'DONE').exists():
    p.mkdir(parents=True,exist_ok=True)
    status=int(os.environ['SPEED_EXIT_STATUS']) or 1
    def write(name,value):
        q=p/(name+'.pending')
        with q.open('w') as f:
            json.dump(value,f);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(q,p/name)
    (p/'cells.jsonl').touch(exist_ok=True)
    rows=sum(1 for _ in (p/'cells.jsonl').open())
    write('progress.json',dict(stage='terminal-failure',cell='FAILED',finished=rows,planned=26,exit_status=status))
    files=[dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(p.rglob('*')) if f.is_file() and f.name not in ('DONE','manifest.json')]
    write('manifest.json',dict(files=files))
    write('DONE',dict(exit_status=status,completed_at=time.time(),cells=rows,manifest_sha256=hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),terminal_supervisor=True))
FAILURE_RECORD
  sleep infinity
}
trap finish EXIT
mkdir -p /speed-state
first_run=1
if ! (set -o noclobber; printf '%s\n' '0c639113f80383fda461f3e776db38eee0efee71' > /speed-state/started); then
  first_run=0
fi
if ! test -x /usr/sbin/sshd; then
  /usr/bin/apt-get update
  /usr/bin/apt-get install -y openssh-server git
fi
mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh
printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
/usr/sbin/sshd
if test "$first_run" = 0 || test -e /speed-source || test -e /speed-env || test -e /speed-work || test -e /speed-results; then
  printf 'RESTART_DETECTED: SSH only; setup and measurement will not restart.\n'
  exit 77
fi
: "${SPEED_ALLOCATED_AT:?Set the actual allocation Unix timestamp}"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/speed-cache/uv
export HF_HOME=/speed-cache/huggingface
export HF_DATASETS_CACHE=/speed-cache/datasets
export RUSTUP_HOME=/root/.rustup
export CARGO_HOME=/root/.cargo
export LC_ALL=C.UTF-8
/usr/bin/git clone --single-branch --branch dr151-compare https://github.com/oborchers/Fast_Sentence_Embeddings.git /speed-source
/usr/bin/git -C /speed-source checkout --detach 0c639113f80383fda461f3e776db38eee0efee71
export SPEED_SETUP_STARTED_AT="$(/usr/local/bin/python -c 'import time; print(time.time())')"
printf 'allocation=%s setup_started=%s\n' "$SPEED_ALLOCATED_AT" "$SPEED_SETUP_STARTED_AT"
/usr/local/bin/python /speed-source/benchmarks/speed-run-2026-11.py --setup /speed-env --allocated-at "$SPEED_ALLOCATED_AT"
/speed-env/main/bin/python /speed-source/benchmarks/speed-run-2026-11.py --run --results /speed-results --work /speed-work --reference-python /speed-env/reference/bin/python --allocated-at "$SPEED_ALLOCATED_AT" &
measurement_pid=$!
printf '%s\n' "$measurement_pid" > /speed-logs/measurement.pid
wait "$measurement_pid"
```

Die Allokationszeit zählt für Geld und Gesamthorizont, einschließlich Provisionierung. `SPEED_SETUP_STARTED_AT` begrenzt die eigentliche Einrichtung samt anschließender Vorbereitung. Langsames Image-Laden verbraucht deshalb Geld, aber nicht bereits das gesamte fünfminütige Einrichtungsfenster.

Während der Einrichtung fehlen die drei Controller-Dateien zunächst. Das ist innerhalb der erwarteten drei Minuten normal. Das Bootstrap-Log liegt bereits unter `/speed-logs/bootstrap.log`. Ein vorzeitiger Prozessfehler erzeugt über den Supervisor die drei Wächterdateien. Danach wartet der Supervisor immer mit `sleep infinity`. Er beendet den Container nicht. Eine atomare Startmarke verhindert erneutes Klonen, Einrichten und Messen nach einem externen Neustart. Dabei startet nur SSH neu. Beim Beginn des Controllers entsteht `progress.json` sofort. Die erste Ergebniszeile folgt nach der ersten vollständigen Zelle.

## Abholung alle zwei Minuten

Der Wächter liest exakt:

1. `/speed-results/cells.jsonl`
2. `/speed-results/progress.json`
3. `/speed-results/DONE`

Alle Rohbelege liegen unter `/speed-results/`: `operations.jsonl`, `checks.jsonl`, `input-identities.jsonl`, `preparation.jsonl`, `quiet.json`, `sized-input.json`, `environment.json`, `setup.json`, Jobdateien, Prozesslogs und Startdiagnosen. Die größten Textbelege sind `operations.jsonl` und die Prozesslogs. Es werden keine Vektormatrizen als Rohbelege gesammelt.

Das Transfermanifest heißt `/speed-results/manifest.json`. `DONE.manifest_sha256` bindet es. Jeder Eintrag nennt relativen Pfad, Bytes und SHA256. Manifest und DONE selbst sind bewusst nicht gegenseitig im Manifest enthalten. Bei der Abholung beide zusätzlich sichern und die Manifest-Prüfsumme gegen DONE prüfen.

Nach DONE vollständig kopieren: `/speed-results/`, zusätzlich `/speed-logs/bootstrap.log`, `/speed-logs/measurement.pid` und `/speed-env/setup.json`, sofern vorhanden. Der identische Setup-Nachweis steht auch im regulären Ergebnismanifest. Erst nach bytegeprüfter Sicherung löscht der Koordinator die Maschine sofort. Modellgewichte, vorbereitete Eingaben, Build-Zwischenstände und Paket-Caches unter `/speed-work`, `/speed-cache` und `/speed-env` nicht abholen. Geprüfte Schreibausgaben werden bereits während des Laufs gelöscht. Ungeprüfte Fehlerausgaben bleiben auf der Maschine und verhindern bei Platzmangel weitere Schreibarbeit.

## Stufen und erwartete Endzeiten

Die Endzeiten beziehen sich auf den Beginn der Einrichtung. Provisionierungs- und SSH-Zeit kommen für die Rechnung davor hinzu. Einzelne Schritte können länger dauern, ohne dass eine begonnene Zelle abgeschnitten wird.

| Stufe | Erwartet, Minuten | Erwartetes Ende | Stufenbudget, Minuten | Kumuliertes Budgetende |
| --- | ---: | --- | ---: | --- |
| Einrichtung und Vorbereitung | 3 | +00:03 | 5 | +00:05 |
| A/A-Ruhekontrolle | 9 | +00:12 | 14 | +00:19 |
| Einmalige Größenschätzung | 10 | +00:22 | 15 | +00:34 |
| 100 Mio. Texte, vier Streaming-Schreibläufe | 18 | +00:40 | 27 | +01:01 |
| Sechs Tabellenzellen | 28 | +01:08 | 42 | +01:43 |
| Gefaltete Klassifikation | 20 | +01:28 | 30 | +02:13 |
| Acht Bibliothekspaare | 108 | +03:16 | 162 | +04:55 |
| Aufrufgrößen und Startdiagnosen | 30 | +03:46 | 45 | +05:40 |
| Abholung | 3 | +03:49 | 5 | +05:45 |

Der Lauf enthält 26 Ergebniszellen. Normale Zellen haben drei rotierende Prozessrunden mit getrennten Erst- und Warmaufrufen. Die Kopfzeile hat zwei A/A-Runden, insgesamt vier echte Schreibläufe. Diese werden ausdrücklich als einzelne Streaming-Aufrufe ausgewiesen, ohne erfundene Warmbeobachtung. Die A/A-Grenze bleibt 10 Prozent. Normale fse-Aufrufe müssen mindestens zwei Sekunden dauern. Die Aufrufgrößen-Serie ist ausdrücklich von dieser Mindestdauer ausgenommen.

Die Datentabelle nutzt eine gemeinsame Eingabe mit 4.620.000 Absätzen. Die Bibliotheksbalken sind getrennte Paarvergleiche. Innerhalb eines Paares sind Textdatei und Digest identisch. POTION-Paare verwenden vorab dieselbe Zeichenkürzung, damit FastEmbed und Model2Vec dieselben Texte verarbeiten. Der Kürzungswert steht bei der Eingabeidentität. Native StaticEmbedding wird ohne nachträgliche fse-Kürzung verglichen. Eine fehlgeschlagene Ausgabeprüfung liefert keine publizierbare Zelle.

Die Klassifikation vergleicht gefaltetes fse, explizites fse encode plus denselben linearen Kopf und Model2Vec plus denselben Kopf. Der feintrainierte Model2Vec-MLP ist nicht dieser Vergleich. Es wird keine neue Genauigkeit berechnet. Historische Genauigkeitswerte bleiben getrennt. Der lineare Kopf wird mit festem Seed auf rohen, unnormalisierten Mitteln rekonstruiert.

## Geld und Grenzen

Compute: 0,96 USD/h. Disk: 150 GB zu 0,10 USD/GB/Monat, geteilt durch 720 Stunden. Zusammen 0,980833 USD/h.

- Erwartete 229 Minuten: ungefähr **3,74 USD**, zusätzlich tatsächliche Provisionierungszeit.
- Planungshorizont 345 Minuten: ungefähr **5,64 USD**.
- Mit zusätzlichen zehn Minuten Abholreserve: ungefähr **5,80 USD**.
- Geldbudget für die Zulassung: **7,30 USD**. Globale Zulassung berücksichtigt zugleich den 345-Minuten-Horizont und zehn Minuten Reserve.

Die Grenze ist nach Olivers Entscheidung weich. Begonnene Arbeit läuft zu Ende. Danach werden weitere Zellen einer erschöpften Stufe als `not_started` markiert. Eine verrauschte Ruhekontrolle beendet den Lauf nicht. Auch ein einzelner Zellfehler beendet spätere Stufen nicht. Ein außergewöhnlich langer Einzelaufruf kann den geplanten Betrag überschreiten. Es wird keine harte Rechnungsgarantie behauptet. Zwei GiB auf dem Datenträger bleiben für Ergebnis- und Abschlussdateien reserviert.

## SIGTERM und echte Hänger

Vor einem Signal liest der Koordinator `/speed-logs/measurement.pid` und prüft die zugehörige Kommandozeile. Das Signal gilt dem Python-Controller, nicht dem Shell-Supervisor und nicht einem zufälligen Kindprozess.

SIGTERM setzt nur die Aufnahmesperre. Die aktive Zelle läuft zu Ende. Danach starten weder neue Vorbereitungsschritte noch neue Ergebniszellen. Verbleibende Zellen erhalten `not_started`. DONE meldet den unvollständigen Abschluss mit Status 1. Der Steuerungs-Probelauf bestätigt dieses Verhalten.

Ein tatsächlich festhängender Kindaufruf wird dadurch nicht beendet. Falls Oliver eine erzwungene Beendigung freigibt, entscheidet der Koordinator über Prozessgruppe und Kindprozesse. Bei einer erzwungenen Beendigung können aktuelle Belege unvollständig sein. Bereits fsync-geschriebene Zeilen bleiben abholbar. Der Supervisor schreibt einen Fehlerabschluss, sofern er selbst noch läuft und noch kein DONE besteht. Es gibt keinen automatischen Neustart.

## Funktionsnachweis und genau ein Review

Der echte lokale Probelauf verwendete 24 Texte, darunter lange und Unicode-Eingaben. Zwanzig Adapter-/Worker-Kombinationen prüften Ausgaben, Faltung und explizite Köpfe. Die korrigierte Rust-Anbindung wurde tatsächlich kompiliert. Vier weitere kleine Streaming-Schreibläufe hatten identische kanonische Daten-Hashes und jeweils 24 Zeilen. Alle vier Python-Bibliotheken wurden getrennt auf ihren eigenen Paketnamen, Pfad und Code-Hash geprüft.

Der zusätzliche Steuerungs-Probelauf erzeugte `cells.jsonl`, `progress.json`, `manifest.json` und DONE wirklich. Seine 26 Aktionen sind simulierte Steuerungsschritte, keine Inferenzraten. Normalabschluss, SIGTERM, erschöpftes Stufenbudget und ausgeschöpftes Geldbudget bestanden. Jede Datei und der Manifest-Digest wurden geprüft. Der Inferenz-Probelauf und der Steuerungs-Probelauf werden nicht als Tempobelege veröffentlicht.

Die lokalen Umgebungen dienen nur der Funktion. Die Linux-Installation wurde nicht auf dem Mac ausgeführt. Die Linux-Abhängigkeitsauflösung wurde für beide Umgebungen separat geprüft. 65 beziehungsweise 35 Paketpins bleiben unverändert und sind nun zusätzlich mit Artefakt-Hashes gebunden. Die endgültige Installation prüft ihre Paketabhängigkeiten erneut nach dem fse-Bau.

Genau ein unabhängiges Review lief mit `gpt-6-astra`, Stufe `high`, auf `8ce4f68`. Es endete regulär. Seine Befunde wurden einzeln bearbeitet. Es wird kein zweites grünes Review erfunden. Die Antwort dokumentiert jeden Befund und den zusätzlichen korrigierten Paketnamenfehler. Image und Python-Patch sind fest, der tatsächliche C-Compiler und die nativen Binär-Hashes werden aufgezeichnet. APT verwendet die offiziellen Paketquellen des festgeschriebenen Images, keinen historischen Paketserver-Snapshot.

Lokale Nachweise, keine zusätzlichen Benchmark-Tests im Repository:

- inference: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/functional-smoke-reviewed/report.json`, SHA256 `dc4a61da72cb9219eed04b8a47ad26945fe86eb170767cbea87bf891e1da39e4`.
- stream_and_identity: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/closure-smoke/report.json`, SHA256 `63b3b1d9c4a540a665ea31589d3368e0814e5b02e5bfafc48941c84e34f7fd9b`.
- steering: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/control-dry-run-handoff/report.json`, SHA256 `62ea7d0a10aa53b6fefb533ae264cd957b3f01849844930c55138845ce8fc1ef`.
- review: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/final-review.last.md`, SHA256 `fd2708e488a35d3d8e955d50737aadb85f22be938d49dadad0261bc66f56b083`.
- response: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/review-response.md`, SHA256 `1d1af8821264dbfb20b0ff5ff4752421b489ef813f365d622ee85b699c89f6ee`.

Die Quellprüfungen Ruff, Formatprüfung und Guardrails bestehen. Es wurden keine neuen Benchmark-Testdateien angelegt. Nach dem einen Lauf folgen Auswertung, Archivierung und das bereits genehmigte Aufräumen. Bestehende Qualitätswerte werden nicht verändert. Kein PR wird gemergt.

## Ergänzung nach dem ersten Einrichtungsfehler

Der erste bezahlte Start lieferte keine Messwerte und kostete nach Koordinatorangabe ungefähr 0,10 USD. Seine fehlerhafte Indexstrategie ist korrigiert. Installation und Lock-Auflösung nutzen jetzt beide `unsafe-best-match`. Alle Pakete bleiben versionsgebunden. `--require-hashes` prüft ihre festgelegten Artefakt-Bytes. Das CPU-Torch ist zusätzlich durch die lokale Version `+cpu` gebunden. Ein gleichnamiges Paket von einem anderen Index kann deshalb keine anderen Bytes einschleusen. Die vorhandenen Paketprüfungen prüfen beide Umgebungen und anschließend fse nochmals.

Die komplette Einrichtung wurde jetzt tatsächlich im selben Image-Digest mit `--platform linux/amd64` unter Docker geprüft. Die lokale Prüfung hatte zwei CPUs und vier GiB RAM. Für die langsamere Emulation betrug nur ihr privates Setup-Zeitfenster 120 Minuten. Die Produktionsdatei behält fünf Minuten. Es wurde kein A/A- oder Zwei-Sekunden-Gate gelockert, weil keinerlei Messzelle aufgerufen wurde.

Die Prüfung installierte beide Umgebungen, baute fse und den Rust-Worker, lud beide Modelle mit allen Tabellen-Datentypen, bereitete den vollständigen Korpus vor und rekonstruierte den linearen Kopf. Alle fünf Vorbereitungsbelege melden `complete`. Die Importe von fse, Model2Vec, FastEmbed, sentence-transformers und CPU-Torch bestanden. Die gesicherten Belege wurden vollständig gegen ihr Manifest geprüft.

Zusätzlich wurde ein früher Bootstrap-Fehler absichtlich ausgelöst. Der Container blieb mit DONE und laufendem SSH erreichbar. Nach einem erzwungenen Neustart blieb DONE unverändert und die Clone-Anzahl bei eins. Der Neustart startete keine Einrichtung und keine Messung.

Kein zusätzliches Review ist erforderlich: Seit dem geprüften Messcode änderte sich in Python ausschließlich die Indexstrategie innerhalb von `setup`. Die weiteren Änderungen betreffen den Bootstrap in dieser Übergabe. Die Messfunktionen sind bytegleich.

Linux-Nachweis: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/linux-setup-proof/complete-results/complete-setup.json`, SHA256 `5a167f0f10287b4259580554a3f9f116531c9a08c852a035a2e3b965a9d834d7`.

Neustart-Nachweis: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/linux-setup-proof/bootstrap-restart-proof.json`, SHA256 `aa27830726acde19a0d6ee3292f9c8dce04b602e5755fbeb57d1fb4c47918832`.

Konsolidierte Prüfung: `/Users/oliverborchers/.herdr/worktrees/Fast_Sentence_Embeddings/dr151-compare/.review/speed-implementation/final-verification-v2.json`. Diese lokalen Prüfprogramme und Rohbelege werden nicht als Werkzeugtests ins Repository aufgenommen.
