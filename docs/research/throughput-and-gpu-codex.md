**PR 6b ist der richtige nächste Schritt.** Langfristig würde ich die Ausführungsgrenze auf „ein Task erzeugt gespeicherte Zeilen“ erweitern, mit einem direkt ausführbaren CPU-Pfad und einer asynchronen GPU-Pipeline. Vollständige Textfusion und ein nativer Executor sollten danach gegen diese Basis antreten.

Gelesen: Architektur, die genannten Implementierungen und MLX-Notizen, Stand `dc97085`. Keine Dateien geändert, keine Tests oder Benchmarks ausgeführt. Die folgenden Effekte sind begründete Erwartungen, keine neuen Messwerte. Kurze Codeangaben beziehen sich auf `src/fse/`.

Die bisherigen Zahlen beweisen noch keinen dominanten Engpass. Insbesondere lassen sich 42 M GPU-Tokens/s nicht direkt mit CPU-Zeilen/s vergleichen: Satzlänge, bekannte Wörter, FastText-Beiträge und Nachverarbeitung unterscheiden sich. Der GPU-Prototyp zeigt Machbarkeit, noch keinen Vorteil gegenüber dem vollständigen CPU-Pfad.

**Die erste Fusion sollte eine äußere Schleife über Zeilen sein.**

Heute erzeugt [`_backends/_cpu.py:85`](src/fse/_backends/_cpu.py:85) bei aktiven Steps zuerst gepoolte Zeilen. `_native_steps.pxi:47` kopiert sie, verarbeitet die Steps einzeln und kopiert das Ergebnis zurück. Anschließend folgen Speicherung und inverse Normen als weitere Aufrufe in `_runtime/_pipeline.py:185`.

Empfehlung: pro Zeile Pooling, Steps, Quantisierung und inverse Normen unmittelbar hintereinander ausführen, mit wiederverwendetem Scratch im Cache. „Ein Loop pro Zeile“ bedeutet dabei weiterhin mehrere notwendige Durchläufe über ihre Dimension:

- `remove_components` braucht Koeffizienten aus der unveränderten Eingangszeile.
- `normalize` braucht zuerst die Norm.
- int8 braucht zuerst `absmax`, danach Codes.
- Die Store-Norm muss aus den **gespeicherten** Werten entstehen. Die Normalisierungsnorm wiederzuverwenden oder die inverse Norm auf 1 zu setzen verletzt den Vertrag.

Beim reinen Normalize-Pfad entfallen damit zwei vollständige float32-Kopien, logisch `16 × Dimension` Bytes Lese-/Schreibverkehr pro Zeile. Dazu kommen weniger Allokationen, Prüfungen und GIL-Wechsel. Das ist nicht automatisch dieselbe Einsparung im DRAM, weil Teile im Cache liegen.

**Kosten/Risiko:** mittel. Rundungspunkte, Tokenreihenfolge, `dot16`, signed zero und FP-Umgebung bleiben unverändert. Zusätzlich muss die bisherige **Fehlerpriorität innerhalb eines Tasks** erhalten bleiben: Heute kann beispielsweise ein Poolingfehler einer späteren Zeile vor einem Nachverarbeitungsfehler einer früheren Zeile auftreten. Ein sofortiger Abbruch beim ersten Fehler der fusionierten Zeilenschleife wäre eine Verhaltensänderung.

**Messung:** Schritte einzeln ablatieren; Allokationen, kopierte Bytes, CPU-Zeit und Enddurchsatz für ohne Steps, Normalize und ungefaltet vollständig vergleichen. Bitvergleich einschließlich Fehlerfällen. Folding existiert bereits (`_encoder/_fit.py:226`); große Projektionsgewinne sind deshalb nicht für jeden Encoder relevant.

**Eine zusätzliche, eigenständige Optimierung betrifft die Store-Normen.**

[`_native_kernel_search.h:28`](src/fse/_native_src/kernel/_native_kernel_search.h:28) summiert Quadrate mit einer einzigen float64-Abhängigkeitskette. Fusion beseitigt den erneuten Datenzugriff, aber nicht diese Latenzkette.

- **float32/float16:** Zwei bis vier Zeilen verschachtelt bearbeiten. Jede Zeile behält ihre Summationsreihenfolge, während unabhängige Ketten parallel laufen können. Aufwand mittel, Risiko Registerdruck und schlechtere Lokalität; bitidentisch möglich.
- **int8:** Quadrate ganzzahlig reduzieren. Für gültige Codes und `MAX_DIM = 2**30` gilt `Σ code² ≤ 127² × 2**30 < 2**44`. Alle Zwischensummen sind auch in float64 exakt. Eine ausreichend breite Integer-Reduktion kann deshalb dieselbe Summe liefern; danach unverändert float64-`sqrt`, Division und float32-Rundung. Aufwand klein bis mittel, Hauptgefahr sind überlaufende SIMD-Teilsummen.

Das ist auch für MLX interessant: int8-Quadratsummen können auf dem Gerät entstehen, nur die finale double-Wurzel und Division bleiben zunächst auf der CPU. Für float32/float16 gibt es diese einfache Umordnungsgarantie nicht.

**Messung:** Normkernel isoliert und Store-Enddurchsatz mit/ohne Normberechnung; lange Dimensionen, Extremwerte und Nullzeilen bitweise vergleichen.

**Python-Arbeit lässt sich zunächst ohne neuen Executor stark reduzieren.**

Der heutige Ablauf pro Task ist:

| Stelle | Arbeit heute | Konkrete Verlagerung |
|---|---|---|
| `_runtime/_executor.py:101` | Iterator, Fehlerlock, Reservierung, `Future`, Queue und Deque | Erst mehrere Deskriptoren gemeinsam einreichen; später nativer Executor |
| `_runtime/_pipeline.py:155` | Python-Closures für Reserve, Work und Commit | Kompakter Task-/Zieldeskriptor |
| `_runtime/_pipeline.py:177` | Thread-local Lookup, Target-Aufteilung, Scratch-Slice, Backend-Lock und Methodenaufrufe | Ein vorbereiteter nativer Aufruf für den CPU-Datenpfad |
| `_text.py:317`, `_text_file.py:80` | Source-Dispatch, Array-Header, Chunkprüfung; bei Dateien mehrere Statusprüfungen | Native Textspanne übergeben; redundante Prüfungen bündeln |
| `_native_text.pxi:276`, `:328` | Offsetkopie, neue Token-/Offsetarrays | Private Buffer mit klarer Besitzdauer direkt weiterreichen |
| `_contract.py:330`, `_native_kernel.pxi:192` | Batchprüfung, danach erneut Offsets, OOV-Maske, `tokens[~unknown]` und Maximum | Interner, ausschließlich nativ erzeugter Batchpfad ohne erneute NumPy-Scans |
| `_backends/_base.py:195`, `_native_kernel.pxi:243` | Wiederholte Shape-/Overlap-Prüfungen | Einmalige Grenzprüfung plus unveränderliche native Deskriptoren |
| `_native_steps.pxi:43` | Step-Dispatch, Prüfungen, Kopien, temporäre Arrays | Steps in `prepare` in einen Ausführungsplan übersetzen |
| `_native_search.pxi:17` | Neues Normarray plus Scratch, anschließend Kopie ins Target | Direkt ins Normtarget schreiben |
| `_store/_sink.py:126` | Targetvergleich, Shardsplitting, gegebenenfalls Kopien, zwei Queue-Submits | Fertige zusammenhängende Commits bündeln und Targets wiederverwenden |

Die Workspace-Puffer werden bereits wiederverwendet. Allerdings kopiert `_native_buffer.pxi:52` ihren Inhalt für jeden Batch erneut in NumPy-Arrays. Genau diese Zwischenprodukte sind vermeidbar.

Wichtig: **Prüfungen nicht einfach abschalten.** Externe Arrays und öffentliche/native Testzugänge bleiben streng geprüft. Der schnelle Pfad benötigt privat besessene Daten, garantierte Lebensdauer und geprüfte Grenzen. Ein frei setzbares `trusted=True` wäre keine belastbare Grenze.

Der erwartete Gewinn betrifft Python-/GIL-Zeit, temporäre Speicherbelegung und wiederholte Scans. Aufwand mittel, Bitidentität bleibt erhalten. Messen: native Aufrufe und Allokationen pro Task, GIL-Zeit, Durchsatz über Taskgröße und Workerzahl.

Ein **nativer Executor** wäre erst die nächste Stufe: C-Taskqueue, Workerzustand und Completion-Ring; Python verarbeitet zusammenhängende fertige Präfixe. Aufwand und Risiko sind hoch wegen Abbruch, Signalen, geordneten Fehlern und beliebigen Python-Sinks. Er lohnt sich erst, wenn nach dem einzelnen nativen Taskaufruf Scheduling messbar dominiert. Python-Iteratoren eines `StreamSource` bleiben ohnehin an den aufrufenden Thread gebunden.

**Textfusion sollte in zwei Varianten erprobt werden.**

[`_native_text.pxi:193`](src/fse/_native_src/text/_native_text.pxi:193) materialisiert zunächst Token-Spannen und danach IDs. `_native_kernel_mean.h:212` konsumiert die IDs.

1. **Direkte Fusion:** Sobald ein endgültiges Token feststeht, Lookup und Akkumulation ausführen. Keine satz- oder taskweite ID-Materialisierung. Das spart Speicherverkehr und Dispatch, verkettet aber Tokenisierung, Hashlookup und Vektorgather stärker.
2. **Kleine native Zwischenpuffer:** Einige Tokens oder Sätze voraus auflösen, Vektoren vorladen und danach poolen. Das spart weiterhin Python-Batches und große Zwischenarrays, erhält aber Memory-Level-Parallelismus.

Ich würde Variante 2 zuerst erproben. Der vorhandene Kernel prefetcht bereits acht Tokens voraus, auch über Satzgrenzen (`_native_kernel_mean.h:217`, `_native_kernel_base.h:58`). Vollständige Fusion kann gerade bei kalter großer Worttabelle langsamer werden.

FastText braucht zudem die Zahl `m` der verwendeten Ngrams, bevor ihr Faktor feststeht. Ein kleiner Ngram-Puffer pro Wort bleibt sinnvoll. Einen fertig gemittelten OOV-Vektor zwischenzuspeichern verändert hingegen die FMA-Reihenfolge und ist nicht bitidentisch.

**Kosten/Risiko:** mittel bis hoch, insbesondere Tokenizerintegration und Cachekonflikte. Beide Varianten können bitidentisch sein. **Messung:** IDs separat, direkter Pfad und Ringpuffer auf demselben Korpus; warme/kalte Tabelle, ASCII/Unicode, kurze/lange Sätze und FastText. Relevant sind Cachemisses, Branch-Misses und Tokens/s einschließlich Tokenisierung.

Das Grundprinzip „Gather direkt in Reduktion, keine Matrix aller Tokenvektoren“ ist etablierte Praxis, etwa bei [PyTorch EmbeddingBag](https://docs.pytorch.org/docs/2.9/generated/torch.nn.EmbeddingBag.html). Diesen Vorteil hat fse heute bereits; neu wäre die Fusion mit dem Textfrontend.

**Die gemeinsame Backend-Grenze sollte Besitz und Fertigstellung ausdrücken.**

Ausgangspunkt ist das synchrone Hostarray-Protokoll in [`_backends/_base.py:110`](src/fse/_backends/_base.py:110). Mein Zielbild:

```text
prepare(EncoderSpec, Numerikprofil) -> Prepared
submit(TaskInput, OutputSpec, BufferLease) -> Completion
Completion -> gespeicherte Zeilen, Scales, Normen, Fehlerstatus
```

`TaskInput` erlaubt Textspannen oder kompakte Tokenbatches. `OutputSpec` beschreibt dtype und benötigte Normen. Der Backend-Ausführungsplan darf CPU-Frontend und Gerät kombinieren. Dateipfade, Hashing und Veröffentlichung bleiben beim Sink.

Für die CPU wird `submit` im Worker synchron durch einen nativen Aufruf erfüllt. Für MLX/CUDA gibt es mehrere CPU-Produzenten und einen Geräte-Owner. **Tokenizerworker warten nach der Übergabe nicht auf das Gerät**, sondern dürfen mit einem freien Buffer weiterarbeiten.

Das heutige `concurrent=False` genügt dafür nicht: `_runtime/_pipeline.py:185` hält den Lock über die ganze Backendarbeit. Mit blockierendem `rows()` wäre höchstens ein Auftrag beim Geräte-Owner, wodurch taskübergreifendes Batching verhindert würde. „Mehrere Einreichungen erlaubt“ und „mehrere Gerätethreads erlaubt“ müssen getrennte Eigenschaften werden.

Der aufrufende Thread reserviert weiterhin in Reihenfolge und committet nur das fertiggestellte Präfix. Ausführung darf ungeordnet fertig werden. Kein Task hinter dem ersten Fehler wird committet. Fertige Präfixe gemeinsam abzuräumen reduziert Synchronisation, ohne Commitreihenfolge aufzugeben. Beim Python-Stream sollte der aufrufende Thread primär produzieren und committen; ihn zusätzlich mit langen Kernels zu beschäftigen kann die Pipeline aushungern.

Aufwand hoch, aber diese Grenze vermeidet spätere Sonderarchitekturen für CUDA. Bitidentität hängt vom gewählten Numerikprofil ab, nicht von der Queue. Messen: Pipelinebelegung, Wartezeit pro Queue, CPU-/GPU-Überlappung und Zeit bis zum letzten Commit.

**Beim Speicherlayout würde ich gezielt experimentieren, statt die Wortmatrix grundsätzlich umzubauen.**

| Vorschlag und Code | Entfernte Grenze / erwarteter Effekt | Kosten, Risiko und Bitidentität | Messung |
|---|---|---|---|
| **Row-major beibehalten**, `_native_kernel.pxi:179` | Aufeinanderfolgende Dimensionen eines Wortes bleiben zusammenhängend, passend für CPU-SIMD und GPU-Lanes | Kein Umbau nötig. Transponieren verschlechtert diese Zugriffe | Gatherdurchsatz bei realer Zugriffshäufigkeit |
| **Privater gepaddeter Row-Stride**, `_native_kernel_base.h:67` | Weniger geteilte Cachelines, günstigere SIMD-Tails | Mittel; zusätzliche Modellkopie und mehr Bytes. Padding nie in `dot16` einrechnen. Bitidentisch | Dimensionen etwa 100/300/768, mehrere Alignments; inklusive Prepare/RSS |
| **Prefetchdistanz und Umfang abstimmen**, `_native_kernel_base.h:87` | Verbirgt Gatherlatenz | Klein; zu viel Prefetch verdrängt nützliche Daten. Schon vorhanden, daher begrenztes Zusatzpotenzial. Bitidentisch | Distanz 0/4/8/16, ganze Zeile gegen Teil-Prefetch |
| **float16-/int8-Wortvektoren nutzen**, `_native_kernel_mean.h:112`, `:136` | Halbierter/geviertelter Vektor-Footprint gegenüber float32, besserer Cachebesatz | Bereits unterstützt. Bitidentisch zum jeweiligen quantisierten Modell, nicht zum ursprünglichen float32-Modell | Warm/kalt, Vokabulargröße, Durchsatz plus STS/MTEB |
| **Dimensionstiles in Registern halten**, `_native_kernel_mean.h:95` | Weniger Akkumulator-Loads/-Stores pro Token | Mittel; IDs werden mehrfach gelesen, Gatherzugriffe verändern sich. Tokenreihenfolge je Element bleibt erhalten | Register-Spills, L1-Verkehr, lange gegen kurze Sätze |
| **Kleine Satzgruppen verschachteln** | Verbirgt Lookup-/Gatherlatenz durch unabhängige Arbeit | Mittel; größerer Working Set. Bitidentisch bei unveränderter Reihenfolge innerhalb jeder Zeile | Gruppen 1/2/4/8, Cachemisses und Durchsatz |
| **Vokabulartabelle mit kompakten Kontrollbytes**, `_native_vocab.pxi:77`, `:93` | Weniger Probe-Traffic und Branch-Misses bei großen Tabellen/OOV | Mittel bis hoch; heutige halbvolle 32-Byte-Slots mit Inline-Keys sind bereits gut. Bitidentisch bei vollständigem Bytevergleich | Probeanzahl, Bytes/Lookup, OOV-Anteil, Tabellen-RSS |
| **Huge Pages nur bei nachgewiesenen TLB-Misses** | Weniger Adressübersetzung bei großen Tabellen | Plattformabhängig, zusätzlicher Speicher-/Page-Fault-Aufwand. Bitidentisch | DTLB-Misses, kalter Start und Dauerbetrieb |

Für die Vokabulartabelle liefert [Abseil Swiss Tables](https://abseil.io/about/design/swisstables) die passende Vorarbeit: SIMD prüft kompakte Hash-Metadaten vor dem vollständigen Schlüsselvergleich. Das rechtfertigt einen Vergleich, keinen pauschalen Ersatz der bestehenden Tabelle. Für Huge Pages beschreibt die [Linux-Dokumentation](https://docs.kernel.org/admin-guide/mm/transhuge.html) sowohl TLB-Vorteile als auch Speicher- und Latenzkosten; das ist keine unmittelbar übertragbare M1-Optimierung.

Task-Cacheblocking sollte zunächst **Arbeitsmenge begrenzen**, nicht Tokens nach Wort-ID sortieren. Sortierung innerhalb eines Satzes verletzt den Vertrag. Ein Cache häufig wiederholter Token-Lookups oder Ngram-ID-Listen bleibt dagegen bitidentisch; sein Nutzen hängt stark von Wiederholung und OOV-Anteil ab.

**Eine Toleranzstufe lohnt sich vor allem dort, wo sie andere Rechenverfahren ermöglicht.**

| Bereich | Potenzial durch Lockerung |
|---|---|
| Viele Suchqueries und Scores | Sehr hoch, bereits durch BLAS demonstriert; `_store/_search.py:157` |
| Ungefaltete Projektion/Komponentenentfernung | Hoch bei großen Batches: Matrixoperationen statt vieler fester `dot16`-Reduktionen |
| Lange Sätze oder viele Ngram-Beiträge | Mittel bis hoch: mehrere Tokenakkumulatoren und parallele Reduktion lösen lange FMA-Ketten |
| Store-Normen auf MLX | Hoch als Architekturvereinfachung: GPU-Reduktion ohne sequenzielle float64-Vorgabe |
| Kurze Sätze mit kalter Worttabelle | Eher begrenzt: Gather, Lookup und Speicherlatenz bleiben |
| Python, Tokenisierung, Queues, Dateisynchronisation | Praktisch kein direkter Nutzen |

Ich würde `exact` beibehalten und ein **versioniertes Numerikprofil** ergänzen. Es gehört in Encoder-/Ausgabeidentität und Store-Kompatibilitätsprüfung, etwa an `_encoder/_format.py:130` und `_store/_layout.py:70`. Bestehende Operationsnamen dürfen keine neue Bedeutung bekommen. Backend, Bibliotheksversion und tatsächlich verwendeter Ausführungspfad gehören zusätzlich zur Provenienz.

„Tolerance“ braucht definierte Zusagen für Unterlauf, Überlauf, signed zero, Akkumulationspräzision und Reproduzierbarkeit. Ein einzelnes `allclose` genügt nicht: Nach nahezu vollständiger Komponentenentfernung kann Normalize kleinste Abweichungen in stark verschiedene Richtungen verwandeln. Prüfen sollte man absolute/relative Fehler, Kosinusabweichung, Top-k-Recall sowie STS/MTEB, einschließlich gezielter Auslöschung.

**Für MLX ist die Numerikentscheidung dringlicher als für CUDA.** Die Notizen zeigen FTZ auf dem M1 auch bei Zwischenresultaten. „Eingaben sind normal“ garantiert deshalb keine Bitidentität. Ein CPU-Fallback muss auch gefährliche Zwischenoperationen konservativ erkennen; ein Blick auf fertige Nullwerte reicht nicht. Diese Erkennung ist bislang unbewiesen. Ein Canary ist nützlich, ersetzt aber weder Beweis noch Konformitätstests.

CUDA erlaubt dagegen explizit korrekt gerundete Operationen mit erhaltenen Subnormals; siehe [NVIDIAs Floating-Point-Dokumentation](https://docs.nvidia.com/cuda/floating-point/index.html). Damit ist ein vollständig strikter CUDA-Pfad grundsätzlich einfacher, bei weiterhin expliziter FMA- und Reduktionsreihenfolge.

**Die GPU sollte kompakte Beiträge konsumieren und fertige Zeilen produzieren.**

Die Pipeline sollte so aussehen:

```text
CPU: Text → Tokenisierung/Lookup → IDs, Offsets, Ngram-Metadaten
                                      ↓ begrenzte Bufferübergabe
GPU: residente Worttabelle → Gather/Pool → Steps → Codes
                                      ↓ Fertigstellung
CPU: nötige exakte Normen → geordneter Commit → Hash/Write/Publish
```

Wortvektoren, Gewichte, Scales und Stepmatrizen bleiben nach `prepare` resident. Pro Task werden keine vollständigen Tokenvektoren übertragen oder als `[Tokens, Dimension]` materialisiert.

Für striktes Pooling bearbeiten benachbarte Lanes benachbarte Dimensionen; jede Dimension durchläuft Tokens in Originalreihenfolge. `dot16` erhält seinen expliziten Baum. Zunächst Pooling, Normalize und Store fusionieren. Eine große ungefaltete Projektion darf ein eigener Kernel bleiben, falls vollständige Fusion Registerdruck und Occupancy verschlechtert.

Die Notizen messen ungefähr 205 µs pro kleinem Kernel einschließlich Synchronisation. Das entspricht bei 3,4 M CPU-Zeilen/s rechnerisch der Zeit für rund 700 Zeilen. Für weniger als 5 % dieses seriellen Zeitbudgets wären etwa 14.000 Zeilen pro Synchronisation nötig. Das ist keine Hardwaregrenze, zeigt aber: **CPU-Taskgröße und GPU-Submissiongröße müssen unabhängig sein.**

Ein Geräte-Owner sollte mehrere Tasks bündeln, begrenzt durch Bytes, Beitragszahl und maximale Wartezeit. Kernelobjekte einmal erstellen; das empfiehlt auch die [MLX-Kerneldokumentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html). Streams müssen im besitzenden Thread entstehen; [MLX dokumentiert diese Threadbindung ausdrücklich](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.new_stream.html). `async_eval` kann Überlappung ermöglichen, ist aber [als experimentell dokumentiert](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.async_eval.html).

Für die Speicherübergabe:

- **MLX:** Die Notizen zeigen kopierfreie Eingaben bei passendem Page-Alignment. Alignment zur Laufzeit bestimmen, Buffer bis zur Fertigstellung unverändert halten. Ergänzend dokumentiert MLX auch [kopierfreie NumPy-Ausgabeviews](https://ml-explore.github.io/mlx/build/html/usage/numpy.html).
- **Sink-Grenze:** Heute verlangt `_store/_sink.py:135` genau das reservierte Target. Für MLX-erzeugte Ausgaben braucht es entweder eine explizite Kopie oder einen optionalen Übergabepfad für fertig erzeugte Buffer samt Besitzer. Zero-copy entsteht nicht allein durch Unified Memory.
- **CUDA:** Wiederverwendete gepinnte Hostbuffer, asynchrone Transfers und Events; Copy-/Compute-Überlappung nach Gerätefähigkeit. Das entspricht [NVIDIAs Empfehlungen](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation).

**Kosten/Risiko:** hoch, besonders Bufferlebensdauer, Backpressure und asynchrone Fehler. **Messung:** CPU-Frontend, Transfer, Encoding/Launch, Gerätelaufzeit, Synchronisation, Normen und Ausgabe getrennt; anschließend vollständig überlappt bis zum letzten Flush. Auf Apple teilen CPU und GPU Speicherbandbreite, daher CPU-Produzenten mitmessen und begrenzen.

**Weitere konkrete Grenzen verdienen kleinere, getrennte Änderungen.**

| Stelle | Vorschlag, Wirkung und Nachweis |
|---|---|
| `_native_scan.pxi:158`, `_text.py:281` | Der Scan zählt Whitespace-Tokens, deren Ergebnis die Runtime verwirft. Internen Scanmodus nur für Validierung und Grenzen anbieten. Weniger Zeichenarbeit, kleiner bis mittlerer Aufwand, bitidentisch. Scanzeit und Enddurchsatz messen. |
| `_native_hash.pxi:35` | FastText hasht überlappende Präfixe wiederholt. FNV-Zustand beim Verlängern eines Ngrams fortführen. Weniger Integerarbeit, mittlerer Aufwand; Byte-Signierung, Reihenfolge und Cap exakt erhalten. OOV-lastige Korpora messen. |
| `_runtime/_pipeline.py:132`, `_native_text.pxi:209` | Das Fenster bilanziert hauptsächlich Zeilen, nicht vollständig Tokens, Ngrams und Workspace-Kapazitäten. Beitrag-/Bytebudget ergänzen. Eine riesige Einzelzeile wird durch das Outputlimit nicht zeitlich begrenzt. Mittlerer Aufwand, bitidentisch; RSS und lange Tasks messen. |
| `_store/_sink.py:119`, `:143` | Tasks an Shardgrenzen teilen, damit Zwischenbuffer und Kopien entfallen. Kleiner bis mittlerer Aufwand, bitidentisch; grenzüberschreitende Tasks und kopierte Bytes zählen. |
| `_store/_shard.py:84`, `_store/_writer.py:28` | Targetpool mit Freigabe erst nach **Hash und Write**; Runtime- und Writerbudgets gemeinsam betrachten. Weniger Allokationen, mittleres Ownership-Risiko; RSS und Queuewartezeiten messen. |
| `_runtime/_executor.py:92`, `_encoder/_batches.py:90` | Pools/Workspaces werden pro `infer` neu angelegt. Explizite Run-Session für viele Batches erwägen. Mittlerer Aufwand und Lifecycle-Risiko, bitidentisch; kleine wiederholte Batches messen. |
| `_runtime/_count.py:53`, `_runtime/_sinks.py:234` | uSIF zählt über vollständigen Lookup; Gramprodukte laufen beim Commit. Reinen Tokenzählpfad und später feste Gramblöcke in Workern ausführen. Mittlerer Aufwand; Gramblockgrenzen und Summenreihenfolge erhalten. Fit separat messen. |
| `_native_search.pxi:67`, `_store/_search.py:188` | Exakte Queries werden je Suchblock erneut geprüft und normalisiert. Einmal vorbereiten und wiederverwenden. Kleiner Aufwand, bitidentisch; viele kleine Suchblöcke messen. |

Beim Store sind Hash-, Schreib- und Publisherthread **bereits vorhanden**, siehe [`_store/_sink.py:96`](src/fse/_store/_sink.py:96). Zusätzliche Threads sind deshalb keine naheliegende Standardlösung. Erst Queuefüllstände zeigen, welcher Abschnitt bremst.

`durable=False` muss alle betreffenden Syncs erreichen: `_store/_shard.py:149`, `:165`, `:189`, Store-Erstellung und Directory-Publish in `_container/_npy_inplace.py:116`. Schreiben, Checksummen und exklusive Veröffentlichung bleiben bestehen. Numerische Bits bleiben identisch; die Persistenzzusage ändert sich. Veröffentlichte Zeilen dürfen dann nicht als gegen Stromausfall dauerhaft bestätigt werden. Aufwand klein bis mittel. Messen bis zum letzten Flush und über genügend Shards, damit der Page Cache nicht nur einen kurzfristigen Gewinn vortäuscht.

Für die nächste Profilrunde würde ich neben den fünf Stufen auch Source-Aufbau, Scan, Python-Prüfungen, Scheduler, Normen, Hashing, Schreiben und Publish erfassen. Pro Abschnitt CPU-Zeit, Walltime, Allokationen und Wartedauer unterscheiden. Die vorhandenen Vergleiche sind nicht vollständig gleichartig: `benchmarks/bench_runtime.py:66` variiert Normalize, während `benchmarks/bench_store.py:76` mit `normalize=False` arbeitet. Dazu gehören reale Unicode-/FastText-Korpora und unterschiedliche Vokabulargrößen.

**Zuerst zu entscheiden ist die Numerikzusage für MLX:** Soll die erste Version vollständig `exact` einschließlich Subnormals erfüllen, oder darf sie mit einem ausdrücklich versionierten Toleranzprofil starten? Meine Empfehlung ist Letzteres, während PR 6b strikt bitidentisch bleibt. Das vermeidet, dass eine noch unbewiesene FTZ-Fallbacktechnik die gesamte Gerätearchitektur bestimmt.

Meine vorläufige Rangfolge nach erwartetem Gewinn zu Aufwand:

1. **Stages 3 bis 5 plus Normen fusionieren.** Bestätigung: weniger vollständige Zeilendurchläufe und Allokationen, höherer Enddurchsatz bei unveränderten Bits.
2. **Ein nativer CPU-Datenpfad pro Task mit privaten Buffern.** Bestätigung: Batchkopien und wiederholte NumPy-Prüfungen verschwinden; GIL-Anteil und Kosten kleiner Tasks sinken.
3. **`durable=False` vollständig umsetzen.** Bestätigung: weniger Publish-Wartezeit und höherer Durchsatz über mehrere Shards bis zum Abschluss, bei klar getrenntem Fortschritt und Dauerhaftigkeit.
4. **Inverse Normen nach dtype optimieren.** Bestätigung: Integer-Reduktion für int8 und verschachtelte Zeilen für float32/float16 verkürzen die Normzeit und verbessern den Store messbar.
5. **Native Mikroblöcke für Lookup, Prefetch und Pooling vergleichen.** Bestätigung: gegenüber dem optimierten ID-Pfad weniger Cache-/Latenzstalls auf realen großen Vokabularen. Vollständig ID-freie Fusion nur übernehmen, wenn sie diesen Vergleich gewinnt.