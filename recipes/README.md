# fse recipes

Run fse in the process that already has your text. These notebooks show offline model preparation and local application patterns.
They add no dependencies to the core package. They are run locally, not in CI.

| Notebook | What it covers |
|---|---|
| [SIF and uSIF](01-sif-usif.ipynb) | Learn on a corpus, fold finished tables, compare GloVe and POTION on STS-B, check the historical GloVe results |
| [Linear heads](02-linear-heads.ipynb) | Train classification and regression heads in scikit-learn, import and save folded models |
| [Files with resume](03-files-with-resume.ipynb) | Bound memory while reading a line file, interrupt and resume without duplicate rows |
| [Kafka and Qdrant](04-kafka-and-qdrant.ipynb) | Fake queue polls, durable offsets, replay reconciliation and local Qdrant upserts |
| [Folder listener](05-folder-listener.ipynb) | Poll local or SMB/NFS inputs, ignore partial writes and resume from a durable prefix |
| [Integrations](06-integrations.ipynb) | DSPy, BERTopic, txtai, Chroma, LanceDB, Qdrant and Polars |
| [Local compute and HTTP](07-local-compute-and-http.ipynb) | Measure compute, localhost request time and 300-dimensional vector sizes |

## Run

Use standard Python 3.13 on macOS or Linux. From the repository root:

```sh
UV_PYTHON=3.13 make install
uv pip install --python .venv/bin/python -r recipes/requirements.txt
uv run --no-sync jupyter execute --inplace --timeout=600 recipes/02-linear-heads.ipynb
```

Replace the filename to execute another notebook. `--no-sync` preserves the recipe extras.
Use the same environment as the notebook kernel. The outputs committed here record the local execution environment.
Do not run these notebooks in CI. The integration packages are intentionally optional and are pinned separately.

All generated data and downloaded files belong to a temporary directory. The SIF and timing notebooks can read exact-revision files from an existing Hub cache.
The SIF notebook downloads GloVe, POTION and NLTK data if they are unavailable locally. The other application examples use a small hand-built table.
No notebook requires credentials, calls a paid service or uploads a model. The upload example is text only.

Run the timing notebook on a quiet machine. It refuses a one-minute load average of 5 or more before or after measurement.
On a shared machine, hold its benchmark lock and pause competing heavy work. Record the machine and date with every measurement.
The HTTP server exists only during the measurement cell and closes before the cell returns. There is no HTTP service deployment recipe.

## Resume assumptions

Use one writer and immutable inputs. A ShardSink reports `first_row` for the existing store and `durable_rows` for the current run.
Only acknowledge the durable prefix. A source checkpoint can lag vector publication, so reconcile it against the store before replay.
The file examples keep their store on a supported local filesystem. Polling network inputs does not extend the store's durability guarantee to SMB/NFS.

The folder listener requires contiguous, monotonically numbered final filenames. Producers close temporary files before renaming them.
A production ingestion manifest is needed for arbitrary names, changed files or multiple rows per file.
