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
recipe_tmp=$(mktemp -d)
IPYTHONDIR="$recipe_tmp/ipython" JUPYTER_RUNTIME_DIR="$recipe_tmp/runtime" \
  JUPYTER_CONFIG_DIR="$recipe_tmp/config" JUPYTER_DATA_DIR="$recipe_tmp/data" \
  XDG_CACHE_HOME="$recipe_tmp/cache" \
  uv run --no-sync jupyter execute --inplace --timeout=600 recipes/02-linear-heads.ipynb
```

Replace the filename to execute another notebook. `--no-sync` preserves the recipe extras.
Use the same environment as the notebook kernel. The temporary environment paths also contain kernel history and runtime files created before the first cell. The outputs committed here record the local execution environment.
Do not run these notebooks in CI. The integration packages are intentionally optional and are pinned separately.

All generated data and downloaded files belong to a temporary directory. The SIF and timing notebooks can read exact-revision files from an existing Hub cache.
The SIF notebook downloads GloVe, POTION and NLTK data if they are unavailable locally. The head notebook downloads SST-5 and uses GloVe. Other application examples use a small hand-built table.
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

## Recorded quality checks

Executed on 2026-11-01 on Apple M1 Pro, macOS 27.0.1, Python 3.13.16, with the pinned recipe packages.
The STS-B comparison fits weights and components on train texts only. These are Pearson correlations times 100 on 1,379 test pairs.

| Table | Mean | SIF-10 | uSIF-5 |
|---|---:|---:|---:|
| GloVe-300 | 45.09 | 70.67 | 66.28 |
| POTION-8M | 77.26 | 75.71 | 76.91 |

On this split, additional uSIF helps GloVe and slightly reduces the POTION score. This does not establish a general ranking.
The largest absolute component error between folded and postprocessed normalized rows was 1.04e-6.

The separate historical GloVe diagnostic fits on test texts to follow the 1.x notebook. It is not a held-out evaluation.
It gives mean 44.4641, SIF-10 71.3792 and uSIF 68.8190 against README values 44.46, 71.43 and 68.82.
Mean and uSIF round to the published values. SIF differs by 0.0508 points. This recipe does not claim exact reproduction of SIF.

SST-5 uses 8,544 training and 2,210 test sentences. The GloVe head gives five-class accuracy 0.4118.
Its test labels agree with scikit-learn on all 2,210 rows. The ordinal regression MAE is 0.8967 label units,
against 1.1411 for the training-mean constant. These figures describe this example, not a tuned sentiment model.
