<div align="center">

# fse recipes

Start with text. Build something you can search, classify, or save.

</div>

**New to fse? Start with [fse 101](00-fse-101.ipynb).**
Each recipe explains the idea, shows the flow, and runs a complete example with real models and texts.
A vector is a list of numbers that represents a text.

| Picture | Recipe and promise | Reading time* |
|---|---|---:|
| <img src="media/00-fse-101-icon.svg" width="46" alt="Recipe number"> | [**fse 101**](00-fse-101.ipynb). Turn sentences into vectors, find similar text, and save your results. | 15 min |
| <img src="media/01-sif-usif-icon.svg" width="46" alt="Recipe number"> | [**SIF and uSIF**](01-sif-usif.ipynb). Make classic word vectors more useful for sentence search. | 15 min |
| <img src="media/02-linear-heads-icon.svg" width="46" alt="Recipe number"> | [**Linear heads**](02-linear-heads.ipynb). Train a text classifier, compare its accuracy, and save it for your app. | 15 min |
| <img src="media/03-files-with-resume-icon.svg" width="46" alt="Recipe number"> | [**Files with resume**](03-files-with-resume.ipynb). Turn a text file into vectors and continue after an interruption. | 10 min |
| <img src="media/04-kafka-and-qdrant-icon.svg" width="46" alt="Recipe number"> | [**Kafka and Qdrant**](04-kafka-and-qdrant.ipynb). Turn queued news into searchable vectors and replay safely. | 10 min |
| <img src="media/05-folder-listener-icon.svg" width="46" alt="Recipe number"> | [**Folder listener**](05-folder-listener.ipynb). Pick up completed files and add each one to your search store. | 10 min |
| <img src="media/06-integrations-icon.svg" width="46" alt="Recipe number"> | [**Integrations**](06-integrations.ipynb). Use one local encoder with your search, topic, and table libraries. | 15 min |
| <img src="media/07-local-compute-and-http-icon.svg" width="46" alt="Recipe number"> | [**Local compute and HTTP**](07-local-compute-and-http.ipynb). See when a local HTTP request costs more than encoding the text. | 5 min |

*Estimates for reading, excluding installation, downloads, and execution.
The first recipe includes a large-data speed demonstration, which needs additional disk space and execution time.

## Run a recipe

Use standard Python 3.13 on macOS or Linux, with Git, Make, a C/C++ compiler, and [uv](https://docs.astral.sh/uv/).
Free-threaded Python builds are unsupported.
From the repository root:

```sh
UV_PYTHON=3.13 make install
uv pip install --python .venv/bin/python -r recipes/requirements.txt
```

Open a recipe in Jupyter or VS Code and select `.venv/bin/python` as the kernel.
Run cells from top to bottom, with `recipes/` as the working directory so the helpers can be imported.
Use `uv run --no-sync` for commands after installation, because syncing removes the optional recipe packages.
For command-line execution:

```sh
recipe_tmp=$(mktemp -d)
IPYTHONDIR="$recipe_tmp/ipython" JUPYTER_RUNTIME_DIR="$recipe_tmp/runtime" \
  JUPYTER_CONFIG_DIR="$recipe_tmp/config" JUPYTER_DATA_DIR="$recipe_tmp/data" \
  uv run --no-sync jupyter execute --inplace --timeout=1800 recipes/00-fse-101.ipynb
```

Replace the filename to run another recipe.
These recipes run locally, outside CI, and their dependencies stay separate from fse core.
fastText 0.9.3 builds from source, which requires the C++ compiler.

## Models and data

**POTION-base-8M supplies real vectors**, including in the file, queue, folder, and integration examples.
Those examples use articles from the published AG News test split.
The SIF recipe also uses GloVe and the repository's STS Benchmark files.
The classification recipe uses SST-5 and a fixed sample of AG News.

Model and dataset revisions are pinned in the recipes and [_helpers.py](_helpers.py).
Read the [POTION model card](https://huggingface.co/minishlab/potion-base-8M), [GloVe model card](https://huggingface.co/fse/glove-wiki-gigaword-300), [SST-5 dataset card](https://huggingface.co/datasets/SetFit/sst5), and [AG News dataset card](https://huggingface.co/datasets/SetFit/ag_news) for their sources and terms.
The model and dataset terms apply separately from fse's license.

The helper puts generated files and new downloads in a temporary folder and reads existing exact-revision Hub snapshots without changing them.
The final cleanup cell removes the temporary results, so choose permanent paths when adapting a recipe.
No recipe uploads data, needs credentials, or calls a paid service.
The queue is simulated, and databases run locally.

## Read the results honestly

**Quality and speed answer different questions.**
The classification recipe reports accuracy and macro-F1 on the same test texts for four training methods.

Model2Vec trains an MLP (multilayer perceptron) with PyTorch and fine-tunes its word vectors, while fse uses fixed POTION vectors and a linear head.
The default MLP has one hidden layer of 512 units with ReLU, which replaces negative hidden values with zero.
Its [deployment head runs in plain NumPy](https://github.com/MinishLab/model2vec/blob/v0.10.0/model2vec/inference/mlp.py), using softmax for single-label probabilities or sigmoid when several labels can apply.

The recipe records accuracy through Model2Vec's training model in PyTorch, rather than exporting the NumPy deployment pipeline.
**The comparison is a fine-tuned MLP against a linear head**, alongside TF-IDF and fastText baselines.
The settings are fixed examples, without a tuning search or repeated-seed confidence intervals.

The SIF recipe fits weights and directions on STS Benchmark training texts only.
It reports test Pearson correlation times 100, then checks the separate historical experiment that used test-fitted directions.
A historical reproduction is not evidence of quality on unseen data.

The fse 101 speed demonstration is labeled **measured on a MacBook in everyday use, not a controlled benchmark**.
It prints its machine, CPU, Python, commit, date, and load averages beside the result.
The HTTP recipe retains its historical 1 November 2026 outputs and [machine provenance](run-environment.json).
Use the [performance guide](../PERFORMANCE.md) for measurements, methods, and their limits.

## Adapt a recipe safely

Use one writer, unchanged inputs, and a supported local filesystem for saved vector stores.
**Acknowledge only durable rows**, meaning rows saved so they survive a process crash.
For a resumed run, add `first_row`, the rows already stored, to `durable_rows`, the newly saved rows.
The queue and folder recipes explain their ordering assumptions where they are used.

The [API and design index](../ARCHITECTURE.md) documents the full contracts.
The diagrams use the maintainer's brand palette with light and dark SVG variants.
Regenerate them with `uv run --no-sync python recipes/_charts.py`.
