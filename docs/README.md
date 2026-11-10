# Documentation

Start with the [README](../README.md): installation, first examples, models and recipes.
This page lists the guides in `docs/` by what you want to do. The links below point to files in this folder, to the [recipes](../recipes/README.md) and to the architecture pages.

## Use fse

| I want to | Read |
| :--- | :--- |
| Choose a model | [Classic models on the Hub](models.md) and the model table in the [README](../README.md#choose-a-model-and-memory-budget) |
| Move an application from fse 1.x | [Moving from fse 1.x to 2.0](migrating-from-1x.md) |
| Encode a file, a stream or a corpus and store the rows | [Recipes](../recipes/README.md), and the [store reference](architecture/store.md) |
| Keep a document ID with each row and search for the best row per document | [Document IDs](document-ids.md) |
| Search with FAISS, USearch or hnswlib and rescore exactly | [External indexes](external-indexes.md) |
| Classify text and keep scores | "Linear heads" in the [README](../README.md#linear-heads-classify-text-and-save-scores) |
| Continue a run after a stop | [Files with resume](../recipes/03-files-with-resume.ipynb) and [fse behind Kafka](../recipes/09-fse-behind-kafka.ipynb) |

## Reference

| Topic | Page |
| :--- | :--- |
| Model files, Hub folders and heads | [formats](architecture/formats.md), [hub](architecture/hub.md) |
| Stores, scores and their durability | [store](architecture/store.md), [scores](architecture/scores.md), [document IDs](architecture/document-ids.md) |
| Runtime, pools and workers | [runtime](architecture/runtime.md) |
| Numerical contract and tokenizers | [numerical-contract](architecture/numerical-contract.md), [tokenizers](architecture/tokenizers.md), [text](architecture/text.md) |
| Persistence names and versions | [persistence](architecture/persistence.md) |
| Logging | [logging](architecture/logging.md) |
| Package layout, build and CI | [process](architecture/process.md) |

## Measurements

| Topic | Page |
| :--- | :--- |
| How fse measures speed and quality | [measurement guide](benchmarks/measurement.md) |
| Evidence behind the README numbers | [README evidence](benchmarks/readme-evidence.md), [comparison with other libraries](fse-comparison.md) |

Contribution rules are in [CONTRIBUTING.md](../CONTRIBUTING.md). The rules that keep the native kernel and the reference consistent are in [ARCHITECTURE.md](../ARCHITECTURE.md).
