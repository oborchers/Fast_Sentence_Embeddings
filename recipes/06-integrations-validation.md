# Recipe 06 validation

The complete [integration recipe](06-integrations.ipynb) executed on 16 November 2026
with Python 3.13.16 on an Apple M1 Pro running macOS 27.0.1.
It used fse 2.1.0 and version 0.2.0 of `langchain-fse`, `llama-index-embeddings-fse` and `fse-haystack`,
all installed from PyPI (DR-434: Veraltete Versionsangaben und Adapter-Pins im Integrationsrezept 06 auf 2.1.0 und 0.2.0 heben).
It used the real POTION-base-8M model and 100 AG News articles.
Model and dataset revisions are pinned in [_helpers.py](_helpers.py).
The claim class is correctness. No speed measurement was made.

## Durable results

- [Executed notebook](06-integrations.ipynb): all 15 code cells completed in order,
  with no error outputs. Only text and install pins changed since the 1 November run;
  every code cell is unchanged.
- [Execution record](06-integrations-run.json): package versions, source commit,
  installation note and the notebook checksum.
- This report: the verified results and the reviewed README text.

LangChain, LlamaIndex and Haystack each retrieved the article used as the query.
Haystack stored all 100 articles. The assertions check the connections for this
example, rather than measuring search quality.

## Local validation

The tested input was the notebook on commit `87f11f9c` with the new version text and pins.

| Step | Verified result |
|---|---|
| Clean Python 3.13.16 environment | Created |
| fse 2.1.0, `langchain-fse` 0.2.0, `llama-index-embeddings-fse` 0.2.0, `fse-haystack` 0.2.0 | Installed from PyPI |
| Recipe packages pinned in `requirements.txt` that recipe 06 imports | Installed |
| Complete notebook execution | Passed, all 15 code cells, no error outputs |

The framework versions were langchain-core 1.7.3, llama-index-core 0.14.25 and
haystack-ai 3.4.1. The environment held only the packages recipe 06 imports, not the
full recipe requirements. The notebook's `pip install` commands are Markdown
instructions and were not executed as cells. The first execution, on 1 November 2026
with fse 2.0.0 and adapters 0.1.0 built from source before publication, is
superseded by this run.

## Reviewed README insertion

The text below is included in the root README after the classifier and folding
material. The prerequisite #189 (2.0: Rewrite the README, CONTRIBUTING and package
metadata) merged as `6c7457b4682ab302a106e963eee6dd2effe03e6e` before this insertion.
The class paths match the package source and the successful notebook imports.

The maintainer requested ready-for-review status after green CI. Merge still
required publication of all three adapters after fse 2.0.0 and separate
maintainer approval. This record authorizes no PR merge, package publication
or infrastructure lifecycle action.

```markdown
## Use fse with your framework

Use the same local encoder in LangChain, LlamaIndex, or Haystack.
Each framework has a separate package, so fse core needs no framework dependency.

| Framework | Install | Embedding interface |
|---|---|---|
| LangChain | `pip install langchain-fse` | `langchain_fse.FseEmbeddings` |
| LlamaIndex | `pip install llama-index-embeddings-fse` | `llama_index.embeddings.fse.FseEmbedding` |
| Haystack | `pip install fse-haystack` | `FseTextEmbedder` and `FseDocumentEmbedder` from `haystack_integrations.components.embedders.fse` |

See [recipe 06](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/recipes/06-integrations.ipynb) for complete local search examples with POTION and real news articles.
```
