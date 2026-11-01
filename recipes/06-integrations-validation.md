# Recipe 06 validation

The complete [integration recipe](06-integrations.ipynb) executed on 1 November 2026
with Python 3.13.16 on an Apple M1 Pro running macOS 27.0.1.
It used the real POTION-base-8M model and 100 AG News articles.
Model and dataset revisions are pinned in [_helpers.py](_helpers.py).

## Durable results

- [Executed notebook](06-integrations.ipynb): all 15 code cells completed in order,
  with no error outputs. All 11 original code cells remain unchanged.
- [Execution record](06-integrations-run.json): package versions, source commits,
  installation substitution and the notebook checksum.
- This report: the verified local gate results and reviewed README text.

LangChain, LlamaIndex and Haystack each retrieved the article used as the query.
Haystack stored all 100 articles. The assertions check the connections for this
example, rather than measuring search quality. No speed measurement was made.

## Local validation

The tested input was commit `a1fec82e6a22d75130d45c704d172ddbde12195a`.
The executed outputs were committed as `db0a7c45e7334c54753f3d186b9419e871147838`,
with every notebook source cell unchanged from the tested input.

| Step | Verified result |
|---|---|
| Core install with Python 3.13 | Passed |
| `make check PYTEST_N=4` | Passed, 9086 tests passed and 515 skipped |
| Separate recipe environment with Python 3.13 | Created |
| Recipe dependencies and local source packages | Installed |
| `uv pip check` | Passed, 231 compatible packages |
| Dependency freeze | Captured during validation |
| Complete notebook execution | Passed, all 15 code cells |

The original temporary logs, dependency freeze, isolated environment and review
transcript were later cleaned away. The gate results above were inspected at
execution and recorded in the draft PR before that cleanup. This summary does
not recreate those raw logs. The committed notebook and execution record remain
available, and their checksum and execution counts were reverified afterward.

## Installation substitution

The notebook's final `pip install` commands are Markdown instructions for use
after publication. They were not executed during local validation.
Instead, validation installed the recipe requirements, fse and all three adapter
packages from parent source commit `2f408ecd8cc164c2d89aa29c36e34f0977fee691`.
The source directories were `integrations/langchain-fse`,
`integrations/llama-index-embeddings-fse` and `integrations/fse-haystack`.
Installed distribution metadata confirmed those local source directories.

`SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FSE=2.0.0` labelled the unreleased fse build
for dependency resolution. It created no tag or release. Each adapter reported
version 0.1.0. The framework versions were langchain-core 1.7.3,
llama-index-core 0.14.25 and haystack-ai 3.4.1.
No PyPI adapter package was installed or published.

The independent read-only review found no API issues. Its installation warning
was addressed with environment activation instructions and adapter version pins.
Optional skipping was rejected because validation must execute every framework
example. No further heavy validation is needed unless a new failure or review
concern appears.

## Reviewed README insertion

The text below belongs in the root README only after
#189 (2.0: Rewrite the README, CONTRIBUTING and package metadata) merges.
Merge master into the tutorial branch, inspect the merged README, then place
this section after the classifier and folding material in the new style.
The class paths match the package source and the successful notebook imports.

This docs PR remains draft. It merges only after the README prerequisite and
publication of all three adapters after fse 2.0.0. This record authorizes no
PR merge, package publication or pod lifecycle action.

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
