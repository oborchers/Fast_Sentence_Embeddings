# langchain-fse

Prepared, unpublished LangChain integration for fse 2.0. The maintainer has not
yet selected the repository layout or approved publication.

This package depends on fse and `langchain-core` only. Framework dependencies never enter
fse core. This folder can move unchanged into a standalone repository.

## Use

After an fse 2.0 build and this package are installed:

```python
from langchain_fse import FseEmbeddings

embeddings = FseEmbeddings(model="minishlab/potion-base-8M", workers=1)
query = embeddings.embed_query("hello world")
rows = embeddings.embed_documents(["hello world", "another document"])
```

For an existing local model, pass `encoder=your_encoder`, built for example with
`fse.Encoder.from_arrays`. No download occurs with an injected encoder.

`FseEmbeddings` implements `langchain_core.embeddings.Embeddings`, including
`aembed_query` and `aembed_documents`. Each document call sends the supplied
batch to fse once. Split larger inputs into batches in your application if needed.
The package runs `langchain-tests==1.1.9` standard embedding tests with a local encoder.
LangChain requires standalone packages for new integrations.

## Framework baseline and guides

The supported minimum for this draft is `langchain-core==1.7.3`, the current stable
release checked on 2026-11-01 from [PyPI](https://pypi.org/project/langchain-core/1.7.3/).
This is the package's tested baseline, not a claim that older framework APIs cannot work.
The runtime dependency allows later versions. Recheck compatibility before publication.

The implementation follows the [official integration guide](https://docs.langchain.com/oss/python/contributing/integrations-langchain).

## Options and behavior

`load_kwargs` passes options unchanged to `Encoder.from_pretrained`. For example,
`{"normalize": False, "revision": "<commit>", "local_files_only": True}` selects
normalization, a Hub revision and offline loading. The default normalization is
`"model"`, which preserves the model's rule. A local fse artifact keeps its saved
steps. See the [fse loader contract](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/hub.md).
Do not pass `load_kwargs` with `encoder=`. Configure that encoder before injection.
Use the Hub credential store rather than putting tokens in application configuration.

`dtype="float32"` (default) or `"float16"` controls fse output precision.
`workers=None` uses fse's worker policy. `workers=1` requests one worker.
The adapter passes text and worker validation to fse without replacing its errors.
Integer embeddings are unsupported because this interface requires float vectors.

Every result converts a numpy array into Python float lists. This copies values
and allocates Python objects, so it takes additional time and memory.
It does not retain the array's compact float16 or float32 storage.
Empty and unknown-only text follows fse's zero-vector behavior.
Async methods delegate encoding and list conversion to a thread.
Loading is synchronous, so construct and warm models before serving async requests.

## Development

Use standard Python 3.13. From this package directory in the fse checkout:

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e '../..' -e '.[test]'
.venv/bin/python -m pytest -m 'not network'
.venv/bin/python -m pytest -m network
uv build --out-dir dist
```

After moving the folder to its own repository, replace `-e '../..'` with an
available fse 2.0 build. The declared requirement is `fse>=2.0.0.dev0`.
The network test downloads `minishlab/potion-base-8M`. Offline tests build a tiny
local encoder with `Encoder.from_arrays` and compare vectors with fse directly.
No model upload or paid service is needed.

## AI assistance

These package sources, tests and documentation were prepared with AI assistance
using Codex. The maintainer remains responsible for reviewing and validating them.
Publication and framework listing submissions require a separate maintainer decision.
