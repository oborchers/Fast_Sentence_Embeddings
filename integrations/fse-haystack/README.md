# fse-haystack

Prepared, unpublished Haystack integration for fse 2.0. The maintainer has not
yet selected the repository layout or approved publication.

This package depends on fse and `haystack-ai` only. Framework dependencies never enter
fse core. This folder can move unchanged into a standalone repository.

## Use

After an fse 2.0 build and this package are installed:

```python
from haystack import Document
from haystack_integrations.components.embedders.fse import (
    FseDocumentEmbedder,
    FseTextEmbedder,
)

query_embedder = FseTextEmbedder(model="minishlab/potion-base-8M", workers=1)
query_embedder.warm_up()
query = query_embedder.run(text="hello world")["embedding"]
document_embedder = FseDocumentEmbedder(workers=1, batch_size=128)
document_embedder.warm_up()
documents = document_embedder.run(documents=[Document(content="hello world")])["documents"]
```

For an existing local model, pass `encoder=your_encoder`, built for example with
`fse.Encoder.from_arrays`. No download occurs with an injected encoder.

The `@component` classes expose `embedding` and `documents` output sockets.
They implement `run_async` as well. Call `warm_up()` before standalone execution.
Haystack pipelines call warm-up themselves. Document calls use batches of at most
`batch_size` texts, preserving input order. Every document must have text content.
The document embedder returns new documents and preserves IDs and metadata.
It embeds content only, without metadata concatenation or implicit query prefixes.

Pipeline serialization stores model and loader options, never the loaded encoder.
An injected encoder cannot be serialized. Save it first, then use its local path
as `model`. Serialization rejects an explicit Hub token to avoid saving credentials.
Follow the [community integration guide](https://github.com/deepset-ai/haystack-integrations#readme)
for a later listing. No listing is submitted in this preparation.

## Framework baseline and guides

The supported minimum for this draft is `haystack-ai==3.4.1`, the current stable
release checked on 2026-11-01 from [PyPI](https://pypi.org/project/haystack-ai/3.4.1/).
This is the package's tested baseline, not a claim that older framework APIs cannot work.
The runtime dependency allows later versions. Recheck compatibility before publication.

The implementation follows the [official integration guide](https://docs.haystack.deepset.ai/docs/custom-components).

## Options and behavior

`load_kwargs` passes options unchanged to `Encoder.from_pretrained`. For example,
`{"normalize": False, "revision": "<commit>", "local_files_only": True}` selects
normalization, a Hub revision and offline loading. The default normalization is
`"model"`, which preserves the model's rule. A local fse artifact keeps its saved
steps. Hub-only options (`revision`, `token`, `cache_dir`, `local_files_only`)
are invalid for local paths. See the [fse loader contract](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/hub.md).
Do not pass `load_kwargs` with `encoder=`. Configure that encoder before injection.
Use the Hub credential store rather than putting tokens in application configuration.

`dtype="float32"` (default) or `"float16"` controls fse output precision.
`workers=None` uses fse's worker policy. `workers=1` requests one worker.
The adapter passes text and worker validation to fse without replacing its errors.
Integer embeddings are unsupported because this interface requires float vectors.

Every result converts a numpy array into Python float lists. This copies values
and allocates Python objects, so it takes additional time and memory.
It does not retain the array's compact float16 or float32 storage.
Empty text and unknown tokens follow the loaded encoder's tokenizer rules.
Async methods delegate encoding and list conversion to a thread.
Construct models before serving async requests. Haystack also provides
`warm_up_async()` so async pipelines can load outside the event loop.

## Development

Use standard Python 3.13. From this package directory in the fse checkout:

```sh
uv venv --python 3.13 .venv
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FSE=2.0.0.dev0 \
  uv pip install --python .venv/bin/python -e '../..' -e '.[test]'
.venv/bin/python -m pytest -m 'not network'
.venv/bin/python -m pytest -m network
uv build --out-dir dist
```

The version override labels the development checkout for dependency resolution.
It does not create a tag, release or published package.

After moving the folder to its own repository, replace `-e '../..'` with an
available fse 2.0 build. The declared requirement is `fse>=2.0.0.dev0`.
The network test downloads `minishlab/potion-base-8M`. Offline tests build a tiny
local encoder with `Encoder.from_arrays` and compare vectors with fse directly.
Tests disable Python network sockets by default and enable them for the marked
network test. No model upload or paid service is needed.

## AI assistance

These package sources, tests and documentation were prepared with AI assistance
using Codex. The maintainer remains responsible for reviewing and validating them.
Publication and framework listing submissions require a separate maintainer decision.
