# llama-index-embeddings-fse

Prepared, unpublished LlamaIndex integration for fse 2.0. The maintainer has not
yet selected the repository layout or approved publication.

This package depends on fse and `llama-index-core` only. Framework dependencies never enter
fse core. This folder can move unchanged into a standalone repository.

## Use

After an fse 2.0 build and this package are installed:

```python
from llama_index.embeddings.fse import FseEmbedding

embeddings = FseEmbedding(model_name="minishlab/potion-base-8M", workers=1, embed_batch_size=128)
query = embeddings.get_query_embedding("hello world")
rows = embeddings.get_text_embedding_batch(["hello world", "another document"])
```

For an existing local model, pass `encoder=your_encoder`, built for example with
`fse.Encoder.from_arrays`. No download occurs with an injected encoder.

`FseEmbedding` implements `BaseEmbedding` query, text and batch hooks, including
async hooks. LlamaIndex's `embed_batch_size` controls calls to fse. Its `num_workers`
controls async batch concurrency, while `workers` controls fse workers per call.
Avoid multiplying both settings without checking your application's resource use.
Callbacks, caches and node embedding use the base class behavior.
The transformation identity includes the encoder fingerprint and output dtype,
so ingestion caches distinguish models and precision. Saved model references and
options survive process-based ingestion. Reconstruction checks the loaded fingerprint.
Injected in-memory encoders support caching but cannot be pickled or reconstructed
without the original encoder. Save them and use a local path for multiprocessing.
Serialized dictionaries omit Hub tokens. Pickling with an explicit token is refused.
Use the Hub credential store for reconstruction.
LlamaIndex requires new integrations to live outside its monorepo.

## Framework baseline and guides

The supported minimum for this draft is `llama-index-core==0.14.25`, the current stable
release checked on 2026-11-01 from [PyPI](https://pypi.org/project/llama-index-core/0.14.25/).
This is our tested minimum and makes no compatibility claim for older framework releases.
The runtime dependency allows later versions. Recheck compatibility before publication.

The implementation follows the [official integration guide](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md).

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
Loading is synchronous, so construct models before serving async requests.

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
