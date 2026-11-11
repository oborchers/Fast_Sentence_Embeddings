# llama-index-embeddings-fse

Version 0.1.0 was published on 2026-11-02. Version 0.2.0 is prepared for fse 2.1 and requires separate maintainer approval before publication.
This package lives in `integrations/llama-index-embeddings-fse/` in the fse repository.

This package depends on fse, NumPy and `llama-index-core`. Framework dependencies never enter
fse core. This package has its own CI environment and release workflow.

## Use

After an fse 2.1 build and this package are installed:

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

The supported framework minimum for version 0.2.0 is `llama-index-core==0.14.25`, the tested lower bound
release checked on 2026-11-01 from [PyPI](https://pypi.org/project/llama-index-core/0.14.25/).
This is our tested minimum and makes no compatibility claim for older framework releases.
The runtime dependency allows later versions. Recheck compatibility before publication.

The implementation follows the [official integration guide](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md).

## Options and behavior

`load_kwargs` passes options unchanged to `Encoder.from_pretrained`.
`load_kwargs={"normalize": False}` returns raw means for every supported model,
including fse artifacts, Model2Vec and supported sentence-transformers models.
`normalize="model"` (the default) preserves the model's normalization rule.
For example, `{"revision": "<commit>", "local_files_only": True}` selects a Hub
revision and offline loading after the files have been cached.
Hub-only options (`revision`, `token`, `cache_dir`, `local_files_only`)
are invalid for local paths. See the [fse loader contract](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/docs/architecture/hub.md).
Do not pass `load_kwargs` with `encoder=`. Configure that encoder before injection.
Use the Hub credential store rather than putting tokens in application configuration.

`dtype="float32"` (default) or `"float16"` controls fse output precision.
`workers=None` uses fse's worker policy. `workers=1` requests one worker.
The adapter validates workers during construction using the fse rule and error text.
Text validation stays with fse.
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
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FSE=2.1.0 \
  uv pip install --python .venv/bin/python -e '../..' -e '.[test]'
.venv/bin/python -m pytest -m 'not network'
.venv/bin/python -m pytest -m network
uv build --out-dir dist
.venv/bin/python examples/retrieval.py
```

The version override labels the development checkout for dependency resolution.
It does not create a tag, release or published package.

The declared requirement is `fse>=2.1.0`. The version override tests the local
unreleased checkout without weakening the published dependency.
The network test downloads `minishlab/potion-base-8M`. Offline tests build a tiny
local encoder with `Encoder.from_arrays` and compare vectors with fse directly.
Tests disable Python network sockets by default and enable them for the marked
network test. No model upload or paid service is needed.

## CI and release

The isolated CI job tests Python 3.11 with the minimum framework and Python 3.13
with the latest framework. Network tests and the real-model retrieval example
run only when requested locally. See [release setup](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/master/integrations/README.md#release-setup)
for the package tag, approval environment and pending PyPI publisher.

## AI assistance

These package sources, tests and documentation were prepared with AI assistance
using Codex. Validation uses offline vector parity, framework API tests and the
real-model retrieval example. The maintainer remains responsible for review and maintenance.
Publication and framework listing submissions require a separate maintainer decision.

## Version 0.2.0 preparation

Version 0.1.0 was published on 2026-11-02. Version 0.2.0 is unreleased and
requires `fse>=2.1.0,<3`. Until fse 2.1.0 is published, test this source against
the local checkout with the documented version override. An unconstrained
resolver may upgrade fse to 2.1 and select adapter 0.2.0. Pinning fse to 2.0.x
while requesting adapter 0.2.0 produces a dependency conflict.

Release fse 2.1.0 first. Before any adapter 0.2.0 tag, test all three adapters
against `fse==2.1.0` installed from PyPI, then permanently pin that version in
the minimum CI job. Only then publish each adapter with separate maintainer
approval. Package changelogs are handwritten, independent of root towncrier.

Configuration validation belongs to each adapter. Invalid types raise
`TypeError`; invalid values raise `ValueError`. Invalid `workers` always raise
`ValueError` using the fse message, already at construction. Text path objects
stay local even when they look like Hub IDs. Serialized local paths are absolute.

Inherited LlamaIndex options are checked before framework conversion or model loading.
`embed_batch_size` requires an integer from 1 to 2048. `num_workers` requires a
positive integer or `None`. Booleans, numeric strings and floats are rejected.
`callback_manager`, `embeddings_cache` and `rate_limiter` require a LlamaIndex
`CallbackManager`, `BaseKVStore` and `BaseRateLimiter`, respectively, or `None`.
A `None` callback manager selects the framework default.
