# Architecture

fse runs finished static embedding models on the CPU, turning text into vectors without training a model.

This document describes how fse is put together and the rules that keep it correct. It is the reference for changes to the contracts, the native kernel and the tests. When a change alters one of these rules, update the relevant architecture document in the same pull request.

master is the development branch of 2.0, the inference engine for finished static embeddings (such as POTION/Model2Vec tables and the StaticEmbedding of sentence-transformers): an encoder is a finished model, its table pooled per sentence by `weighted_mean@1` with an optional `normalize@1`. fse 2.0 trains nothing (maintainer decision of 29.10.2026): word weights, removed components and projections of a model such as SIF or uSIF are folded into its table offline, by a notebook with the 1.1.0 recipe that follows before the release. A linear head after the mean (`fse.Classifier`, `fse.Regressor`, PR 13) is imported from where it was trained (scikit-learn or arrays), never trained by fse, and runs in the worker as an operation of its own (`linear@1`, [linear_head 1](docs/architecture/formats.md#linear_head-1)). The 1.x API (`Average`, `SIF`, `uSIF`, the inputs and `SentenceVectors`) was removed from master; it lives on the branch `1.x` and in fse 1.1.0 on PyPI, and accessing one of its names on `fse` raises an `AttributeError` that says so. Items marked 2.1 are additions that must fit without changing the architecture contracts.

## Rules

### Validate strictly, never fail silently

Invalid input raises a clear exception at the API boundary. Nothing is coerced, ignored or truncated without an error, and an unexpected keyword argument is an error (`pydantic.ValidationError` at validated entry points). A slower but correct fallback is logged as a warning ([Logging](docs/architecture/logging.md#logging)).

Public entry points whose own work dwarfs a few microseconds use `fse._validation.validate_arguments`, a strict pydantic `validate_call`, with constraints such as `PositiveInt` in the signatures. Methods called per element in loops, and parameters that hold bulk data, check explicitly instead, because validation would add measurable overhead or copy the data; such parameters stay `Any` in validated signatures. Integer constraints accept numpy integer scalars, float constraints reject infinity and NaN.

## Documents

The index and these documents are the single source of truth. Each section states its scope.
Sections formerly under "2.0 design" are binding for every 2.0 module.

| Document | Sections |
|---|---|
| [process.md](docs/architecture/process.md) | Package Layout, Build, Tests, Layout, Baseline |
| [numerical-contract.md](docs/architecture/numerical-contract.md) | Native kernel, float32 throughout, Numerical contract (`transform_version` 2) |
| [text.md](docs/architecture/text.md) | Text sources |
| [logging.md](docs/architecture/logging.md) | Logging |
| [overview.md](docs/architecture/overview.md) | Principle and stages, Batch |
| [backends.md](docs/architecture/backends.md) | Backends |
| [persistence.md](docs/architecture/persistence.md) | Identifiers and versions, Identities, Persistence names replace class paths, Migration |
| [tokenizers.md](docs/architecture/tokenizers.md) | Tokenizers, wordpiece@1, unigram@1 |
| [runtime.md](docs/architecture/runtime.md) | Runtime, Defaults |
| [formats.md](docs/architecture/formats.md) | Artifacts, safetensors 1, NpyContainer 1, word_vectors 1, encoder 1, linear_head 1 |
| [hub.md](docs/architecture/hub.md) | Model2Vec models and from_pretrained, model2vec compatibility (fse.StaticModel), sentence-transformers models, Hub folders |
| [store.md](docs/architecture/store.md) | sentence_vectors 1 |

## Former sections

Old fragment links land here. Follow the section link to its current location.

| Former section | Current section |
|---|---|
| <a id="package-layout"></a>Package Layout | [Package Layout](docs/architecture/process.md#package-layout) |
| <a id="native-kernel"></a>Native kernel | [Native kernel](docs/architecture/numerical-contract.md#native-kernel) |
| <a id="text-sources"></a>Text sources | [Text sources](docs/architecture/text.md#text-sources) |
| <a id="float32-throughout"></a>float32 throughout | [float32 throughout](docs/architecture/numerical-contract.md#float32-throughout) |
| <a id="logging"></a>Logging | [Logging](docs/architecture/logging.md#logging) |
| <a id="build"></a>Build | [Build](docs/architecture/process.md#build) |
| <a id="tests"></a>Tests | [Tests](docs/architecture/process.md#tests) |
| <a id="20-design"></a>2.0 design | [2.0 design](#documents) |
| <a id="principle-and-stages"></a>Principle and stages | [Principle and stages](docs/architecture/overview.md#principle-and-stages) |
| <a id="numerical-contract-transform_version-2"></a>Numerical contract (`transform_version` 2) | [Numerical contract (`transform_version` 2)](docs/architecture/numerical-contract.md#numerical-contract-transform_version-2) |
| <a id="identifiers-and-versions"></a>Identifiers and versions | [Identifiers and versions](docs/architecture/persistence.md#identifiers-and-versions) |
| <a id="batch"></a>Batch | [Batch](docs/architecture/overview.md#batch) |
| <a id="tokenizers"></a>Tokenizers | [Tokenizers](docs/architecture/tokenizers.md#tokenizers) |
| <a id="wordpiece1"></a>wordpiece@1 | [wordpiece@1](docs/architecture/tokenizers.md#wordpiece1) |
| <a id="unigram1"></a>unigram@1 | [unigram@1](docs/architecture/tokenizers.md#unigram1) |
| <a id="backends"></a>Backends | [Backends](docs/architecture/backends.md#backends) |
| <a id="runtime"></a>Runtime | [Runtime](docs/architecture/runtime.md#runtime) |
| <a id="artifacts"></a>Artifacts | [Artifacts](docs/architecture/formats.md#artifacts) |
| <a id="safetensors-1"></a>safetensors 1 | [safetensors 1](docs/architecture/formats.md#safetensors-1) |
| <a id="npycontainer-1"></a>NpyContainer 1 | [NpyContainer 1](docs/architecture/formats.md#npycontainer-1) |
| <a id="word_vectors-1"></a>word_vectors 1 | [word_vectors 1](docs/architecture/formats.md#word_vectors-1) |
| <a id="encoder-1"></a>encoder 1 | [encoder 1](docs/architecture/formats.md#encoder-1) |
| <a id="model2vec-models-and-from_pretrained"></a>Model2Vec models and from_pretrained | [Model2Vec models and from_pretrained](docs/architecture/hub.md#model2vec-models-and-from_pretrained) |
| <a id="model2vec-compatibility-fsestaticmodel"></a>model2vec compatibility (fse.StaticModel) | [model2vec compatibility (fse.StaticModel)](docs/architecture/hub.md#model2vec-compatibility-fsestaticmodel) |
| <a id="sentence-transformers-models"></a>sentence-transformers models | [sentence-transformers models](docs/architecture/hub.md#sentence-transformers-models) |
| <a id="linear_head-1"></a>linear_head 1 | [linear_head 1](docs/architecture/formats.md#linear_head-1) |
| <a id="hub-folders"></a>Hub folders | [Hub folders](docs/architecture/hub.md#hub-folders) |
| <a id="identities"></a>Identities | [Identities](docs/architecture/persistence.md#identities) |
| <a id="persistence-names-replace-class-paths"></a>Persistence names replace class paths | [Persistence names replace class paths](docs/architecture/persistence.md#persistence-names-replace-class-paths) |
| <a id="migration"></a>Migration | [Migration](docs/architecture/persistence.md#migration) |
| <a id="sentence_vectors-1"></a>sentence_vectors 1 | [sentence_vectors 1](docs/architecture/store.md#sentence_vectors-1) |
| <a id="layout"></a>Layout | [Layout](docs/architecture/process.md#layout) |
| <a id="defaults"></a>Defaults | [Defaults](docs/architecture/runtime.md#defaults) |
| <a id="baseline"></a>Baseline | [Baseline](docs/architecture/process.md#baseline) |
| <a id="writing"></a>Writing | [sentence_vectors 1](docs/architecture/store.md#sentence_vectors-1) |
| <a id="reading-stored-rows"></a>Reading stored rows | [Numerical contract (`transform_version` 2)](docs/architecture/numerical-contract.md#numerical-contract-transform_version-2) |
| <a id="from_pretrained"></a>from_pretrained | [Model2Vec models and from_pretrained](docs/architecture/hub.md#model2vec-models-and-from_pretrained) |
| <a id="model2vec-models"></a>Model2Vec models | [Model2Vec models and from_pretrained](docs/architecture/hub.md#model2vec-models-and-from_pretrained) |
| <a id="compatibility-limit-of-persistence"></a>Compatibility limit of persistence | [model2vec compatibility (fse.StaticModel)](docs/architecture/hub.md#model2vec-compatibility-fsestaticmodel) |
| <a id="tasks"></a>Tasks | [Runtime](docs/architecture/runtime.md#runtime) |
| <a id="errors"></a>Errors | [Runtime](docs/architecture/runtime.md#runtime) |
| <a id="linear-heads"></a>Linear heads | [linear_head 1](docs/architecture/formats.md#linear_head-1), [Numerical contract (`transform_version` 2)](docs/architecture/numerical-contract.md#numerical-contract-transform_version-2), [Runtime](docs/architecture/runtime.md#runtime) |
| <a id="quantized-models"></a>Quantized models | [encoder 1](docs/architecture/formats.md#encoder-1) |
| <a id="numerical-contract"></a>Numerical contract | [Numerical contract (`transform_version` 2)](docs/architecture/numerical-contract.md#numerical-contract-transform_version-2) |
| <a id="model2vec-compatibility"></a>model2vec compatibility | [model2vec compatibility (fse.StaticModel)](docs/architecture/hub.md#model2vec-compatibility-fsestaticmodel) |
| <a id="20"></a>2.0 | [Principle and stages](docs/architecture/overview.md#principle-and-stages) |
