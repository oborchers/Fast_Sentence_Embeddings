# Integration audit

Checked 2026-11-01 against the merged fse master API and the current framework
releases. This records preparation for 0.1.0, not publication or listing approval.

## Sources and interface comparison

All sources below were retrieved on 2026-11-01. Upstream main branches can change.

### LangChain

- [Implementation guide](https://docs.langchain.com/oss/python/contributing/implement-langchain):
  embedding integrations subclass `Embeddings`. `FseEmbeddings` does so.
- [Interface source](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/embeddings/embeddings.py):
  required query/document methods return float lists. Async overrides are optional.
  Our adapter implements all four methods, with async work in threads. A document
  call encodes its supplied batch once. Applications choose the batch size.
- [Reference integration](https://github.com/langchain-ai/langchain/blob/master/libs/partners/huggingface/langchain_huggingface/embeddings/huggingface.py):
  the provider class follows the embeddings interface and exposes model options.
  Pydantic inheritance is a provider choice, not an `Embeddings` requirement.
  fse uses its own strict constructor validation and loader options.
- [Standard embeddings tests](https://github.com/langchain-ai/langchain/blob/master/libs/standard-tests/langchain_tests/integration_tests/embeddings.py):
  `EmbeddingsIntegrationTests` includes inherited initialization tests and sync/async
  query/document checks. Our test class runs this suite using a local encoder.
- [Publication guide](https://docs.langchain.com/oss/python/contributing/publish-langchain)
  and [contribution policy](https://docs.langchain.com/oss/python/contributing/overview#acceptable-uses-of-llms):
  use an independent PyPI package, `langchain-<provider>` naming, verified examples
  and reviewed AI contributions. `langchain-fse` fits. Listing follows publication
  through the listing issue form. A hosted guide needs separate eligibility.
  No upstream issue or documentation submission is part of this change.

### LlamaIndex

- [Embedding guide](https://developers.llamaindex.ai/python/framework/module_guides/models/embeddings/)
  and [base source](https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/base/embeddings/base.py):
  `FseEmbedding` subclasses `BaseEmbedding`, implements query/text hooks and all
  async counterparts, and overrides batch hooks. Base behavior supplies callbacks,
  public batch scheduling, node transforms and caching. `embed_batch_size` controls
  framework batches. `workers` controls fse work within each call.
- [FastEmbed reference](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/embeddings/llama-index-embeddings-fastembed/llama_index/embeddings/fastembed/base.py):
  the adapter uses `PrivateAttr` for the loaded model, a provider `class_name`,
  and sync/async query and text hooks. fse follows that pattern. Our persistence
  also checks encoder fingerprints and preserves options across process ingestion.
- [Contribution guide](https://github.com/run-llama/llama_index/blob/main/CONTRIBUTING.md):
  new packages must live outside their monorepo and publish independently.
  `llama-index-embeddings-fse` and `llama_index.embeddings.fse` follow the reference
  naming. The guide uses package pytest suites and mocked remote dependencies.
  No generic third-party embedding conformance suite is prescribed there.
  Our offline tests cover public APIs, async execution, ingestion cache identity,
  process ingestion, serialization and local model loading. AI scope and validation
  are disclosed. Upstream listing or discovery changes need separate review later.

### Haystack

- [Custom component guide](https://docs.haystack.deepset.ai/docs/custom-components):
  both embedders use `@component`, typed `run` inputs and output dictionaries with
  declared sockets. Document inputs remain unchanged. Async methods and warm-up
  support pipeline execution. Content-only embedding and explicit standalone
  warm-up are documented choices, not missing mandatory features.
- [FastEmbed reference](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/fastembed/src/haystack_integrations/components/embedders/fastembed/fastembed_document_embedder.py)
  and [its tests](https://github.com/deepset-ai/haystack-core-integrations/blob/main/integrations/fastembed/tests/test_fastembed_document_embedder.py):
  fse follows the provider namespace, separate text/document components, warm-up,
  serialization and copied-document pattern. Tests cover batching, metadata/ID
  preservation, option roundtrips, async calls and pipeline warm-up. The reference
  uses package-specific pytest tests, not a reusable embedding standard-test class.
- [Community index guide](https://github.com/deepset-ai/haystack-integrations/blob/main/README.md):
  community integrations remain author-maintained. A later listing needs Markdown
  with required name, description, authors and layout, plus installation and usage.
  `fse-haystack` follows the reference package naming. No listing is submitted here.
- [Contribution policy](https://github.com/deepset-ai/haystack/blob/main/CONTRIBUTING.md):
  contributors must review generated work and test it. Fully generated upstream
  PRs need a disclaimer. The package README discloses AI assistance and validation.

## Cross-package packaging and maintenance

| Concern | Audit result |
|---|---|
| Package discovery | Setuptools discovers `src/`; LlamaIndex and Haystack retain namespace packages. No parent namespace `__init__.py` shadows framework packages. |
| Entry points | These APIs use direct class imports. None of the cited interfaces requires a plugin entry point. |
| Versions | Each package declares 0.1.0, with release tags checked against that exact version. Each requires `fse>=2.0.0`. |
| Framework ranges | Lower bounds match current stable PyPI releases: langchain-core 1.7.3, llama-index-core 0.14.25, haystack-ai 3.4.1. Later releases remain allowed and tested by the latest CI job. No older-version claim. |
| License | Each package includes its own MIT license and declares its license file in metadata. Upstream reference implementation licenses do not impose their license on independent adapter code. |
| README | Each documents installation, public names, loader behavior, async/batching, float-list costs, framework baseline, validation, release boundary and AI assistance. |
| Maintenance | The repository owner reviews and maintains the adapters. CI keeps framework environments separate and detects future release incompatibility. Listing is a later maintainer decision. |
| Source archives | Explicit manifests include changelogs, complete tests and retrieval scripts/results. Wheels contain only the discovered package namespace and distribution metadata. |
| fse API | All adapters use current `Encoder.from_pretrained`, `encode`, `fingerprint` and `validate_arguments`. Loader options pass through. Tests use current `from_arrays` and safetensors persistence. No removed 1.x API is used. |
| Network boundary | Offline tests use local tiny encoders and disabled sockets. Only marked network tests and opt-in examples download POTION. |

Current-release evidence: [LangChain PyPI](https://pypi.org/pypi/langchain-core/json),
[LlamaIndex PyPI](https://pypi.org/pypi/llama-index-core/json),
[Haystack PyPI](https://pypi.org/pypi/haystack-ai/json), retrieved 2026-11-01.

## Every-file review

The audit read every tracked file under `integrations/`, not only changed lines.
For each of the three package directories, the following checks apply:

| File | Inspection |
|---|---|
| `src/.../__init__.py` | Every constructor, public hook, encoding call, async delegation, validation and serialization path compared with current fse and framework sources above. |
| `tests/test_api.py` | Every assertion checked for actual vector parity or framework behavior. Process ingestion really uses two processes. Local artifact options remain valid. Network coverage now runs the full retrieval example. |
| `tests/conftest.py` | Tiny float32 encoder fixture uses the current API and independent pytest fixture lifetime. Haystack telemetry is disabled in its tests. |
| `pyproject.toml` | Build backend, namespace discovery, name/version, stable fse requirement, framework minimum, test extras, strict warnings and socket markers reviewed. |
| `.gitignore` | Only local environments, build outputs and caches are ignored. Sources, tests and examples remain tracked. |
| `MANIFEST.in` | Includes changelog, complete tests and examples in sdist. |
| `README.md` | All code snippets and options reviewed against the current implementation. Removed stale layout-decision and prerelease-dependency wording. |
| `LICENSE` | Complete MIT grant present in every package. |
| `CHANGELOG.md` | Unpublished 0.1.0 plan, stable fse dependency, CI/release preparation and retrieval coverage recorded. |
| `examples/retrieval.py` | Full indexing and retrieval app, real POTION model, three distinct documents and asserted matching top result. No LLM or service API key needed. |
| `examples/retrieval-output.txt` | Generated only after a successful fresh-environment application run; records framework version, dimensions and retrieval outcome. |

The common `README.md` was reviewed for the settled in-repo layout and exact
release setup requirements. This audit file records the research and comparison.
Validation results and final commit are recorded in the delivery report.
