# fse integration packages

The packages live in this repository under `integrations/`. Each package has
independent build metadata, tests, MIT license and documentation. fse core gains
no framework dependency. Version 0.1.0 is prepared for publication after fse 2.0.0.
Publication and framework listing submissions require separate maintainer approval.
This work can merge before publication.

| Package | Interface | Runtime dependencies |
|---|---|---|
| langchain-fse | `FseEmbeddings` | fse, langchain-core |
| llama-index-embeddings-fse | `FseEmbedding` | fse, llama-index-core |
| fse-haystack | `FseTextEmbedder`, `FseDocumentEmbedder` | fse, haystack-ai |

## Development and CI

Follow each package README from its directory. Use a separate environment per
framework. The local fse version override satisfies `fse>=2.0.0` while testing
unreleased source. It changes neither published metadata nor git tags.

Root `make check` checks core and the `integrations` guardrail path. It does not
install frameworks. `.github/workflows/integrations.yml` selects affected packages
on pull requests and master pushes. A package change runs only that package.
Changes to fse source or its build metadata run all packages. Every selected
package tests Python 3.11 with its framework minimum and Python 3.13 with the
latest framework, then builds and checks its sdist and wheel. Network tests stay
opt-in. `examples/retrieval.py` in each package runs a real POTION search app.

## Release setup

No environments, PyPI publishers, tags or releases are created by this change.
The maintainer must complete this setup before any approved publication:

| PyPI project | Workflow filename | GitHub environment | Tag pattern |
|---|---|---|---|
| langchain-fse | release-langchain-fse.yml | release-langchain-fse | langchain-fse-v* |
| llama-index-embeddings-fse | release-llama-index-embeddings-fse.yml | release-llama-index-embeddings-fse | llama-index-embeddings-fse-v* |
| fse-haystack | release-fse-haystack.yml | release-fse-haystack | fse-haystack-v* |

1. In GitHub, create each named environment with required maintainer reviewers,
   prevent self-review, and disable administrator bypass. Restrict deployments
   to tags matching that package's pattern. Environment names alone do not
   enforce approval. Configure these rules before registering PyPI trust.
2. In PyPI's pending publishers form, enter the exact project name above,
   owner `oborchers`, repository `Fast_Sentence_Embeddings`, workflow filename
   above (without `.github/workflows/`), and matching environment name. If the
   project already exists under the maintainer's control, add its trusted
   publisher instead. Package name availability is confirmed by PyPI at setup.
3. Publish fse 2.0.0 first under its separate approval. Recheck framework versions,
   package metadata, offline tests and real-model examples before publication.
4. Only with explicit publication approval, create a matching version tag on a
   commit contained in master. The workflow checks tag/version equality and
   master ancestry, builds sdist and wheel and runs `twine check`. A separate
   publish job waits for environment approval and uses PyPI trusted publishing.

Only the publish job receives `id-token: write`. It downloads this run's package
artifact and executes no checkout code. Package tags do not match fse's `v*`
release trigger, and fse tags do not trigger package releases. There is no manual
or pull-request publication trigger. No API tokens or package credentials are used.

Sources: [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
and [GitHub environment protections](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments),
checked 2026-11-01.
