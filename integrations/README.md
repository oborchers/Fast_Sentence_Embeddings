# Prepared fse integrations

These three unpublished packages are prepared for the maintainer's layout decision.
Do not merge or publish them before that decision.

| Package | Interface | Runtime dependencies |
|---|---|---|
| langchain-fse | `FseEmbeddings` | fse, langchain-core |
| llama-index-embeddings-fse | `FseEmbedding` | fse, llama-index-core |
| fse-haystack | `FseTextEmbedder`, `FseDocumentEmbedder` | fse, haystack-ai |

Each folder has independent build metadata, tests, MIT license and documentation.
Each framework runs in its own environment. fse core gains no dependency.
Package READMEs document the tested minimum framework versions and official guides.
No package imports code from another integration folder.

If these folders move to separate repositories, move each folder's contents as its
repository root. Add its own CI, release permissions, tags and package credentials.
Update repository links when those repositories exist. Do not create them yet.

If they stay here, retain the `integrations` guardrail path and hook coverage.
Before merging, add isolated package CI jobs for the documented test and build
commands, with separate dependency resolution and minimum-version checks.
Keep publication independent from fse's `v*` tag workflow. Select package-specific
tags and explicit release approval before implementing a publication workflow.

The only layout-dependent enforcement changes in this draft are the guardrail
path and pre-commit guardrail trigger. Root `make check` continues to check core.
Package tests run separately with their own dependencies. This draft does not add
publication automation or framework listing submissions.
