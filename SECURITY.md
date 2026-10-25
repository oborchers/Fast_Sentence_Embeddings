# Security Policy

## Supported versions

Only the latest release of fse receives fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/oborchers/Fast_Sentence_Embeddings/security/advisories/new), not as a public issue. fse is in maintenance mode, so responses are best effort.

## Loading models

`Vectors.from_pretrained`, `FTVectors.from_pretrained` and the `load` methods of all models use pickle through gensim. Unpickling a file can execute arbitrary code, so only load models from sources you trust.
