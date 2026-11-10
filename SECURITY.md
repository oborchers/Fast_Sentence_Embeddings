# Security Policy

## Supported versions

Security fixes go to the latest 2.x release. The branch `1.x` (fse 1.1.x) receives bug and compatibility fixes.

## Reporting a vulnerability

Please report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/oborchers/Fast_Sentence_Embeddings/security/advisories/new), not as a public issue. Responses are best effort.

## Loading models

fse 2 loads no pickle files. fse 1.x loads some models through gensim, which uses pickle: load only files you trust. Files you load with gensim yourself can also run code.
