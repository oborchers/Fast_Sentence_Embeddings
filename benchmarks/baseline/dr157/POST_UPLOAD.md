# Public Hub readback protocol

This worker has performed no Hub writes. Oliver must approve the exact cards,
repository names, licenses and complete file inventory in the immutable publication
manifest before the coordinator creates any PUBLIC repo. There is no private stage.
FastText retains its separate publication hold.

After the approved uploads finish, the coordinator supplies a JSON receipt:

```json
{
  "publication_manifest_sha256": "SHA256 of the approved publication-manifest.json",
  "approved_repositories": ["fse/glove-wiki-gigaword-300-raw"],
  "repositories": [
    {"repo": "fse/glove-wiki-gigaword-300-raw", "revision": "full 40-character Hub commit"}
  ]
}
```

The lists must contain every approved uploaded repository exactly once. Approval is
an external human decision, never inferred from a receipt or a manifest flag.
The coordinator uploads the exact payload, including the reviewed `.gitattributes`.
The remote file list must equal the complete local file list.

Run with the same fse 2.0 CPU installation used for the local references:

```sh
.venv/bin/python benchmarks/verify_published_hub.py \
  --manifest /Users/oliverborchers/fse-dr157/publication-manifest.json \
  --upload-receipts /Users/oliverborchers/fse-dr157/upload-receipts.json \
  --output /Users/oliverborchers/fse-dr157/public-readback-01 \
  --scratch /Users/oliverborchers/fse-dr157/public-readback-scratch
```

The runner uses one new child process and an empty `HF_HOME` per repo. All HF caches
point into that owned directory. Token use is disabled. It verifies public visibility,
the uploaded revision, exact file inventory, sizes and SHA256, then executes the
literal approved Python card example through the public API. The downloaded
preparation receipt supplies the folded SIF component count. Metadata must equal the
local build, including the encoder fingerprint.

The probe uses 256 fixed STS-B pairs (512 raw sentences), seed 0, saved tokenization,
float32 finite output and the expected shape. The allowed Pearson difference is
0.001 points on a 0 to 100 scale. CPU vectors must also match the local reference
bit for bit. The model never fits on these texts.

After each child exits, the runner reads back its scratch ownership and exact
contents, deletes only that owned cache, confirms removal and records allocated
bytes removed. Evidence and logs stay outside scratch. A failure stops that repo's
checks, records the failed stage and allows the remaining repos to be checked.
No failure triggers Hub deletion, overwrite or repair.

Commit `evidence.json`, `report.md`, per-repo JSON and logs to this evidence folder.
Report the complete per-repo table and the evidence commit to fse-main. Actual
post-upload checks remain pending until approval and completed upload receipts exist.
