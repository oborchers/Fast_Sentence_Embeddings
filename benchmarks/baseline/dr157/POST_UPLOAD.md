# Public Hub readback protocol

This worker has performed no Hub writes. Oliver approved content, repository names
and licenses of all 30 repos under Option B, including FastText. Upload waits until
release 2.0.0 and regenerated cards. Every repo is PUBLIC from creation.
Run `RELEASE_CARDS.md` after the release and upload the resulting release payload.
The manifest and local card paths below must then reference that new output.
The explicit Option B decision supplies authorization; a manifest flag does not
supply or revoke it. Changes beyond that approved scope require review.

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
/Users/oliverborchers/fse-dr157/release-env/bin/python benchmarks/verify_published_hub.py \
  --manifest /Users/oliverborchers/fse-dr157/release-2.0.0/publication-manifest.json \
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

Hand raw `evidence.json`, reports, per-repo JSON and logs to dr151-compare, the sole
writer of `benchmark-evidence-2.0`, for archival under
`benchmarks/baseline/results/dr157-20261101/`. Verify the published archive bytes.
Only a compact summary and immutable archive reference belong in this folder.
Report the complete per-repo table and the archive commit to fse-main. Actual
post-upload checks remain pending until release 2.0.0, release cards and completed
upload receipts exist.
