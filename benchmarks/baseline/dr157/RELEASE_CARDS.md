# Option B: release cards after v2.0.0

Oliver approved the content, repository list and licenses of all 30 repos, including
FastText. DR-146 (Über FastText in 2.0 entscheiden (12f)) is Done. The six FastText
`publication_hold` values are null. Upload waits until the 2.0.0 release and release
cards. This worker performs no Hub writes.

Current development-card manifest SHA256:
`ac85cae41b05788bc278d68fd3231489de1615c7c15b8b79229844c3d0dabe69`.
Only the six `publication_hold` fields changed in that manifest. All 135 payload
file specifications, card bytes and model bytes remain unchanged.

After the published release, run from this repository with the original benchmark
fixture commit available. Use a separate environment containing the actual release:

```sh
uv venv --python 3.13 /Users/oliverborchers/fse-dr157/release-env
uv pip install --python /Users/oliverborchers/fse-dr157/release-env/bin/python \
  'fse==2.0.0' 'scipy==1.18.1'
/Users/oliverborchers/fse-dr157/release-env/bin/python benchmarks/release_hub_classics.py \
  --manifest /Users/oliverborchers/fse-dr157/publication-manifest.json \
  --manifest-sha256 ac85cae41b05788bc278d68fd3231489de1615c7c15b8b79229844c3d0dabe69 \
  --output /Users/oliverborchers/fse-dr157/release-2.0.0
```

Reserve the shared heavy-work lock and check at least 25 GiB free before this step.
No release tag, package installation or real release rendering is performed now.
The runner refuses a development version and requires the remote `origin` tag
`v2.0.0`. Both the distribution version and `fse.__version__` must equal `2.0.0`.
The input manifest must match the supplied exact SHA256.

Before creating output, the runner checks all 30 repositories and every source
file SHA256. The installed release must load each local model through
`Encoder.from_pretrained`, match its metadata and reproduce the frozen 512-sentence
probe. This verification does not fit or change a model.

The new release payload is an exclusive directory. Non-README files are hardlinks
to the original files, requiring the same filesystem. No copying or writing of a
model occurs. The runner replaces the development Usage notice with the released
2.0.0 requirement and the Git install pin with `pip install fse`. The literal Python
example, measured claims, licensing and fit provenance remain unchanged. The writer
line continues to quote the immutable artifact's historical `fse.created_by` value.
It does not falsely claim the model was rebuilt by 2.0.0.

The runner recomputes output SHA256 and requires that only `README.md` differs.
It preserves metadata, fingerprints, source pins, probes and every non-README file
hash. Model files remain immutable after rendering because release views share
those inodes with the original models. Do not edit either view's model files.

Outputs under `release-2.0.0`:

- `models/<repo>/`: the upload payloads with new README files.
- `cards/*.md`: the new card previews.
- `publication-manifest.json`: new paths, README hashes, sizes and release identity.
- `publication-inventory.md`: all new file sizes/SHA256 and totals.
- `release-transition.json`: basis/final manifest SHA256, remote tag commit and a
  per-repo proof of unchanged model SHA256 and `changed_payload_files: [README.md]`.

The old payload and old manifest remain intact. A failed output is never reused or
removed automatically. Preserve any failure evidence, inspect the owned output and
clean unneeded hardlinks using the binding targeted-cleanup protocol.

The manifest flag does not grant upload permission. The coordinator applies Oliver's
Option B decision after release. It uploads all 30 PUBLIC repos from the new release
payload, then supplies upload receipts referencing the NEW manifest SHA256.
Use that new manifest with `verify_published_hub.py`. Follow `POST_UPLOAD.md` for
fresh per-repo download, SHA256, literal card example, metadata/probe verification,
cache cleanup and committed JSON evidence. No error triggers Hub repair writes.
