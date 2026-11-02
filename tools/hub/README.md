# Classic Hub publication

`run.py` is the entry point for preparation, byte checks, explicit publication and
anonymous readback. Run it from the repository root with the project environment.
The published manifest is frozen at SHA256
`04a4453fecdff450f4bc48b1e46710f4bd12c29f38b411ada405442bb6aade34`.
Its historical `local_folder` fields are evidence, never input paths for this tool.

## Local preparation

`--models` identifies the read-only preparation intermediates: the ten prepared
float32 encoder folders, the historical dtype headers, build records, fit receipts
and verification measurements from the original preparation. The build does not
refit SIF or rerun benchmarks. It loads each prepared float32 table, recomputes
float16/int8 storage when requested and serializes a new model. Historical writer
provenance is preserved, so a new checkout can reproduce the original bytes.
Cards are generated from model metadata and recorded measurements, without
reading previous cards. Config, attributes and preparation receipts are generated
again. Every generated file must match the frozen manifest by size and SHA256.

```sh
uv run python tools/hub/run.py cards \
  --manifest tools/hub/published-manifest.json \
  --models "$HOME/fse-dr157/models" --output /tmp/classic-cards

nice -n 10 uv run python tools/hub/run.py build \
  --manifest tools/hub/published-manifest.json \
  --models "$HOME/fse-dr157/models" --output /tmp/classic-rebuild \
  --discard-verified-output
```

The output directory must be new. Without `--repo`, all 30 entries are selected.
Use `--repo fse/glove-wiki-gigaword-300-raw-int8` for one entry. Without
`--discard-verified-output`, payloads remain for a later `check` or `upload`.
With that switch, each payload is checked, recorded durably and removed before
the next starts. Failed payloads remain for inspection. No source is moved,
linked, overwritten or deleted. Allow the next payload's size plus 8 GiB of disk
reserve. Retaining all payloads needs 39,426,332,256 bytes plus that reserve.
No throughput measurement runs during preparation.

## Publication and readback

`check` takes generated payload folders under `--models` and writes a receipt
under the new `--output` directory. `readback` anonymously resolves each public
repository to an exact revision, downloads into a fresh temporary cache, compares
all files and loads both default normalization and `normalize=False`. It also
loads both examples through the public repository ID in a separate cache and
compares the recorded 512-sentence probe bit for bit. This probe reads the original
STS-B fixture from immutable Git history, so use a checkout with history. The Hub
mirror rounds some labels and cannot reproduce the original Pearson exactly.
Only its own temporary cache is removed.

`upload` additionally requires `--upload`. It refuses existing repository names,
including the 30 already published repositories in this manifest. It creates new
public repositories only after checking all local payloads and remote names.
It records each creation and commit before continuing, then performs readback.
Before each creation the ledger records that its outcome is unknown. A lost
response never proves that no remote write occurred. A failed run never deletes
or repairs a remote repository. Inspect the durable
receipt before any retry. Hub authentication belongs to the installed CLI or its
approved environment, never to a file in this repository. The default local
commands perform no Hub writes.

The existing preparation and quality-comparison math remains under `benchmarks/`
for the measurement tools that import it. The superseded publication scripts were preserved in a verified local archive
before removal. The maintainer requested local archival only.

## Recorded validation

`validation.json` records the maintainer-approved five-repository sample, covering
all five sources, raw and SIF-10, and float32, float16 and int8. Every file matched
the frozen manifest. Each newly built payload was removed after its receipt was
secured. All 30 cards matched independently. The complete 30-payload rebuild was
not run, following the maintainer's narrower validation instruction.

The recorded anonymous public readback of GloVe raw int8 also passed, including
all file hashes, both Hub loading modes and the exact 512-sentence vector checksum.
One independent Codex review found five issues. The follow-up commits protect LF
manifest bytes, restore all STS-B readers, use the public Hub loading path, record
uncertain creation outcomes and retain the Hub CI path trigger.
