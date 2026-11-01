# Evidence cleanup size report

The archive retains every original raw file. Compact summaries preserve their selected measured values without recalculation. The original eight 1.1 baseline result files and fixture generator remain on the code branch.

The five existing master studies total 1,067,802 raw bytes. Their summaries total 95,533 bytes. Each summary is below 50,000 bytes and links its exact original file by evidence commit, path, size and SHA256. Verification receipt: `benchmarks/baseline/dr151_existing_studies_migration_20261101.json`.

The following counts describe pinned changes against each branch's merge base. Head-file bytes sum all changed files present at the pinned head, including code, tests and summaries. Net tree bytes subtract the corresponding base file sizes. They do not describe Git history storage or compressed downloads.

| Study branch | Head | Merge base | Changed files | Added lines | Removed lines | Changed head-file bytes | Net tree bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Comparison | bc1d79b0aef6d88590629825d9bcdcae387f110d | 81e64c0d5f92ea6270522f0d97a009979fc678d3 | 205 | 33,190 | 48,743 | 1,783,980 | 536,587 |
| README | 219b49831dd43a7b03f366ee65eb8e7a29925848 | dd08673d62459d72979f865307417e2a526f8da6 | 44 | 6,370 | 287 | 1,169,325 | 1,042,318 |

Later provenance reports add small text files beyond these pinned snapshots. Raw history on working branches does not reach master through the coordinator's approved squash merges. The orphan evidence branch must never be merged.

Fifteen focused evidence tests, Ruff and guardrails passed after the five-study change. A complete local make check, model run and timing run were not performed under the Mac resource restriction. Remote CI is separate and was still pending when checked.

Three completed review checkouts were removed after checking their exact path, detached commit and clean state. Their reports remain preserved. The removed regular files totalled 31,992,348 logical bytes. The shared machine and APFS allocation prevent attributing a precise free-space delta. No shared cache, result or report was deleted.
