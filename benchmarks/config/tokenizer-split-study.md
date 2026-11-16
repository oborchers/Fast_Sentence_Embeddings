# DR-445 (Tokenizer- und Laufzeitanteil von potion-base-8M auf dem M1 aufschlüsseln)

Question: which tokenizer and runtime costs should guide the next per-core work?
Claim classes: descriptive timing and correctness probes. Local Apple M1 only,
`publishable: false`. No Linux comparison run. No product changes.

Decision: recommend an order for hash fusion, fixed sentence costs, vocabulary
layout and chunk locality work. Use exclusive production sample categories:
char cut/validation, added tokens, normalization, pre-tokenization, hash, probe,
trie, sentence bookkeeping, array copies and unattributed. Never sum inclusive
function shares. Keep noinline diagnostic shares separate from production.

Workload: pinned potion-base-8M, float32 table/output, one worker, STS-B timing
texts in source order, 32 complete corpus cycles (552,192 rows). Reuse the
pooling study's fixed 31/25/12 tokenizer/pooling/encode iterations. Five fixed
repetitions with two warmups each. At least two seconds per measured group.
The chunk adapter uses the same pipeline with automatic, 64 KiB and 16 KiB
input byte targets. It checks all output bits against public encode outside
timing, records actual task ranges, and includes scan/planning in every call.
Chunk size changes overhead and locality together. It cannot isolate cache misses.

Boundaries: normalization plus pre-tokenization >=40% supports fusion priority;
hash/probe >=30% raises vocabulary layout priority. High fixed sentence costs
instead favor fixed-cost work. Recovering half the historical 0.114 s residual
with smaller chunks is only a preliminary signal for locality work.

Budget: 45 minutes including preparation and load/lock waits, starting
2026-11-16 16:30 CET, ending 17:15 CET. Estimated cost USD 0. Stop safely at the
budget boundary, preserve partial evidence, report inconclusive; no automatic
extension or retry. Initial one-minute load 8.15; timing admission requires <5.
External data: /private/tmp/dr445-profile. Previous pooling data at
/private/tmp/dr440-profile is read-only input. No raw profiles or models in Git.
