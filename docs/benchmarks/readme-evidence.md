# README evidence

## Headline: AMD EPYC 9575F (fse 2.0)

The headline uses the approved 16-worker file-to-vector run on AMD EPYC 9575F with 32 allocated vCPU, RunPod, 2 November 2026.
It embeds 100 million sampled, repeated and shuffled WikiText-103 sentences with POTION-8M (256 dimensions).
There are 3,755,995 distinct source sentences, repeated about 26.6 times.
Average text length is 28.73384302 POTION tokenizer IDs (25.57447294 words).
The model table is float32 and output storage is int8.

Three encoding-and-write calls took 13.40701300802175, 13.48732629197184 and 13.458833114011213 seconds.
The median is 13.458833114011213 seconds, or 7,430,064.638805557 texts/s.
The README rounds the duration to 13.46 seconds and says more than 7 million, without rounding the rate up.
The same machine's 32-worker run took a median of 16.75060166104231 seconds, rounded to 16.75 in the footnote.
The calls include writing through sink close, but exclude model loading and the later explicit filesystem sync.
Durability is `none`; the headline does not claim crash-safe storage at 13.46 seconds.

The collected 16-worker manifest SHA256 is `5fb185267692e6b14cb3831972b4324f9bd599a922a72a37ac3c7bd44d042172`.
The 32-worker manifest SHA256 is `079581ae37e056565d2d60ec01061d65b6fa0fc7ead77bc1037787a092ba8f31`.
Both manifests and all listed member hashes were checked locally against the collected results.
The fixed [sentence-run archive](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/608cf02a9557ef8bade83ae6c0c2fadd5b8d5012/archives/sentence-speed-2026-11.tar.gz) on `benchmark-evidence-2.0` contains both collected studies and `audit/headline-summary.json`.
Archive SHA256: `443e7ff798d38b41d14e3952128d8080b2d698e6005991b147c34a69ea88f1c7`.

The library comparisons, the dtype table and the large runs come from AWS EC2 measurements of fse 2.1, described below.
They use other machines and workloads than the headline. Their rates are not combined with it or used to derive a speedup.
Missing cells are not filled from an older run.

## AWS machines for fse 2.1

All 2.1 rows were measured on 14 November 2026 in AWS eu-west-1 with the two central measurement entries and the publication protocol of [the measurement guide](measurement.md#aws-ec2-for-readme-speed-numbers).
Each allocation ran the pinned `python:3.13` image in Docker with 32 CPUs (`--cpus=32 --cpuset-cpus=0-31`), exactly 128000000000 bytes of memory, 64 GB of shared memory and the allocation's machine check, including the three-minute host quiet check.

| Machine | CPU | Allocated CPUs | Storage | README rows |
|---|---|---|---|---|
| m8a.8xlarge | AMD EPYC 9R45, 32 cores without SMT | 32 | gp3 EBS | Library comparisons, dtype table |
| m8id.8xlarge | Intel Xeon 6975P-C, 16 cores with SMT | 32 | local NVMe instance store | Large runs |

The write cells run on m8id.8xlarge because the gp3 volume of m8a.8xlarge limited the 100-million write cell to about 140 MB/s.
Every measured fse call lasts at least two seconds. Each rate is the input count divided by the median of three calls.
The speedup is the median of the three paired time ratios, so it need not equal the quotient of the separately rounded rates.

## Library comparisons: m8a.8xlarge

`readme-speed.json` keeps the eight library pairs of the README graphic under `comparisons` and the two bekko-static-embedding-v1 pairs under `bekko`.
Each pair uses the same pinned model, input and normalized float32 output for both libraries.
No peak-RSS comparison is made.

| Cells | Texts | Input SHA256 |
|---|---:|---|
| `libraries-model2vec-1`, `libraries-rust-1`, `libraries-fastembed-1` | 2,500,000 | `b37670ab565c659d0c22bc73ee4e35cbefdf087da0d68b61918de6a3c962ac0e` |
| `libraries-st-1`, `bekko-st-1` | 2,000,000 | `fa525410db83d7405759ba34b06288d769286ab6730e5ac4e19167e84cba5024` |
| `libraries-model2vec-16`, `libraries-rust-16`, `libraries-fastembed-16` | 20,000,000 | `16a11230cc72f5a68c1d474d996ccd98ca88c0aca15cf7ee9078d056b0eb6be6` |
| `libraries-st-16` | 10,000,000 | `1bd11e132c8ec642f5ba9019c7a01d862e6460b89d46f6ca0400742caf98070e` |
| `bekko-st-16` | 8,000,000 | `5a673189202a6901a6eb70a96a375c20225d97cb42c5adcd62da2d11ceb4a724` |

All inputs are sampled and shuffled WikiText-103 sentences with about 26 words on average.
The Sentence Transformers inputs include eight mandatory rows with unknown tokens, so a larger input is not a prefix of a smaller one.
Sentence Transformers with one worker uses the default batch size 32. With 16 workers it uses the recommended reused CPU process pool with batch size 1024.
Model2Vec with 16 allowed CPUs uses its default thread path. This is the library's default scheduling, not sixteen fully occupied worker processes.
POTION-8M has 256 output dimensions. static-retrieval-mrl-en-v1 and bekko-static-embedding-v1 have 1024; their absolute rates are not compared as if they were POTION-8M.
fse loads bekko-static-embedding-v1 with its float16 table; both libraries return float32 vectors.

The main allocation (manifest `229b073c0e4711b26ecfc0474074933f0a761526c62edf011845c58accd619a0`) measured 27 cells.
Its `libraries-st-16` and `bekko-st-16` cells at 5 million texts are excluded: one hung on an 8 GB shared memory, the other had fse calls below two seconds.
Their replacements come from a second allocation on the same instance with exactly pinned larger inputs (manifest `c2f49679e7430d28a81cc1b934e1e32c2b0669a5abca67760631823f0fc4fa6d`).
`heads-sinks-16` is excluded too: its fse calls took 2.15 seconds in the median, and its protocol fixes the input at 10 million texts. It has no README row in 2.1.

CPU and memory profiles of the representative deciding configuration (`libraries-model2vec-16`) are valid: `profile-libraries-cpu` with py-spy and `profile-libraries-memory` with memray.
The `heads-sinks` profiles failed and support no README row.

## Dtype table: m8a.8xlarge

The six values come from cells `dtype-paragraph-float32-1` through `dtype-paragraph-int8-16` of the main allocation.
Each cell runs a second fse arm as control; the control ratios lie between 0.997 and 1.004.
One worker uses 600,000 WikiText-103 passages (input SHA256 `19f2bf81…`), 16 workers use 5,000,000 (`6d7aee48…`); the full digests are in `readme-speed.json`.
Passages average 119.0 words. The model is POTION-8M, the output is normalized float32, the model table precision varies by row.
These values replace the 2.0 dtype table from AMD EPYC 9965. Machine and input differ, so the two are not compared.

## Large runs: m8id.8xlarge

| Cell | Texts | Median | Min | Max | Texts/s |
|---|---:|---:|---:|---:|---:|
| `headline-100m` | 100,000,000 | 15.294 s | 15.211 s | 15.304 s | 6,538,517 |
| `headline-1b` | 1,000,000,000 | 217.385 s | 216.791 s | 218.763 s | 4,600,136 |
| `headline-100m-predict` | 100,000,000 | 5.837 s | | | 17,132,557 |
| `headline-100m-predict-proba` | 100,000,000 | 5.707 s | | | 17,521,793 |

All four cells use 16 workers and sampled, repeated and shuffled WikiText-103 sentences with 28.73 POTION tokenizer IDs on average.
The 100-million input has SHA256 `2765496afe36095473776599284b29462fb1798744b3b25abbc06e005fa167ef`; the one-billion input `8b00be8ea93a08628da5a57bbb35b69577f9d31fd17828a8fe46fea6a8e1b612`.
The write cells use POTION-8M with a float32 table and int8 output and durability `none`. Each call includes writing through sink close.
The prediction cells load [fse/bekko-civil-comments-toxicity-classifier](https://huggingface.co/fse/bekko-civil-comments-toxicity-classifier) at revision `2400389fa61db6fd2ec7c7d8f7561d3b0ec53c0c`, a standalone head with its own folded table, labels `toxicity` and `insult`. A sample of each output matches the cpu and numpy backends bit for bit.
The write cells come from the headline allocation (manifest `657102c3b2d50c22f8680d6848ad94608f29ba128e3c00406070c289b606bc48`). Its prediction cells failed because their warmup on 20 million texts took less than two seconds.
The prediction rows come from a second allocation on the same instance with a warmup of 50 million texts (manifest `e2a10358b61f0c48cac5a1cd27c59def2f5cb71dbb13dc0f92e80be0e9779d02`).

## Evidence archives for 2.1

The collected results are kept as compact archives without the sampled output vectors (`.npz`). Each archive contains a member manifest with the SHA256 of every file.

| Archive | Bytes | SHA256 |
|---|---:|---|
| `aws-readme-2026-11-14.tar.gz` | 2,732,863 | `bf9fba1009d5e13a6915a61584c53cc718a219b4127c8cd91746d2766bf97925` |
| `aws-readme-profiles-2026-11-14.tar.gz` | 8,184,576 | `1e7f26664f50e718eba3cc0adb7848a8b60184f46476f04518a853db727acee6` |
| `aws-st16-rerun-2026-11-14.tar.gz` | 335,390 | `c11a393f9f6d8ece44f045f012871f2b30ffb206aa75311cebecbf34193836f7` |
| `aws-headline-2026-11-14.tar.gz` | 879,426 | `e629d3557ab9ff2aac5e9c7056b6182926fb045c611b793edb895358afbb2808` |
| `aws-headline-predict-2026-11-14.tar.gz` | 205,303 | `c4ea4520ace22d384f5a8e6776dcb4c8291b80bbd5f2ee05037bab1ce3c6da48` |

They are not yet published on an evidence branch.

## Quality and file size

[readme-quality.json](readme-quality.json) is the single local summary for all eleven supported models in the README.
It records unrounded 40-task means, the model identities and exact float32 file sizes used for display.
The scores use MTEB 2.26.25, measured 1 November 2026. MTEB(eng, v2) is 40 of 41 tasks because MIND was not computed.
The summary preserves MIND's recorded memory-limit failure (two attempts, ENOSPC and status 137).
It pins the compact quality source and archived source records through their SHA256 identifiers.

Historical SIF-10 benchmark scores use the WikiText-103 fit. Linked Hub SIF-10 models use wordfreq weights and a FineWeb fit.
Those model versions have separate results on their cards. File size is not process memory.

## Example verification

Run the displayed Python examples unchanged with the documented dependencies and real models. Each README section is one run in its own working directory; the examples of "Linear heads" build on each other and share one directory and one interpreter. Examples that need fse 2.1 run against the checked-out master until 2.1 is released.
Check output shape and dtype, the saved int8 table, classifier labels and save/reload behavior.
For folding, check `classifier.normalized is False`; a saved classifier must also load without scikit-learn.
These are documentation checks, not additional library unit tests or speed benchmarks.
The last complete real-model execution is recorded in the PR description. Library API/conformance tests remain in the normal suite.
