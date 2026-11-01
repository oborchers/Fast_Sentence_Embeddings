# 100 million text showcase, 1 November 2026

These three metadata files are byte-identical copies from the [committed Pod 3 result folder](https://github.com/oborchers/Fast_Sentence_Embeddings/tree/cbcad5a5d1724456c2a712ffae4694a2dc29b86f/benchmarks/baseline/results/dr151-pod3-20261101/showcase-100m).
No model tensors or output-vector files were copied. The original downloaded metadata and committed files match byte for byte.

## Approved headline and caption

Compute nearly 3 million text embeddings per second.

100 million short texts (28.7 tokens on average) embedded and written to disk in 35 seconds, int8, 32 vCPU AMD EPYC 7713, RunPod, 1 November 2026.

The primary observed rate is **2,832,638.6960565858 texts/s**, displayed as 2,832,638.70 when an exact rate is needed.
The primary input-read-through-sink-close duration is **35.30277269007638 seconds**, displayed as 35.30 in evidence.
The caption's 35 seconds and 28.7 tokens are the maintainer's approved rounded wording. It does not say “under 35 seconds”.

## What ran

POTION-8M revision `bf8b056651a2c21b8d2565580b8569da283cab23`, a **float32 model table**, encoded a native text file with 32 fse workers.
The output was an **int8 ShardSink**, with durability `none`. The caption's int8 describes output storage, not an int8 model table.
The cell is `showcase/100m/int8/none/32`, series `85778f47f4e04ae6a3c5f2531abdc9c8`.
The assigned CPU budget was 32 vCPU on an AMD EPYC 7713 host. Source execution commit is `3ca670db94ae8924bf6e4c59fc9d4b03b364c633`.
The loaded fse runtime fingerprint is `xxh3-128:d2dc5ced22cb83dc2df3f15624ad2e69`.

Timing includes reading the input, encoding, writing and closing the sink. Model loading, sink construction and checksum/output verification are excluded.
Durability `none` does not wait for crash-safe storage. Both runs published all **100,000,000 rows**, with zero durable rows reported under that mode.
Each wrote **26,400,136,189 bytes**. The recorded output digests agree, and the source verifies row count, stored checksums and finite decoded values after timing.
Input checksum verification warms the filesystem cache before timing. fse does not use a sentence-result cache in this WordPiece path.
This measures the recorded file-to-vector workload, not a general network service or unique-text corpus.

## Repetition and length

The input is **repeated and shuffled WikiText-103 passages**, from the pinned train split of Salesforce/wikitext.
It contains **3,755,995 distinct texts**, repeated about **26.6 times** and shuffled with NumPy default_rng seed 0.
There are 3,863,020 qualifying source positions; natural duplicates make that count different from the distinct-text count.
The mean is **28.73027215 consumed POTION tokens per text**, from the exact total of 2,873,027,215 tokens over 100,000,000 rows.
These are tokenizer pieces after the recorded limits, not whitespace-separated words.
The input file contains 13,639,186,350 bytes. Use `bytes_written` for the output size; the operation's `file_bytes` field is overwritten with the output count.
The input identity/checksum is retained separately in `input-identity.json`.

## Repeatability and scope

The maintainer authorized **one primary timed run plus one A/A repeat** for this showcase, an explicit exception to the three-pair matrix protocol.
The repeat took 33.771484442055225 seconds. The symmetric A/A band is 0.9566241365383857 to 1.0453426396061603, a **4.534263960616025%** spread.
This is within the 10% band. Use the primary rate, not a pooled two-run median or the faster repeat.
It is not a three-pair matrix cell, a library speedup, or evidence for a 32/16 scaling decision.
The separate one-core/16-core/32-core panels still require their own accepted paired cells and direct 1/16/32 selection cells.

The independent metadata check recomputed texts/duration, token totals/rows and the A/A ratio, verified both output row counts/digests, and checked the table/output dtype distinction.
No new inference, model loading or local timing was performed for that check.

Sources: [full selected journal](speed-results.jsonl), [source summary](summary.json), [input identity](input-identity.json),
[writer boundary](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/3ca670db94ae8924bf6e4c59fc9d4b03b364c633/benchmarks/quality/large_write_paths.py),
[post-run verification](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/3ca670db94ae8924bf6e4c59fc9d4b03b364c633/benchmarks/quality/pod_large_child.py).
