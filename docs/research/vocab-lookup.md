# Static vocabulary lookup for fse: research report

## 1. Minimal perfect hashing (MPHF) for static string sets

An MPHF maps the n keys to 0..n-1 without collisions. For string-to-id lookup you also store a permutation `id[p]` (u32), usually a fingerprint per position, and you compare the key bytes once against `vocab[offsets[id]]`. Unknown keys get an arbitrary position, so verification is mandatory.

- **PtrHash (Groot Koerkamp, 2025; Rust, MIT, no C bindings).**
  - On 300M random strings of 10 to 50 bytes, built on 6 threads:
    - default: 2.40 bits/key, 32 ns/key to build, 37 ns per query (23 ns streaming);
    - fast: 2.99 bits/key, 27 ns/key, 33 ns per query (16 ns streaming);
    - compact: 2.12 bits/key, 63 ns/key, 35 ns per query.
  - On integer keys a query takes 12 ns in a loop and about 8 ns streaming. The paper says string hashing is the bottleneck.
  - The README reports 3.2 to 3.7 ns/query for 10M integer keys (FastPtrHash and DefaultPtrHash, plain loop).
  - Same 300M-string table, for comparison:
    - PTHash: 2.17 to 3.19 bits/key, 403 to 765 ns/key to build, 77 to 156 ns per query.
    - PHOBIC: 1.86 to 4.14 bits/key, 62 to 446 ns/key, 108 to 179 ns.
    - CHD: 2.27 bits/key, 1059 ns/key single-threaded, 542 ns.
    - FMPH: 2.80 to 3.40 bits/key, 168 to 236 ns.
    - SIMDRecSplit: 1.81 to 2.96 bits/key, 258 to 310 ns.
    - Consensus-RecSplit: 1.49 to 1.58 bits/key, 528 to 565 ns.
  - At 300M keys none of these structures fits in cache. At 3M keys PtrHash is about 0.9 MB (my arithmetic: 2.4 bits × 3M) and should stay L2/L3-resident.
- **BBHash (C++, MIT).** About 3 bits/key. The README's sample run builds 100M keys in 10.25 s on 8 threads and reports 243.84 ns average per query, on unspecified hardware. Strings are supported only on the `alltypes` branch, which the README calls slower.
- **RecSplit.** 1.56 bits/key and under 2 µs/key to build at that setting (abstract; the source writes 2 ms, which I read as a typo). The abstract gives no lookup time. The `sux` implementation is GPLv3 with the GCC runtime exception, which makes it unattractive to vendor.
- **PTHash (C++, MIT).** Supports strings (its README example uses 39M URLs). The README itself gives no numbers.
- **cmph (C, LGPL/MPL 1.1).** Contains CHD (about 2.07 bits/key), BDZ (about 2.6 bits/key), BMZ, CHM and FCH, all built in linear time. It publishes no timings. It is the only C library here, but the license is the least convenient.

**Takeaway:** no permissively licensed C MPHF exists. PtrHash's algorithm (buckets, 8-bit pilots, plain or Elias-Fano remap) is small enough to reimplement. At 3M keys and 32 ns/key, the build takes about 0.1 s.

## 2. Swiss tables and compact static layouts

- **Abseil SwissTable.** 1 control byte per slot: one bit of state plus a 7-bit H2 taken from the hash, with the remaining 57 bits (H1) choosing the position. Lookups scan 16 control bytes with SSE2 (`cmpeq` and `movemask`), and every match is then compared in full. The design page states no load factor.
- **hashbrown.** Its source caps load at 7/8 (`bucket_mask_to_capacity` returns `(buckets/8)*7`). Groups are 16 bytes with SSE2 and 8 bytes with NEON and the generic word-sized version.
- **Folly F14.** Chunks of 14 slots, each with a 7-bit tag (1/128 false-positive chance per tag), plus an overflow counter per chunk. Maximum load is 12/14. The F14Vector variant keeps only a 4-byte index in the hash array, with 12 slots per chunk filling one cache line and a 10/12 load limit. The article gives no benchmark numbers.

**Compact static variant for fse:** 3M keys in 2^22 slots gives load 0.715.

- 1-byte tags plus u32 indices: 4 + 16 = **20 MB**.
- 8-byte slots of {u32 id, u32 hash fingerprint}: **32 MB**.

Both reuse the existing vocabulary bytes and offsets. Because the table is built once, inserting keys in hash order gives the Robin Hood invariant for free, so a miss can stop early. That is my reasoning, not a sourced measurement.

## 3. What existing libraries do

| Library | Structure (verified in source) |
|---|---|
| HF tokenizers (WordPiece, WordLevel, BPE) | `AHashMap<String,u32>` (hashbrown with aHash). BPE merges are `AHashMap<(u32,u32),(rank,new_id)>`. WordPiece's greedy longest match calls the vocab hash once for each shrinking substring. The Unigram trie is a nested `AHashMap` per node. |
| tiktoken | `FxHashMap<Vec<u8>, Rank>`, looked up with byte slices during merging. |
| fastText | `word2int_`: an int32 array of fixed size 30M (120 MB), 32-bit FNV-1a hash, linear probing, comparing `words_[id].word` (a std::string). |
| SentencePiece | `PieceToId` uses `absl::flat_hash_map<string_view,int>`. Unigram uses a darts-clone double-array trie (`commonPrefixSearch`) for prefix matching. |
| darts-clone (C++ header, BSD-2) | 4-byte units. On EN3 (13.6M English unigrams), the trie is 193 MB, takes 421 ns/key to build, and an exact lookup costs 453 ns with random order and 40.5 ns sorted, on a 2009 Core 2. |
| model2vec | Delegates to HF `tokenizers` (`encode_batch_fast`). |
| gensim | `key_to_index` is a Python dict. |
| spaCy StringStore | 64-bit MurmurHash keyed into preshed `PreshMap` (linear probing, resize at 60% load). It never verifies the bytes, so a 64-bit hash collision counts as a match. That is not exact rejection. |
| fst crate (BurntSushi) | Ordered map to u64. Gutenberg terms (3.54M keys, 41 MB raw) compress to 22 MB and build in 2.04 s sorted or 5.58 s unsorted. The article states no lookup ns; lookup cost grows with key length, not set size. |

None of them publishes memory per word or lookup ns for its hash map. A per-bucket figure for HF can be derived, not cited: a 24-byte String plus a u32 pads to 32 bytes, plus 1 control byte, plus a heap allocation per word.

## 4. Fast rejection and short-string tricks

- **Tags or fingerprints in the first cache line touched.**
  - A 7-bit tag gives 1/128 false positives per occupied candidate (F14).
  - A 16-bit fingerprint gives about 1/65536.
  - A 32-bit fingerprint makes the trip to the vocabulary bytes vanishingly rare for unknown words.
- **Length check.** Compare a stored length (or length bits folded into the tag) before `memcmp`.
- **Separate filters.**
  - Binary fuse filters are within 13% of the space lower bound, against 23% for xor and 44% for Bloom filters (abstract).
  - The C single-header library (Apache-2.0) reports 9.04 bits/entry and about 0.36 s for 10M keys in sample output.
  - Its query timings look like placeholder sample values, so I am not citing them.
  - A filter in front of a table whose first line already holds a fingerprint only adds a cache line, so it is not recommended.
- **Inline prefixes.** Store the length and the first 4 to 8 bytes in the slot, as Umbra's 16-byte strings do with a 4-byte prefix. Words of 8 bytes or fewer, most of GloVe, then verify without leaving the slot.

## 5. Future tokenizers (WordPiece, Unigram, BPE)

All three designs below give **exact lookup only**. None enumerates prefixes.

- **WordPiece.** The HF style makes O(m) failed lookups of shrinking substrings per position, so the miss path dominates. A fingerprint in the first line touched keeps each miss to about one cache line. The alternative is Google's Fast WordPiece: a vocabulary trie with Aho-Corasick-style links, strictly O(n), on average 8.2× faster than HF tokenizers and 5.1× faster than TF Text (Song et al., EMNLP 2021). That trie would sit next to the table, which still maps final pieces to ids.
- **Unigram.** Needs all vocabulary prefixes at each position. That means a trie (a double array, as in SentencePiece) next to the table. Emulating it with one lookup per prefix length costs O(m) lookups per position.
- **Byte-level BPE.** Needs a separate integer-pair table mapping (u32, u32) to (rank, new id), as HF does. The string table only maps initial byte pieces and pre-tokenized chunks. That integer-key table can reuse the same fingerprint-slot code.

## 6. Ranked recommendation

Notes on the cost estimates:

- "Probe" means one random cache-line access; current = 1 probe per hit for words of 16 bytes or fewer.
- The current table is 270 MB, so in practice nearly every probe also misses the TLB and L3.
- All three designs add the existing `offsets` array (24 MB, int64) and the vocabulary bytes to the hit path.

1. **Static fingerprint table: {u32 id, u32 fingerprint}, 2^22 slots, linear probing in Robin Hood order (my recommendation).**
   - Memory: 32 MB, plus the shared offsets and bytes.
   - Miss: about 1 line, with false positives near 2^-32 per probed slot.
   - Hit: slot, then `offsets[id]`, then the bytes, about 3 dependent lines against 1 today. The lines are in a structure 8× smaller, and hot ids cluster because the vocabularies are frequency-ordered.
   - Build: one pass, well under a second (my estimate).
   - Complexity: low, and close to the current code. Portable, with no SIMD needed.
   - Risk: hit latency on long, cold words.
   - Variant: 16-byte slots {u32 id, u8 len, 3-byte fingerprint, 8-byte inline prefix}, 64 MB, which verify short words inside the slot.
2. **Swiss-style SoA: 1-byte tags in groups of 16 plus a u32 index array.**
   - Memory: 20 MB at load 0.715. It could fit 2^22 slots up to load 7/8.
   - Miss: 1 tag line, but about 8% of misses touch index and bytes (my estimate: about 11 occupied tags × 1/128).
   - Hit: about 4 lines.
   - Complexity: medium, because it needs SSE2/NEON plus a scalar fallback.
   - Risk: worse for WordPiece-heavy miss traffic than design 1.
3. **PtrHash-style MPHF plus {u32 id, u16 fingerprint, u16 len}.**
   - Memory: about 0.9 MB + 24 MB.
   - Lookups never probe. A miss costs 1 pilot read (cache-resident) plus 1 slot line, rejected at about 1/65536. A hit costs about 3 to 4 lines.
   - Build: about 0.1 s at the cited 32 ns/key, extrapolated from 300M keys.
   - Complexity: highest, since it means a C reimplementation and handling of seed retries.
   - Risk: build-failure paths and more code. It wins only if memory matters more than about 8 MB.

**What to measure.** Use `bench_tokenize`, the 3M-word GoogleNews vocabulary and 400k-word GloVe, and real text: Zipfian hits plus a realistic unknown rate.

- ns per token for the current table and designs 1 to 3, at 1, 4 and all threads, alternating paired runs.
- Miss-only throughput on WordPiece-style substring floods.
- Lines touched and LLC/TLB misses (perf/Instruments), with and without huge pages.
- Build time at 3M words.
- Peak memory.

Decide on design 1 if its hits are not slower than today's.

## Sources
- PtrHash paper: https://arxiv.org/abs/2502.15539 (HTML: https://arxiv.org/html/2502.15539) ; code: https://github.com/RagnarGrootKoerkamp/ptrhash
- BBHash: https://github.com/rizkg/BBHash
- PTHash: https://github.com/jermp/pthash
- RecSplit: https://arxiv.org/abs/1910.06416 ; sux license: https://github.com/vigna/sux
- cmph: https://cmph.sourceforge.net/
- Abseil SwissTables: https://abseil.io/about/design/swisstables
- hashbrown source (`src/raw.rs`, `src/control/group/*.rs`): https://github.com/rust-lang/hashbrown
- Folly F14: https://engineering.fb.com/2019/04/25/developer-tools/f14/
- HF tokenizers source (`tk-encode/src/models/{wordpiece,wordlevel,bpe,unigram}`): https://github.com/huggingface/tokenizers
- tiktoken `src/lib.rs`: https://github.com/openai/tiktoken
- fastText `src/dictionary.{h,cc}`: https://github.com/facebookresearch/fastText
- SentencePiece `src/model_interface.{h,cc}`, `src/unigram_model.h`: https://github.com/google/sentencepiece
- darts-clone and evaluation: https://github.com/s-yata/darts-clone , https://github.com/s-yata/darts-clone/blob/master/doc/ja/Evaluation.md
- model2vec `model2vec/model.py`: https://github.com/MinishLab/model2vec ; gensim `keyedvectors.py`: https://github.com/piskvorky/gensim
- spaCy `spacy/strings.pyx`: https://github.com/explosion/spaCy ; preshed `maps.pyx`: https://github.com/explosion/preshed
- FST blog: https://burntsushi.net/transducers/
- Binary fuse filters: https://arxiv.org/abs/2201.01174 ; C header: https://github.com/FastFilter/xor_singleheader
- Fast WordPiece (Song et al., EMNLP 2021): https://aclanthology.org/2021.emnlp-main.160/
- Umbra strings (Neumann & Freitag, CIDR 2020): https://www.cidrdb.org/cidr2020/papers/p29-neumann-cidr20.pdf (cited from memory, not re-fetched)