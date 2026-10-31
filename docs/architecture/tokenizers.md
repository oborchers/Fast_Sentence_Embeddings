## Tokenizers

Binding for every 2.0 module (formerly under "2.0 design").

The manifest key `fse.tokenizer` holds `external` or the identifier of a native tokenizer; reading anything else raises `UnsupportedArtifactError` naming the supported ones (`fse._contract.check_tokenizer`). A tokenizer is part of the encoder contract like its aggregation: any change of its rules, its tables or its Unicode version is a new identifier. `fse._reference_text` defines the tokens of every tokenizer, and the native code gives the same bytes.

- `external`: the caller gives the tokens. Text input is split at the whitespace W below, as `str.split()` does, and the tokens are not changed.
- `fse_words@1`: the native tokenizer specified here.
- `wordpiece@1`: the WordPiece tokenizer of BERT vocabularies (POTION/Model2Vec, the static models of sentence-transformers), specified below; its parameters are data of the model.
- `unigram@1`: the Unigram tokenizer of SentencePiece vocabularies as Hugging Face tokenizers runs it (potion-multilingual-128M), specified below; its parameters, float64 scores and charsmap are data of the model.

**Input.** One sentence is a `str` that UTF-8 can encode (a lone surrogate raises), or one line of a UTF-8 file: a line ends at the byte `\n`, which is not part of it; the last line needs no `\n`, a final `\n` starts no empty line, and an empty file has no lines. A file must not start with a UTF-8 byte order mark, which would otherwise stick to the first token as U+FEFF; it raises. Bytes are decoded strictly, as CPython's UTF-8 decoder does, and invalid UTF-8 raises. Sentences are not split further and not normalized: NFC and NFD spellings stay different.

**Output.** A list of non-empty tokens, matched to the vocabulary by the exact equality of their UTF-8 bytes. `fse_words@1` computes it in five steps.

1. **Lowercase.** Every code point is replaced by its simple lowercase mapping (field 13 of `UnicodeData.txt` of Unicode 15.1.0, from the checked-in table `fse._unicode`); code points without a mapping stay. There is no SpecialCasing and no context: U+0130 becomes `i`, and Σ always becomes σ.
2. **Classes** of the lowercased code points:
   - whitespace W: U+0009 to U+000D, U+001C to U+0020, U+0085, U+00A0, U+1680, U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F, U+3000;
   - separators S: ``! " # $ % & ( ) * ; < > ? @ [ ] ` { }`` and U+00AB, U+00BB, U+2012 to U+2015, U+2018, U+2019, U+201C, U+201D, U+201E;
   - token characters: every other code point, subject to step 3;
   - word characters: `a` to `z`, `0` to `9`, `_`, and every code point from U+0080 that is in neither W nor S;
   - digits: `0` to `9`.
3. **Context.** Applied from left to right; t(i) says whether s[i] is a token character.
   - A `.` directly before or after another `.` is not a token character; the same holds for `-`.
   - A `,` or `:` is a token character exactly if the next code point is a digit.
   - A `'` directly before or after another `'` is not a token character. Otherwise it is not one if i > 0, t(i-1) holds, s[i-1] is not a word character, and s[i+1] exists and is a word character.
4. **Spans.** Every maximal run of token characters is a candidate u.
5. **Per candidate u.** Strip `.` and `:` from both ends. Split u into parts by the first rule that applies:
   - the table: `cannot` to can, not; `gimme` to gim, me; `gonna` to gon, na; `gotta` to got, ta; `lemme` to lem, me; `wanna` to wan, na; `more'n` to more, n; `d'ye` to d, ye; `'tis` to t, is; `'twas` to t, was;
   - u is `n't`: the part `not`;
   - u ends with `n't` after a non-empty rest that does not end with `'`: the rest and `not`;
   - u ends with the first matching one of `'s`, `'m`, `'d`, `'ll`, `'re`, `'ve` after a non-empty rest that does not end with `'`: the rest and the clitic without `'`;
   - otherwise u itself.

   From every part, strip `.`, `:` and `'` from both ends; drop it if it holds no `a` to `z` and no `0` to `9`; otherwise split it at every `-` and emit the non-empty pieces in order.

Apart from W, S, the lowercase table and the word characters from U+0080, every rule refers to ASCII only, so an implementation may apply them to UTF-8 bytes.

**Stage 2** of `fse_words@1` and `external`: a token whose bytes are a vocabulary word becomes its id. Otherwise, with FastText it becomes `OOV_TOKEN` with the first `max_ngrams` `fasttext@1` buckets of `<token>` (possibly none, which still counts for the length); without FastText it is no token.

| Input | `fse_words@1` |
|---|---|
| `I don't know.` | i, do, not, know |
| `Can't, won't` | ca, not, wo, not |
| `'Tis` | t, is |
| `O'Reilly's` | o'reilly, s |
| `James' book` | james, book |
| `rock'n'roll` / `rock’n’roll` | rock'n'roll / rock, n, roll |
| `x='abc'` | x=, abc |
| `U.S. e.g.` | u.s, e.g |
| `Wait... what?` | wait, what |
| `1,000 and 10:30` / `a,b c:d` | 1,000, and, 10:30 / a, b, c, d |
| `http://example.com/a-b?x=1` | http, //example.com/a, b, x=1 |
| `well-known --- a--b -x` | well, known, a, b, x |
| `café 中国 Paris` | café, paris |
| `İstanbul` / `ΣΟΦΟΣ 2Σ` | istanbul / 2σ |
| `snake_case foo/bar a+b` | snake_case, foo/bar, a+b |
| `...`, an empty or blank sentence | none |

The rules reproduce the preprocessing of the published 1.x tables on all STS sentences; on WikiText 99.98 % of the paragraphs agree, the rest differ only by the simple lowercase of İ and final sigma.

### wordpiece@1

Binding for every 2.0 module (formerly under "2.0 design").

`wordpiece@1` gives the ids that Hugging Face tokenizers 0.23.3 gives for a `tokenizer.json` with a `BertNormalizer` (all four flags on), a `BertPreTokenizer` and a `WordPiece` model, whose added tokens are matched on the raw text (`normalized: false`, without `lstrip`, `rstrip` and `single_word`), encoded without special tokens (`add_special_tokens=False`), as model2vec and sentence-transformers call it. The rules are reimplemented from the source of tokenizers at tag v0.23.3 (`tokenizer/added_vocabulary.rs`, `normalizers/bert.rs`, `pre_tokenizers/bert.rs`, `models/wordpiece/mod.rs`) and from the Unicode data; no code is copied. Its parameters are data of the model, in `fse.tokenizer_config` (encoder 1), all required and no others:

| Parameter | Meaning |
|---|---|
| `added_tokens` | `[[id, content], ...]`, sorted by id: the tokens matched on the raw text; each content is the vocabulary string of its id, non-empty and unique |
| `unk_id` | the id of a word WordPiece cannot split |
| `continuing_subword_prefix` | the prefix of every piece of a word after its first, non-empty (`##`) |
| `max_input_chars_per_word` | a word of more code points is `unk_id` (100), at least 1 |
| `clean_text`, `handle_chinese_chars`, `strip_accents`, `lowercase` | the flags of the normalizer; this fse supports only all four true (every target model), so another value raises `UnsupportedArtifactError` |

Every id is an int (not a bool) with `0 <= id < n`, n the number of tokens. The encoder adds `fse.unknown` (`drop`: `unk_id` adds nothing and does not count in n, as model2vec; `keep`: it is a row like any other, as sentence-transformers), and optionally `fse.max_tokens` and `fse.max_chars`, the length rule of the format the model came from (model2vec: 512 tokens and 512 times the median token length in code points).

Input as for `fse_words@1`. Per sentence:

1. **Character cut.** With `fse.max_chars`, keep the first `max_chars` code points. The whole sentence is still decoded: invalid UTF-8 after the cut raises.
2. **Added tokens.** Split the text at the added tokens, leftmost-longest over the raw code points: at the first position where an added token starts, the longest one starting there is taken and gives its id; the search goes on after it. Each piece between them goes through steps 3 to 5 on its own.
3. **Normalize** a piece:
   1. remove U+0000, U+FFFD and every code point of general category Cc, Cf or Co in Unicode 8.0.0 except U+0009, U+000A and U+000D; then replace U+0009, U+000A, U+000D and every code point with the property White_Space of Unicode 17.0.0 by U+0020;
   2. put U+0020 before and after every code point in U+4E00 to U+9FFF, U+3400 to U+4DBF, U+20000 to U+2A6DF, U+2A700 to U+2B73F, U+2B740 to U+2B81F, U+2B920 to U+2CEAF, U+F900 to U+FAFF and U+2F800 to U+2FA1F (fixed ranges, not Unicode data);
   3. NFD of Unicode 9.0.0 over the whole piece: the full canonical decomposition (Hangul syllables by the algorithm of the standard), then canonical ordering: every maximal run of code points with a nonzero canonical combining class (9.0.0) is sorted stably by class;
   4. remove every code point of category Mn in Unicode 8.0.0. This follows step 3, so a removed mark of class 0 (U+034F COMBINING GRAPHEME JOINER) has ended a run, and the marks on either side of it were not reordered across it;
   5. replace every code point by its full lowercase mapping of Unicode 17.0.0: field 13 of `UnicodeData.txt`, replaced by the unconditional entries of `SpecialCasing.txt` (U+0130 becomes `i` U+0307), without context (Σ always becomes σ).
4. **Pre-tokenize.** Split at White_Space (17.0.0), which is removed; every ASCII punctuation character (`!` to `/`, `:` to `@`, `[` to `` ` ``, `{` to `~`) and every code point of category Pc, Pd, Ps, Pe, Pi, Pf or Po in Unicode 8.0.0 is a word of its own.
5. **WordPiece** per word. A word of more than `max_input_chars_per_word` code points is `unk_id`. Otherwise the word is split from the left: the longest prefix of the rest that is a token (a piece after the first with `continuing_subword_prefix` in front), shortened one code point at a time, is the next id; if no prefix is a token at some point, the whole word is one `unk_id`.
6. **Token cut and unknown tokens.** With `fse.max_tokens`, keep the first `max_tokens` ids of the whole sequence of steps 2 to 5 (added tokens and `unk_id` count). Every word is split whole before the cut, so a word whose split fails after the cut is one `unk_id` (vocabulary `[UNK]`, `a`, text `ab`, one token: `[UNK]`, not `a`). Then `fse.unknown=drop` removes every `unk_id`.

Golden ids (`tests/test_data/v2/wordpiece.json`, written by tokenizers 0.23.3 with a small vocabulary that holds the letters, digits and their continuation pieces; shown as tokens):

| Input | `wordpiece@1` |
|---|---|
| `un[MASK]able x[MASK]y [cls]` | un, [MASK], able, x, [MASK], y, [UNK], c, ##l, ##s, [UNK] |
| `İstanbul ISTANBUL` | istanbul, istanbul (İ decomposes to I and U+0307 before lowercasing) |
| `ǰ` (U+01F0) | j (decomposed, the mark removed) |
| 101 times `a` / 100 times `a` | [UNK] / a, ##a, ... |
| `한` (U+D55C) | ᄒ, ##ᅡ, ##ᆫ (Hangul as conjoining jamo) |
| `豈中国` (U+F900 first) | 豈 (U+8C48), 中, 国 |
| `aẔ9͏ᅄ6` | a, ##z, ##9, ##ᅄ, ##6 (U+0331 and U+034F removed) |
| `a` U+1E949 U+11446 / `a` U+1E949 U+034F U+11446 | a, ##U+11446, ##U+1E949 / a, ##U+1E949, ##U+11446 (classes 7 and 230 reordered, not across U+034F) |
| `ΣΟΦΟΣ σοφος` | σ, ##ο, ##φ, ##ο, ##σ, σ, ##ο, ##φ, ##ο, ##ς (no final sigma by context) |

**Stage 2** of `wordpiece@1`: token id i is row i. The vocabulary of the table (`vocab.bytes`, `vocab.offsets`) holds the token strings in id order, continuation pieces with their prefix, so every id from 0 to n - 1 exists and the checks of word_vectors 1 cover the tokens.

The rules mix three Unicode versions, those of the libraries tokenizers 0.23.3 is built with:

| Rule | In tokenizers 0.23.3 | Unicode |
|---|---|---|
| Cc, Cf (clean), Mn (accents), P* (punctuation) | crate `unicode_categories` 0.1.1 | 8.0.0; Co by fixed ranges, unassigned code points are kept |
| decomposition and combining classes | crate `unicode-normalization-alignments` 0.1.12 | 9.0.0 |
| lowercase, White_Space | the Rust standard library that built the wheel | 17.0.0 |

So a code point that became Mn, Cf or P* after 8.0.0 is kept, and the 83 code points with a nonzero class in 9.0.0 that are not Mn in 8.0.0 survive and are reordered. A tokenizers wheel built with another Rust lowercases some rare code points differently; `wordpiece@1` stays fixed to these tables, and any change is a new identifier. `tools/make_wordpiece_tables.py` reads the five UCD files (pinned by sha256) and writes `fse._unicode_bert`, the data for the reference `fse._reference_wordpiece`, and `_native_unicode_bert.h`, the same composed per code point for the native code (steps 3.1 to 3.5 of one code point, with its combining class; only the ordering of runs is computed per piece).

### unigram@1

Binding for every 2.0 module (formerly under "2.0 design").

`unigram@1` gives the ids that Hugging Face tokenizers 0.23.3 gives for a `tokenizer.json` with a `Unigram` model, a normalizer made of the steps below, a `Metaspace` pre-tokenizer and added tokens matched on the raw text, encoded without special tokens (the tokenizer of minishlab/potion-multilingual-128M). The rules are reimplemented from tokenizers at tag v0.23.3 (`models/unigram/model.rs`, `normalizers/precompiled.rs`, `normalizers/replace.rs`, `normalizers/strip.rs`, `pre_tokenizers/metaspace.rs`), from the crates its Cargo.lock pins (spm_precompiled 0.1.4 for the charsmap, unicode-segmentation 1.13.3 for graphemes, Oniguruma through onig 6.5.3 for the regexes) and from the Unicode data; no code is copied. `fse._reference_unigram` defines the ids, and the native tokenizer (`_native_unigram.h`, `_native_unigram.pxi`, `_native_unigram_steps.pxi`) gives the same without the GIL: the graphemes from the table `_native_unicode_graphemes.h`, the bounds-checked search of the charsmap, every step through a growing buffer that checks each write, the candidates of a start by a byte walk of the vocabulary double-array trie (the same builder as wordpiece@1, built at load and never persisted), accepting terminal nodes only at code point boundaries in increasing length, then offering the unknown node when no piece covers the first code point, and the Viterbi in binary64 (checked when compiled: no extended precision) only in functions never inlined, inside the floating-point environment of the kernels, so a thread with another rounding mode gets the same ids. An encoder holds the parameters in `fse.tokenizer_config`, the scores in the F64 tensor `tokenizer.scores` and the charsmap verbatim in the U8 tensor `tokenizer.charsmap` (encoder 1).

Unigram constructs the temporary trie from sorted binary keys and their original ids.
Adjacent keys reuse their common prefix through a node stack, avoiding a sibling search
for every byte. Temporary key references are freed before the shared double-array
placement. Sorting never changes vocabulary ids.

The trie path reserves the candidate buffer once per start, including every candidate
already stored and at most `min(bytes_remaining, max_piece_bytes) + 1` new candidates.
The addition and byte-size multiplication are checked before allocation. A private
`_generic=True` state retains the hash and proper-prefix tables for tests. Only this
path builds the Python proper-prefix set and its VocabTable. The private `_candidates`
test hook exposes the ordered `(end, id, score)` offers before Viterbi, so tests compare
the trie with both that generic path and the independent reference.

Its parameters are data of the model: the added tokens and `unk_id` as for `wordpiece@1`; `byte_fallback`, which must be false; `normalizer`, a list of steps; `metaspace`, the replacement (`▁`, U+2581), the prepend scheme (`always`) and `split` (false), the only values this fse supports. The score of every piece is a float64 (tokenizers parses and sums them in float64), and the charsmap of a `precompiled` step is the decoded blob of the file, kept as it is. Only these forms are read; any other normalizer, regex, pre-tokenizer value, `byte_fallback: true` or `unk_id: null` raises `UnsupportedArtifactError` naming it.

Input as for `fse_words@1`. Per sentence:

1. **Character cut** and 2. **added tokens** as steps 1 and 2 of `wordpiece@1`. Every piece between added tokens goes through steps 3 to 5 on its own; an empty piece gives no id.
3. **Normalize** the piece by its steps, in order:
   - `precompiled`: cut the piece into extended grapheme clusters by UAX #29 of Unicode 17.0.0 (Grapheme_Cluster_Break, Extended_Pictographic and InCB of 17.0.0; rules GB3 to GB13 and GB999, with GB9c). A grapheme of fewer than 6 UTF-8 bytes is searched in the charsmap with its bytes: if any key is a prefix of it, the **shortest** such key wins and the whole grapheme becomes that key's value, so the rest of the grapheme is dropped (`Ａ` U+0301 becomes `A`). Otherwise every code point of the grapheme is searched alone and becomes its key's value, or stays. This is tokenizers' rule, not SentencePiece's (which takes the longest match over the text, without graphemes).
   - `replace`: every occurrence of a non-empty `pattern` becomes `content`, left to right without overlap (tokenizers' `Replace` with a `String` pattern).
   - `runs`: every maximal run of at least `min` code points of a class becomes `content`; the class `space` is U+0020 (the regex ` {2,}` is `min` 2), the class `white_space` is White_Space of Unicode 17.0.0 (the regex `\s+` is `min` 1: Oniguruma's `\s` matches exactly these 25 code points).
   - `strip`: leading and trailing White_Space code points (17.0.0) are removed.
4. **Metaspace.** Every U+0020 becomes the replacement, which is put in front of the piece unless it starts with it; the piece stays one pre-token.
5. **Unigram** on the UTF-8 bytes `b` of the pre-token, n bytes. `unk_score = min(scores) - 10.0` and every sum below are float64, round to nearest. `best[0] = 0.0`. For each code point boundary s from 0 up, with m the length of the code point at s: every piece that is a prefix of `b[s:]`, by increasing length, offers `score + best[s]` for its end; it is taken if that end holds nothing yet or the offer is strictly greater. If no piece is exactly the m bytes at s, `unk_score + best[s]` is offered for s + m in the same way, with the id `unk_id`. So a tie goes to the earliest start. Going back from n, consecutive nodes whose id is `unk_id` (unknown code points, and a piece that has that id) are joined into one string, and every string is the id of the piece it equals, else `unk_id`.
6. **Token cut and unknown tokens** as step 6 of `wordpiece@1`: every pre-token is tokenized whole, then the first `max_tokens` ids are kept and `fse.unknown=drop` removes every `unk_id`.

The native normalizer combines a maximal run of consecutive single-byte Replace
patterns into one byte-mapping pass only when the patterns are distinct and no later
pattern occurs in an earlier replacement content. Otherwise the steps remain separate.
The generic test path always runs the steps separately. This changes neither the
normalizer rules nor persisted parameters.

**The charsmap** (the blob of spm_precompiled 0.1.4, SentencePiece's double array): a little-endian u32 `size`, `size / 4` little-endian u32 units, then the values as UTF-8 strings, each ended by NUL. A unit u has `leaf = (u >> 8) & 1`, `label = u & (2**31 | 255)`, `offset = (u >> 10) << ((u & 512) >> 6)` and `value = u & (2**31 - 1)`. The search starts at `p = offset(a[0])` and takes the bytes of the key in turn, ending at a byte 0: `p ^= c`, it ends unless `label(a[p]) == c`, then `p ^= offset(a[p])`, and if that unit had `leaf`, `value(a[p])` is the offset of a match in the values, whose string runs to the next NUL or the end of the values. The keys are byte strings. A blob is checked completely when it is read, as tokenizers panics on a broken one: `size` a positive multiple of 4 within the blob, the values valid UTF-8, and every node reachable from the root (darts-clone shares the nodes of equal suffixes, so a node is checked once, and a cycle is refused) with every probe `p ^ c` inside the array and every value offset at most the length of the values and on a code point boundary.

| Input (potion-multilingual-128M) | `unigram@1` |
|---|---|
| `Hello, World!` | `▁Hello` `▁` `,` `▁World` `▁!` |
| `a[UNK]b` | `▁a` `[UNK]` `▁b` |
| `Ａ` U+0301 | `▁A` |
| `x` U+3000 `y` | `▁x` `▁y` |
| U+0000 `a` / `x` U+180E `y` | `▁` `[UNK]` `a` / `▁x` `[UNK]` `y` (no key, no White_Space) |
| `…` | `▁` `.` `▁` `.` `▁` `.` |
| an empty or blank sentence | none |

`tests/test_data/v2/unigram.json` holds such cases for small tokenizers written by tokenizers 0.23.3 (`make_unigram_golden.py`), including a tie (pieces `a`, `ab`, `bc`, `c` of score -1: `abc` is `a`, `bc`), joined unknowns that are a piece (scores above 20, so two unknown nodes beat the piece `xy`: `axya` is `a`, `xy`, `a`) and the case that needs float64 scores: `a` and `b` of score -1 and `ab` of `-2 - 2**-30` make `ab` `a`, `b`, while scores rounded to float32 tie and give `ab`. Scores rounded to float32 agree on a million texts for the target model, yet 16 of its Viterbi decisions on MASSIVE lie within one float32 rounding of a flip; float32 sums change 380 of those texts.

The Unicode data: `tools/make_grapheme_tables.py` reads `GraphemeBreakProperty.txt`, `emoji-data.txt`, `DerivedCoreProperties.txt` and `PropList.txt` of 17.0.0 (pinned by sha256) and writes `fse._unicode_graphemes`; `tests/test_data/GraphemeBreakTest-17.0.0.txt`, pinned the same way, holds the 766 cases of UAX #29 the segmentation passes. A tokenizers wheel with another unicode-segmentation, or a new form of Precompiled, can change ids; `unigram@1` stays fixed to these rules, and a change is a new identifier.
