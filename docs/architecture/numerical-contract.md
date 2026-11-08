# Native kernel

Scope: outside the former "2.0 design" section.

`_native_kernel.h` and the headers it includes implement `weighted_mean@1` (pool and accumulate), `dot16`, the step `normalize@1`, the linear score of the numerical contract below, storage quantization, inverse norms, cosine search and, for linear heads, the bias of folded logits, `exp@1`, the links and the labels (`_native_kernel_head.h`: scalar float32 code, one rounding per operation, the same on every instruction set). Each is one C loop, instantiated per ISA: portable (`fmaf`), AVX2 with FMA and F16C on x86 (selected at run time by cpuid and xgetbv, so one wheel serves every x86 machine) and NEON on arm64. `_native.pyx` validates all arrays before it releases the GIL, then calls the loop of the chosen ISA (`available_isas`, `default_isa`; `isa=` selects one for tests). The rules that make it bit-identical to `fse._reference`:

- Every multiply-add is an explicit fused multiply-add in token order; the compiler must not contract or reorder anything else. The extension is built with `-ffp-contract=off -fno-fast-math` (`/fp:precise` on MSVC), see `setup.py`. MinGW is refused, because its runtime computes `fmaf` with two roundings.
- The floating-point environment is fixed for the duration of every call: round to nearest, no flush-to-zero or denormals-are-zero (MXCSR on x86, FPCR on arm64). The previous state is restored afterwards, also after an error. Each Python entry point sets it for the whole call, so the numpy conversions of its arguments (such as `bias`) run in it too, and each C entry point sets it again and calls its arithmetic only through functions marked never to be inlined, so no compiler can move the arithmetic before it.
- Dimensions are limited to `MAX_DIM` (2**30), because the loops index a row with `int`. Batches are read-only views; the kernel checks their offsets again before it reads through raw pointers, and `out` must not overlap the rows, weights or the batch. `pool_text` (`_native_task.pxi`), which pools a batch straight from the buffers of a `Workspace`, makes the same checks in C on those buffers (the rules and messages of `Batch`, the sentence count of the chunk, then those of `_check_batch`), and the overlap of `out` with the kernel's arrays by their byte ranges, which is the bounds test of `np.may_share_memory` for C-contiguous arrays; a batch in a workspace's own buffers cannot overlap `out`.
- Reductions over the dimension use `dot16`: 16 lanes, element i in lane i % 16, then a tree 8/4/2/1. Lanes are not padded with zeros, because a lane can hold -0.0.
- Word rows are prefetched 8 tokens ahead, also across sentence boundaries; prefetching changes no result.
- The NEON float32 loop (`_native_kernel_tiles.h`) cuts a sentence into tiles of at most K = 16 KiB / row bytes (at most 64) contributions and keeps the accumulators of each block of 32 values (then 16, 8, 4 and single values) in registers over a tile, storing them to the out row between tiles. Per value this is the same sequence of fused multiply-adds from +0.0 in token order, and a partial sum of -0.0 is stored and reloaded as such, so the bits are those of the row loop. Rows of more than 1024 values, and the other loops, add one row at a time into the out row. The header states the invariants of the tile builder (every contribution once, in order; g once per unknown word; the token count as divisor; the prefetches of the row loop), and no load reads past a row.
- A row that is not finite raises with its row number, as the reference does.

`tests/test_conformance.py` and `tests/test_conformance_steps.py` compare every CPU backend in `BACKENDS` (`tests/support/conformance.py`) with the reference bit for bit, for every dtype, step list, empty, long and unknown-word sentences and subnormal values. Its meta-tests check that the bit-identical suite fails for a CPU kernel without fma, with `acc / n`, with an EPS start or with zero-padded `dot16`, so a weakened comparison cannot pass unnoticed. `tests/test_conformance_head.py` runs every CPU backend through the runtime of linear heads against `fse._reference_head` bit for bit on every path (folded, normalized, unfolded with the vectors, standalone) for float32, float16 and int8 tables, FastText, k = 1, 2 and 17 and ties, with the errors of the fold and of `normalize@1`, and checks folded against unfolded logits within the bound of `tests/support/head_bound.py`; its meta-tests (`tests/test_conformance_head_meta.py`) show the cases catch a bias folded into the table, added before `1/n` or twice, a skipped norm or its errors, ties to the larger index, the naive sigmoid and another order of the softmax sum. `tests/test_conformance_tiles.py` adds block and tile edges, -0.0 witnesses in every block and across tiles, a late overflow and rows next to pages without access; its meta-tests run a numpy model of the tiled loop and check that the cases catch the mistakes of a tile builder and of the accumulators, in every block and for every K.

## float32 throughout

Scope: outside the former "2.0 design" section.

Word vectors, weights and sentence vectors are computed in float32. Word vectors may be stored as float16 or int8 and are widened exactly; every sum runs in float32. The kernel takes raw pointers, so arrays must have the expected dtype and be C-contiguous, which `_native.pyx` checks before the GIL is released.

## Numerical contract (`transform_version` 2)

Binding for every 2.0 module (formerly under "2.0 design").

`fse._contract` holds the identifiers and versions, `fse._reference` the numpy reference that computes every rule below bit for bit; it is the oracle for every CPU backend, and tests compare by bit pattern (`view(np.uint32)`), because `array_equal` treats -0.0 and +0.0 as equal.

- **Arithmetic.** IEEE-754 float32 with round to nearest even and subnormals kept (no flush to zero). `fma(a, b, c)` rounds once. Native code calls `fmaf` or FMA intrinsics explicitly and is built without other contractions (`-ffp-contract=off`). Division and square root are correctly rounded; no reciprocal approximations or fast math.
- **Stage 2.** A known word w contributes the stored row `x` (float32 as is, float16 widened exactly, int8 codes converted exactly) with the factor `f = weight[w]` (1 for every word of a finished table), or `f = weight[w] * scale[w]` with one rounding for int8. An unknown FastText word with m >= 1 ngram buckets (in the order of the hashing, capped at `max_ngrams`) contributes each ngram row with `g = (1/m) * oov_weight`, each product rounded, times the ngram scale for int8; with m = 0 it counts for the length and adds nothing. Without FastText, unknown words are no tokens. The scale of int8 rows always goes into the factor, never into the row.
- **`weighted_mean@1`.** The accumulator starts at +0.0 per value; every contribution does `acc[i] = fma(f, x[i], acc[i])` in token order (ngrams in bucket order); `n` counts the tokens. The row is `(1/n) * acc` with `1/n` rounded first, not `acc / n`. An empty sentence is the zero row; there is no EPS.
- **`dot16`**, the dot product of every row reduction: 16 partial sums start at +0.0; index i goes into lane `i % 16` by fma in index order; the lanes are then added 8, 4, 2 and 1 lanes apart, and lane 0 is the result. It vectorizes on NEON, AVX2 and AVX-512. A lane can hold -0.0 (a product that rounds to zero keeps its sign), so implementations must not pad the reduction with zeros: a tail shorter than a vector updates only its own lanes.
- **`normalize@1`**: `s = dot16(row, row)`; an all-zero row stays zero; a nonzero row whose `s` underflows to 0 or overflows raises; otherwise every value is multiplied by `1 / sqrt(s)`.
- **Step lists.** None, or `normalize` once; anything else is rejected when an encoder is built or loaded. After stage 4 every row must be finite, otherwise the row id is reported.
- **Stage 5, storage.** float32 as is; float16 rounds to nearest even and raises beyond +-65504; int8 is symmetric per row: `s = absmax / 127` in float32, stepped down once with `nextafter` if `127 * s` overflows, `code = clip(rint(x / s), -127, 127)`; a row whose scale is 0 (a zero row, or one whose largest value divided by 127 underflows) has codes 0. Reading gives `code * s` with one rounding. These are the 1.x rules (`fse.models._quantize` in 1.1.0), which `fse._codes` and the reference reproduce bit for bit.
- **Reading stored rows.** The inverse norm of a stored row is `float32(1 / sqrt(s))`, where s is the sum of the squares of the stored values (int8 codes without their scale) in float64 and index order; a zero row has 0. A linear `score(w, bias)` of a row is `dot16(stored, w) * scale + bias` (scale 1 for float32 and float16). A cosine search normalizes the query by `normalize@1` (a zero query, or one whose norm underflows or overflows, raises) and scores `dot16(stored, q) * inverse norm`; the scale cancels and is not used. A score, inverse norm or cosine beyond float32 raises; top-k ties go to the smaller row id.
- **`linear@1`**, a head of k rows `a_j` and biases `b_j` (float32) on the pooled rows r of an encoder: *folded* (a head on raw means: `predict`, `predict_proba`, `decision_function` and standalone files) through the folded table `t[w, j] = score(stored_w, scale_w, a_j, 0)` of every word and ngram row (the int8 scale is in it; the factors are those of a float32 table, 1 per word and `(1/m) * oov_weight` per ngram), `z_j = weighted_mean@1(t)_j + b_j`, the bias added once after pooling; *normalized*: `z_j = dot16(normalize@1(r), a_j) + b_j`, every error of `normalize@1` kept; *unfolded* (`return_vectors=True`, and store mode): `z_j = dot16(r', a_j) + b_j`, r' the raw or normalized row by the head's flag, on a store `score(stored, scale, a_j, b_j)` of the stored rows. An empty sentence gives `z = b` on every path; a logit that is not finite raises with its row. Labels: for k = 1 `labels[1]` if `z > 0`, else `labels[0]` (scikit-learn's rule; -0.0 and 0.0 give `labels[0]`); for k >= 2 the first largest logit.
- **`exp@1`**, for x <= 0: below -104 it is +0.0; else `k = rint(x * log2e)` (to nearest even), `r = fma(-k, ln2_hi, x)`, `r = fma(-k, ln2_lo, r)`, `p` the polynomial of degree 7 in r by Horner with fused multiply-adds from `c_7` down, and `(p * 2**(k + 64)) * 2**-64`: the first product is exact, the second rounds once, also into the subnormal range. Constants (float32): log2e `0x1.715476p+0`, ln2_hi `0x1.62e430p-1`, ln2_lo `-0x1.05c610p-29`, `c_0 = c_1 = 1`, `c_2 = 0x1p-1`, `c_3 = 0x1.555556p-3`, `c_4 = 0x1.555556p-5`, `c_5 = 0x1.111112p-7`, `c_6 = 0x1.6c16c2p-10`, `c_7 = 0x1.a01a02p-13` (1/i! rounded). The contract promises 2 ulps against exp; over all 1,120,927,745 float32 inputs from -0.0 to -104 it is within 0.9371 ulp (`tools/check_exp.py`). Libraries' exp functions are not bit-identical across platforms, which is why the links use this one.
- **Links**, on finished logits, never adding the bias: `sigmoid@1`, with `e = exp@1(-|z|)`, is `1 / (1 + e)` for z >= 0 (-0.0 included) and `e / (1 + e)` else, and gives `[1 - p, p]` for one logit; `softmax@1` is `e_j / s` with `e_j = exp@1(z_j - max)` and s their sum in index order from +0.0, and of one logit z it is the softmax of `[-z, z]` (scikit-learn's binary multinomial model, whose decision is the one logit); `ovr@1` divides the `sigmoid@1` of every logit by their sum in index order, and a row whose sum is 0 is `1/k` everywhere (scikit-learn's one-vs-rest rule). A lookup table of the sigmoid would be an identifier of its own, made only if a profile shows the link.

What is bit-identical across what:

| Across | Guarantee |
|---|---|
| worker counts, job and batch sizes, input classes with the same tokens, shards, restarts | bit-identical: every row is a pure function of its sentence |
| CPU instruction sets and operating systems (portable, NEON, AVX2) | bit-identical, through explicit fma and `dot16` |
| the numpy reference and every CPU backend | bit-identical for every profile the conformance suite checks |
| each path of a linear head (folded, normalized, unfolded, standalone, store mode) and its reference | bit-identical on every CPU backend; a standalone file gives the bits of its complete model |
| each head of a group (`fse.Heads`) and the same head alone, on the same path | bit-identical on every CPU backend and instruction set: a group adds no arithmetic, the folded table shared by raw heads is the columns of their single tables side by side (`weighted_mean@1` is defined per value) and the shared normalization is `normalize@1` of the raw pooled rows |
| the folded and the unfolded path of a head on raw means (`predict` against `return_vectors=True` or a float32 store) | within the rounding of the fold, not bit-identical: folding changes the order of the roundings, so about half the logits differ in their last bits (maintainer decision); a head on normalized means has one path, so its store mode gives the bits of its texts |
| a head in fse and in scikit-learn | within the bound of `tests/support/head_bound.py` (the rounding of A and b, dot16, the fold, the pooling, the bias, in float64 per row); a label differs only where the top two logits are within their bounds, a probability by at most the bound through the link (`|dz| / 4` for sigmoid, `max |dz| / 2` for softmax) plus 8 u (sigmoid) or (k + 8) u (softmax) for the link's own roundings |
| 1.x and 2.0 | not identical (EPS, `saxpy`, BLAS in the removal); compared with tolerances, STS and MTEB |

### Groups of heads (`fse.Heads`)

A group has no identifier and no `transform_version` of its own: it adds no arithmetic. Every head computes with the functions of its own path (`fold`, `weighted_mean@1`, `normalize@1`, `dot16`, the bias, the link, the label rule, the cutoffs) and gets the bits of the same head alone on the same backend, instruction set and path. The group shares only what does not depend on the head: the tokenization, the pooling of one table that holds the folded rows of all raw heads side by side (the score of a row with a weight row depends on that row and weight alone, and `weighted_mean@1` is defined per value, so a column has the bits of the table of its head alone, also across the blocks of the NEON kernel), and the `normalize@1` of the pooled raw rows. A lone head is a group of one. The reference of a group is the sequence of the single references (`fse._reference_head`); no function is added to it. A head on raw means takes the folded path without the vectors and the unfolded path with them (`Heads.paths`), the two only within the rounding of the fold, as for one head.

### Label rule `linear@2` (`multilabel@1` and single-label heads)

The aggregation, fold, dot16 and bias rules of linear@1 remain unchanged.
`multilabel@1` applies sigmoid@1 independently to each finished logit, with no
normalization across outputs and no negative-class expansion for a single column.
`linear@2` compares each finished F32 probability with its stored F32 threshold
using strict greater-than. Equality, including probability 0 or 1 at the same
threshold, is false. Every row produces a bool vector in class order.

The reference is `fse._reference_head.multilabel` for decisions and its `sigmoid`
for probabilities. CPU backends reuse the existing SIMD sigmoid@1 implementation.
The threshold comparison is a whole-array comparison, with no per-row Python hook.
The CPU backend performs that comparison under its protected floating-point environment
and restores the caller's state afterwards. Positive subnormal probabilities must remain
greater than a zero cutoff even when the caller enables flush-to-zero or denormals-as-zero.
This introduces no rounding operation beyond the existing sigmoid. The old
linear@1 label rule, all existing links, and transform_version 2 remain unchanged.

#### Single-label heads under `linear@2`

For a single-label head with the link `sigmoid@1`, `softmax@1` or `ovr@1`, `linear@2`
is the index of the first largest finished F32 probability of `predict_proba`
(`argmax(proba(z, link), axis=1)`), so `predict == classes_[argmax(predict_proba)]` bit
for bit on every path. For one logit the probabilities are the columns `[1 - p, p]`. The
rule is undefined for the link `none` and for regression heads. No new rounding is
added: the probabilities are those of the link, the argmax is a comparison, and the
maximum of a row is at least 1/k and normal, so flush-to-zero cannot change it. The
reference is `fse._reference_head.predicted`; the runtime writes the probabilities into
a worker buffer of the width of `predict_proba` and takes the argmax into the label
array, in the store path too.

In exact arithmetic `linear@1` and `linear@2` agree. In F32 they differ only on ties of
the probabilities: saturation (ovr logits 30 and 40 both give 1.0, `linear@2` takes the
first), complete underflow (all ovr logits below -104 give the uniform row and class 0)
and a binary logit so small that p is exactly 0.5 (class 0). `linear@2` decides by the
same bits that `predict_proba` returns. `predict` costs more than under `linear@1`: it
computes the probabilities (`exp@1` per output, and for ovr the sum and quotient) before
the argmax.

`v2.0.0` is released and contains `linear@1`, `sigmoid@1`, `softmax@1`, `ovr@1` and
`exp@1`, not `linear@2` and not `multilabel@1`. The single-label rule is an additive
definition for combinations that every earlier reader refuses (format version 2 without
`multilabel@1`, and `linear@2` with a single-label link). Every combination that was
valid keeps its meaning and its bits, so no version 3 is needed.

#### Imported calibrations

Platt scaling (`CalibratedClassifierCV(method="sigmoid", ensemble=False)`) and a softmax
temperature are affine maps of the logits and fold into A and b on import, not at
inference. With `a` and `c` the slope and intercept of a sigmoid calibrator (probability
`sigmoid(-(a z + c))` of the response `z` of the base model) the head is
`W' = (-a) W` and `b' = (-a) b + (-c)` in float64; a temperature T gives `W / T` and
`b / T`. Negation is exact, so the bits are those of `-(a W)` and `-(a b + c)`, except that a
zero may have the other sign (the probabilities do not change). The inputs are converted to
float64 inside the same environment. A
`StandardScaler` outside folds afterwards in float64. The float64 values are rounded
once to F32. The arithmetic runs on the backend (`affine64`, `quotient64`), under the
floating-point environment of the kernels for the `cpu` backend, so a caller's
flush-to-zero or rounding mode cannot change it. The calibration is not stored: the
artifact holds the folded head and the rule `linear@2`.

The comparison with scikit-learn is the sum of two bounds. fse against the float64
folded head is the bound of `tests/support/head_bound.py`. That head against scikit-learn's
own float64 evaluation of the calibrated model is a chain of `d + 6` float64 roundings of
the magnitudes (`tests/support/sklearn_calibrated.py`). For one-vs-rest the probabilities
are normalized by their sum, so a logit error `d` changes a component by the factor
`e^d - 1` plus the rounding of `exp@1` (2 ulps), the sum (`gamma(k - 1)`) and the
quotient; a component below `ln 2**-126` has an absolute error of at most `2**-126`. Rows
whose F32 sum is below `4 k 2**-126` (the deep underflow, for example all logits below
-104) are not compared with scikit-learn: fse returns the uniform row and class 0
there, where scikit-learn continues in float64. This limit is stated in the docstring of
`Classifier.from_sklearn`.

The new conformance cases cover all available instruction sets and the NumPy
backend, quantized tables, FastText, folded/normalized/vector/standalone paths,
k=1 and lane tails, threshold ties, saturation, subnormals and float-environment
restoration. Existing conformance cases and numerical bounds remain in force.

## int8 prefilter (no new arithmetic)

An exact search of an int8 store scores every selected row with `dot16(codes, u) * inv` (see "Reading stored rows"). The prefilter drops the rows that a bound from integer sums proves to lie outside the top k, so only the rest is scored by that rule. It adds no rounding rule, identifier or `transform_version`, and a search that uses it returns what one without it returns: the same ids and the same score bits. The reference is `fse._reference_prefilter` (`bounds`, `estimates`, `int8_pass`, `merge_heaps`, `survivors`); the native kernel (`_native_kernel_prefilter.h`, with the integer sums of `_native_kernel_int8.h`) matches the reference bit for bit in every variant, and the search that uses the rule follows. The constants 1.003 and 2^-14, and the limits below, are fixed; only the maintainer changes them. Only int8 stores are covered.

**Quantities.** A stored row has int8 codes `x` (each in [-128, 127]), `S = sum x_i^2` (exact) and the stored float32 inverse norm `inv`, the same array the exact path reads. A query is `u = normalize@1(q)` (float32), stored by `quantize_rows@1` as codes `qi` (int8, in [-127, 127]) with the float32 scale `s`. `v = s * qi` is exact (24 + 7 bits), `e = u - v`, `eps = ||e||`, and `d <= 1024`. `c = fl32(dot16(x, u) * inv)` is the cosine of the contract. Sums such as `||e||` without `fl` are real numbers.

**Rule.** Every operation in float64, in this grouping, in the IEEE standard environment (round to nearest even, no flush to zero, no contraction):

1. Per query: `n2` is the sum of the `u_i^2` and `eps_hat` the square root of the sum of the `(u_i - s * qi_i)^2`, both in index order. `E = 1.003 * eps_hat + 2^-14`. The query is *in the domain* if `n2` is finite and `n2 <= 1.002`.
2. Per row: `D = sum x_i * qi_i` and `S` are integers, exact. The row is *safe* if `inv` is finite, `inv >= 0` and `(inv * inv) * S <= 1 + 2^-20` (a NaN fails the comparison). `a = (s * D) * inv + 0.0`; the sum with `+0.0` only turns `-0.0` into `+0.0`, so equal estimates have equal bits.
3. `T` is the k-th largest `a` over all selected rows, `-inf` if there are fewer than k. The survivors are `C = { r : fl(a_r + E) >= fl(T - E) }`, with an inclusive comparison.
4. The exact rule scores the rows of `C`, at their global positions, and takes the top k.

If any query is outside the domain or any selected row is not safe, the rule does not run: the whole call takes the exact path over all rows, with the bits and errors of that path. The rule claims nothing for `d > 1024`.

**Theorem.** If every query is in the domain and every selected row is safe, `C` contains every row with `c >= K`, where K is the k-th largest exact score (ties at K included). The top k over `C` therefore have the ids and the score bits of the top k over all selected rows.

**Proof.**

*Zero rows.* If `S = 0` then `x = 0`, every product in `dot16` is a signed zero, so `c = +-0`, and `a = +-0` (inv is finite): `c = a` as numbers. From here on `S >= 1`.

*Identity.* In real numbers `p = x . u = x . v + x . e`. With `a*` the value of `a` before rounding, `a* = inv * (x . v)`, so `inv * p - a* = inv * (x . e)`, and by Cauchy-Schwarz `|inv * p - a*| <= inv * ||x|| * eps = nu * eps` with `nu = inv * sqrt(S)`.

*(a) Row norm.* `inv * inv` is exact in float64 (48 bits) and its product with `S` rounds once, so the test gives `inv^2 * S <= (1 + 2^-20) / (1 - 2^-53) < 1 + 2^-19`; hence `nu <= 1 + 2^-19` and, as `S >= 1`, `inv <= nu`. Only an upper bound on `nu` is needed: a small `inv` shrinks every error below. Every finite `inv` that passes is covered, whether or not it is `float32(1 / sqrt(S))`, so the proof does not depend on where the store came from. The sign condition makes `nu >= 0`, as the Cauchy-Schwarz step above uses it, and keeps corrupt stores out.

*(b) Query quantization.* `eps_hat` has one subtraction and one square per term, at most `d - 1` additions and one square root. A nonzero `e_i` is a multiple of 2^-149 (so are `u_i` and `s`), so no square underflows in float64. Hence `eps_hat = eps * (1 + theta)` with `|theta| < 2^-40`, and `nu * eps <= (1 + 2^-19)(1 + 2^-39) * eps_hat < 1.000003 * eps_hat` (both sides are 0 if `eps_hat = 0`, which a query that the int8 grid holds exactly reaches: then `E = 2^-14`).

*(c) Kernel rounding.* `dot16` passes each product through one fused multiply-add of its lane (at most `ceil(d/16)` per lane) and 4 additions, so every term meets at most `m = ceil(d/16) + 4 <= 68` roundings of relative error at most 2^-24. With `t` the float32 result, `|t - p| <= gamma_m * sum |x_i u_i| <= gamma_m * ||x|| * ||u||` (Cauchy-Schwarz); `c = fl32(t * inv)` adds one rounding. So `|c - inv * p| <= nu * ||u|| * (gamma_m (1 + 2^-24) + 2^-24)` plus the terms of (d). The test `n2 <= 1.002` (a sum of at most 1024 exact squares, relative error below 2^-43) gives `||u|| <= 1.001`, and with `gamma_68 < 4.0532 * 10^-6` the bound is below `1.0010019 * 4.1128 * 10^-6 < 4.2 * 10^-6 < 2^-17`. This is the only step that uses `||u||`. A query whose `n2` is outside the limit is outside the proven domain even where the slack would still cover it: `[2^-74, 2^-75]` has a squared norm of 2^-148, which `normalize@1` accepts but computes as subnormal, and its unit vector has `||u||^2 = 1.25`.

*(d) Underflow.* A rounding is `fl(z) = z (1 + delta) + eta` with `|delta| <= 2^-24`, `|eta| <= 2^-150`. At most `d + 15` operations of `dot16` and the last multiplication carry an `eta`; an `eta` is amplified by less than 1.0001 on its way through the other operations and multiplied by `inv <= 1 + 2^-19`. Together they stay below `1042 * 2^-150 < 2^-139`.

*(e) float64 evaluation.* `s * D` is exact (24 + 25 bits). The product with `inv`, `fl(a + E)` and `fl(T - E)` each round once. With `h = 2^-53`, `|a| <= (1 + h) * nu * (||u|| + eps) <= 1.002 + 1.001 * eps_hat` and `E <= (1 + h)((1 + h) * 1.003 * eps_hat + 2^-14) <= 1.0031 * eps_hat + (1 + h) * 2^-14`, the three roundings are below `2^-53 (3 max|a| + 2 E) < rho = 2^-50 (1 + eps_hat)`.

*Total.* Let `B = nu * eps + (c) + (d) < 1.000003 * eps_hat + 4.2 * 10^-6 + 2^-139`. Then `B + rho < 1.00001 * eps_hat + 4.3 * 10^-6`, while `E >= 1.0029 * eps_hat + 2^-14 * (1 - 2^-52)` (one rounding of `1.003 * eps_hat` and one of the sum). So `B + rho <= E`, with a margin of `0.0028 * eps_hat + 5.6 * 10^-5`; the absolute term is more than 14 times what is needed. For a safe row `|c - a*| <= B`. The roundings of (e) move `a` by `2^-53 |a|`, `fl(a + E)` by `2^-53 |a + E|` and `fl(T - E)` by `2^-53 |T - E|`, together below `rho` (`T` is the `a` of some row, so `|T| <= max |a|`). Hence for every safe row r:

    c_r <= fl(a_r + E)        and        fl(T - E) <= c_r   if   a_r >= T.

The first follows from `c_r <= a* + B <= a + B + 2^-53 |a|` and `fl(a + E) >= a + E - 2^-53 |a + E|`; the second from `c_r >= a* - B` and `fl(T - E) <= T - E + 2^-53 |T - E|` with `T <= a_r`.

*Completeness.* If there are fewer than k selected rows, `T = -inf`, `fl(T - E) = -inf`, and every row stays. Otherwise let R be k rows with the largest `a` (any tie-break), so `a_r >= T` for each. By the second inequality, `fl(T - E) <= c_r` for the k rows of R: at least k rows have `c >= fl(T - E)`, so `K >= fl(T - E)`. For every row with `c_r >= K` (ties at K included) `fl(a_r + E) >= c_r >= K >= fl(T - E)`, so `r` is in `C`. No k-th value per block is needed, so this holds for blocks of any size, also smaller than k.

*Monotonicity.* A pass may use a running threshold `max(T_in, k-th largest a of the rows seen so far)`, with `T_in` at most the k-th largest `a` of all rows (the k-th largest of a subset never exceeds that of the whole set), so it is at most `T`. As `y -> fl(y - E)` does not decrease, the floor of a running threshold is at most `fl(T - E)`: the rows kept under it contain `C`. Cutting them once more with the final `T` leaves exactly `C`, which depends on the store, the selection and the query, not on the order of the blocks or the threads. A row that is not safe never enters a heap or a list: the call ends and takes the exact path.

*Bits.* The score of a row is a function of the row, its inverse norm and the query alone, not of the other rows scored, the shards, the workers or a row selection (see the table above). The exact order is total: larger score first, then smaller position, `-0.0` equal to `+0.0`. The top k are the first k in that order, each with `c >= K`, so each lies in `C`; the first k of any set that holds them are the same rows with the same score bits. QED.

**Not covered.** `d > 1024`; float32 and float16 stores; `max_per_doc`, whose floor is a document score and not a row score; and every call that falls back, which gets the exact path as before.

**Tests.** `tests/test_reference_prefilter.py` checks the reference against exact fractions, the grouping of the rule and the domain rules (queries and rows outside it), `fl(a - E) <= c <= fl(a + E)` for every pair of random, extreme and adversarial stores of small size, where an adversarial row reaches `|c - a| = 0.99999 * eps_hat`, and the whole search against the exact path for ids and score bits, with ties, equal and zero rows and other block orders. `tests/test_reference_prefilter_meta.py` runs fixed witnesses on the rule and on the rule with one mistake, and the mistake must lose the top row (other ids), lose the fallback, or change the set of survivors: a bound of `0.9 * eps_hat` or zero; the threshold `T` instead of `fl(T - E)`; an exclusive comparison on an exact tie; no row test, and an unsafe row in a heap; no query test; a cut against the running threshold instead of the final one. `tests/test_conformance_prefilter.py` holds every native variant (portable, NEON SDOT, AVX2, each where it is built and the CPU runs it) and every backend to the reference bit for bit: the slack, the heaps, the kept rows and their bounds, the statuses, with the remainders of the tiles of four rows and two queries and of the 16-byte blocks of the dimension, the largest integer sums, rows against pages without access, every kind of unsafe row at the edges of the tiles, the overflow of the list and its compaction, and the floating-point environment of the caller (flush to zero and rounding toward +inf change no bit and are restored on every exit). `tests/test_conformance_prefilter_meta.py` runs the witnesses above on the kernel. The tests sample; the guarantee rests on the derivation above.

## Sign bits (`sign_bits@1`)

A binary prefilter of a search ranks rows by one bit per dimension. The rule adds no arithmetic: it has no floating point operation, no rounding rule and no `transform_version`. The reference is `fse._reference_bits` (`sign_bits`, `query_bits`, `hamming`, `candidates`, and for variant B `int4_codes`, `signed_sums`, `candidates_signed`). The bits of a shard are stored as the companion artifact `sign_bits-<first row>` ([store.md](store.md#sign-bits)); a search that uses them follows. The rule is fixed; only the maintainer changes it.

**Bit rule.** The bit of a stored value is 1 exactly if its integer bit pattern, read as a signed integer, is greater than 0:

- float32: the pattern as `int32` is greater than 0.
- float16: the pattern as `int16` is greater than 0.
- int8: the code is greater than 0.

For a finite value this is "the value is greater than 0": the sign bit is 0 and the value is not +0.0. `-0.0`, `+0.0`, the code 0 and a row of zeros give 0. A positive subnormal gives 1 and a negative subnormal gives 0, in every state of the thread. A floating point comparison would depend on flush to zero (FTZ) and denormals are zero (DAZ), which the numpy reference takes from the calling thread; reading the pattern does not. NaN does not occur, because stores hold finite values.

**Packing.** `np.packbits(bits, axis=1, bitorder="big")`: dimension 0 is the highest bit of byte 0, a row has `ceil(d / 8)` bytes (also for `d` below 8 and not a multiple of 8), and the padding bits of the last byte are 0. For finite values and the standard environment this equals `ubinary` of sentence-transformers.

**Query.** The bits of a query come from the float32 pattern of the query by the same rule.

**Distance and order.** The distance of a row is `popcount(b_row XOR b_query)`, an int32. The reference sums a table of 256 popcounts in int32 (`ndarray.sum(dtype=np.int32)`, which calls `np.add.reduce`) in blocks of rows, queries and bytes, and does not use `np.bitwise_count` (NumPy 1.26 has none). Candidates are ordered by ascending distance; at equal distance the smaller global position comes first. The order is total, so it depends on neither the blocks nor the threads. For a selection, the position of a candidate is `rows[local]`, never `first_row + local`, and the result of a call is padded once, with the widest list of the whole call, never per block of queries.

**Variant B (asymmetric, no format change).** With `s = max_j |q_j|` in float32 (greater than 0; a zero query has no scale), the int4 codes are `u_j = rint(float64(q_j) * 7 / float64(s))` in -7 to 7, ties to even. The rank is the integer `S = sum_j u_j * b_j`; a larger `S` comes first, at equal `S` the smaller position. It ranks like `sum_j u_j * (2 * b_j - 1)`, because `sum_j u_j` is the same for every row of a query. The dimension is at most `2^24 / 7 = 2396745`, which the reference enforces, so `S` (at most `7 * d`) is exact in int32 and in float32, and so is a Hamming distance (at most `d`).

**Tests.** `tests/test_reference_bits.py` checks the bit rule against a loop over `struct` bit patterns for float32, float16 and int8 (`-0.0`, positive and negative subnormals, code 0, a row of zeros; d = 1, 7, 8, 9, 63, 64, 65, 256, 1024), that FTZ and DAZ change no bit, the distance against a count of the bits, the order, selections and the int4 codes at the rounding boundaries. `tests/test_reference_bits_meta.py` runs fixed witnesses on the reference and on the reference with one mistake, and the mistake must change the answer: bit order `little`, a float comparison under DAZ, a threshold of at least 0, a tie to the larger position, set padding bits, `first_row + local` for `rows[local]`, padding per block.
