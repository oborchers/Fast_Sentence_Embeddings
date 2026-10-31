## Principle and stages

Binding for every 2.0 module (formerly under "2.0 design").

A frozen encoder is a pure function from one sentence to one finished row. Every row is computed in five stages, all inside the worker that owns the sentence:

| Stage | What | Where | Owned by |
|---|---|---|---|
| 1 tokenize | split the text into token spans | CPU | the Source |
| 2 token rows and factors | look up vocabulary ids, hash FastText ngrams; the factor of each row is 1 for a word of the finished table (times its int8 scale), `oov_weight` shared by the ngrams of an unknown FastText word | CPU for ids and ngrams, the backend kernel applies the factor | Source (ids), Encoder (factors) |
| 3 aggregation | combine the weighted rows of a sentence into one row | backend | an `Aggregation` |
| 4 postprocessing | `normalize@1` or nothing | backend | the Encoder's steps |
| 4b head (optional) | a linear head `z = A x + b` and its outputs (logits, probabilities, labels); folded into the table for a head on raw means | backend (`logits`, `outputs`) | a Classifier or Regressor |
| 5 store | quantize per row and write | backend `store`, in the worker; then the Sink | the Sink |

The tokenizer is part of the encoder contract: the manifest names it with an identifier like the aggregation (`fse.tokenizer`), and reading an encoder with an unknown tokenizer raises. `external` means the tokens were given by the caller, who then owns reproducing them; `fse_words@1` is the native tokenizer specified under [Tokenizers](tokenizers.md#tokenizers). A text Source runs stages 1 and 2 in one native pass without the GIL.

There are no Python hooks per row: they would hold the GIL, could not run on a device, could not be saved and would break bit-identity. Two escape hatches remain: own tokenization in Python inside the Source (stage 1, once per sentence, slower), and own processing of the finished output arrays. Models in which a row depends on more than its own sentence (attention over neighbors, sentence pairs) are out of scope.

Three rules keep new models possible without a new layer:

1. **An aggregation is not a sum.** It is defined by an initial accumulator, a combine step `combine(acc, f, x)` per weighted token row, a finish step `finish(acc, n)`, the result of an empty sentence and its output dimension. There is no group mode (many sentences to one row, as `CIndexedList` with repeated indices in 1.x): a document is encoded as one text, a mean over all its tokens, and a mean of equally weighted sentences is a reduction of their rows (`np.add.reduceat` over the group starts, then normalize), also over the batches of a stream; the migration guide shows both. It only adds, so it can come in a later 2.x if it is asked for.
2. **Factors (stage 2) are separate from the aggregation (stage 3).** Stage 2 supplies a factor and a stored row per contribution; the aggregation never sees weights or scales. A finished table has the factor 1 for every word, internally a ones array, so the kernels keep their arithmetic; removing that array from the kernel is a later optimization of its own, measured and compared bit for bit.
3. **The output dimension belongs to the encoder.** The aggregation gives the dimension of its rows (`Aggregation.output_dim(d)`: d for the weighted mean; a concatenation such as SWEM would give 2d), and `normalize` keeps it. The result is written into the manifest, and Sinks and Store read it from there, never from the word vectors. A head changes the dimension of its outputs (k instead of d): its rows are no encoder rows, so it is no step of the encoder but stage 4b of a model of its own, with its own targets and terms in the window ([Runtime](runtime.md#runtime)).

**Finished tables.** The weighted mean is linear: word weights, the removal of components and a projection (SIF, uSIF, PCA) applied to every word and ngram row before averaging give the same rows, within float32 rounding, as applied to every averaged row. A model therefore holds them in its table, made offline (the SIF and uSIF notebook with the 1.1.0 recipe), and encoding only averages; `normalize` is not linear and stays a step per row. Two promises are kept apart: a finished table gives, on every CPU backend, the bits of the reference applied to that table; how close it is to the model it was made from (rounding of the fold, quantization of float16 and int8 tables after it, the normalization rule of another library, such as the small value Model2Vec adds to the norm) is checked when the table is made, not by fse. `normalize` scales a row that a removal nearly cancels up to unit length, so for such rows the residues of rounding or quantization decide the direction. Heads: a head `A x` on the mean folds into the table, `A x + b` folds `A` and adds `b` separately (an empty sentence still gives `b`), and a head trained on normalized means runs as `A normalize(x) + b` on the full mean, a fused execution of the same algebra, never a different normalization; `normalize(A x)` is not `A normalize(x)`.

## Batch

Binding for every 2.0 module (formerly under "2.0 design").

A `Batch` is the input of an aggregation: `first_row` (the row of its first sentence), `offsets` (int64, n + 1 entries, from 0 to the number of tokens), `tokens` (uint32 vocabulary ids, or `OOV_TOKEN` for an unknown FastText word) and, for FastText, `ngram_offsets` and `ngrams` (uint32 buckets, one range per unknown word, in order). Sentence s is row `first_row + s`. Weights are not part of a batch, so a Source does not depend on the encoder; 2.1 adds token ids from Arrow and `TokenIds` as further Sources of the same batches.

