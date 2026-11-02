# Why the GloVe SIF-10 numbers differ

This is a read-only protocol audit, without a new fit or inference run. The 1.x source is pinned at `fc3cbfb0509f56dfe2374c4c94fd68024f8810d6`.

| Evaluation | Correlation times 100 |
|---|---:|
| fse 1.1 README, historical GloVe SIF-10 | 71.43 Pearson |
| Existing fse 2.0 recipe output reproducing the historical setup | 71.3792 Pearson |
| New WikiText-fitted GloVe SIF-10, recovered native MTEB cache | 66.8440 Pearson |
| Same new model, published MTEB main score | 62.7461 Spearman |

The apparent 8.68-point gap combines different metrics and different fitted models. About 4.10 points are the Pearson/Spearman difference for the new model itself. The remaining Pearson gap is about 4.59 points. No ablation has assigned that remainder to individual choices.

## Verified common points

Both procedures use the test split. All 1,379 text pairs and labels match in the same order between the 1.x CSV and the pinned MTEB Arrow test split, without label rescaling. This is not a validation-versus-test discrepancy. Both use alpha 0.001, ten uncentered components and cosine comparison. Lowercased NLTK tokenization and hyphen/contraction handling follow the legacy procedure; the new boundary drops empty tokens explicitly. Normalization alone does not explain a cosine-quality difference.

## Verified protocol differences

The [1.x notebook](https://github.com/oborchers/Fast_Sentence_Embeddings/blob/fc3cbfb0509f56dfe2374c4c94fd68024f8810d6/notebooks/STS-Benchmarks.ipynb) loads `evaluation/sts-test.csv`, combines its two sentence columns in `CSplitIndexedList`, calls the SIF constructor with `lang_freq="en"`, then calls `m.train(sentences)` on those test texts. The constructor uses English wordfreq frequencies. The ten components are consequently fitted on the evaluation texts themselves, without labels. This is a transductive historical setup, not an independently held-out fit.

The new benchmark fits vocabulary probabilities and components using the pinned WikiText-103 training corpus only. It therefore evaluates a different finished SIF model. The original fitting scripts are preserved in the source archive. The weights use observed WikiText counts, rather than wordfreq. The components use all 859,346 fit paragraphs and do not see STS test texts.

The committed `recipes/01-sif-usif.ipynb` contains a separate historical diagnostic. It uses wordfreq, fits components on the test sentences, folds them into a 2.0 table and reports 71.3792 Pearson. Its code checks migrated vectors against its unfurled reference. This is existing notebook output, not a fresh reproduction in this audit. It demonstrates why the new held-out score must not be described as an established nine-point regression of the 2.0 engine. It does not attribute every residual difference or prove exact bitwise reproduction of 1.1.

## Sources and next protocol

[The compact quality summary](../benchmarks/baseline/quality-2026-11.json) preserves the old source pin, recovered correlations and exact protocol comparison. Original checks and cache receipts remain in the pinned source archive. The lost invocation journal is not reconstructed or given invented timestamps.

The maintainer requested wordfreq for subsequent SIF preparation. New wordfreq-weighted tables must receive new fit receipts and hashes. Keep components fitted on WikiText training unless the maintainer explicitly requests a separate historical diagnostic. Do not overwrite or relabel the existing WikiText-frequency results, and do not present them as a matched wordfreq comparison. The completed historical classic scores remain a separate fitted state.
