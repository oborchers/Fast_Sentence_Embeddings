# DR-151 A/A repeatability audit

All observations remain in their raw journals. A complete speed cell still requires
three valid rotating pairs and endpoint load strictly below five. A symmetric
A/A upper bound above 1.10 is labeled `noisy, re-measure in a quiet window`.
Do not headline its throughput until a quiet re-measurement supports it.
Incomplete cells have diagnostic A/A values only, never accepted speed values.

| Cell | Speed series | A/A B/A median (range) | Symmetric A/A band | Label |
|---|---|---|---|---|
| [dr151-model2vec-massive-1](../benchmarks/baseline/results/dr151-model2vec-massive-1.jsonl) | complete | 1.0078 (1.0025 to 1.0315) | 0.9695 to 1.0315 | within 10% A/A band |
| [dr151-model2vec-massive-10](../benchmarks/baseline/results/dr151-model2vec-massive-10.jsonl) | incomplete | 0.9537 (0.8825 to 1.0249) | 0.8825 to 1.1331 | noisy, re-measure in a quiet window |
| [dr151-model2vec-wikitext-1](../benchmarks/baseline/results/dr151-model2vec-wikitext-1.jsonl) | complete | 1.0141 (0.8998 to 1.0704) | 0.8998 to 1.1114 | noisy, re-measure in a quiet window |
| [dr151-model2vec-wikitext-10](../benchmarks/baseline/results/dr151-model2vec-wikitext-10.jsonl) | complete | 1.0941 (0.9862 to 1.1562) | 0.8649 to 1.1562 | noisy, re-measure in a quiet window |
| [dr151-matrix-fastembed-float32-w1](../benchmarks/baseline/results/dr151-matrix-fastembed-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-fastembed-float32-w10](../benchmarks/baseline/results/dr151-matrix-fastembed-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-glove-wiki-gigaword-300-float16-w1](../benchmarks/baseline/results/dr151-matrix-glove-wiki-gigaword-300-float16-w1.jsonl) | incomplete | 0.8921 (0.8921 to 0.8921) | 0.8921 to 1.1209 | noisy, re-measure in a quiet window |
| [dr151-matrix-glove-wiki-gigaword-300-float16-w10](../benchmarks/baseline/results/dr151-matrix-glove-wiki-gigaword-300-float16-w10.jsonl) | incomplete | 1.0497 (0.9967 to 1.1026) | 0.9069 to 1.1026 | noisy, re-measure in a quiet window |
| [dr151-matrix-glove-wiki-gigaword-300-float32-w1](../benchmarks/baseline/results/dr151-matrix-glove-wiki-gigaword-300-float32-w1.jsonl) | complete | 1.0404 (0.9471 to 1.0407) | 0.9471 to 1.0559 | within 10% A/A band |
| [dr151-matrix-glove-wiki-gigaword-300-float32-w10](../benchmarks/baseline/results/dr151-matrix-glove-wiki-gigaword-300-float32-w10.jsonl) | complete | 1.0290 (0.7850 to 1.1500) | 0.7850 to 1.2739 | noisy, re-measure in a quiet window |
| [dr151-matrix-glove-wiki-gigaword-300-int8-w1](../benchmarks/baseline/results/dr151-matrix-glove-wiki-gigaword-300-int8-w1.jsonl) | complete | 0.9284 (0.8730 to 0.9633) | 0.8730 to 1.1455 | noisy, re-measure in a quiet window |
| [dr151-matrix-glove-wiki-gigaword-300-int8-w10](../benchmarks/baseline/results/dr151-matrix-glove-wiki-gigaword-300-int8-w10.jsonl) | incomplete | 0.9888 (0.9888 to 0.9888) | 0.9888 to 1.0113 | within 10% A/A band |
| [dr151-matrix-minilm-float32-w1](../benchmarks/baseline/results/dr151-matrix-minilm-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-minilm-float32-w10](../benchmarks/baseline/results/dr151-matrix-minilm-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-32M-float16-w1](../benchmarks/baseline/results/dr151-matrix-potion-base-32M-float16-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-32M-float16-w10](../benchmarks/baseline/results/dr151-matrix-potion-base-32M-float16-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-32M-float32-w1](../benchmarks/baseline/results/dr151-matrix-potion-base-32M-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-32M-float32-w10](../benchmarks/baseline/results/dr151-matrix-potion-base-32M-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-32M-int8-w1](../benchmarks/baseline/results/dr151-matrix-potion-base-32M-int8-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-32M-int8-w10](../benchmarks/baseline/results/dr151-matrix-potion-base-32M-int8-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-8M-float16-w1](../benchmarks/baseline/results/dr151-matrix-potion-base-8M-float16-w1.jsonl) | complete | 0.9931 (0.8899 to 1.0134) | 0.8899 to 1.1237 | noisy, re-measure in a quiet window |
| [dr151-matrix-potion-base-8M-float16-w10](../benchmarks/baseline/results/dr151-matrix-potion-base-8M-float16-w10.jsonl) | incomplete | 1.0261 (1.0261 to 1.0261) | 0.9745 to 1.0261 | within 10% A/A band |
| [dr151-matrix-potion-base-8M-float32-w1](../benchmarks/baseline/results/dr151-matrix-potion-base-8M-float32-w1.jsonl) | incomplete | 0.8588 (0.8588 to 0.8588) | 0.8588 to 1.1645 | noisy, re-measure in a quiet window |
| [dr151-matrix-potion-base-8M-float32-w10](../benchmarks/baseline/results/dr151-matrix-potion-base-8M-float32-w10.jsonl) | complete | 1.2597 (1.0567 to 1.9543) | 0.5117 to 1.9543 | noisy, re-measure in a quiet window |
| [dr151-matrix-potion-base-8M-int8-w1](../benchmarks/baseline/results/dr151-matrix-potion-base-8M-int8-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-base-8M-int8-w10](../benchmarks/baseline/results/dr151-matrix-potion-base-8M-int8-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-multilingual-128M-float16-w1](../benchmarks/baseline/results/dr151-matrix-potion-multilingual-128M-float16-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-multilingual-128M-float16-w10](../benchmarks/baseline/results/dr151-matrix-potion-multilingual-128M-float16-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-multilingual-128M-float32-w1](../benchmarks/baseline/results/dr151-matrix-potion-multilingual-128M-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-multilingual-128M-float32-w10](../benchmarks/baseline/results/dr151-matrix-potion-multilingual-128M-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-multilingual-128M-int8-w1](../benchmarks/baseline/results/dr151-matrix-potion-multilingual-128M-int8-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-multilingual-128M-int8-w10](../benchmarks/baseline/results/dr151-matrix-potion-multilingual-128M-int8-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-retrieval-32M-float16-w1](../benchmarks/baseline/results/dr151-matrix-potion-retrieval-32M-float16-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-retrieval-32M-float16-w10](../benchmarks/baseline/results/dr151-matrix-potion-retrieval-32M-float16-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-retrieval-32M-float32-w1](../benchmarks/baseline/results/dr151-matrix-potion-retrieval-32M-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-retrieval-32M-float32-w10](../benchmarks/baseline/results/dr151-matrix-potion-retrieval-32M-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-retrieval-32M-int8-w1](../benchmarks/baseline/results/dr151-matrix-potion-retrieval-32M-int8-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-potion-retrieval-32M-int8-w10](../benchmarks/baseline/results/dr151-matrix-potion-retrieval-32M-int8-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-rust-float32-w1](../benchmarks/baseline/results/dr151-matrix-rust-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-rust-float32-w10](../benchmarks/baseline/results/dr151-matrix-rust-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-retrieval-mrl-en-v1-float16-w1](../benchmarks/baseline/results/dr151-matrix-static-retrieval-mrl-en-v1-float16-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-retrieval-mrl-en-v1-float16-w10](../benchmarks/baseline/results/dr151-matrix-static-retrieval-mrl-en-v1-float16-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-retrieval-mrl-en-v1-float32-w1](../benchmarks/baseline/results/dr151-matrix-static-retrieval-mrl-en-v1-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-retrieval-mrl-en-v1-float32-w10](../benchmarks/baseline/results/dr151-matrix-static-retrieval-mrl-en-v1-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-retrieval-mrl-en-v1-int8-w1](../benchmarks/baseline/results/dr151-matrix-static-retrieval-mrl-en-v1-int8-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-retrieval-mrl-en-v1-int8-w10](../benchmarks/baseline/results/dr151-matrix-static-retrieval-mrl-en-v1-int8-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-similarity-mrl-multilingual-v1-float16-w1](../benchmarks/baseline/results/dr151-matrix-static-similarity-mrl-multilingual-v1-float16-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-similarity-mrl-multilingual-v1-float16-w10](../benchmarks/baseline/results/dr151-matrix-static-similarity-mrl-multilingual-v1-float16-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-similarity-mrl-multilingual-v1-float32-w1](../benchmarks/baseline/results/dr151-matrix-static-similarity-mrl-multilingual-v1-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-similarity-mrl-multilingual-v1-float32-w10](../benchmarks/baseline/results/dr151-matrix-static-similarity-mrl-multilingual-v1-float32-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-similarity-mrl-multilingual-v1-int8-w1](../benchmarks/baseline/results/dr151-matrix-static-similarity-mrl-multilingual-v1-int8-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-static-similarity-mrl-multilingual-v1-int8-w10](../benchmarks/baseline/results/dr151-matrix-static-similarity-mrl-multilingual-v1-int8-w10.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-word2vec-google-news-300-float16-w1](../benchmarks/baseline/results/dr151-matrix-word2vec-google-news-300-float16-w1.jsonl) | incomplete | 0.9519 (0.9519 to 0.9519) | 0.9519 to 1.0505 | within 10% A/A band |
| [dr151-matrix-word2vec-google-news-300-float16-w10](../benchmarks/baseline/results/dr151-matrix-word2vec-google-news-300-float16-w10.jsonl) | incomplete | 1.1181 (0.9794 to 1.2568) | 0.7957 to 1.2568 | noisy, re-measure in a quiet window |
| [dr151-matrix-word2vec-google-news-300-float32-w1](../benchmarks/baseline/results/dr151-matrix-word2vec-google-news-300-float32-w1.jsonl) | incomplete | unavailable | unavailable | unavailable, incomplete |
| [dr151-matrix-word2vec-google-news-300-float32-w10](../benchmarks/baseline/results/dr151-matrix-word2vec-google-news-300-float32-w10.jsonl) | incomplete | 1.4809 (1.4809 to 1.4809) | 0.6752 to 1.4809 | noisy, re-measure in a quiet window |
| [dr151-matrix-word2vec-google-news-300-int8-w1](../benchmarks/baseline/results/dr151-matrix-word2vec-google-news-300-int8-w1.jsonl) | complete | 1.0458 (0.9527 to 1.0693) | 0.9352 to 1.0693 | within 10% A/A band |
| [dr151-matrix-word2vec-google-news-300-int8-w10](../benchmarks/baseline/results/dr151-matrix-word2vec-google-news-300-int8-w10.jsonl) | complete | 0.9840 (0.9302 to 1.0281) | 0.9302 to 1.0751 | within 10% A/A band |

Source summaries: [initial cells](../benchmarks/baseline/dr151_summary_20261101.json), [precision matrix](../benchmarks/baseline/dr151_matrix_summary_20261101.json).
The band is an empirical repeatability diagnostic, not a confidence interval.
For complete cells it uses the accepted speed series. For incomplete cells it
uses valid fse A/A operations from retained attempts and is marked diagnostic.
