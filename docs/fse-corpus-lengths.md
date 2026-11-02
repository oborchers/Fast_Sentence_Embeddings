# Speed corpus lengths

Exact counts from the original corpus-length statistics preserved in [the source archive](../benchmarks/baseline/evidence-2026-11.json). Words use whitespace splitting. Characters are Unicode code points.
POTION tokens follow pinned model truncation and unknown-token removal. No throughput is measured here. The 2M recipe uses source positions once; natural duplicate strings can occur.

| Corpus | Texts | Words mean / median | Characters mean / median | POTION tokens mean / median |
| --- | ---: | ---: | ---: | ---: |
| wikitext encode | 5,352 | 85.03 / 71 | 453.62 / 387 | 95.61 / 81 |
| wikitext pipeline | 5,352 | 85.03 / 71 | 453.62 / 387 | 95.61 / 81 |
| massive encode | 151,674 | 5.80 / 5 | 34.48 / 32 | 13.49 / 12 |
| massive pipeline | 151,674 | 5.80 / 5 | 34.48 / 32 | 13.49 / 12 |
| fse 101 unique 2M train chunks | 2,000,000 | 25.61 / 24 | 135.30 / 125 | 28.77 / 26 |
| setfit/20_newsgroups evaluation | 7,532 | 175.34 / 82 | 1096.86 / 483.5 | 170.65 / 119 |
| setfit/ade_corpus_v2_classification evaluation | 5,879 | 18.33 / 17 | 130.89 / 121 | 30.08 / 27 |
| setfit/ag_news evaluation | 7,600 | 37.72 / 37 | 235.30 / 231 | 50.75 / 48 |
| setfit/amazon_counterfactual_en evaluation | 5,000 | 19.53 / 17 | 102.04 / 86 | 24.17 / 20 |
| setfit/bbc-news evaluation | 1,000 | 384.51 / 334 | 2231.07 / 1951.5 | 387.45 / 390.5 |
| setfit/emotion evaluation | 2,000 | 19.15 / 17 | 96.59 / 86 | 20.21 / 18 |
| setfit/enron_spam evaluation | 2,000 | 304.15 / 149.5 | 1481.31 / 706.5 | 231.20 / 173 |
| setfit/hate_speech_offensive evaluation | 2,000 | 14.42 / 14 | 86.45 / 84 | 28.18 / 26 |
| setfit/imdb evaluation | 25,000 | 228.53 / 172 | 1293.79 / 962 | 269.49 / 228 |
| setfit/amazon_massive_scenario_en-US evaluation | 2,974 | 6.82 / 6 | 34.56 / 32 | 7.39 / 7 |
| setfit/SentEval-CR evaluation | 753 | 20.73 / 19 | 98.88 / 87 | 22.51 / 20 |
| setfit/sst5 evaluation | 2,210 | 19.19 / 18 | 102.71 / 99 | 23.00 / 22 |
| setfit/student-question-categories evaluation | 5,000 | 41.68 / 36 | 231.13 / 195 | 87.40 / 70 |
| setfit/subj evaluation | 2,000 | 24.80 / 23 | 129.66 / 120.5 | 28.38 / 26 |
| setfit/tweet_sentiment_extraction evaluation | 3,534 | 12.76 / 12 | 67.76 / 63 | 18.56 / 17 |
| fineweb/fineweb evaluation | 46,743 | 584.16 / 386 | 3530.06 / 2339 | 386.79 / 491 |

SetFit rows use the complete pinned test split. Training selection is recorded separately in the JSON. The provenance label correction changes no length value.
