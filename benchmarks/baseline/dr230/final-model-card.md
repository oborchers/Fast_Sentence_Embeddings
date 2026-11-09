---
license: mit
language: [en]
library_name: fse
pipeline_tag: text-classification
datasets: [google/civil_comments]
base_model: hotchpotch/bekko-static-embedding-v1
tags: [multilabel, toxicity, linear-classifier]
---

# bekko-civil-comments-toxicity-classifier

An English two-label multilabel classifier for `toxicity` and `insult`, in that order. Training happened outside fse in scikit-learn. The frozen Bekko encoder was not trained. fse imports and runs the finished affine head. This preparation has not been uploaded.

## Requirements and use

Requires an fse master build with multilabel support, or fse 2.1 when released. fse 2.0.0 cannot load this artifact. The standalone folder contains the tokenizer and folded score table. The complete folder also retains encoder vectors.

```python
import fse

model = fse.Classifier.from_pretrained("/path/to/standalone")
scores = model.predict_proba(["A calm discussion of the proposed policy."])
mask = model.predict(["A calm discussion of the proposed policy."])
print(model.classes_)
```

## Data, training and selection

[Civil Comments](https://huggingface.co/datasets/google/civil_comments) contains historical English news-site discussion. The [source targets](https://www.tensorflow.org/datasets/catalog/civil_comments) are annotator fractions, binarized at >= 0.5 for training and evaluation.

- Dataset revision `f2970eb3a55777454c94069077cc8d9b5866312d` (CC0-1.0).
- Encoder `hotchpotch/bekko-static-embedding-v1` revision `c18167b8f244c1bc6de21ce9b710bac8a29c4615` (MIT), normalize=False.
- Training rows: 1,804,874. Exactly 3 full epochs.
- StandardScaler statistics were fitted incrementally on training vectors only. The standardized float32 matrix was streamed in batches of 8,192.
- Six independent SGDClassifier estimators used log_loss, L2, inverse-scaling learning rate (eta0=0.01, power_t=0.5), averaged coefficients, seed 17, and fixed train-wide balanced class weights N/(2 * class_count).
- Selected alpha: 5.5405529693485529e-07, from alpha=1/(C*N), C in {0.1, 1, 10}.
- Selection used the entire validation split only. The full six-label head's macro ROC AUC was 0.926405 versus 0.910239 for the small Bekko head, a gain of 0.016166, exceeding the predeclared 0.01 replacement boundary. An exact tie between C references 1 and 10 chose the larger alpha, corresponding to C=1.
- After validation, the maintainer chose only the already fitted toxicity and insult estimators. No refit or new alpha selection followed. The scaler was folded in float64 and imported through Classifier.from_arrays(link="multilabel").
- Source revision `980a5147e2b9ab7c0d9c4f5504920a7f69930b59`; scikit-learn 1.9.1, NumPy 2.5.4, Python 3.13.16.

## Thresholds and removed labels

Each retained threshold maximizes F1 on validation only, choosing the highest float32 cutoff on a tie. Inference is strictly score > cutoff, excluding equality. These thresholds were stored before test. Scores are not calibrated probabilities of toxicity or annotator agreement. No calibration study was performed.

| Retained label | Stored threshold |
|---|---:|
| toxicity | 0.78959250450134277 |
| insult | 0.8461187481880188 |

`severe_toxicity` was removed before model selection: only 13 of 1,804,874 training rows were positive. No validation estimate was computed for that removed label.

The maintainer removed `obscene`, `threat`, `identity_attack` and `sexual_explicit` after reviewing validation. Their maximum-F1 validation scores were lower than the retained labels. The table preserves all six full-head validation results to document that decision. These thresholds were optimized on this same split and are not independent estimates.

| Label | Positive | ROC AUC | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| toxicity | 7671 | 0.883862 | 0.452638 | 0.408919 | 0.563030 | 0.473756 |
| obscene | 525 | 0.905140 | 0.140071 | 0.168328 | 0.289524 | 0.212885 |
| threat | 224 | 0.950650 | 0.086428 | 0.140436 | 0.258929 | 0.182104 |
| insult | 5769 | 0.902546 | 0.454770 | 0.435791 | 0.527648 | 0.477340 |
| identity_attack | 664 | 0.953641 | 0.150269 | 0.177180 | 0.486446 | 0.259751 |
| sexual_explicit | 201 | 0.962589 | 0.106585 | 0.148438 | 0.283582 | 0.194872 |
| Macro | | 0.926405 | 0.231794 | 0.246515 | 0.401526 | 0.300118 |

## Official test

Exactly one evaluation on the official test split (97,320 rows) followed the frozen two-label decision and threshold persistence. All test scores came from the saved standalone fse model. No model or threshold changed afterwards. AP means non-interpolated average precision. ROC AUC means area under the receiver operating characteristic curve. Undefined precision/F1 is zero; absent positive or negative classes abort evaluation.

| Label | Positive | ROC AUC | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| toxicity | 7777 | 0.883407 | 0.447505 | 0.412795 | 0.557541 | 0.474372 |
| insult | 5776 | 0.904925 | 0.449595 | 0.436526 | 0.523892 | 0.476235 |
| Macro | | 0.894166 | 0.448550 | 0.424661 | 0.540717 | 0.475304 |

## Limits, license and evidence

Use for assisted screening and research with human review. Other languages, domains and current usage were not evaluated. Static embeddings can miss context, negation, quotation and intent. Identity-related bias is a known [corpus concern](https://arxiv.org/abs/1903.04561). No identity subgroup or calibration metrics were measured. This head does not establish legality or harmfulness, and does not output the five removed labels.

Both the fitted head and encoder use MIT. Source data is CC0-1.0. metrics.json records selection, versions, correctness checks and test metrics. SHA256SUMS.json covers the prepared files. Numerical import bounds and complete/standalone bit identity passed. Labels with fewer than 100 test positives are marked unreliable: none. Evidence class: correctness. No Linux comparison or speed claim.
