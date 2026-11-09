# SGD validation comparison

DR-311 (Toxizitäts-Classifier aus google/civil_comments außerhalb von fse trainieren und mit Modellkarte vorbereiten).

Correctness evidence from preparation revision a3294292486bfcf80336fafd15e03927fd74b1b5. No Linux comparison or speed claim.

All candidates completed three epochs over 1,804,874 training rows and evaluation on all 97,320 validation rows. Wrapper exit 0, heavy lock released, deadline observer ended before its deadline. Peak RSS 1,763,377,152 bytes (1.6423 GiB).

Selected alpha: 5.5405529693485529e-07 (C reference 1.0). C references 1 and 10 tie exactly in validation macro ROC AUC, so the predeclared larger-alpha tie-break chooses C reference 1. Full gains 0.016165567508, above the required 0.01, and replaces small under the prior rule. Existing artifacts are unchanged; the full checkpoint is retained outside Git. Label scope and official test remain held for Oliver's decision.

| C reference | alpha | Validation macro ROC AUC | Status |
|---:|---:|---:|---|
| 0.1 | 5.5405529693485523e-06 | 0.926404823620 | complete |
| 1.0 | 5.5405529693485529e-07 | 0.926404851520 | complete |
| 10.0 | 5.5405529693485526e-08 | 0.926404851520 | complete |

## Validation-selected thresholds

ROC AUC is area under the receiver operating characteristic curve. AP is average precision. Precision, recall and F1 use the stated thresholds. All scores come from fse float32 outputs.

| Label | Head | ROC AUC | AP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| toxicity | small | 0.881034 | 0.451563 | 0.413058 | 0.548429 | 0.471214 |
| toxicity | full | 0.883862 | 0.452638 | 0.408919 | 0.563030 | 0.473756 |
| obscene | small | 0.889689 | 0.114040 | 0.178025 | 0.243810 | 0.205788 |
| obscene | full | 0.905140 | 0.140071 | 0.168328 | 0.289524 | 0.212885 |
| threat | small | 0.913647 | 0.056528 | 0.116832 | 0.263393 | 0.161866 |
| threat | full | 0.950650 | 0.086428 | 0.140436 | 0.258929 | 0.182104 |
| insult | small | 0.898151 | 0.452061 | 0.452652 | 0.489686 | 0.470441 |
| insult | full | 0.902546 | 0.454770 | 0.435791 | 0.527648 | 0.477340 |
| identity_attack | small | 0.937043 | 0.145406 | 0.174190 | 0.453313 | 0.251672 |
| identity_attack | full | 0.953641 | 0.150269 | 0.177180 | 0.486446 | 0.259751 |
| sexual_explicit | small | 0.941871 | 0.083915 | 0.150307 | 0.243781 | 0.185958 |
| sexual_explicit | full | 0.962589 | 0.106585 | 0.148438 | 0.283582 | 0.194872 |
| macro | small | 0.910239 | 0.217252 | 0.247511 | 0.373735 | 0.291157 |
| macro | full | 0.926405 | 0.231794 | 0.246515 | 0.401526 | 0.300118 |

## Default threshold 0.5

ROC AUC is area under the receiver operating characteristic curve. AP is average precision. Precision, recall and F1 use the stated thresholds. All scores come from fse float32 outputs.

| Label | Head | ROC AUC | AP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| toxicity | small | 0.881034 | 0.451563 | 0.284016 | 0.756746 | 0.413020 |
| toxicity | full | 0.883862 | 0.452638 | 0.182020 | 0.919306 | 0.303874 |
| obscene | small | 0.889689 | 0.114040 | 0.034725 | 0.716190 | 0.066238 |
| obscene | full | 0.905140 | 0.140071 | 0.011137 | 0.948571 | 0.022015 |
| threat | small | 0.913647 | 0.056528 | 0.031118 | 0.656250 | 0.059418 |
| threat | full | 0.950650 | 0.086428 | 0.005277 | 0.995536 | 0.010499 |
| insult | small | 0.898151 | 0.452061 | 0.257746 | 0.772924 | 0.386579 |
| insult | full | 0.902546 | 0.454770 | 0.141022 | 0.934477 | 0.245062 |
| identity_attack | small | 0.937043 | 0.145406 | 0.066443 | 0.804217 | 0.122745 |
| identity_attack | full | 0.953641 | 0.150269 | 0.016840 | 0.987952 | 0.033116 |
| sexual_explicit | small | 0.941871 | 0.083915 | 0.033866 | 0.716418 | 0.064675 |
| sexual_explicit | full | 0.962589 | 0.106585 | 0.004631 | 0.995025 | 0.009220 |
| macro | small | 0.910239 | 0.217252 | 0.117986 | 0.737124 | 0.185446 |
| macro | full | 0.926405 | 0.231794 | 0.060155 | 0.963478 | 0.103964 |

## Proposed thresholds

Each cutoff maximizes F1 on validation only. Ties choose the highest float32 cutoff. These tuning-set results are not independent estimates. No proposed cutoff has been written into existing model artifacts.

| Label | Small | Full |
|---|---:|---:|
| toxicity | 0.72634363174438477 | 0.78959250450134277 |
| obscene | 0.99799209833145142 | 0.97370743751525879 |
| threat | 0.99948179721832275 | 0.99309051036834717 |
| insult | 0.83302062749862671 | 0.8461187481880188 |
| identity_attack | 0.97911280393600464 | 0.95431977510452271 |
| sexual_explicit | 0.99995118379592896 | 0.99226701259613037 |

All three candidates passed the unchanged data-dependent import bound on 256 validation texts, including probabilities, masks, strict equality and the unfolded path. Checkpoint SHA256 digests were checked against each complete validation receipt. The prior functional probe verified complete/standalone reloads and score/vector stores. Final full-model export and persistence verification are still pending.

Official test hold is active. No test-start marker or prepared payload exists. No retries or extensions occurred. The deadline and completion observers have ended. Full-run directory: /Users/oliverborchers/fse-dr311/sgd-full-01 (7.7 MiB). Source matrix/scaler retained separately. Final make check and coordinator diff review remain pending.
