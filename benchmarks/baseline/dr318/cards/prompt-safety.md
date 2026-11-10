---
license: mit
language: [en]
library_name: fse
pipeline_tag: text-classification
datasets: [nvidia/Aegis-AI-Content-Safety-Dataset-2.0]
base_model: hotchpotch/bekko-static-embedding-v1
---

# fse/bekko-aegis2-prompt-safety-classifier

A linear prompt classifier trained outside fse using scikit-learn. This is a complete head, retaining its encoder for use with fse.Heads. The local artifact has not been uploaded.

## Use

```python
import fse

safety = fse.Classifier.from_pretrained("/path/to/safety-complete")
categories = fse.Classifier.from_pretrained("/path/to/categories-complete")
heads = fse.Heads(safety=safety, categories=categories)
scores = heads.predict(["How do I bake sourdough bread?"], output="proba")
```

Both complete heads use the same encoder and share one tokenization and pooling pass. Independently folded standalone heads cannot be grouped. No standalone files are supplied.

Output labels, in order: `safe`, `unsafe`.

## Targets and provenance

Safety predicts safe or unsafe. The source maps Needs Caution to safe (permissive policy). Categories are positive only when prompt_label is unsafe AND that category is named. The two heads make independent predictions, so their predictions can disagree. Threat and Sexual (minor) are excluded by the reviewed protocol.

[NVIDIA Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) is CC BY 4.0. Credit: Shaona Ghosh, Prasoon Varshney, Makesh Narsimhan Sreedhar, Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan Varghese and Christopher Parisien (NVIDIA), [Aegis2.0](https://arxiv.org/abs/2501.09004). Human conversation-level annotations supply prompt labels and category unions (at least one annotator). Response-bearing rows can reflect the response as well. Only prompt strings are encoded. Response strings and response labels are discarded. Refusal files are never loaded.

Credit: Bhaktipriya Radharapu, Kevin Robinson, Lora Aroyo and Preethi Lahoti, Google, [AART](https://arxiv.org/abs/2311.08592). AART remains in training. Its PaLM-generated prompts are identified only by exact CSV text matches. The deleted Google source is evidenced by its archived HEAD and the [same-commit copy](https://github.com/xxxx-dddd/aart-ai-safety-dataset/tree/fcdb2a8a3343a992cd8cbfd53eac797ac047c26f), whose LICENSE states CC BY 4.0. This provenance does not prove that all AART-derived prompts are detected. PaLM API terms were not assessed. This is a provenance report, not legal advice.

Other source credits: [Anthropic hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) (MIT), [DAN by Xinyue Shen and colleagues](https://github.com/verazuo/jailbreak_llms) (MIT; source says "intended for research purposes only"), and [Do-Not-Answer by Yuxia Wang and colleagues](https://github.com/Libr-AI/do-not-answer) (Apache-2.0). NVIDIA distributes their mixture under CC BY 4.0. The fitted head and encoder are MIT, with source attribution retained here.

Encoder: `hotchpotch/bekko-static-embedding-v1` at `c18167b8f244c1bc6de21ce9b710bac8a29c4615`, normalize=False. No dimension truncation. Dataset revision: `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`.

| Output | Source category |
|---|---|
| hate_identity_hate | Hate/Identity Hate |
| sexual | Sexual |
| suicide_self_harm | Suicide and Self Harm |
| violence | Violence |
| guns_illegal_weapons | Guns and Illegal Weapons |
| pii_privacy | PII/Privacy |
| criminal_planning_confessions | Criminal Planning/Confessions |
| harassment | Harassment |
| controlled_regulated_substances | Controlled/Regulated Substances |
| profanity | Profanity |

## Training and selection

All 22537 cleaned training rows are used, at natural class proportions, with balanced class weights. StandardScaler precedes LogisticRegression (lbfgs), with OneVsRestClassifier for categories, n_jobs=1 and BLAS/OpenMP threads=1. Seed 17, max_iter=1000, selected C=0.1.

Both encoders use C=[0.1, 1.0, 10.0]. Safety selects unsafe Average Precision (AP). Categories select macro AP over labels with at least 20 validation positives. Ties choose smaller C. Bekko replaces POTION only when both AP gains are at least 0.01. No test result changes these choices. Failed convergence candidates are recorded and excluded.

Normalization for leakage and grouping: NFC, collapse whitespace, trim, casefold (prompt-normalize@1). Train excludes normalized matches in validation or test. Validation excludes test matches. Train drops conflicting-label groups, merges same-label categories, and chooses the smallest ID. AART and has_response flags use OR. REDACTED, reconstructable and empty rows are excluded. The data report gives each stage.

Planner inspected test counts and structure before training. Preparation reads test for frozen source filters and leakage exclusion only. No test-driven selection.

## Held-out evaluation

One evaluation after model freeze, using the saved complete fse heads. AP is sklearn.metrics.average_precision_score, not trapezoidal PR area. AP intervals are descriptive 95% percentile intervals, resampling normalized prompt groups 1,000 times with seed 17. Draws without positives are counted and omitted. Rows below 20 positives are exploratory and strongly uncertain. The full metrics receipt includes negative counts, groups, F1, ROC AUC and omitted draws.

The binary rule is logit > 0 (equality selects safe). Category scores must be strictly > 0.5, with equality false. Balanced weights make scores uncalibrated. They are not probabilities of unsafe intent.

| Subset | Label | Rows | Positive | AP | AP 95% interval | Precision | Recall |
|---|---|---:|---:|---:|---|---:|---:|
| all | safe | 1928 | 889 | 0.805655 | [0.780198, 0.831410] | 0.732240 | 0.753656 |
| all | unsafe | 1928 | 1039 | 0.822219 | [0.793350, 0.850069] | 0.783810 | 0.764196 |
| prompt_only | safe | 1115 | 551 | 0.854725 | [0.826409, 0.879979] | 0.771127 | 0.794918 |
| prompt_only | unsafe | 1115 | 564 | 0.836653 | [0.802517, 0.871672] | 0.793419 | 0.769504 |
| without_exact_aart | safe | 1763 | 875 | 0.811870 | [0.785202, 0.838670] | 0.734873 | 0.763429 |
| without_exact_aart | unsafe | 1763 | 888 | 0.788948 | [0.756649, 0.822702] | 0.757611 | 0.728604 |

prompt_only excludes response-bearing rows. without_exact_aart means without exactly recognized AART texts, not proof of source independence. Both subsets are reported for both heads without another selection round.

## Compatibility and limits

The category head needs a master build with multilabel support, or fse 2.1. A fresh fse 2.0.0 environment was checked: Safety loads with identical probability bits and predictions; categories raises UnsupportedArtifactError.

Only this predominantly English dataset has been evaluated. It also contains non-English prompts. Other language performance is unknown. Annotations are noisy and conversation context can influence them. Static embeddings can miss intent, negation, quotation and context. Use for assisted screening with human review, not automatic blocking without human oversight. The source contains harmful text. Examples here are harmless and independently written.

Claim class: correctness. No Linux comparison or speed measurement was run. The receipts include source checksums, environment, length summaries, RAM admission, numerical bounds, reference bits, grouped predictions, persistence and ScoreStore checks.

Preparation source commit: `5da66fe1f6ec549f391351da18fd4360c38b3a3e`.
