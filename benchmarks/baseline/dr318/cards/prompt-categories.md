---
license: mit
language: [en]
library_name: fse
pipeline_tag: text-classification
datasets: [nvidia/Aegis-AI-Content-Safety-Dataset-2.0]
base_model: hotchpotch/bekko-static-embedding-v1
---

# fse/prompt-categories-aegis2-bekko

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

Output labels, in order: `hate_identity_hate`, `sexual`, `suicide_self_harm`, `violence`, `guns_illegal_weapons`, `pii_privacy`, `criminal_planning_confessions`, `harassment`, `controlled_regulated_substances`, `profanity`.

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
| all | hate_identity_hate | 1928 | 175 | 0.497992 | [0.426266, 0.587190] | 0.375000 | 0.788571 |
| all | sexual | 1928 | 99 | 0.500558 | [0.406320, 0.616632] | 0.302682 | 0.797980 |
| all | suicide_self_harm | 1928 | 38 | 0.291452 | [0.172559, 0.449754] | 0.170455 | 0.789474 |
| all | violence | 1928 | 176 | 0.399036 | [0.330241, 0.479580] | 0.308235 | 0.744318 |
| all | guns_illegal_weapons | 1928 | 62 | 0.348649 | [0.244922, 0.490957] | 0.262295 | 0.774194 |
| all | pii_privacy | 1928 | 99 | 0.581605 | [0.477148, 0.686430] | 0.356808 | 0.767677 |
| all | criminal_planning_confessions | 1928 | 502 | 0.770811 | [0.728799, 0.810404] | 0.653968 | 0.820717 |
| all | harassment | 1928 | 152 | 0.373130 | [0.299031, 0.455923] | 0.236902 | 0.684211 |
| all | controlled_regulated_substances | 1928 | 112 | 0.655274 | [0.558082, 0.749680] | 0.410714 | 0.821429 |
| all | profanity | 1928 | 98 | 0.510367 | [0.408914, 0.606536] | 0.253776 | 0.857143 |
| prompt_only | hate_identity_hate | 1115 | 100 | 0.536055 | [0.433625, 0.647564] | 0.393035 | 0.790000 |
| prompt_only | sexual | 1115 | 49 | 0.500668 | [0.357432, 0.690900] | 0.272727 | 0.795918 |
| prompt_only | suicide_self_harm | 1115 | 25 | 0.405832 | [0.267639, 0.626454] | 0.264368 | 0.920000 |
| prompt_only | violence | 1115 | 81 | 0.375170 | [0.291671, 0.494035] | 0.297030 | 0.740741 |
| prompt_only | guns_illegal_weapons | 1115 | 19 | 0.296083 | [0.139838, 0.530915] | 0.168675 | 0.736842 |
| prompt_only | pii_privacy | 1115 | 44 | 0.705800 | [0.557681, 0.827978] | 0.359223 | 0.840909 |
| prompt_only | criminal_planning_confessions | 1115 | 300 | 0.822333 | [0.771917, 0.866379] | 0.711048 | 0.836667 |
| prompt_only | harassment | 1115 | 73 | 0.369291 | [0.273551, 0.495995] | 0.235294 | 0.657534 |
| prompt_only | controlled_regulated_substances | 1115 | 53 | 0.663203 | [0.526900, 0.797458] | 0.400000 | 0.867925 |
| prompt_only | profanity | 1115 | 49 | 0.443307 | [0.322991, 0.592504] | 0.264706 | 0.918367 |
| without_exact_aart | hate_identity_hate | 1763 | 153 | 0.459499 | [0.387907, 0.550377] | 0.352239 | 0.771242 |
| without_exact_aart | sexual | 1763 | 91 | 0.486169 | [0.384367, 0.605807] | 0.295833 | 0.780220 |
| without_exact_aart | suicide_self_harm | 1763 | 38 | 0.292463 | [0.174092, 0.433929] | 0.172414 | 0.789474 |
| without_exact_aart | violence | 1763 | 148 | 0.338572 | [0.274316, 0.421972] | 0.276596 | 0.702703 |
| without_exact_aart | guns_illegal_weapons | 1763 | 57 | 0.346794 | [0.248783, 0.488129] | 0.254335 | 0.771930 |
| without_exact_aart | pii_privacy | 1763 | 98 | 0.585835 | [0.484387, 0.701234] | 0.367647 | 0.765306 |
| without_exact_aart | criminal_planning_confessions | 1763 | 359 | 0.688815 | [0.635274, 0.743985] | 0.583158 | 0.771588 |
| without_exact_aart | harassment | 1763 | 139 | 0.348525 | [0.270375, 0.446017] | 0.227384 | 0.669065 |
| without_exact_aart | controlled_regulated_substances | 1763 | 83 | 0.581130 | [0.461514, 0.703634] | 0.340426 | 0.771084 |
| without_exact_aart | profanity | 1763 | 98 | 0.512216 | [0.410208, 0.606124] | 0.257669 | 0.857143 |

prompt_only excludes response-bearing rows. without_exact_aart means without exactly recognized AART texts, not proof of source independence. Both subsets are reported for both heads without another selection round.

## Compatibility and limits

The category head needs a master build with multilabel support, or fse 2.1. A fresh fse 2.0.0 environment was checked: Safety loads with identical probability bits and predictions; categories raises UnsupportedArtifactError.

Only this predominantly English dataset has been evaluated. It also contains non-English prompts. Other language performance is unknown. Annotations are noisy and conversation context can influence them. Static embeddings can miss intent, negation, quotation and context. Use for assisted screening with human review, not automatic blocking without human oversight. The source contains harmful text. Examples here are harmless and independently written.

Claim class: correctness. No Linux comparison or speed measurement was run. The receipts include source checksums, environment, length summaries, RAM admission, numerical bounds, reference bits, grouped predictions, persistence and ScoreStore checks.

Preparation source commit: `5da66fe1f6ec549f391351da18fd4360c38b3a3e`.
