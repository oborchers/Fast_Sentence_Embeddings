"""Model cards generated from the complete Aegis evaluation receipt only."""

from __future__ import annotations


def render(record, name):
    if record["status"] != "complete":
        raise ValueError("incomplete runs cannot produce a prepared card")
    config, selection = record["config"], record["selection"]
    spec = next(s for s in config["encoders"] if s["name"] == selection["encoder"])
    suffix = "potion-8m" if selection["encoder"] == "potion" else "bekko"
    model_id = f"fse/{suffix}-aegis2-prompt-{name}-classifier"
    lines = [
        "---",
        "license: mit",
        "language: [en]",
        "library_name: fse",
        "pipeline_tag: text-classification",
        "datasets: [nvidia/Aegis-AI-Content-Safety-Dataset-2.0]",
        f"base_model: {spec['repo']}",
        "---",
        "",
        f"# {model_id}",
        "",
        "A linear prompt classifier trained outside fse using scikit-learn. "
        "This is a complete head, retaining its encoder for use with fse.Heads. "
        "The local artifact has not been uploaded.",
        "",
        "## Use",
        "",
        "```python",
        "import fse",
        "",
        'safety = fse.Classifier.from_pretrained("/path/to/safety-complete")',
        'categories = fse.Classifier.from_pretrained("/path/to/categories-complete")',
        "heads = fse.Heads(safety=safety, categories=categories)",
        'scores = heads.predict(["How do I bake sourdough bread?"], output="proba")',
        "```",
        "",
        "Both complete heads use the same encoder and share one tokenization and pooling pass. "
        "Independently folded standalone heads cannot be grouped. No standalone files are supplied.",
        "",
        "Output labels, in order: " + ", ".join(f"`{s}`" for s in config["labels"][name]) + ".",
        "",
        "## Targets and provenance",
        "",
        "Safety predicts safe or unsafe. The source maps Needs Caution to safe (permissive policy). "
        "Categories are positive only when prompt_label is unsafe AND that category is named. "
        "The two heads make independent predictions, so their predictions can disagree. "
        "Threat and Sexual (minor) are excluded by the reviewed protocol.",
        "",
        "[NVIDIA Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) "
        "is CC BY 4.0. Credit: Shaona Ghosh, Prasoon Varshney, Makesh Narsimhan Sreedhar, "
        "Aishwarya Padmakumar, Traian Rebedea, Jibin Rajan Varghese and Christopher Parisien "
        "(NVIDIA), [Aegis2.0](https://arxiv.org/abs/2501.09004). "
        "Human conversation-level annotations supply prompt labels and category unions "
        "(at least one annotator). Response-bearing rows can reflect the response as well. "
        "Only prompt strings are encoded. Response strings and response labels are discarded. "
        "Refusal files are never loaded.",
        "",
        "Credit: Bhaktipriya Radharapu, Kevin Robinson, Lora Aroyo and Preethi Lahoti, "
        "Google, [AART](https://arxiv.org/abs/2311.08592). AART remains in training. "
        "Its PaLM-generated prompts are identified only by exact CSV text matches. "
        "The deleted Google source is evidenced by its archived HEAD and the "
        "[same-commit copy](https://github.com/xxxx-dddd/aart-ai-safety-dataset/tree/"
        "fcdb2a8a3343a992cd8cbfd53eac797ac047c26f), whose LICENSE states CC BY 4.0. "
        "This provenance does not prove that all AART-derived prompts are detected. "
        "PaLM API terms were not assessed. This is a provenance report, not legal advice.",
        "",
        "Other source credits: [Anthropic hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) "
        "(MIT), [DAN by Xinyue Shen and colleagues](https://github.com/verazuo/jailbreak_llms) "
        '(MIT; source says "intended for research purposes only"), and '
        "[Do-Not-Answer by Yuxia Wang and colleagues](https://github.com/Libr-AI/do-not-answer) "
        "(Apache-2.0). NVIDIA distributes their mixture under CC BY 4.0. "
        "The fitted head and encoder are MIT, with source attribution retained here.",
        "",
        f"Encoder: `{spec['repo']}` at `{spec['revision']}`, normalize=False. "
        "No dimension truncation. Dataset revision: `d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`.",
        "",
        "| Output | Source category |",
        "|---|---|",
    ]
    lines.extend(f"| {label} | {category} |" for label, category in config["categories"].items())
    chosen = selection["selected"][name]
    lines += [
        "",
        "## Training and selection",
        "",
        f"All {record['data']['train']['final']['rows']} cleaned training rows are used, "
        "at natural class proportions, with balanced class weights. "
        "StandardScaler precedes LogisticRegression (lbfgs), with OneVsRestClassifier "
        "for categories, n_jobs=1 and BLAS/OpenMP threads=1. "
        f"Seed {config['seed']}, max_iter={config['max_iter']}, selected C={chosen['C']}.",
        "",
        f"Both encoders use C={config['C']}. Safety selects unsafe Average Precision (AP). "
        "Categories select macro AP over labels with at least 20 validation positives. "
        "Ties choose smaller C. Bekko replaces POTION only when both AP gains are at least 0.01. "
        "No test result changes these choices. Failed convergence candidates are recorded and excluded.",
        "",
        "Normalization for leakage and grouping: NFC, collapse whitespace, trim, casefold "
        "(prompt-normalize@1). Train excludes normalized matches in validation or test. "
        "Validation excludes test matches. Train drops conflicting-label groups, merges "
        "same-label categories, and chooses the smallest ID. AART and has_response flags use OR. "
        "REDACTED, reconstructable and empty rows are excluded. The data report gives each stage.",
        "",
        record["data"]["test_access"],
        "",
        "## Held-out evaluation",
        "",
        "One evaluation after model freeze, using the saved complete fse heads. "
        "AP is sklearn.metrics.average_precision_score, not trapezoidal PR area. "
        "AP intervals are descriptive 95% percentile intervals, resampling normalized prompt "
        "groups 1,000 times with seed 17. Draws without positives are counted and omitted. "
        "Rows below 20 positives are exploratory and strongly uncertain. "
        "The full metrics receipt includes negative counts, groups, F1, ROC AUC and omitted draws.",
        "",
        "The binary rule is logit > 0 (equality selects safe). "
        "Category scores must be strictly > 0.5, with equality false. "
        "Balanced weights make scores uncalibrated. They are not probabilities of unsafe intent.",
        "",
        "| Subset | Label | Rows | Positive | AP | AP 95% interval | Precision | Recall |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for subset, outputs in record["test"].items():
        summary = outputs[name]
        for label, values in summary["per_label"].items():
            low, high = values["ap_percentile_95"]
            lines.append(
                f"| {subset} | {label} | {summary['rows']} | {values['positive']} | "
                f"{values['average_precision']:.6f} | [{low:.6f}, {high:.6f}] | "
                f"{values['precision']:.6f} | {values['recall']:.6f} |"
            )
    lines += [
        "",
        "prompt_only excludes response-bearing rows. without_exact_aart means "
        "without exactly recognized AART texts, not proof of source independence. "
        "Both subsets are reported for both heads without another selection round.",
        "",
        "## Compatibility and limits",
        "",
        "The category head needs a master build with multilabel support, or fse 2.1. "
        "A fresh fse 2.0.0 environment was checked: " + record["compatibility"]["summary"],
        "",
        "Only this predominantly English dataset has been evaluated. It also contains "
        "non-English prompts. Other language performance is unknown. "
        "Annotations are noisy and conversation context can influence them. Static embeddings "
        "can miss intent, negation, quotation and context. Use for assisted screening with human "
        "review, not automatic blocking without human oversight. The source contains harmful text. "
        "Examples here are harmless and independently written.",
        "",
        "Claim class: correctness. No Linux comparison or speed measurement was run. "
        "The receipts include source checksums, environment, length summaries, RAM admission, "
        "numerical bounds, reference bits, grouped predictions, persistence and ScoreStore checks.",
        "",
        f"Preparation source commit: `{record['environment']['git_revision']}`.",
        "",
    ]
    return "\n".join(lines)
