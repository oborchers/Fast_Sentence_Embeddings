"""Temporary workspaces and pinned public downloads for the recipes."""

import json
import os
import shutil
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

POTION = "minishlab/potion-base-8M"
POTION_REVISION = "bf8b056651a2c21b8d2565580b8569da283cab23"
DATASETS = {
    "SST-5": ("SetFit/sst5", "e51bdcd8cd3a30da231967c1a249ba59361279a3"),
    "AG News": ("SetFit/ag_news", "ca5ba619eb034211db5f70932b6702efd21e7c73"),
}
ROOT = Path(__file__).resolve().parent.parent


class Recipe:
    """Own generated files. Read existing Hub snapshots without changing them."""

    def __init__(self):
        self.scratch = TemporaryDirectory(prefix="fse-recipe-")
        self.work = Path(self.scratch.name)
        self.previous = {}
        settings = {
            "XDG_CACHE_HOME": str(self.work / "cache"),
            "NUMBA_CACHE_DIR": str(self.work / "numba"),
            "MPLCONFIGDIR": str(self.work / "matplotlib"),
            "DSPY_CACHEDIR": str(self.work / "dspy"),
            "ANONYMIZED_TELEMETRY": "False",
            "TOKENIZERS_PARALLELISM": "false",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
        for name, value in settings.items():
            self.previous[name] = os.environ.get(name)
            os.environ[name] = value
        print(f"Recipe run: {date.today().isoformat()}. Generated files are temporary.")

    def source_file(self, repo, revision, filename, *, repo_type="model"):
        from huggingface_hub import hf_hub_download

        prefix = "models" if repo_type == "model" else "datasets"
        cached = (
            Path.home()
            / ".cache/huggingface/hub"
            / (prefix + "--" + repo.replace("/", "--"))
            / "snapshots"
            / revision
            / filename
        )
        if cached.is_file():
            return cached
        return Path(
            hf_hub_download(
                repo,
                filename,
                revision=revision,
                repo_type=repo_type,
                cache_dir=self.work / "hub",
                token=False,
            )
        )

    def model_folder(self):
        # Keep the three Model2Vec files together, even with a partial existing cache.
        files = ["config.json", "model.safetensors", "tokenizer.json"]
        paths = [self.source_file(POTION, POTION_REVISION, f) for f in files]
        folder = self.work / "potion"
        folder.mkdir(exist_ok=True)
        for filename, path in zip(files, paths, strict=True):
            target = folder / filename
            if not target.exists():
                shutil.copyfile(path, target)
        return folder

    def encoder(self):
        import fse

        return fse.Encoder.from_pretrained(self.model_folder())

    def dataset(self, name, split):
        import numpy as np

        repo, revision = DATASETS[name]
        path = self.source_file(repo, revision, f"{split}.jsonl", repo_type="dataset")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        return [r["text"] for r in rows], np.array([r["label"] for r in rows])

    def news(self, count):
        texts, _ = self.dataset("AG News", "test")
        return texts[:count]

    def close(self):
        self.scratch.cleanup()
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
