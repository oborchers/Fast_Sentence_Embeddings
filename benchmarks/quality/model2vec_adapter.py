"""Native Model2Vec semantics, distinct from the MTEB Sentence Transformers loader."""

from __future__ import annotations

from typing import Any

import numpy as np
from mteb.models.abs_encoder import AbsEncoder


class NativeModel2VecEncoder(AbsEncoder):
    """Use the source library's character cut, unknown-token rule and normalization."""

    def __init__(self, model_name: str, revision: str, **_kwargs: Any) -> None:
        from compare_snapshots import cached_snapshot
        from model2vec import StaticModel

        repo = model_name.removeprefix("model2vec/")
        folder = cached_snapshot(repo, revision)
        self.model = StaticModel.from_pretrained(folder)

    def encode(
        self,
        inputs: Any,
        *,
        task_metadata: Any,
        hf_split: str,
        hf_subset: str,
        prompt_type: Any = None,
        **kwargs: Any,
    ) -> np.ndarray:
        del task_metadata, hf_split, hf_subset, prompt_type, kwargs
        texts = [text for batch in inputs for text in batch["text"]]
        # Quality jobs use the bounded tokenizer pool, not joblib's all-core pool.
        return self.model.encode(texts, use_multiprocessing=False)
