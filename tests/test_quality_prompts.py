"""Requested regression for ST's consuming tokenizer dispatch, without model weights."""

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

pytest.importorskip("sentence_transformers")
from sentence_transformers.base.modules.transformer import Transformer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def observer(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    from _quality.prompts import Observer

    cell = {
        "id": "smoke-ref-all-MiniLM-L6-v2_STSBenchmark",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "task": "STSBenchmark",
    }
    result = Observer(cell, None)
    result.active = True
    return result


def test_real_st_dispatch_consumes_text_but_observer_keeps_actual_tokens(observer):
    """Call ST 6.2's real dispatch and tokenizer, without constructing a Transformer model."""
    words = ["[UNK]", "A", "short", "sentence", "."]
    tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    processor = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")
    first = SimpleNamespace(processor=processor)
    first._call_single_modality_processor = MethodType(
        Transformer._call_single_modality_processor, first
    )
    original = MethodType(Transformer._call_processor, first)
    actual_texts = ["A short sentence."]
    inputs = {"text": actual_texts}
    returned = []

    def delegate(*args, **kwargs):
        assert args[1] is inputs
        assert inputs["text"] is actual_texts
        features = original(*args, **kwargs)
        returned.append(features)
        return features

    features = observer.processing(delegate)(
        "text",
        inputs,
        {"text": {}, "audio": {}, "video": {}, "image": {}},
        {"return_tensors": "pt"},
    )
    assert "text" not in inputs  # The actual third-party pop(), not a simulated mutation.
    assert features is returned[0]
    assert actual_texts == ["A short sentence."]
    records = observer.finish()  # Before the fix this raises: no prompt observations.
    assert len(records) == 1
    assert records[0]["prompt_type"] is None
    assert records[0]["prompt"] is None
    assert records[0]["prefix_occurrences_min"] == records[0]["prefix_occurrences_max"] == 0
    assert records[0]["first_token_ids"] == features["input_ids"][0, :8].tolist()
    assert records[0]["first_token_ids"]


def test_missing_smoke_observations_still_fail(observer):
    with pytest.raises(ValueError, match="missing smoke prompt observations"):
        observer.finish()
