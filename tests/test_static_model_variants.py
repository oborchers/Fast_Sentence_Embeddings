"""The encoders of one model under different rules: shared table and vocabulary, the
bounded cache that keeps the model's own, and threads asking at once."""

import threading

import numpy as np
import pytest

import fse
from fse._encoder._model2vec_rules import UNSET
from fse._static_model._variants import Variants
from support import static_model as sm

pytest.importorskip("model2vec")
pytest.importorskip("safetensors")


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return fse.StaticModel.from_pretrained(sm.model_folder(tmp_path_factory.mktemp("m") / "m"))


def test_every_variant_shares_the_table_and_the_vocabulary(model):
    variants = Variants(model._rules)
    own = variants.own(512, True)
    others = [variants.get(m, n) for m in (None, 3, 512) for n in (True, False)]
    assert variants.get(512, True) is own
    for encoder in others:
        assert encoder.word_vectors is own.word_vectors
        assert np.shares_memory(
            encoder.word_vectors.stored_vectors, own.word_vectors.stored_vectors
        )
        assert encoder._vocabulary is own._vocabulary
        assert encoder._vocabulary is not None
    spec = variants.get(3, False)._content.tokenizer
    assert (spec.max_tokens, spec.max_chars, spec.unknown) == (
        3,
        3 * model.median_token_length,
        "drop",
    )
    assert variants.get(None, False)._content.tokenizer.max_tokens is None
    assert variants.get(3, False).steps == ()
    assert variants.get(3, True).steps == ("normalize@1",)


def test_the_cache_is_bounded_and_keeps_the_own_encoder(model):
    variants = Variants(model._rules, limit=3)
    own = variants.own(512, True)
    for length in range(1, 12):
        variants.get(length, False)
        assert len(variants) <= 3
    assert variants.get(512, True) is own  # never evicted
    assert len(variants) <= 3


def test_the_least_recently_used_goes_first(model):
    variants = Variants(model._rules, limit=3)
    first = variants.get(1, True)
    second = variants.get(2, True)
    variants.get(3, True)
    assert variants.get(1, True) is first  # now the most recent
    variants.get(4, True)  # evicts 2
    assert variants.get(1, True) is first
    assert variants.get(2, True) is not second


def test_a_new_own_encoder_replaces_the_pinned_one(model):
    variants = Variants(model._rules, limit=2)
    variants.own(5, True)
    variants.own(6, True)
    for length in range(10, 16):
        variants.get(length, True)
    assert len(variants) == 2
    assert variants.get(6, True) is variants.get(6, True)


def test_threads_with_other_settings_get_the_rows_of_a_serial_run(model):
    texts = sm.texts(count=120)
    settings = [(UNSET, None), (3, None), (None, False), (7, True), (2, False), (512, None)]
    serial = {
        key: model.encode(texts, max_length=key[0], normalize=key[1]).tobytes() for key in settings
    }
    errors: list[BaseException] = []

    def work(key):
        try:
            for _ in range(5):
                got = model.encode(texts, max_length=key[0], normalize=key[1]).tobytes()
                assert got == serial[key]
        except BaseException as error:
            errors.append(error)

    pool = [threading.Thread(target=work, args=(key,)) for key in settings * 3]
    for thread in pool:
        thread.start()
    for thread in pool:
        thread.join()
    assert not errors
    assert (model.max_length, model.normalize) == (512, True)


def test_a_setter_that_fails_pins_nothing(model):
    variants = Variants(model._rules, limit=2)
    own = variants.own(512, True)
    with pytest.raises(ValueError, match="max_tokens"):
        variants.own(2**62, True)
    for length in range(1, 8):
        variants.get(length, False)
    assert variants.get(512, True) is own
