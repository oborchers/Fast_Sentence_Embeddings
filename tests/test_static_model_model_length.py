"""The two cuts of a text, at the boundary: the first max_length * median_token_length
characters (code points, not bytes or UTF-16 units), then max_length tokens; and the
defaults of the model after every per-call override."""

import pytest

import fse
from support import static_model as sm

model2vec = pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
PATTERN = "ab \U0001f600 hello xẕ ée " * 30


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    folder = sm.model_folder(tmp_path_factory.mktemp("m") / "m")
    return fse.StaticModel.from_pretrained(folder), model2vec.StaticModel.from_pretrained(folder)


@pytest.mark.parametrize("length", [1, 2, 3, 6, 40])
def test_the_character_cut_is_by_code_points_and_matches_model2vec(models, length):
    ours, theirs = models
    cut = length * ours.median_token_length
    assert cut == length * theirs.median_token_length
    texts = [PATTERN[:n] for n in range(max(cut - 3, 0), cut + 4)]
    want = theirs.encode(texts, max_length=length)
    assert sm.close(ours, texts, want, max_length=length) <= 1
    # everything past the cut is ignored, one character after it as well as many
    at, above, far = (ours.encode([PATTERN[:n]], max_length=length) for n in (cut, cut + 1, 10**4))
    assert at.tobytes() == above.tobytes() == far.tobytes()


def test_the_token_cut_comes_after_the_character_cut_and_counts_unknown_ids(models):
    ours, theirs = models
    texts = ["zzzq " * 40, "hello zzzq " * 40, "[UNK] [UNK] hello world " * 30]
    for length in (1, 3, 8):
        want = theirs.encode(texts, max_length=length)
        assert sm.close(ours, texts, want, max_length=length) <= 1


def test_the_defaults_of_the_model_survive_every_override(models):
    ours, _ = models
    texts = sm.texts(count=30)
    before = ours.encode(texts).tobytes()
    for kwargs in (
        {"max_length": 2},
        {"max_length": None},
        {"normalize": False},
        {"normalize": True},
    ):
        ours.encode(texts, **kwargs)
        assert (ours.max_length, ours.normalize) == (512, True)
        assert ours.config["max_length"] == 512
        assert ours.encode(texts).tobytes() == before
    assert ours.encoder._content.tokenizer.max_tokens == 512
