"""fse.StaticModel.encode_as_sequence and tokenize against model2vec 0.10.0: the rows
equal bit for bit, the ids equal, the two cuts of a text at their boundaries (the
character cut only with an explicit ``max_length``, the token cut of ``tokenize`` by the
model's), and the arguments they refuse. The models are tiny, so nothing is downloaded."""

import numpy as np
import pytest

import fse
from support import model2vec as mv
from support import static_model as sm
from support import unigram as uni

model2vec = pytest.importorskip("model2vec")
pytest.importorskip("safetensors")
TEXTS = sm.texts()
PATTERN = "ab \U0001f600 hello xẕ ée " * 30


@pytest.fixture(scope="module")
def folders(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    return {layout: sm.model_folder(root / layout, layout) for layout in sm.LAYOUTS}


def _both(folder, **kwargs):
    return (
        fse.StaticModel.from_pretrained(folder, **kwargs),
        model2vec.StaticModel.from_pretrained(folder, **kwargs),
    )


def _same_sequences(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == np.float32
        assert g.shape == w.shape
        assert g.tobytes() == w.astype(np.float32).tobytes()


@pytest.mark.parametrize("layout", sm.LAYOUTS)
def test_the_sequences_are_those_of_model2vec_bit_for_bit(folders, layout):
    ours, theirs = _both(folders[layout])
    for max_length in (None, 1, 7, 512):
        _same_sequences(
            ours.encode_as_sequence(TEXTS, max_length=max_length),
            theirs.encode_as_sequence(TEXTS, max_length=max_length),
        )
    _same_sequences(ours.encode_as_sequence(TEXTS), theirs.encode_as_sequence(TEXTS))
    assert ours.tokenize(TEXTS) == theirs.tokenize(TEXTS)


def test_a_text_gives_one_array_a_list_a_list_and_no_text_no_array(folders):
    ours, theirs = _both(folders["model2vec"])
    one = ours.encode_as_sequence("hello world")
    assert isinstance(one, np.ndarray)
    assert one.tobytes() == theirs.encode_as_sequence("hello world").tobytes()
    assert one.ndim == 2
    assert one.shape[1] == ours.dim
    for texts in (["hello world", "the cat"], ("hello world", "the cat"), np.array(["a b", "c"])):
        got = ours.encode_as_sequence(texts)
        assert isinstance(got, list)
        _same_sequences(got, theirs.encode_as_sequence(list(texts)))
    assert ours.encode_as_sequence([]) == []
    assert ours.tokenize([]) == []
    # rows of the table: writable copies, not views of the table
    one[0, 0] = 5.0
    assert ours.encode_as_sequence("hello world")[0, 0] != 5.0


def test_a_text_without_tokens_is_an_empty_float32_array_where_model2vec_has_float64(folders):
    ours, theirs = _both(folders["model2vec"])
    for text in ("", "zzzq", "[UNK] [UNK]", " \t "):
        got, want = ours.encode_as_sequence(text), theirs.encode_as_sequence(text)
        assert got.shape == want.shape == (0, ours.dim)
        assert (got.dtype, want.dtype) == (np.float32, np.float64)
    first, second = ours.encode_as_sequence(["", ""])
    assert first is not second


def test_the_sequences_ignore_the_max_length_of_the_model_as_model2vec_does(folders):
    ours, theirs = _both(folders["model2vec"])
    ours.max_length = theirs.max_length = 3
    _same_sequences(ours.encode_as_sequence(TEXTS), theirs.encode_as_sequence(TEXTS))
    _same_sequences(
        ours.encode_as_sequence(TEXTS, max_length=5), theirs.encode_as_sequence(TEXTS, max_length=5)
    )
    assert ours.tokenize(TEXTS) == theirs.tokenize(TEXTS)
    assert all(len(ids) <= 3 for ids in ours.tokenize(TEXTS))
    assert any(len(s) > 3 for s in ours.encode_as_sequence(TEXTS))
    ours.max_length = theirs.max_length = None
    assert ours.tokenize(TEXTS) == theirs.tokenize(TEXTS)
    assert max(map(len, ours.tokenize(TEXTS))) > 512


@pytest.mark.parametrize("length", [1, 2, 3, 6, 40])
def test_encode_as_sequence_cuts_characters_then_tokens_only_with_its_own_max_length(
    folders, length
):
    ours, theirs = _both(folders["model2vec"])
    cut = length * ours.median_token_length
    texts = [PATTERN[:n] for n in range(max(cut - 3, 0), cut + 4)]
    _same_sequences(
        ours.encode_as_sequence(texts, max_length=length),
        theirs.encode_as_sequence(texts, max_length=length),
    )
    # one character past the cut changes nothing, as do many
    at, above, far = (
        ours.encode_as_sequence(PATTERN[:n], max_length=length) for n in (cut, cut + 1, 10**4)
    )
    assert at.tobytes() == above.tobytes() == far.tobytes()
    # no cut by default: the same text is longer
    assert len(ours.encode_as_sequence(PATTERN)) > len(far)


@pytest.mark.parametrize("length", [1, 2, 3, 6, 40])
def test_tokenize_cuts_tokens_by_the_model_and_no_characters(folders, length):
    ours, theirs = _both(folders["model2vec"], max_length=length)
    cut = length * ours.median_token_length
    texts = [PATTERN[:n] for n in range(max(cut - 3, 0), cut + 4)]
    assert ours.tokenize(texts) == theirs.tokenize(texts)
    # past the character cut tokenize still sees tokens, up to the model's max_length
    assert ours.tokenize([PATTERN[: cut + 3]]) == theirs.tokenize([PATTERN[: cut + 3]])
    assert len(ours.tokenize([PATTERN])[0]) <= length
    assert len(ours.tokenize([PATTERN])[0]) >= len(
        ours.encode_as_sequence(PATTERN, max_length=length)
    )


def test_the_unknown_id_counts_for_the_token_cut_and_is_dropped_after_it(tmp_path):
    folder = mv.save_model(tmp_path / "m", max_length=6)
    ours, theirs = _both(folder)
    texts = ["zzzq " * 5 + "hello world", "hello " * 5 + "zzzq world", "zzzq " * 9]
    assert ours.tokenize(texts) == theirs.tokenize(texts)
    assert [len(i) for i in ours.tokenize(texts)] == [1, 6, 0]
    _same_sequences(
        ours.encode_as_sequence(texts, max_length=6), theirs.encode_as_sequence(texts, max_length=6)
    )


def test_a_unigram_model_gives_the_sequences_and_ids_of_model2vec(tmp_path):
    ours, theirs = _both(mv.save_unigram_model(tmp_path / "uni"))
    texts = [*uni.corpus(300, seed=2), "", "[UNK] " * 600 + "word", "x" * 4000]
    for max_length in (None, 4):
        _same_sequences(
            ours.encode_as_sequence(texts, max_length=max_length),
            theirs.encode_as_sequence(texts, max_length=max_length),
        )
    assert ours.tokenize(texts) == theirs.tokenize(texts)
    ours.max_length = theirs.max_length = 5
    assert ours.tokenize(texts) == theirs.tokenize(texts)


def test_the_defaults_of_the_model_survive_the_calls(folders):
    ours, _ = _both(folders["model2vec"])
    before = ours.encode(TEXTS).tobytes()
    ours.encode_as_sequence(TEXTS, max_length=2)
    ours.tokenize(TEXTS)
    ours.tokenizer.encode_batch(TEXTS, add_special_tokens=False)
    assert (ours.max_length, ours.normalize) == (512, True)
    assert ours.config["max_length"] == 512
    assert ours.encode(TEXTS).tobytes() == before
    assert ours.encoder._content.tokenizer.max_tokens == 512


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"normalize": True}, TypeError, "unexpected keyword argument"),
        ({"max_length": 0}, ValueError, "max_length must be at least 1"),
        ({"max_length": "5"}, TypeError, "max_length must be an int"),
        ({"max_length": False}, TypeError, "max_length must be an int"),
        ({"show_progress_bar": 1}, TypeError, "show_progress_bar must be a bool"),
        ({"use_multiprocessing": None}, TypeError, "use_multiprocessing must be a bool"),
        ({"batch_size": 0}, ValueError, "batch_size must be at least 1"),
        ({"batch_size": 1.5}, TypeError, "batch_size must be an int"),
        ({"multiprocessing_threshold": -1}, ValueError, "at least 0"),
    ],
)
def test_encode_as_sequence_refuses_what_it_would_not_do(folders, kwargs, error, match):
    ours, _ = _both(folders["model2vec"])
    with pytest.raises(error, match=match):
        ours.encode_as_sequence(["hello"], **kwargs)


def test_encode_as_sequence_takes_the_neutral_values_and_numpy_integers(folders):
    ours, _ = _both(folders["model2vec"])
    want = ours.encode_as_sequence(["hello world"], max_length=5)
    got = ours.encode_as_sequence(
        ["hello world"],
        max_length=np.int64(5),
        batch_size=np.int32(1),
        multiprocessing_threshold=np.int8(0),
        use_multiprocessing=False,
        show_progress_bar=False,
    )
    _same_sequences(got, want)


@pytest.mark.parametrize(
    "sentences", [iter(["a"]), (s for s in ["a"]), {"a"}, b"a", 5, None, [1, 2], ["a", None]]
)
def test_the_sequences_take_a_text_or_a_sequence_of_texts_only(folders, sentences):
    ours, _ = _both(folders["model2vec"])
    with pytest.raises((TypeError, ValueError)):
        ours.encode_as_sequence(sentences)
    with pytest.raises((TypeError, ValueError)):
        ours.tokenize(sentences)


def test_tokenize_takes_no_single_text(folders):
    ours, _ = _both(folders["model2vec"])
    with pytest.raises(TypeError, match="list, tuple or numpy array of str, not str"):
        ours.tokenize("hello world")
    assert ours.tokenize(("hello world",)) == ours.tokenize(np.array(["hello world"]))


def test_the_surrogate_of_a_text_is_named_by_its_row(folders):
    ours, _ = _both(folders["model2vec"])
    with pytest.raises(UnicodeEncodeError, match="string 1"):
        ours.encode_as_sequence(["fine", "bad \ud800"])
