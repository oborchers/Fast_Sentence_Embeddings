#!/usr/bin/env python

# Author: Oliver Borchers
# Copyright (C) Oliver Borchers


"""Automated tests for checking the input methods."""

import itertools
import pickle
import random
import sys

import numpy as np
import pytest
from gensim.models import FastText, Word2Vec

import fse
from fse import CorpusFile, TextList
from fse.inputs import (
    BaseIndexedList,
    CIndexedList,
    CSplitCIndexedList,
    CSplitIndexedList,
    IndexedLineDocument,
    IndexedList,
    SplitCIndexedList,
    SplitIndexedList,
)


@pytest.fixture
def list_a():
    return ["the dog is good", "it's nice and comfy"]


@pytest.fixture
def capitalized():
    return ["The Dog is good", "it's nice and comfy"]


def _lower_split(text):
    return text.lower().split()


def _assert_immutable(il):
    with pytest.raises(NotImplementedError):
        il.__delitem__(0)
    with pytest.raises(NotImplementedError):
        il.__setitem__(0, "the")
    with pytest.raises(NotImplementedError):
        il.insert(0, "the")
    with pytest.raises(NotImplementedError):
        il.append("the")
    with pytest.raises(NotImplementedError):
        il.extend(["the", "dog"])


class TestBaseIndexedList:
    @pytest.fixture
    def list_b(self):
        return ["lorem ipsum dolor", "si amet"]

    @pytest.fixture
    def base(self, list_a):
        return BaseIndexedList(list_a)

    @pytest.fixture
    def multi(self, list_a, list_b):
        return BaseIndexedList(list_a, list_b, [s.split() for s in list_a])

    def test_init(self, list_a):
        _ = BaseIndexedList(list_a)

    def test_init_mult_arg(self, multi):
        assert len(multi.items) == 6

    def test_init_ndarray(self, list_a):
        _ = BaseIndexedList(np.array(list_a))

    @pytest.mark.parametrize("value", [1, "Hello"])
    def test__check_list_type(self, base, value):
        with pytest.raises(TypeError):
            base._check_list_type(value)

    def test__check_str_type(self, base):
        assert base._check_str_type("Hello") == 1

    @pytest.mark.parametrize("value", [1, []])
    def test__check_str_type_wrong_type(self, base, value):
        with pytest.raises(TypeError):
            base._check_str_type(value)

    def test__len(self, base):
        assert len(base) == 2

    def test__str(self, base):
        assert str(base) == "['the dog is good', \"it's nice and comfy\"]"

    def test__getitem(self, base):
        with pytest.raises(NotImplementedError):
            base[0]

    def test__delitem(self, multi):
        multi.__delitem__(0)
        assert len(multi) == 5

    def test__setitem(self, multi):
        multi.__setitem__(0, "is it me?")
        assert multi.items[0] == "is it me?"

    def test_append(self, multi):
        multi.append("is it me?")
        assert multi.items[-1] == "is it me?"

    def test_extend(self, multi, list_a):
        multi.extend(list_a)
        assert len(multi) == 8

        with pytest.raises(TypeError, match="A set has no order"):
            multi.extend({"hello there", "its a set"})
        assert len(multi) == 8

    def test_extend_ndarr(self):
        base = BaseIndexedList(np.array([str(i) for i in [1, 2, 3, 4]]))
        base.extend(np.array([str(i) for i in [1, 2, 3, 4]]))
        assert len(base) == 8


class TestIndexedList:
    def test_init(self, list_a):
        _ = IndexedList(list_a)

    def test_getitem(self, list_a):
        il = IndexedList(list_a, [s.split() for s in list_a])
        assert il[0] == ("the dog is good", 0)

    def test_split(self, list_a):
        base = SplitIndexedList(list_a)
        assert base[0][0] == ["the", "dog", "is", "good"]


class TestCIndexedList:
    @pytest.fixture
    def il(self, capitalized):
        return CIndexedList(capitalized, custom_index=[1, 1])

    def test_cust_index(self, il):
        assert il[0][1] == 1

    def test_wrong_len(self, capitalized):
        with pytest.raises(RuntimeError):
            CIndexedList(capitalized, custom_index=[1])

    def test_mutable_funcs(self, il):
        _assert_immutable(il)


class TestCSplitIndexedList:
    def test_getitem(self, capitalized):
        il = CSplitIndexedList(capitalized, custom_split=_lower_split)
        assert il[0][0] == ["the", "dog", "is", "good"]


class TestSplitCIndexedList:
    @pytest.fixture
    def il(self, capitalized):
        return SplitCIndexedList(capitalized, custom_index=[1, 1])

    def test_getitem(self, il):
        assert il[0] == (["The", "Dog", "is", "good"], 1)

    def test_mutable_funcs(self, il):
        _assert_immutable(il)


class TestCSplitCIndexedList:
    @pytest.fixture
    def il(self, capitalized):
        return CSplitCIndexedList(capitalized, custom_split=_lower_split, custom_index=[1, 1])

    def test_getitem(self, il):
        assert il[0] == (["the", "dog", "is", "good"], 1)

    def test_mutable_funcs(self, il):
        _assert_immutable(il)


class TestIndexedLineDocument:
    @pytest.fixture
    def doc(self, corpus_path):
        return IndexedLineDocument(corpus_path)

    def test_getitem(self, doc):
        assert doc[0] == "Good stuff i just wish it lasted longer"
        assert doc[19] == "Save yourself money and buy it direct from lg"
        assert doc[-1] == "I am not sure if it is a tracfone problem or the battery"

    def test_yield(self, doc):
        first = (["Good", "stuff", "i", "just", "wish", "it", "lasted", "longer"], 0)
        last = (
            [
                "I",
                "am",
                "not",
                "sure",
                "if",
                "it",
                "is",
                "a",
                "tracfone",
                "problem",
                "or",
                "the",
                "battery",
            ],
            99,
        )
        items = list(doc)
        assert len(items) == 100
        assert items[0] == first
        assert items[99] == last


def test_indexed_list_rejects_set():
    """A set has no order, so the sentence indices would be arbitrary."""
    from fse import IndexedList

    with pytest.raises(TypeError, match="A set has no order"):
        IndexedList({"the dog", "a cat"})


def test_indexed_line_document_validates_arguments(corpus_path):
    from pydantic import ValidationError

    from fse import IndexedLineDocument

    assert len(list(IndexedLineDocument(corpus_path))) == 100
    with pytest.raises(ValidationError):
        IndexedLineDocument(corpus_path, get_able="yes")


@pytest.mark.parametrize("as_array", [False, True], ids=["list", "ndarray"])
@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (IndexedList, {}),
        (CIndexedList, {"custom_index": [3, 1, 3]}),
        (SplitIndexedList, {}),
        (SplitCIndexedList, {"custom_index": np.array([3, 1, 3])}),
        (CSplitIndexedList, {"custom_split": _lower_split}),
        (CSplitCIndexedList, {"custom_split": _lower_split, "custom_index": [0, 0, 2]}),
    ],
)
def test_iter_matches_getitem(cls, kwargs, as_array):
    """The fast __iter__ yields exactly what indexing every position returns."""
    items = ["The dog is good", "it's nice and comfy", "Ünïcödé\u2003words"]
    il = cls(np.array(items) if as_array else items, **kwargs)
    by_index = [il[i] for i in range(len(il))]
    assert list(il) == by_index
    assert [type(s) for s, _ in il] == [type(s) for s, _ in by_index]
    assert list(il) == list(il)


def test_iter_uses_a_subclass_getitem():
    """A subclass that changes __getitem__ is iterated through it, as before."""

    class Lower(SplitIndexedList):
        def __getitem__(self, i):
            words, index = super().__getitem__(i)
            return [w.lower() for w in words], index

    il = Lower(["The Dog", "A Cat"])
    assert list(il) == [(["the", "dog"], 0), (["a", "cat"], 1)]


# ---------------------------------------------------------------------------------------
# CorpusFile and TextList, read natively without the GIL
# ---------------------------------------------------------------------------------------

SPACES = "".join(chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace())

EDGE_FILES = [
    b"",
    b"\n",
    b"a",
    b"a\n",
    b"a\n\n",
    b"\n\na b\n",
    b"a\r\nb\rc\n",
    b"\xef\xbb\xbfa b\n",
    b"a\x00b c\n",
    "x\u3000y\u00a0z\u2028w\x1cv\n\u0085".encode(),
    ("t " * 12_000 + "\n").encode(),
]


def _write(tmp_path, data, name="corpus.txt"):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _corpus_text(seed=0, lines=4000):
    """Lines with known and unknown words, empty lines, Unicode spaces and a long line."""
    rng = random.Random(seed)
    words = ["the", "dog", "is", "good", "ä", "東京", "\U0001f600", "unknownword", "naïve"]
    out = []
    for i in range(lines):
        if i % 97 == 0:
            out.append("")
            continue
        n = rng.randrange(1, 15)
        sep = [rng.choice(" \t" + SPACES) for _ in range(n)]
        out.append("".join(rng.choice(words) + s for s in sep))
    out.insert(lines // 2, " ".join(["dog"] * 30_000))  # longer than a chunk
    return "\n".join(out) + "\n"


@pytest.fixture
def native_w2v():
    model = Word2Vec(min_count=1, vector_size=8, seed=1, workers=1)
    model.build_vocab([["the", "dog", "is", "good", "ä", "東京", "\U0001f600", "naïve"]])
    model.wv.vectors[:] = np.random.default_rng(0).random(model.wv.vectors.shape, dtype=np.float32)
    return model


@pytest.fixture
def native_ft():
    model = FastText(min_count=1, vector_size=8, seed=1, workers=1, bucket=500)
    model.build_vocab([["the", "dog", "is", "good", "ä", "東京"]])
    rng = np.random.default_rng(1)
    model.wv.vectors_vocab[:] = rng.random(model.wv.vectors_vocab.shape, dtype=np.float32)
    model.wv.vectors_ngrams[:] = rng.random(model.wv.vectors_ngrams.shape, dtype=np.float32)
    return model


class TestCorpusFile:
    @pytest.mark.parametrize("data", EDGE_FILES)
    def test_sentences_equal_indexed_line_document(self, tmp_path, data):
        path = _write(tmp_path, data)
        assert list(CorpusFile(path)) == list(IndexedLineDocument(path))
        assert len(CorpusFile(path)) == len(list(IndexedLineDocument(path)))

    @pytest.mark.parametrize("data", EDGE_FILES)
    def test_scan_statistics_equal_indexed_line_document(self, tmp_path, data, native_w2v):
        path = _write(tmp_path, data)
        model = fse.Average(native_w2v, workers=3)
        assert model.scan_sentences(CorpusFile(path)) == model.scan_sentences(
            IndexedLineDocument(path)
        )

    def test_getitem_equals_indexed_line_document(self, tmp_path):
        path = _write(tmp_path, b"a b \n\nc\r\nlast")
        corpus, reference = CorpusFile(path), IndexedLineDocument(path)
        assert [corpus[i] for i in range(4)] == [reference[i] for i in range(4)]
        assert corpus[-1] == "last"
        with pytest.raises(IndexError):
            corpus[4]

    @pytest.mark.parametrize("name", ["corpus.txt.gz", "corpus.bz2", "corpus.XZ", "corpus.zst"])
    def test_rejects_compressed_files(self, tmp_path, name):
        path = _write(tmp_path, b"a\n", name)
        with pytest.raises(ValueError, match="compressed"):
            CorpusFile(path)

    def test_rejects_missing_files_and_directories(self, tmp_path):
        with pytest.raises(ValueError, match="not a file"):
            CorpusFile(tmp_path / "missing.txt")
        with pytest.raises(ValueError, match="not a file"):
            CorpusFile(tmp_path)

    def test_pickles_without_native_state(self, tmp_path, native_w2v):
        path = _write(tmp_path, b"the dog\n")
        corpus = CorpusFile(path)
        fse.Average(native_w2v).train(corpus)
        restored = pickle.loads(pickle.dumps(corpus))
        assert restored._scan_state is None
        assert list(restored) == list(corpus)

    def test_change_after_the_scan_is_an_error(self, tmp_path, native_w2v):
        path = _write(tmp_path, b"the dog\n")
        corpus = CorpusFile(path)
        fse.Average(native_w2v).scan_sentences(corpus)
        path.write_bytes(b"the dog is good\n")
        with pytest.raises(RuntimeError, match="changed between the scan and training"):
            corpus._native_chunks()

    def test_reads_paths_with_non_ascii_characters(self, tmp_path, native_w2v):
        directory = tmp_path / "Ümlaut 東京"
        directory.mkdir()
        path = _write(directory, b"the dog\n\nis good\n", "Satz.txt")
        native, reference = fse.Average(native_w2v), fse.Average(native_w2v)
        native.train(CorpusFile(path))
        reference.train(IndexedLineDocument(path))
        assert np.array_equal(native.sv.vectors, reference.sv.vectors)


class TestTextList:
    def test_sentences_equal_split_indexed_list(self):
        strings = ["the dog", "", "  ä\u3000東京 ", "a\nb"]
        assert list(TextList(strings)) == list(SplitIndexedList(strings))
        assert len(TextList(strings)) == 4
        assert TextList(strings)[2] == strings[2]

    def test_copies_the_strings(self):
        strings = ["the dog"]
        texts = TextList(strings)
        strings[0] = "changed"
        assert texts[0] == "the dog"

    @pytest.mark.parametrize("value", [{"a", "b"}, "a b", ["a", 1]])
    def test_rejects_invalid_input(self, value):
        with pytest.raises(TypeError):
            TextList(value)

    def test_accepts_numpy_strings(self):
        texts = TextList(np.array(["the dog", "is good"]))
        assert list(texts) == [(["the", "dog"], 0), (["is", "good"], 1)]
        assert all(type(s) is str for s in texts.items)

    def test_rejects_unencodable_strings(self):
        """Raised when the TextList is created, with and without the extension."""
        with pytest.raises(UnicodeEncodeError, match="in string 1"):
            TextList(["the dog", "the \ud800"])

    def test_pickles_without_native_state(self, native_w2v):
        texts = TextList(["the dog"])
        fse.Average(native_w2v).train(texts)
        restored = pickle.loads(pickle.dumps(texts))
        assert restored._scan_state is None
        assert list(restored) == list(texts)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
@pytest.mark.parametrize("model_name", ["Average", "SIF", "uSIF"])
@pytest.mark.parametrize("kind", ["native_w2v", "native_ft"])
def test_native_inputs_give_the_vectors_of_the_python_inputs(
    tmp_path, native_w2v, native_ft, kind, model_name, workers
):
    """Bit-identical vectors for train and infer, with chunks and ranges across the file."""
    text = _corpus_text()
    path = _write(tmp_path, text.encode())
    lines = text.splitlines()
    wv = native_w2v if kind == "native_w2v" else native_ft
    cls = getattr(fse, model_name)
    kwargs = {"lang_freq": "en"} if model_name != "Average" else {}

    def train(data):
        model = cls(wv, workers=workers, **kwargs)
        model.train(data)
        return model

    file_native, file_python = train(CorpusFile(path)), train(IndexedLineDocument(path))
    assert np.array_equal(file_native.sv.vectors, file_python.sv.vectors)
    assert np.array_equal(
        file_native.infer(CorpusFile(path)), file_python.infer(IndexedLineDocument(path))
    )

    list_native, list_python = train(TextList(lines)), train(SplitIndexedList(lines))
    assert np.array_equal(list_native.sv.vectors, list_python.sv.vectors)


def test_native_inputs_use_native_chunks(tmp_path, native_w2v, monkeypatch):
    from fse.models import average

    calls = []
    original = average.run_chunk
    monkeypatch.setattr(average, "run_chunk", lambda *a: calls.append(1) or original(*a))
    path = _write(tmp_path, _corpus_text(lines=200).encode())
    fse.Average(native_w2v, workers=2).train(CorpusFile(path))
    assert calls


@pytest.mark.parametrize("kind", ["w2v", "ft"])
@pytest.mark.parametrize("input_name", ["CorpusFile", "TextList"])
def test_fallback_without_the_extension_gives_the_same_vectors(
    tmp_path, native_w2v, native_ft, monkeypatch, kind, input_name
):
    """With the scan, the kernel and train_average replaced by numpy, results match."""
    from fse import inputs
    from fse.models import average

    text = _corpus_text(lines=500)
    path = _write(tmp_path, text.encode())

    def make():
        return CorpusFile(path) if input_name == "CorpusFile" else TextList(text.splitlines())

    wv = native_w2v if kind == "w2v" else native_ft
    native = fse.Average(wv, workers=2)
    native_counts = native.train(make())

    calls = []

    def numpy_only(*args, **kwargs):
        calls.append(1)
        return average.train_average_np(*args, **kwargs)

    monkeypatch.setattr(inputs, "scan_spans", None)
    monkeypatch.setattr(average, "train_average", numpy_only)
    monkeypatch.setattr(fse.Average, "_kernel_context", lambda self, target: None)
    fallback = fse.Average(wv, workers=2)
    assert fallback.train(make()) == native_counts
    assert calls
    np.testing.assert_allclose(native.sv.vectors, fallback.sv.vectors, rtol=1e-5, atol=1e-7)


def test_update_appends_after_the_existing_rows(tmp_path, native_w2v):
    first = _write(tmp_path, b"the dog\nis good\n", "first.txt")
    second = _write(tmp_path, b"good dog\n\nthe\n", "second.txt")
    native, reference = fse.Average(native_w2v), fse.Average(native_w2v)
    native.train(CorpusFile(first))
    native.train(CorpusFile(second), update=True)
    reference.train(IndexedLineDocument(first))
    reference.train(IndexedLineDocument(second), update=True)
    assert native.sv.vectors.shape == (5, 8)
    assert np.array_equal(native.sv.vectors, reference.sv.vectors)


@pytest.mark.parametrize(
    "data",
    [
        b"the\nab\xffcd\n",
        b"\xc0\xaf",
        b"ok\n\nx \xed\xa0\x80\n",
        b"a\n\xf4\x90\x80\x80",
        b"a b\n\xe2\x82",
    ],
)
def test_invalid_utf8_raises_like_python(tmp_path, native_w2v, data):
    path = _write(tmp_path, data)
    with pytest.raises(UnicodeDecodeError):
        list(IndexedLineDocument(path))
    bad_line = next(i for i, line in enumerate(data.split(b"\n")) if not _decodes(line))
    with pytest.raises(UnicodeDecodeError, match=f"sentence {bad_line} "):
        fse.Average(native_w2v, workers=2).train(CorpusFile(path))


def _decodes(line):
    try:
        line.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def test_random_text_splits_like_python(tmp_path, native_w2v):
    """Random Unicode text: scan statistics and vectors equal the Python inputs."""
    rng = random.Random(1)
    alphabet = ["the", "dog", "ä", "東京", "\U0001f600", "﻿", "\x00", "\r", "\n", *SPACES]
    for case in range(60):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 200)))
        path = _write(tmp_path, text.encode(), f"random{case}.txt")
        model = fse.Average(native_w2v, workers=2)
        assert model.scan_sentences(CorpusFile(path)) == model.scan_sentences(
            IndexedLineDocument(path)
        )
        assert _train_outcome(native_w2v, CorpusFile(path)) == _train_outcome(
            native_w2v, IndexedLineDocument(path)
        )


def _train_outcome(native_w2v, data):
    """The trained vectors as bytes, or the type and message of the error."""
    model = fse.Average(native_w2v, workers=2)
    try:
        model.train(data)
    except Exception as err:
        return type(err), str(err)
    return model.sv.vectors.tobytes()


def test_random_bytes_fail_where_python_fails(tmp_path, native_w2v):
    rng = random.Random(2)
    for case in range(300):
        data = bytes(
            rng.choice([rng.randrange(256), 10, 32, 97]) for _ in range(rng.randrange(1, 40))
        )
        path = _write(tmp_path, data, f"bytes{case}.txt")
        lines = data.split(b"\n")
        bad = [i for i, line in enumerate(lines) if not _decodes(line)]
        model = fse.Average(native_w2v)
        if bad:
            with pytest.raises(UnicodeDecodeError, match=f"sentence {bad[0]} "):
                model.scan_sentences(CorpusFile(path))
        else:
            assert model.scan_sentences(CorpusFile(path)) == model.scan_sentences(
                IndexedLineDocument(path)
            )


def test_scan_hook_is_skipped_for_replaced_validators(tmp_path, native_w2v):
    path = _write(tmp_path, b"the dog\n")
    model = fse.Average(native_w2v)

    def reject(iter_pos, obj):
        raise ValueError("custom validator")

    model._check_indexed_sent_valid = reject
    with pytest.raises(ValueError, match="custom validator"):
        model.scan_sentences(CorpusFile(path))


@pytest.mark.parametrize("where", ["instance", "subclass"])
def test_own_validators_see_every_sentence_and_training_still_works(tmp_path, native_w2v, where):
    """A custom validator forces the Python scan, and training then reads tuples as well."""
    path = _write(tmp_path, _corpus_text(lines=300).encode())
    seen = []

    def validate(self, iter_pos, obj):
        seen.append(iter_pos)
        return fse.Average._check_indexed_sent_valid(self, iter_pos, obj)

    if where == "subclass":
        model = type("Checked", (fse.Average,), {"_check_indexed_sent_valid": validate})(native_w2v)
    else:
        model = fse.Average(native_w2v)
        model._check_indexed_sent_valid = validate.__get__(model)
    model.train(CorpusFile(path))
    reference = fse.Average(native_w2v)
    reference.train(IndexedLineDocument(path))
    assert len(seen) == len(CorpusFile(path))
    assert np.array_equal(model.sv.vectors, reference.sv.vectors)


def test_utf8_errors_carry_the_attributes_of_python(tmp_path, native_w2v):
    data = b"ok\nab\xff cd\nthe \xe2\x82"
    path = _write(tmp_path, data)
    with pytest.raises(UnicodeDecodeError) as native:
        fse.Average(native_w2v).scan_sentences(CorpusFile(path))
    with pytest.raises(UnicodeDecodeError) as python:
        b"ab\xff cd".decode("utf-8")
    assert (native.value.object, native.value.start, native.value.end) == (
        python.value.object,
        python.value.start,
        python.value.end,
    )
    assert "sentence 1 (byte 5 of the source)" in str(native.value)


def test_replaced_file_with_the_same_size_and_time_is_detected(tmp_path, native_w2v):
    """The fingerprint includes device and inode, not only size and modification time."""
    import os

    path = _write(tmp_path, b"the dog\n")
    corpus = CorpusFile(path)
    fse.Average(native_w2v).scan_sentences(corpus)
    stat = os.stat(path)
    replacement = _write(tmp_path, b"dog the\n", "replacement.txt")
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(replacement, path)
    with pytest.raises(RuntimeError, match="changed between the scan and training"):
        corpus._native_chunks()


def test_change_during_training_is_detected(tmp_path, native_w2v, monkeypatch):
    """A file modified after the chunks were handed out raises at the end of training."""
    import os

    from fse.models import average

    path = _write(tmp_path, _corpus_text(lines=200).encode())
    original = average.run_chunk

    def touch_then_run(*args):
        os.utime(path, ns=(1, 1))
        return original(*args)

    monkeypatch.setattr(average, "run_chunk", touch_then_run)
    with pytest.raises(RuntimeError, match="changed during training"):
        fse.Average(native_w2v, workers=2).train(CorpusFile(path))


@pytest.mark.parametrize(
    ("offsets", "first", "last", "message"),
    [
        (None, 0, 99, "outside"),
        (None, 3, 9, "not the start of a line"),
        (None, -1, 2, "outside"),
        (np.array([0, 4], dtype=np.int64), 0, 2, "outside"),
        (np.array([0, 99], dtype=np.int64), 0, 1, "outside the source"),
        (np.array([0, 5, 2, 9], dtype=np.int64), 0, 3, "decrease"),
    ],
)
def test_native_entry_points_validate_ranges(native_w2v, offsets, first, last, message):
    from fse.models.average_inner import KernelContext, VocabTable, run_chunk, scan_spans

    data = b"the dog\nis\n"
    with pytest.raises(ValueError, match=message):
        scan_spans(data, offsets, first, last, 1 << 16)
    model = fse.Average(native_w2v)
    model.prep.prepare_vectors(sv=model.sv, total_sentences=3)
    context = KernelContext(model, model.sv.vectors)
    table = VocabTable(model.wv.key_to_index)
    with pytest.raises(ValueError, match=message):
        run_chunk(context, table, data, offsets, first, last, 0, model._get_thread_working_mem()[0])


def test_text_list_infer_and_update_match_split_indexed_list(native_w2v):
    first, second = ["the dog", "", "is good"], ["good dog", "ä 東京"]
    native, python = fse.Average(native_w2v), fse.Average(native_w2v)
    assert native.train(TextList(first)) == python.train(SplitIndexedList(first))
    native.train(TextList(second), update=True)
    python.train(SplitIndexedList(second), update=True)
    assert np.array_equal(native.sv.vectors, python.sv.vectors)
    assert np.array_equal(native.infer(TextList(second)), python.infer(SplitIndexedList(second)))


def test_text_list_rejects_str_subclasses():
    class Shouting(str):
        def split(self):
            return [self.upper()]

    with pytest.raises(TypeError, match="SplitIndexedList"):
        TextList([Shouting("a b")])


def test_text_list_with_many_strings_per_chunk(native_w2v):
    """More strings than bytes per chunk: the chunk table is sized by bytes, not strings."""
    from fse.models.average_inner import scan_spans

    strings = ["the dog", "is", "", "good ä"] * 50_000
    texts = TextList(strings)
    source = texts._open_source()
    _, _, _, chunks, error, _ = scan_spans(source.data, source.offsets, 0, len(strings), 1 << 16)
    assert error == -1
    assert chunks[0][0] == 0
    assert chunks[-1][1] == len(strings)
    assert all(a[1] == b[0] for a, b in itertools.pairwise(chunks))
    native, python = fse.Average(native_w2v, workers=1), fse.Average(native_w2v, workers=1)
    native.train(texts)
    python.train(SplitIndexedList(strings))
    assert np.array_equal(native.sv.vectors, python.sv.vectors)
