"""Independent tokenizer references for small adapter preflights."""

from __future__ import annotations


def reference_batch(model, texts):
    import numpy as np

    from fse import _reference_text, _reference_wordpiece
    from fse._contract import Batch

    spec = model._spec.tokenizer
    words = model.word_vectors._words()
    if spec.name in ("fse_words@1", "external"):
        if model.word_vectors.fasttext is not None:
            raise ValueError("local preflight requires an explicit FastText reference adapter")
        return _reference_text.encode_texts(texts, words, tokenizer=spec.name)
    vocab = {word.decode("utf-8"): index for word, index in words.items()}
    config = spec.config
    rules = dict(
        added={word: index for index, word in config.added_tokens},
        max_tokens=spec.max_tokens,
        max_chars=spec.max_chars,
        drop_unknown=spec.unknown == "drop",
    )
    if spec.name == "wordpiece@1":
        return _reference_wordpiece.encode_texts(
            texts,
            vocab,
            unk_id=config.unk_id,
            prefix=config.continuing_subword_prefix,
            max_input_chars_per_word=config.max_input_chars_per_word,
            **rules,
        )
    if spec.name != "unigram@1":
        raise ValueError("unknown tokenizer reference for local preflight")
    from fse import _reference_unigram

    pieces = [model.word_vectors.word(index) for index in range(len(model.word_vectors))]
    reference = _reference_unigram.Model(pieces, config.scores_array(), config.unk_id)
    rows = [
        _reference_unigram.encode_ids(
            text, reference, normalizer=config.steps(), charsmap=config._checked_charsmap, **rules
        )
        for text in texts
    ]
    offsets = np.array([0, *np.cumsum([len(row) for row in rows])], dtype=np.int64)
    return Batch(0, offsets, np.array([token for row in rows for token in row], dtype=np.uint32))
