"""Untimed whole-word misses through the independent WordPiece reference."""

from __future__ import annotations

from collections import Counter


def audit(job):
    from _development_encode import output_identity
    from _development_stages import inputs, tokenizer
    from _development_token_reference import reference_batch

    from fse import _native
    from fse import _reference_wordpiece as reference
    from fse._text import make_vocabulary

    model, texts = inputs(job)
    spec = model._spec.tokenizer
    if spec.name != "wordpiece@1":
        raise ValueError("whole-word audit requires wordpiece@1")
    config = spec.config
    vocab = {word.decode("utf-8"): index for word, index in model.word_vectors._words().items()}
    counts = Counter(words=0, whole_word_hits=0, trie_words=0, overlong_bypass=0, added_tokens=0)
    expected = [
        token for text in texts for token in _sentence(text, counts, vocab, config, spec, reference)
    ]
    native = tokenizer(model, texts)()
    if output_identity(native) != output_identity(reference_batch(model, texts)):
        raise ValueError("whole-word audit reference differs from native batch")
    if native.tokens.tolist() != expected:
        raise ValueError("instrumented reference differs from native token IDs")
    vocabulary = make_vocabulary(model.word_vectors)
    data, offsets, _ = model.word_vectors._vocabulary_snapshot()
    state = _native.WordPieceState(vocabulary.table, data, offsets, *spec.native_arguments())
    return dict(
        counts=dict(counts),
        trie_fraction=counts["trie_words"] / counts["words"] if counts["words"] else None,
        rows=len(texts),
        output=output_identity(native),
        reference_matches_native_bits=True,
        table_nbytes=vocabulary.table.nbytes,
        table_words=len(vocabulary.table),
        trie_bytes=state._trie_bytes,
        untimed=True,
        corpus=job["corpus"],
        artifact=job["artifact"],
    )


def _sentence(text, counts, vocab, config, spec, reference):
    ids = []
    cut = text if spec.max_chars is None else text[: spec.max_chars]
    added = {word: index for index, word in config.added_tokens}
    for piece, token in reference.split_added(cut, added):
        if token is not None:
            counts["added_tokens"] += 1
            ids.append(token)
        else:
            _words(piece, ids, counts, vocab, config, spec, reference)
        if spec.max_tokens and len(ids) >= spec.max_tokens:
            break
    if spec.max_tokens:
        ids = ids[: spec.max_tokens]
    return [token for token in ids if token != config.unk_id] if spec.unknown == "drop" else ids


def _words(piece, ids, counts, vocab, config, spec, reference):
    for word in reference.pre_tokenize(reference.normalize(piece)):
        if spec.max_tokens and len(ids) >= spec.max_tokens:
            break
        counts["words"] += 1
        if len(word) > config.max_input_chars_per_word:
            counts["overlong_bypass"] += 1
        elif word in vocab:
            counts["whole_word_hits"] += 1
        else:
            counts["trie_words"] += 1
        ids.extend(
            reference.wordpiece(
                word,
                vocab,
                config.unk_id,
                config.continuing_subword_prefix,
                config.max_input_chars_per_word,
            )
        )
