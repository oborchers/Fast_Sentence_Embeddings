"""Write the golden encoders of format 2 in tests/test_data/v2 (never rewrite them).

    uv run python tests/test_data/v2/make_encoder_golden.py

They pin the format "encoder 2": an unfolded SIF encoder on the golden float32 word
vectors and a folded uSIF encoder that holds its int8 word vectors. tests/test_encoder_
encoder.py loads them and compares their fingerprints and the rows they give.

encoder_sif_unfolded.safetensors and encoder_usif_folded_int8.safetensors are the same
encoders in the pre-release format 1 with sha256 checksums, fitted at commit 7f3cb32 on
CORPUS: ``SIF(components=1, fold=False)`` on the float32 vectors, ``USIF(components=1,
length=3)`` on the int8 ones with the float32 counts. They are kept byte for byte as files
that must be rejected. Fit depends on BLAS, so the files of format 2 are not fitted again
but transcoded: every tensor is taken byte for byte from format 1, and each fingerprint
they name is made anew only after its sha256 value of format 1 was recomputed from the
same content and matched. ``fse.quantized_from`` of the folded encoder names folded
float32 rows the file does not hold; they are folded again with the kernel and must give
the stored rows and the recorded sha256 value, or nothing is written. Every checksum and
the fingerprint of a file of format 1 are checked before its tensors are used. The new word vectors
(make_golden.py) must exist first.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from fse._container import SafetensorsContainer, checksum, fingerprint, tensor_code
from fse._contract import RemoveComponents
from fse._encoder._fold import BLOCK_ROWS, _folded_block, fold, linear_map
from fse._wordvectors import WordVectors

HERE = Path(__file__).parent
CORPUS = ["the é 日本", "the the é", "日本 x", "é", "the 日本 é é", "zz the", "", "日本 日本"]
VOCAB = ("vocab.bytes", "vocab.offsets")
EMBEDDED = ("vectors", "scales", "vocab.bytes", "vocab.offsets", "ngrams", "ngram_scales")
EMBEDDED_KEYS = ("fse.dim", "fse.dtype", "fse.quantization", "fse.quantized_from")
EMBEDDED_KEYS += ("fse.fasttext", "fse.ngram_hash")


def sha256_fingerprint(tensors, metadata):
    """The fingerprint of format 1: sha256 of the canonical JSON of scheme 1."""
    rows = [
        [n, tensor_code(a), list(a.shape), hashlib.sha256(a.tobytes()).hexdigest()]
        for n, a in sorted(tensors.items())
    ]
    content = {k: v for k, v in metadata.items() if k != "fse.created_by"}
    document = {"fingerprint": 1, "metadata": content, "tensors": rows}
    text = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def new_fingerprint(tensors, metadata):
    return fingerprint(
        {n: (tensor_code(a), a.shape, checksum(a)) for n, a in tensors.items()}, metadata
    )


def read_v1(name):
    """The tensors of a file of format 1 and its metadata without checksums and fingerprint,
    read with the safetensors package (fse rejects format 1)."""
    from safetensors import safe_open
    from safetensors.numpy import load_file

    with safe_open(str(HERE / name), "np") as f:
        metadata = f.metadata()
    tensors = load_file(str(HERE / name))
    plain = {k: v for k, v in metadata.items() if k != "fse.fingerprint"}
    plain = {k: v for k, v in plain.items() if not k.startswith("fse.checksum.")}
    # Check the file of format 1 before its hashes are dropped.
    keys = {k.removeprefix("fse.checksum.") for k in metadata if k.startswith("fse.checksum.")}
    if keys != set(tensors):
        raise SystemExit(f"{name}: the checksum keys do not match the tensors")
    for n, a in tensors.items():
        if hashlib.sha256(a.tobytes()).hexdigest() != metadata[f"fse.checksum.{n}"]:
            raise SystemExit(f"{name}: the checksum of {n} does not match")
    if sha256_fingerprint(tensors, plain) != metadata["fse.fingerprint"]:
        raise SystemExit(f"{name}: the fingerprint of format 1 does not match")
    return tensors, plain


def renamed(old, recomputed, new, key):
    """The new fingerprint of ``key``, after the recomputed value of format 1 matched."""
    if recomputed != old[key]:
        raise SystemExit(f"{key}: recomputed {recomputed}, the file of format 1 has {old[key]}")
    return new


def vocabulary(tensors, sha256):
    vocab = {n: tensors[n] for n in VOCAB}
    meta = {"fse.kind": "vocabulary"}
    return sha256_fingerprint(vocab, meta) if sha256 else new_fingerprint(vocab, meta)


def word_vectors_v1(name):
    tensors, metadata = read_v1(name)
    return sha256_fingerprint(tensors, metadata), tensors


def same_word_vectors(name_v1, name_v2):
    """Stop unless the word vectors of format 2 hold the tensors and keys of format 1."""
    from safetensors import safe_open
    from safetensors.numpy import load_file

    old, old_keys = read_v1(name_v1)
    new = load_file(str(HERE / name_v2))
    with safe_open(str(HERE / name_v2), "np") as f:
        new_keys = f.metadata()
    if set(old) != set(new) or any(
        old[n].dtype != new[n].dtype
        or old[n].shape != new[n].shape
        or old[n].tobytes() != new[n].tobytes()
        for n in old
    ):
        raise SystemExit(f"{name_v2} does not hold the tensors of {name_v1}")
    if "fse.quantized_from" in old_keys:  # an identity: of the float32 vectors of each format
        float32_v1, _ = word_vectors_v1("word_vectors_float32.safetensors")
        float32 = WordVectors.load(HERE / "word_vectors2_float32.safetensors")
        if (old_keys["fse.quantized_from"], new_keys.get("fse.quantized_from")) != (
            float32_v1,
            float32.fingerprint,
        ):
            raise SystemExit(f"fse.quantized_from of {name_v2} does not name its float32 vectors")
    ignored = ("fse.format_version", "fse.created_by", "fse.fingerprint", "fse.quantized_from")

    def content(keys):
        return {
            k: v for k, v in keys.items() if k not in ignored and not k.startswith("fse.checksum.")
        }

    if content(old_keys) != content(new_keys):
        raise SystemExit(f"{name_v2} does not hold the keys of {name_v1}")


def folded_rows(int8, steps):
    """The folded float32 rows of the int8 vectors, as fold computes them, never stored."""
    a = linear_map(steps, int8.dim)
    out = {}
    for rows, scales in (("vectors", "scales"), ("ngrams", "ngram_scales")):
        source, source_scales = getattr(int8, rows), getattr(int8, scales)
        blocks = [
            _folded_block(np.array(source[b : b + BLOCK_ROWS]), source_scales, b, a)
            for b in range(0, len(source), BLOCK_ROWS)
        ]
        out[rows] = np.concatenate(blocks)
    return out


def quantized_from(old_tensors, old, int8, steps):
    """The new ``fse.quantized_from``; stops if folding again does not reproduce the file."""
    again = fold(int8, steps)
    if any(
        getattr(again, n).tobytes() != old_tensors[n].tobytes() for n in EMBEDDED if n not in VOCAB
    ):
        raise SystemExit("folding again does not give the stored rows; nothing is written")
    rows = {**folded_rows(int8, steps), **{n: old_tensors[n] for n in VOCAB}}
    keys = {"fse.kind": "word_vectors", "fse.dim": old["fse.dim"], "fse.dtype": "float32"}
    keys |= {k: old[k] for k in ("fse.fasttext", "fse.ngram_hash")}
    if sha256_fingerprint(rows, {**keys, "fse.format_version": "1"}) != old["fse.quantized_from"]:
        raise SystemExit(
            "the folded float32 rows do not match the sha256 value; nothing is written"
        )
    new = new_fingerprint(rows, {**keys, "fse.format_version": "2"})
    assert new == again.metadata["fse.quantized_from"], "fold and the transcoder disagree"
    return new


def transcode_unfolded():
    tensors, old = read_v1("encoder_sif_unfolded.safetensors")
    wv_v1, wv_tensors = word_vectors_v1("word_vectors_float32.safetensors")
    same_word_vectors("word_vectors_float32.safetensors", "word_vectors2_float32.safetensors")
    wv = WordVectors.load(HERE / "word_vectors2_float32.safetensors")
    new = {**old, "fse.format_version": "2"}
    new["fse.word_vectors"] = renamed(old, wv_v1, wv.fingerprint, "fse.word_vectors")
    new["fse.vocabulary"] = renamed(
        old, vocabulary(wv_tensors, True), wv.vocabulary_fingerprint, "fse.vocabulary"
    )
    SafetensorsContainer().write(HERE / "encoder2_sif_unfolded.safetensors", tensors, new)


def transcode_folded():
    tensors, old = read_v1("encoder_usif_folded_int8.safetensors")
    int8_v1, _ = word_vectors_v1("word_vectors_int8.safetensors")
    same_word_vectors("word_vectors_int8.safetensors", "word_vectors2_int8.safetensors")
    int8 = WordVectors.load(HERE / "word_vectors2_int8.safetensors")
    new = {**old, "fse.format_version": "2"}
    new["fse.folded_from"] = renamed(old, int8_v1, int8.fingerprint, "fse.folded_from")
    new["fse.vocabulary"] = renamed(
        old, vocabulary(tensors, True), vocabulary(tensors, False), "fse.vocabulary"
    )
    steps = [RemoveComponents(tensors["folded.0.components"], tensors["folded.0.weights"])]
    assert json.loads(old["fse.folded"]) == ["remove_components@1"]
    origin = quantized_from(tensors, old, int8, steps)
    embedded = {n: tensors[n] for n in EMBEDDED}
    keys = {k: old[k] for k in EMBEDDED_KEYS}
    recomputed = sha256_fingerprint(
        embedded, {"fse.kind": "word_vectors", "fse.format_version": "1", **keys}
    )
    keys["fse.quantized_from"] = new["fse.quantized_from"] = origin
    content = {"fse.kind": "word_vectors", "fse.format_version": "2", **keys}
    new["fse.word_vectors"] = renamed(
        old, recomputed, new_fingerprint(embedded, content), "fse.word_vectors"
    )
    SafetensorsContainer().write(HERE / "encoder2_usif_folded_int8.safetensors", tensors, new)


def main():
    transcode_unfolded()
    transcode_folded()


if __name__ == "__main__":
    main()
