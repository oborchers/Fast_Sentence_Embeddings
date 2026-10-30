"""Write the golden models in tests/test_data/v2/finished (never rewrite them once committed).

    uv run python tests/test_data/v2/make_golden.py

They pin the formats "word_vectors 1", "encoder 1" and "linear_head 1" in "safetensors 1":
tests/test_wordvectors_format.py, tests/test_encoder_format.py and
tests/test_head_format.py read them and compare the arrays, the fingerprints and the
bits of the rows an encoder gives and of the outputs of a head. The writers never
overwrite a file, so this writes only the files that are missing.

The files directly in tests/test_data/v2 (word_vectors_*, encoder_*) and in sha256/ were
written by pre-releases of 2.0, with word counts, recipes, weights and learned steps
(and in sha256/ with sha256 checksums); they are kept byte for byte, only as files that
must be rejected. Their generators were removed with fit.
"""

from pathlib import Path

import numpy as np

from fse._encoder import Encoder
from fse._head import Classifier, Regressor
from fse._wordvectors import FastTextParams, WordVectors

HERE = Path(__file__).parent / "finished"
# The sentences whose rows tests/test_encoder_format.py pins for every golden encoder.
SENTENCES = ["the é", "日本 thé zz", "", "THE"]


def arrays():
    rows = np.array([[1.0, -0.0, 0.5], [2.0, -3.0, 0.25], [1e-30, 4.0, -1.0]], dtype=np.float32)
    ngrams = np.array([[0.5, 0.5, 0.5], [-1.0, 2.0, 0.0]], dtype=np.float32)
    return {
        "vectors": rows,
        "words": ["the", "é", "日本"],
        "fasttext": FastTextParams(2, 4, 2),
        "ngrams": ngrams,
    }


def word_vectors():
    a = arrays()
    return WordVectors.from_arrays(
        a["vectors"], words=a["words"], fasttext=a["fasttext"], ngrams=a["ngrams"]
    )


def encoders(wv: WordVectors, int8: WordVectors) -> dict[str, Encoder]:
    return {
        "encoder_float32": Encoder.from_word_vectors(wv, oov_weight=0.5, max_ngrams=3),
        "encoder_int8": Encoder.from_word_vectors(int8, tokenizer="external", normalize=False),
    }


def heads(encoders: dict[str, Encoder]) -> dict[str, tuple]:
    """The golden heads: (model, standalone) by file name."""
    raw = Encoder.from_word_vectors(word_vectors(), normalize=False, max_ngrams=3)
    three = np.array([[1.0, -2.0, 0.5], [0.25, 0.0, -1.0], [-0.5, 1.5, 2.0]], dtype=np.float32)
    return {
        "classifier_float32": (
            Classifier.from_arrays(
                encoders["encoder_float32"], three, [0.1, -0.0, 0.3], labels=["a", "b", "c"]
            ),
            False,
        ),
        "regressor_int8": (
            Regressor.from_arrays(encoders["encoder_int8"], three[:2], [1.0, -1.0]),
            False,
        ),
        "classifier_standalone": (
            Classifier.from_arrays(raw, three[0], 0.25, labels=[0, 1]),
            True,
        ),
    }


def _missing(name: str) -> Path | None:
    """The path of a golden file that does not exist yet, or None: none is rewritten."""
    path = HERE / f"{name}.safetensors"
    return None if path.exists() else path


def main():
    HERE.mkdir(exist_ok=True)
    wv = word_vectors()
    int8 = wv.quantize("int8")
    for name, vectors in (("word_vectors_float32", wv), ("word_vectors_int8", int8)):
        if path := _missing(name):
            vectors.save(path)
    made = encoders(wv, int8)
    for name, encoder in made.items():
        if path := _missing(name):
            encoder.save(path)
    for name, (model, standalone) in heads(made).items():
        if path := _missing(name):
            model.save(path, standalone=standalone)


if __name__ == "__main__":
    main()
