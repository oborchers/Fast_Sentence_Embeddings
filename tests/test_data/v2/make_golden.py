"""Write the golden word vectors in tests/test_data/v2 (never rewrite them once committed).

    uv run python tests/test_data/v2/make_golden.py

They pin the format "word_vectors 1" in "NpyContainer 1": tests/test_wordvectors.py
reads them and compares the arrays and the fingerprint, and the export of fse 1.2.0
writes the same vectors and must give the same fingerprint.
"""

from pathlib import Path

import numpy as np

from fse._wordvectors import FastTextParams, WordVectors

HERE = Path(__file__).parent


def arrays():
    rows = np.array([[1.0, -0.0, 0.5], [2.0, -3.0, 0.25], [1e-30, 4.0, -1.0]], dtype=np.float32)
    ngrams = np.array([[0.5, 0.5, 0.5], [-1.0, 2.0, 0.0]], dtype=np.float32)
    return {
        "vectors": rows,
        "words": ["the", "é", "日本"],
        "counts": np.array([30, 20, 10], dtype=np.int64),
        "fasttext": FastTextParams(2, 4, 2),
        "ngrams": ngrams,
    }


def main():
    a = arrays()
    wv = WordVectors.from_arrays(
        a["vectors"],
        words=a["words"],
        counts=a["counts"],
        fasttext=a["fasttext"],
        ngrams=a["ngrams"],
    )
    wv.save(HERE / "word_vectors_float32")
    wv.quantize("int8").save(HERE / "word_vectors_int8")


if __name__ == "__main__":
    main()
