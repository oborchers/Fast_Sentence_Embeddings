"""Write the golden word vectors in tests/test_data/v2 (never rewrite them once committed).

    uv run python tests/test_data/v2/make_golden.py

They pin the format "word_vectors 2" in "safetensors 2": tests/test_wordvectors_format.py
reads them and compares the arrays and the fingerprint, and the export of fse 1.2.0
writes the same vectors and must give the same fingerprint. The writers never overwrite a
file, so this only writes files that are missing.

word_vectors_{float32,int8}.safetensors are the same vectors in the pre-release format 1
with sha256 checksums, written by this script at commit ca0d0b0; they are kept byte for
byte as files that must be rejected.
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
    wv.save(HERE / "word_vectors2_float32.safetensors")
    wv.quantize("int8").save(HERE / "word_vectors2_int8.safetensors")


if __name__ == "__main__":
    main()
