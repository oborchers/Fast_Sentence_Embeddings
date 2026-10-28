"""Write the golden encoders in tests/test_data/v2 (never rewrite them once committed).

    uv run python tests/test_data/v2/make_encoder_golden.py

They pin the format "encoder 1": an unfolded SIF encoder on the golden float32 word
vectors and a folded uSIF encoder that holds its int8 word vectors. tests/test_encoder_
encoder.py loads them and compares their fingerprints and the rows they give. This writes
only new files; the golden word vectors of make_golden.py stay as they are.
"""

from pathlib import Path

import fse

HERE = Path(__file__).parent
CORPUS = ["the é 日本", "the the é", "日本 x", "é", "the 日本 é é", "zz the", "", "日本 日本"]


def main():
    float32 = fse.WordVectors.load(HERE / "word_vectors_float32.safetensors")
    int8 = fse.WordVectors.load(HERE / "word_vectors_int8.safetensors")
    sif = fse.SIF(components=1, fold=False).fit(float32, CORPUS, workers=1)
    sif.save(HERE / "encoder_sif_unfolded.safetensors")
    usif = fse.USIF(components=1, length=3).fit(int8, CORPUS, counts=float32.counts, workers=1)
    usif.save(HERE / "encoder_usif_folded_int8.safetensors")


if __name__ == "__main__":
    main()
