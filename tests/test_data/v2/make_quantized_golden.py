"""Write the golden quantized encoders in tests/test_data/v2/finished (never rewrite them
once committed).

    uv run python tests/test_data/v2/make_quantized_golden.py

They pin encoder 1 with a float16 or int8 table of a tokenizer with parameters, made by
``Encoder.quantize`` (quantize_rows@1): tests/test_encoder_format.py reads them and compares
the fingerprints and the bits of the rows they give. The float32 tables are built from
exact binary fractions, so every platform gives the same files. The writers never
overwrite a file, so this writes only the files that are missing.
"""

from pathlib import Path

import numpy as np

from fse._encoder import Encoder
from fse._tokenizer_spec import TokenizerSpec, WordPieceConfig
from fse._tokenizer_unigram import UnigramConfig, step_of
from fse._wordvectors import WordVectors

HERE = Path(__file__).parent / "finished"
WORDPIECE = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "##b", "un", "##able", "x"]
UNIGRAM = ["[PAD]", "[UNK]", "▁a", "▁b", "▁ab", "a", "b", "▁"]
SCORES = [-10.0, -10.0, -2.5, -2.5, -3.75, -4.5, -4.5, -1.0]
# The sentences whose rows tests/test_encoder_format.py pins for every golden encoder.
SENTENCES = ["a b", "un ab a", "", "x unable b", "a  ab \t"]


def rows(count: int) -> np.ndarray:
    """Exact binary fractions of different sizes, one row of 4 values per word."""
    i, j = np.arange(count)[:, None], np.arange(4)[None, :]
    return ((i * 7 + j * 3) % 11 - 5).astype(np.float32) / np.float32(4) * np.float32(1 + i % 3)


def float32_tables() -> dict[str, WordVectors]:
    return {
        "encoder_wordpiece_int8": WordVectors.from_arrays(rows(len(WORDPIECE)), words=WORDPIECE),
        "encoder_unigram_float16": WordVectors.from_arrays(rows(len(UNIGRAM)), words=UNIGRAM),
    }


def encoders() -> dict[str, Encoder]:
    tables = float32_tables()
    wordpiece = TokenizerSpec(
        "wordpiece@1", WordPieceConfig(((0, "[PAD]"), (1, "[UNK]"), (4, "[MASK]")), 1), "drop", 512
    )
    steps = (
        step_of({"type": "runs", "class": "white_space", "min": 1, "content": " "}),
        step_of({"type": "strip", "left": True, "right": True}),
    )
    scores = np.array(SCORES, dtype="<f8").tobytes()
    config = UnigramConfig(((0, "[PAD]"), (1, "[UNK]")), 1, steps, scores)
    unigram = TokenizerSpec("unigram@1", config, "keep", 8, 40)
    made = {
        "encoder_wordpiece_int8": (tables["encoder_wordpiece_int8"], wordpiece, "int8"),
        "encoder_unigram_float16": (tables["encoder_unigram_float16"], unigram, "float16"),
    }
    return {
        name: Encoder._converted(wv, tokenizer, normalize=True).quantize(dtype)
        for name, (wv, tokenizer, dtype) in made.items()
    }


def main():
    HERE.mkdir(exist_ok=True)
    for name, encoder in encoders().items():
        path = HERE / f"{name}.safetensors"
        if not path.exists():
            encoder.save(path)


if __name__ == "__main__":
    main()
