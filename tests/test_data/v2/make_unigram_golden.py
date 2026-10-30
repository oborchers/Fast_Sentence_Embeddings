"""Write unigram.json: the ids of unigram@1 for fixed texts, from Hugging Face tokenizers.

The ids come from tokenizers 0.23.3 (the oracle of the identifier), built in code from
small models that are part of the file: their pieces and float64 scores, the unknown id,
the added tokens, the normalizer steps and the charsmap as a dict (its blob is built by
tests/support/charsmap.py; the layout of a blob does not change ids). Written once and
checked in; the tests compare the reference and tokenizers with it, so a change of either
is noticed. Never regenerate it for a released identifier.

    uv run python tests/test_data/v2/make_unigram_golden.py
"""

import json
import sys
from pathlib import Path

import tokenizers

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))


def specs() -> list[dict]:
    """The small models, each with the texts whose ids it gives."""
    from support.unigram import PUNCTUATION, REPLACEMENT, STEPS

    r = REPLACEMENT
    # The chain of potion-multilingual-128M on a small model and charsmap.
    chain = {
        "name": "chain",
        "pieces": [
            ["[PAD]", -13.5],
            ["[UNK]", -13.5],
            [r, -3.9],
            [r + "Hello", -8.0],
            [r + "World", -8.5],
            [r + "a", -5.0],
            [r + "b", -6.0],
            [r + "x", -6.5],
            [r + "y", -6.5],
            [r + "A", -7.0],
            ["a", -4.0],
            ["b", -4.5],
            ["y", -5.5],
            [".", -3.6],
            [",", -3.5],
            [r + "!", -6.0],
            [r + "fi", -7.5],
            ["\u00e9", -6.0],
            [r + "\u00e9", -7.0],
            ["\U0001f600", -9.0],
            [r + "ab", -6.2],
            ["ab", -6.3],
            ["SMILE", -9.5],
            [r + "SMILE", -10.0],
            *([r + c, -6.0] for c in PUNCTUATION if c not in "!"),
        ],
        "unk_id": 1,
        "added": ["[PAD]", "[UNK]"],
        "steps": STEPS,
        "charsmap": {
            "\u00a0": " ",
            "\u3000": " ",
            "\t": " ",
            "\x01": "",
            "\uff21": "A",
            "\ufb01": "fi",
            "e\u0301": "\u00e9",
            "\u2026": "...",
            "\U0001f600": "SMILE",
            "\u00e9\u0302\u0303": "LONG",
        },
        "texts": [
            "Hello, World!",
            "a[UNK]b [PAD][PAD] [pad]",
            "\uff21\u0301 \uff21 e\u0301 e\u0301\u0302 \u00e9\u0302\u0303",
            "x\u3000y a\u00a0\u00a0b \t a \x01b",
            "\u2026 \ufb01 \U0001f600 \U0001f600\u0301",
            "\x00a x\u180ey abab q",
            "",
            "   ",
            r + "a " + r + r + "b",
        ],
    }
    # Pieces of equal score: the earliest start wins a tie.
    tie = {
        "name": "tie",
        "pieces": [["<unk>", -10.0], ["a", -1.0], ["ab", -1.0], ["bc", -1.0], ["c", -1.0]],
        "unk_id": 0,
        "added": [],
        "steps": [],
        "charsmap": {},
        "texts": ["abc", "abcabc", "cab"],
    }
    # Scores above 20, so two unknown nodes (each min - 10) beat the piece xy: joined, they
    # are that piece.
    joined = {
        "name": "joined",
        "pieces": [["<unk>", 25.0], ["xy", 21.0], ["a", 30.0], [r, 30.0]],
        "unk_id": 0,
        "added": [],
        "steps": [],
        "charsmap": {},
        "texts": ["axya", "xyz", "xy"],
    }
    # A piece with the id unk_id is joined with unknown nodes next to it.
    by_id = {
        "name": "by_id",
        "pieces": [["a", -1.0], ["UNK", -1.0], ["b", -1.0], [r, -1.0]],
        "unk_id": 1,
        "added": [],
        "steps": [],
        "charsmap": {},
        "texts": ["aUNKz", "xUNKy", "UNKUNK"],
    }
    # Rounded to float32, ab ties a + b and wins as the earlier start; float64 keeps a, b.
    float64 = {
        "name": "float64",
        "pieces": [["<unk>", -10.0], ["a", -1.0], ["b", -1.0], ["ab", -2.0 - 2.0**-30], [r, -1.0]],
        "unk_id": 0,
        "added": [],
        "steps": [],
        "charsmap": {},
        "texts": ["ab", "abab"],
    }
    return [chain, tie, joined, by_id, float64]


def main() -> None:
    from support.unigram import hf_tokenizer

    models = []
    for spec in specs():
        tok = hf_tokenizer(
            [tuple(p) for p in spec["pieces"]],
            steps=spec["steps"],
            charsmap=spec["charsmap"],
            unk_id=spec["unk_id"],
            added=spec["added"],
        )
        cases = [[t, tok.encode(t, add_special_tokens=False).ids] for t in spec["texts"]]
        models.append({**{k: v for k, v in spec.items() if k != "texts"}, "cases": cases})
    payload = {"tokenizer": "unigram@1", "tokenizers": tokenizers.__version__, "models": models}
    text = json.dumps(payload, ensure_ascii=True, indent=0) + "\n"
    (HERE / "unigram.json").write_text(text)
    print(f"{sum(len(m['cases']) for m in models)} cases in {len(models)} models")


if __name__ == "__main__":
    main()
