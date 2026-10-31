"""Write wordpiece.json: the ids of wordpiece@1 for fixed texts, from Hugging Face tokenizers.

The ids come from tokenizers 0.23.3 (the oracle of the identifier), built in code from a
small vocabulary that is part of the file. Written once and checked in; the tests compare
the reference, the native tokenizer and tokenizers with it, so a change of any of them is
noticed. Never regenerate it for a released identifier.

    uv run python tests/test_data/v2/make_wordpiece_golden.py
"""

import json
from pathlib import Path

import tokenizers
from tokenizers import AddedToken, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer

HERE = Path(__file__).parent
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TOKENS = [
    *SPECIAL,
    *"abcdefghijklmnopqrstuvwxyz0123456789",
    *("##" + ch for ch in "abcdefghijklmnopqrstuvwxyz0123456789"),
    "un", "##believ", "##able", "able", "istanbul", "##stan", "##bul", "is",
    "\u0307", "##\u0307", "\u4e2d", "\u56fd", "\u8c48", "\u1100", "##\u1161", "##\u11ab",
    "\u1112", "##\u116e", "##\u11af", "\u1100", "\u110b", "##\u1165", "\u1144",
    "\u03c3", "##\u03c3", "\u03bf", "##\u03bf", "##\u03c6", "\u00e9", "caf", "##\u00e9",
    "!", ".", ",", "'", "?", "-", "#", "(", ")", "\u00bf", "\u300c", "\u300d",
    "\U0001f600", "x\U0001f600", "hello", "world", "##ing", "##s", "the", "new",
    "##\u1144", "##\u11a8", "\u1100", "##\u116e", "##\u03c2", "##\U0001e949", "##\U00011446",
]  # fmt: skip
TOKENS = list(dict.fromkeys(TOKENS))
# Each exercises a rule of docs/architecture/tokenizers.md, "wordpiece@1".
TEXTS = [
    "Hello, world! The new world.",
    "un[MASK]able [MASK] x[MASK]y [UNK] [CLS] [cls] [MASK",
    "\u0130stanbul ISTANBUL istanbul",
    "\u01f0 \u01f0a J\u030c",
    "a" * 99 + " " + "a" * 100 + " " + "a" * 101,
    "\u00e9" * 100 + " " + "\u00e9" * 101,
    "\ud55c\uad6d \ud55c\uad6d\uc5b4 \u1112\u1161\u11ab",
    "\uf900\u4e2d\u56fd \u8c48",
    "a\u1e94" + "9\u034f\u1144" + "6",
    "a\U0001e949\u034f\U00011446 a\U0001e949\U00011446 a\u0301\u0327 a\u0327\u0301",
    "\u03a3\u039f\u03a6\u039f\u03a3 \u03c3\u03bf\u03c6\u03bf\u03c2",
    "caf\u00e9 cafe\u0301 CAF\u00c9",
    "tab\there new\u2028line\u0085x a\x00b \x01 \u00a0nbsp\u3000ideographic \u200bzw",
    "\u00bfqu\u00e9? \u300cx\u300d a-b a\u2014b (x) #1 ##ing",
    "\ue000private \U000f0000x \u0378unassigned \u0890x",
    "\U0001f600 x\U0001f600 \U0001f600x",
    "abc" * 40,
    "",
]


def tokenizer() -> Tokenizer:
    model = WordPiece(
        {t: i for i, t in enumerate(TOKENS)},
        unk_token="[UNK]",
        max_input_chars_per_word=100,
        continuing_subword_prefix="##",
    )
    tok = Tokenizer(model)
    tok.normalizer = BertNormalizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=None, lowercase=True
    )
    tok.pre_tokenizer = BertPreTokenizer()
    tok.add_special_tokens(
        [
            AddedToken(t, single_word=False, lstrip=False, rstrip=False, normalized=False)
            for t in SPECIAL
        ]
    )
    return tok


def main() -> None:
    tok = tokenizer()
    cases = [[text, tok.encode(text, add_special_tokens=False).ids] for text in TEXTS]
    payload = {
        "tokenizer": "wordpiece@1",
        "tokenizers": tokenizers.__version__,
        "config": {
            "added_tokens": [[TOKENS.index(t), t] for t in SPECIAL],
            "clean_text": True,
            "continuing_subword_prefix": "##",
            "handle_chinese_chars": True,
            "lowercase": True,
            "max_input_chars_per_word": 100,
            "strip_accents": True,
            "unk_id": TOKENS.index("[UNK]"),
        },
        "tokens": TOKENS,
        "cases": cases,
    }
    text = json.dumps(payload, ensure_ascii=True, indent=0) + "\n"
    (HERE / "wordpiece.json").write_text(text)
    print(f"{len(cases)} cases, {len(TOKENS)} tokens")


if __name__ == "__main__":
    main()
