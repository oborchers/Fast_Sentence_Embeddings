"""Write fse_words.json: the tokens of fse_words@1 for fixed sentences, from the reference.

Written once and checked in; the tests compare the reference and the native tokenizer
with it, so a change of either is noticed. Never regenerate it for a released identifier.
The original synthetic corpus replaces separately licensed prose before the 2.0 release.

    uv run python tests/test_data/v2/make_fse_words_golden.py
"""

import json
from pathlib import Path

from fse._reference_text import tokenize_fse_words
from fse._unicode import UNICODE_VERSION

HERE = Path(__file__).parent

# Sentences that exercise every rule, the lowercase table and UTF-8 lengths.
EXTRA = [
    "I don't know. Can't, won't 'Tis O'Reilly's James' book",
    "rock'n'roll rock\u2019n\u2019roll x='abc' ''quoted'' =' ='a /'s",
    "U.S. e.g. Wait... what? 3.14 a.b. .5 5. -- - a- well-known --- a--b -x",
    "1,000 and 10:30 a,b c:d 12:30: 1,2, ,3 :4",
    "http://example.com/a-b?x=1 snake_case foo/bar a+b",
    "cannot gonna gimme lemme wanna gotta more'n d'ye 'twas n't isn't o''n't",
    "I'm he'd we'll they're I've it's",
    "caf\u00e9 \u4e2d\u56fd Paris \u0130stanbul \u03a3\u039f\u03a6\u039f\u03a3 2\u03a3 5\u212a",
    "\u00bfqu\u00e9? \u00abs\u00ed\u00bb \u201cso\u201d \u2018it\u2019 \u201elow\u201c a\u2013b a\u2014b",
    "X\u023aY \u023a \u1e9e\u00df \u0391\u0392 e\u0301 \u00e9\u0301 \U0001f600 x\U0001f600",
    "tab\there new\u2028line\u0085x a\x00b \x01 \u00a0nbsp\u3000ideographic",
]


def main() -> None:
    # Original templates and unique labels exercise a vocabulary at least as large
    # as the former STS corpus, including known and unknown halves in lookup tests.
    subjects = [
        "Our robot",
        "A sailor",
        "The otter",
        "My neighbour",
        "The pilot",
        "A weaver",
        "The clock",
    ]
    actions = ["inspects", "carries", "paints", "measures", "repairs", "finds", "circles"]
    endings = [
        "near café.",
        "at 10:30, isn't it?",
        "in room 3.14; we'll return.",
        "by a well-known tree!",
        "in Zürich, 中国 and Αθήνα.",
        "under sign_7... don't move?",
        "beside e\u0301 and 😀.",
    ]
    texts = [*EXTRA, "Quiet morning."]
    # 31 tokens and more than 215 UTF-8 bytes, the former corpus' upper bounds.
    texts.append("The cartographer inspects " + " ".join(f"instrument{i}" for i in range(28)))
    texts.extend(
        f"{subjects[i % 7]} {actions[(i // 7) % 7]} crate{i} for team{i} {endings[(i // 49) % 7]}"
        for i in range(2756)
    )
    cases = [[text, tokenize_fse_words(text)] for text in texts]
    payload = {"tokenizer": "fse_words@1", "unicode": UNICODE_VERSION, "cases": cases}
    (HERE / "fse_words.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=0) + "\n"
    )
    print(f"{len(cases)} cases")


if __name__ == "__main__":
    main()
