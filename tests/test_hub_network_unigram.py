"""minishlab/potion-multilingual-128M of the Hub at a pinned commit: ids, vectors and the
fingerprint of its encoder (unigram@1).

Marked ``network`` (``make test-network``): it downloads the model (a table of 512 MB).
The ids equal those of the model's own tokenizer (tokenizers 0.23.3), the vectors those of
model2vec 0.10.0 within the bound of tests/support/converted.py, and converting the commit
always gives the same encoder.
"""

import random

import pytest

from fse._encoder import Encoder
from support import converted as cv
from support import unigram as uni

pytestmark = pytest.mark.network
REPO = "minishlab/potion-multilingual-128M"
REVISION = "73908c3438cf03b6a01bcb9611d62b23d0726f08"
FINGERPRINT = "xxh3-128:6bdec1ad83263940d62a9e6224f0a0d4"
FILES = ["config.json", "model.safetensors", "tokenizer.json", "modules.json", "README.md"]
EDGE = [
    "Hello, World!",
    "a[UNK]b [PAD][PAD]",
    "\uff21\u0301 e\u0301 \u00e9\u0302\u0303 \u2026 \ufb01",
    "x\u3000y a\u200bb \x00a x\u180ey",
    "Das Wetter in M\u00fcnchen ist heute sch\u00f6n.",
    "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002 \u0627\u0644\u0637\u0642\u0633 \u062c\u0645\u064a\u0644",
    "\u0915\u094d\u0937\u0924\u094d\u0930\u093f\u092f \U0001f468\u200d\U0001f469\u200d\U0001f467 \U0001f1e9\U0001f1ea",
    "",
    "   ",
]


def _texts():
    rng = random.Random(11)
    long = " ".join(
        rng.choice(["mot", "\u5b57", "\u0441\u043b\u043e\u0432\u043e", "word"]) for _ in range(3000)
    )
    return EDGE + uni.corpus(3000, seed=13) + [long]


def test_potion_multilingual_gives_the_ids_and_vectors_of_model2vec():
    model2vec = pytest.importorskip("model2vec")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from tokenizers import Tokenizer

    folder = huggingface_hub.snapshot_download(REPO, revision=REVISION, allow_patterns=FILES)
    encoder = Encoder.from_pretrained(REPO, revision=REVISION)
    assert encoder.tokenizer == "unigram@1"
    assert encoder._content.source.revision == REVISION
    assert encoder._content.license == "mit"
    assert encoder.fingerprint == FINGERPRINT
    spec = encoder._content.tokenizer
    assert (spec.unknown, spec.max_tokens, spec.max_chars) == ("drop", 512, 3584)
    texts = _texts()
    whole = Encoder.from_pretrained(REPO, revision=REVISION, truncate=False, max_length="model")
    hf = Tokenizer.from_file(f"{folder}/tokenizer.json")
    want = [
        [i for i in e.ids if i != 1] for e in hf.encode_batch_fast(texts, add_special_tokens=False)
    ]
    assert [ids.tolist() for ids in cv.ids_of(whole, texts)] == want
    rows = model2vec.StaticModel.from_pretrained(folder).encode(texts)
    worst = cv.assert_close(encoder.encode(texts), rows, encoder, texts, True)
    print(f"{REPO}: {len(texts)} texts, the largest error is {worst:.2e} of the bound")
    assert worst <= 1
