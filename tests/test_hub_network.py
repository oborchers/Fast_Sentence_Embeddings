"""The POTION models of the Hub at pinned commits: ids, vectors and fingerprints.

Marked ``network`` (``make test-network``): they download the models. The ids equal those
of each model's own tokenizer; the vectors equal model2vec 0.10.0's within the bound of
tests/support/model2vec.py; converting a commit always gives the same encoder.
"""

import csv
import random
from pathlib import Path

import pytest

from fse._encoder import Encoder
from support import model2vec as mv
from support import wordpiece as wp

pytestmark = pytest.mark.network
ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "minishlab/potion-base-8M": (
        "bf8b056651a2c21b8d2565580b8569da283cab23",
        "xxh3-128:7aed2ca9bf7f9073ffa6778672a7575f",
    ),
    "minishlab/potion-base-32M": (
        "1e5a03f8eeb2c98b928fbbd846f22f816360919f",
        "xxh3-128:99737b0f0d2dc7cfb4ee240faa2e278f",
    ),
    "minishlab/potion-retrieval-32M": (
        "6fc8051fab2a1e0ee76689cf08c853792ac285e7",
        "xxh3-128:6f78af26bab53a3d464e97d9189f5b20",
    ),
}


def _texts(tokens):
    with open(ROOT / "evaluation" / "sts-test.csv", encoding="utf-8", newline="") as f:
        sts = [s for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) for s in row[5:7]]
    rng = random.Random(7)
    words = rng.sample([t for t in tokens if not t.startswith("[")], 2000)
    edge = ["hello [MASK] [PAD] [UNK] [CLS] world [SEP]", "x" * 101, "İstanbul", ""]
    long = " ".join(rng.choice(words) for _ in range(5000))  # past both cuts
    return sts + wp.corpus(words, count=2000, seed=3) + edge + [long]


@pytest.mark.parametrize("repo", sorted(MODELS))
def test_a_potion_model_gives_the_ids_and_vectors_of_model2vec(repo):
    model2vec = pytest.importorskip("model2vec")
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from tokenizers import Tokenizer

    revision, fingerprint = MODELS[repo]
    folder = Path(huggingface_hub.snapshot_download(repo, revision=revision))
    encoder = Encoder.from_pretrained(repo, revision=revision)
    assert encoder._content.source.revision == revision
    assert encoder._content.license == "mit"
    assert encoder.fingerprint == fingerprint
    tokens = [encoder.word_vectors.word(i) for i in range(len(encoder.word_vectors))]
    texts = _texts(tokens)
    whole = Encoder.from_pretrained(repo, revision=revision, truncate=False, max_length="model")
    hf = Tokenizer.from_file(str(folder / "tokenizer.json"))
    want = [e.ids for e in hf.encode_batch_fast(texts, add_special_tokens=False)]
    unk = hf.token_to_id("[UNK]")
    got = [ids.tolist() for ids in mv.ids_of(whole, texts)]
    assert got == [[i for i in ids if i != unk] for ids in want]
    rows = model2vec.StaticModel.from_pretrained(folder).encode(texts)
    worst = mv.assert_close_to_model2vec(encoder.encode(texts), rows, encoder, texts, True)
    print(f"{repo}: {len(texts)} texts, the largest error is {worst:.2e} of the bound")
    assert worst <= 1
