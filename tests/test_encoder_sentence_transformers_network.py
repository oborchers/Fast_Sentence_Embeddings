"""The static models of sentence-transformers on the Hub at pinned commits: ids, vectors
and fingerprints against sentence-transformers 6.2.0.

Marked ``network`` (``make test-network``): they download both models (about 560 MB).
The ids equal those of each model's own tokenizer, ``[UNK]`` kept; the vectors equal
``SentenceTransformer.encode`` within the bound of tests/support/converted.py, whole and
with ``truncate_dim=256``; converting a commit always gives the same encoder, also from
the cache alone.
"""

import csv
import random
from pathlib import Path

import pytest

from fse._encoder import Encoder
from support import converted as cv
from support import wordpiece as wp

pytestmark = pytest.mark.network
ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "sentence-transformers/static-retrieval-mrl-en-v1": (
        "f60985c706f192d45d218078e49e5a8b6f15283a",
        "xxh3-128:6cb1abbcb0649e6362ff38e5ee29e0b4",
    ),
    "sentence-transformers/static-similarity-mrl-multilingual-v1": (
        "b68f4122911bcffcd6e1f695f2d99cd6788972d8",
        "xxh3-128:d0a49427e9aae947660d4a1e0c78846a",
    ),
}
# Sentences in several scripts, for the multilingual vocabulary.
SCRIPTS = [
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Le café est très chaud, n'est-ce pas ?",
    "El niño comió paella en Valencia.",
    "Быстрая коричневая лиса прыгает через ленивую собаку.",
    "\u0397 γρήγορη καφέ αλεπού πηδά πάνω από τον τεμπέλη σκύλο.",
    "敏捷的棕色狐狸跳过了懒狗。",
    "素早い茶色の狐がのろまな犬を飛び越える。",
    "빠른 갈색 여우가 게으른 개를 뛰어넘는다.",
    "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    "השועל החום המהיר קופץ מעל הכלב העצלן.",
    "H\u0131zl\u0131 kahverengi tilki tembel köpeğin üzerinden atlar.",
]


def _texts(tokens):
    with open(ROOT / "evaluation" / "sts-test.csv", encoding="utf-8", newline="") as f:
        sts = [s for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE) for s in row[5:7]]
    rng = random.Random(7)
    words = rng.sample([t for t in tokens if not t.startswith("[")], 2000)
    edge = ["hello [MASK] [PAD] [UNK] [CLS] world [SEP]", "x" * 101, "İstanbul", "", "zzqxj"]
    long = [" ".join(rng.choice(words) for _ in range(n)) for n in (600, 5000)]
    return sts + SCRIPTS + wp.corpus(words, count=2000, seed=3) + edge + long


@pytest.mark.parametrize("repo", sorted(MODELS))
def test_a_static_model_gives_the_ids_and_vectors_of_sentence_transformers(repo):
    sentence_transformers = pytest.importorskip("sentence_transformers")
    pytest.importorskip("huggingface_hub")

    revision, fingerprint = MODELS[repo]
    encoder = Encoder.from_pretrained(repo, revision=revision)
    assert encoder._content.source.revision == revision
    assert encoder._content.license == "apache-2.0"
    assert encoder.fingerprint == fingerprint
    offline = Encoder.from_pretrained(repo, revision=revision, local_files_only=True)
    assert offline.fingerprint == fingerprint
    model = sentence_transformers.SentenceTransformer(repo, revision=revision, device="cpu")
    tokens = [encoder.word_vectors.word(i) for i in range(len(encoder.word_vectors))]
    texts = _texts(tokens)
    want = [e.ids for e in model[0].tokenizer.encode_batch(texts, add_special_tokens=False)]
    assert [ids.tolist() for ids in cv.ids_of(encoder, texts)] == want
    worst = cv.assert_close(encoder.encode(texts), model.encode(texts), encoder, texts, False)
    cut = Encoder.from_pretrained(repo, revision=revision, truncate_dim=256)
    rows = model.encode(texts, truncate_dim=256)
    worst = max(worst, cv.assert_close(cut.encode(texts), rows, cut, texts, False))
    assert cut._content.source.dims == (1024, 256)
    print(f"{repo}: {len(texts)} texts, the largest error is {worst:.2e} of the bound")
    assert worst <= 1
