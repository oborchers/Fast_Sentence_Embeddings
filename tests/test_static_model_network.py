"""fse.StaticModel on the static models of the Hub at pinned commits, against model2vec
0.10.0: the attributes and the rows of ``encode`` (whole, with settings of one call, and
with ``dimensionality=256`` for the two MRL models), also for the two repositories of
sentence-transformers, which model2vec reads with its own rules.

Marked ``network`` (``make test-network``): they download the models (about 1 GB).
model2vec takes no revision, so it reads the snapshot of the pinned commit.
"""

import pytest

import fse
from support import static_model as sm

pytestmark = pytest.mark.network
MODELS = {
    "minishlab/potion-base-8M": "bf8b056651a2c21b8d2565580b8569da283cab23",
    "minishlab/potion-base-32M": "1e5a03f8eeb2c98b928fbbd846f22f816360919f",
    "minishlab/potion-retrieval-32M": "6fc8051fab2a1e0ee76689cf08c853792ac285e7",
    "sentence-transformers/static-retrieval-mrl-en-v1": "f60985c706f192d45d218078e49e5a8b6f15283a",
    "sentence-transformers/static-similarity-mrl-multilingual-v1": (
        "b68f4122911bcffcd6e1f695f2d99cd6788972d8"
    ),
}
MRL = {name for name in MODELS if name.startswith("sentence-transformers/")}
SCRIPTS = [
    "Der schnelle braune Fuchs springt über den faulen Hund.",
    "Быстрая коричневая лиса.",
    "敏捷的棕色狐狸跳过了懒狗。",
    "الثعلب البني السريع.",
]


@pytest.fixture(autouse=True)
def online(monkeypatch):
    """The tests of the fake Hub set the offline mode, and huggingface_hub reads it once:
    these tests need the network, whatever ran before them in the process."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)


@pytest.mark.parametrize("repo", sorted(MODELS))
def test_a_hub_model_is_the_static_model_of_model2vec(repo):
    model2vec = pytest.importorskip("model2vec")
    huggingface_hub = pytest.importorskip("huggingface_hub")

    revision = MODELS[repo]
    folder = huggingface_hub.snapshot_download(repo, revision=revision)
    ours = fse.StaticModel.from_pretrained(repo, revision=revision)
    theirs = model2vec.StaticModel.from_pretrained(folder)
    assert ours.encoder._content.source.revision == revision
    texts = sm.hub_texts(ours, SCRIPTS)
    worst = sm.assert_the_model_of_model2vec(ours, theirs, texts)
    if repo in MRL:
        cut_ours = fse.StaticModel.from_pretrained(repo, revision=revision, dimensionality=256)
        cut_theirs = model2vec.StaticModel.from_pretrained(folder, dimensionality=256)
        assert cut_ours.dim == 256
        worst = max(worst, sm.assert_the_model_of_model2vec(cut_ours, cut_theirs, texts))
    offline = fse.StaticModel.from_pretrained(repo, revision=revision, local_files_only=True)
    assert offline.encode(texts[:50]).tobytes() == ours.encode(texts[:50]).tobytes()
    print(f"{repo}: {len(texts)} texts, the largest error is {worst:.2e} of the bound")
    assert worst <= 1
