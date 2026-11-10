"""LangChain embeddings backed by fse."""

from asyncio import to_thread

from langchain_core.embeddings import Embeddings

from fse import Encoder
from fse.typing import ModelRef, WorkerCount

from ._checks import configuration

__all__ = ["FseEmbeddings"]


class FseEmbeddings(Embeddings):
    """Load a pretrained encoder or wrap an existing local encoder.

    ``load_kwargs`` goes to ``Encoder.from_pretrained`` unchanged. Encoding uses
    one fse call for the supplied batch. fse controls its internal scheduling.
    """

    def __init__(
        self,
        model: ModelRef = "minishlab/potion-base-8M",
        *,
        encoder: Encoder | None = None,
        dtype: str = "float32",
        workers: WorkerCount = None,
        load_kwargs: dict[str, object] | None = None,
    ):
        model, workers, load_kwargs = configuration(model, encoder, dtype, workers, load_kwargs)
        self.encoder = (
            encoder
            if encoder is not None
            else Encoder.from_pretrained(model, **(load_kwargs or {}))
        )
        self.dtype = dtype
        self.workers = workers

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        return self.encoder.encode(texts, dtype=self.dtype, workers=self.workers).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await to_thread(self.embed_query, text)
