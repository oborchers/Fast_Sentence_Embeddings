"""LlamaIndex embeddings backed by fse."""

from asyncio import to_thread

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr

from fse import Encoder
from fse._validation import validate_arguments


class FseEmbedding(BaseEmbedding):
    """Use LlamaIndex batching and fse encoding without a transformer runtime."""

    _encoder: Encoder = PrivateAttr()
    _dtype: str = PrivateAttr()
    _workers: int | None = PrivateAttr()

    @validate_arguments
    def __init__(
        self,
        model_name: str = "minishlab/potion-base-8M",
        *,
        encoder: Encoder | None = None,
        dtype: str = "float32",
        workers: int | None = None,
        load_kwargs: dict | None = None,
        **kwargs,
    ):
        if dtype not in ("float32", "float16"):
            raise ValueError("dtype must be float32 or float16 for float-list embeddings")
        if encoder is not None and load_kwargs:
            raise ValueError("load_kwargs cannot be used with an existing encoder")
        unknown = kwargs.keys() - BaseEmbedding.model_fields.keys()
        if unknown:
            raise TypeError(f"Unknown BaseEmbedding options: {sorted(unknown)}")
        super().__init__(model_name=model_name, **kwargs)
        self._encoder = (
            encoder
            if encoder is not None
            else Encoder.from_pretrained(model_name, **(load_kwargs or {}))
        )
        self._dtype = dtype
        self._workers = workers

    @classmethod
    def class_name(cls) -> str:
        return "FseEmbedding"

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode(texts, dtype=self._dtype, workers=self._workers).tolist()

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_text_embedding(query)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await to_thread(self._get_text_embeddings, texts)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return await to_thread(self._get_text_embedding, text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await to_thread(self._get_query_embedding, query)
