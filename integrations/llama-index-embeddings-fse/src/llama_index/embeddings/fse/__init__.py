"""LlamaIndex embeddings backed by fse."""

from asyncio import to_thread
from pathlib import Path

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr

from fse import Encoder
from fse._validation import validate_arguments


class FseEmbedding(BaseEmbedding):
    """Use LlamaIndex batching and fse encoding without a transformer runtime."""

    _encoder: Encoder = PrivateAttr()
    dtype: str = "float32"
    workers: int | None = None
    load_kwargs: dict = Field(default_factory=dict, exclude=True, repr=False)
    encoder_fingerprint: str = ""
    encoder_injected: bool = False

    @validate_arguments
    def __init__(
        self,
        model_name: str = "minishlab/potion-base-8M",
        *,
        encoder: Encoder | None = None,
        dtype: str = "float32",
        workers: int | None = None,
        load_kwargs: dict | None = None,
        encoder_fingerprint: str = "",
        encoder_injected: bool = False,
        **kwargs,
    ):
        if dtype not in ("float32", "float16"):
            raise ValueError("dtype must be float32 or float16 for float-list embeddings")
        if encoder is not None and load_kwargs:
            raise ValueError("load_kwargs cannot be used with an existing encoder")
        if encoder_injected and encoder is None:
            raise ValueError("Reconstruction requires the injected encoder or a saved model path")
        unknown = kwargs.keys() - BaseEmbedding.model_fields.keys()
        if unknown:
            raise TypeError(f"Unknown BaseEmbedding options: {sorted(unknown)}")
        super().__init__(
            model_name=model_name,
            dtype=dtype,
            workers=workers,
            load_kwargs=dict(load_kwargs or {}),
            encoder_fingerprint=encoder_fingerprint,
            encoder_injected=encoder is not None,
            **kwargs,
        )
        loaded = (
            encoder
            if encoder is not None
            else Encoder.from_pretrained(model_name, **(load_kwargs or {}))
        )
        if encoder_fingerprint and loaded.fingerprint != encoder_fingerprint:
            raise ValueError("Loaded encoder fingerprint differs from the serialized encoder")
        self.encoder_fingerprint = loaded.fingerprint
        self._encoder = loaded

    @classmethod
    def class_name(cls) -> str:
        return "FseEmbedding"

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data["load_kwargs"] = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.load_kwargs.items()
            if k != "token"
        }
        return data

    def __getstate__(self) -> dict:
        if self.encoder_injected:
            raise ValueError(
                "Save the injected encoder and use its local path before multiprocessing"
            )
        if "token" in self.load_kwargs:
            raise ValueError("Do not pickle Hub tokens. Use the Hub credential store")
        return {"config": {name: getattr(self, name) for name in type(self).model_fields}}

    def __setstate__(self, state: dict) -> None:
        self.__init__(**state["config"])

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._encoder.encode(texts, dtype=self.dtype, workers=self.workers).tolist()

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
