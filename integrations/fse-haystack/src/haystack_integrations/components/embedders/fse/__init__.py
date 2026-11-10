"""Haystack text and document embedders backed by fse."""

from asyncio import to_thread
from dataclasses import replace
from pathlib import Path
from typing import Self

from haystack import Document, component, default_from_dict, default_to_dict

from fse import Encoder
from fse.typing import ModelRef, WorkerCount

from ._checks import configuration

__all__ = ["FseDocumentEmbedder", "FseTextEmbedder"]


class _FseEmbedder:
    def __init__(
        self,
        model: ModelRef = "minishlab/potion-base-8M",
        *,
        encoder: Encoder | None = None,
        dtype: str = "float32",
        workers: WorkerCount = None,
        batch_size: int = 128,
        load_kwargs: dict[str, object] | None = None,
    ):
        model, workers, load_kwargs = configuration(model, encoder, dtype, workers, load_kwargs)
        if type(batch_size) is not int:
            raise TypeError("batch_size must be a positive integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.model = model
        self.dtype = dtype
        self.workers = workers
        self.batch_size = batch_size
        self.load_kwargs = dict(load_kwargs or {})
        self._encoder = encoder
        self._injected = encoder is not None

    def warm_up(self) -> None:
        """Load once before running a pipeline. An injected encoder is already ready."""
        if self._encoder is None:
            self._encoder = Encoder.from_pretrained(self.model, **self.load_kwargs)

    async def warm_up_async(self) -> None:
        await to_thread(self.warm_up)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._encoder is None:
            raise RuntimeError("Call warm_up() before run()")
        rows = []
        for start in range(0, len(texts), self.batch_size):
            rows.extend(
                self._encoder.encode(
                    texts[start : start + self.batch_size], dtype=self.dtype, workers=self.workers
                ).tolist()
            )
        return rows

    def to_dict(self) -> dict[str, object]:
        """Serialize loader options. In-memory encoders must be saved separately."""
        if self._injected:
            raise ValueError("Save the encoder and use its local path before serialization")
        if "token" in self.load_kwargs:
            raise ValueError("Do not serialize Hub tokens. Use the Hub credential store")
        return default_to_dict(
            self,
            model=str(self.model),
            dtype=self.dtype,
            workers=self.workers,
            batch_size=self.batch_size,
            load_kwargs={
                k: str(v) if isinstance(v, Path) else v for k, v in self.load_kwargs.items()
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        return default_from_dict(cls, data)


@component
class FseTextEmbedder(_FseEmbedder):
    """Embed one query. Output uses Haystack's ``embedding`` socket."""

    @component.output_types(embedding=list[float])
    def run(self, text: str) -> dict[str, list[float]]:
        return {"embedding": self._embed([text])[0]}

    @component.output_types(embedding=list[float])
    async def run_async(self, text: str) -> dict[str, list[float]]:
        return await to_thread(self.run, text)


@component
class FseDocumentEmbedder(_FseEmbedder):
    """Embed document content, preserving order and leaving inputs unchanged."""

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list) or any(not isinstance(d, Document) for d in documents):
            raise TypeError("documents must be a list of Haystack Documents")
        texts = [doc.content for doc in documents]
        if any(text is None for text in texts):
            raise ValueError("Every document must have text content")
        rows = self._embed(texts)
        return {
            "documents": [
                replace(doc, embedding=row) for doc, row in zip(documents, rows, strict=True)
            ]
        }

    @component.output_types(documents=list[Document])
    async def run_async(self, documents: list[Document]) -> dict[str, list[Document]]:
        return await to_thread(self.run, documents)
