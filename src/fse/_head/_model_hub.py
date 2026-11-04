"""Hub persistence of a linear model, shared by classifiers and regressors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from fse._encoder import _folder
from fse._validation import PathLike, validate_arguments


class HubModel(ABC):
    """The Hub operations of models implementing save."""

    __slots__ = ()

    @abstractmethod
    def save(self, path: PathLike, standalone: bool = False) -> None:
        """Write the concrete model artifact."""

    @validate_arguments
    def save_pretrained(
        self,
        path: PathLike,
        *,
        standalone: bool = False,
        license: str | list[str] | None = None,
        language: list[str] | None = None,
        model_name: str | None = None,
        extra_files: dict[str, PathLike] | None = None,
    ) -> Path:
        """Write the model as a Hugging Face folder at the new ``path`` (never half written
        or replaced): ``config.json``, ``model.safetensors`` (as ``save`` writes it,
        complete or ``standalone``), a ``README.md`` model card of that file with the
        license its encoder records (or ``license`` if it records none) and
        ``extra_files`` (name to local text file, such as LICENSE). ``from_pretrained``
        reads it. docs/architecture/hub.md, "Hub folders"."""

        def write(file: Path) -> None:
            self.save(file, standalone=standalone)

        card: dict[str, Any] = {"license": license, "language": language, "extra": extra_files}
        return _folder.save(path, write, model_name=model_name, **card)[0]

    @validate_arguments
    def push_to_hub(
        self,
        repo_id: str,
        *,
        standalone: bool = False,
        token: str | bool | None = None,
        private: bool | None = None,
        revision: str | None = None,
        create_pr: bool = False,
        commit_message: str | None = None,
        replace: bool = False,
        license: str | list[str] | None = None,
        language: list[str] | None = None,
        extra_files: dict[str, PathLike] | None = None,
    ) -> str:
        """Publish the folder of ``save_pretrained`` to the Hub repository ``repo_id`` in
        one commit and return its URL, as ``Encoder.push_to_hub`` does: a license is
        required, other content under the same names and other files are refused unless
        ``replace=True``, license files already there are kept, every file is checked at
        the new commit."""

        def write(file: Path) -> None:
            self.save(file, standalone=standalone)

        hub = {"token": token, "private": private, "revision": revision, "replace": replace}
        hub |= {"create_pr": create_pr, "commit_message": commit_message}
        card: dict[str, Any] = {"license": license, "language": language, "extra": extra_files}
        return _folder.push(
            repo_id, write, type(self).__name__.lower(), hub, card | {"model_name": repo_id}
        )

