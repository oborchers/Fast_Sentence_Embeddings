"""Path protocol implementations that are deliberately not pathlib.Path."""

import os


class OwnPath:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = os.fspath(path)

    def __fspath__(self) -> str:
        return self.path


class BytesPath:
    def __fspath__(self) -> bytes:
        return b"not-a-text-path"


class CountingPath(OwnPath):
    def __init__(self, path: str | os.PathLike[str]) -> None:
        super().__init__(path)
        self.calls = 0

    def __fspath__(self) -> str:
        self.calls += 1
        return self.path
