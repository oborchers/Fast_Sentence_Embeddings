"""Path protocol implementations that are deliberately not pathlib.Path."""

import os


class OwnPath:
    def __init__(self, path):
        self.path = os.fspath(path)

    def __fspath__(self) -> str:
        return self.path


class BytesPath:
    def __fspath__(self) -> bytes:
        return b"not-a-text-path"
