"""Exercise installed PyPI adapters 0.1.0 against the current fse checkout."""

import sys
from importlib import import_module
from importlib.metadata import version

import numpy as np

from fse import Encoder


def main(package):
    assert version(package) == "0.1.0"
    encoder = Encoder.from_arrays(np.array([[1, 0, 2]], dtype=np.float32), words=["hello"])
    expected = encoder.encode(["hello"]).tolist()[0]
    if package == "langchain-fse":
        adapter = import_module("langchain_fse").FseEmbeddings

        assert adapter(encoder=encoder).embed_query("hello") == expected
    elif package == "llama-index-embeddings-fse":
        adapter = import_module("llama_index.embeddings.fse").FseEmbedding

        assert adapter(encoder=encoder).get_query_embedding("hello") == expected
    elif package == "fse-haystack":
        adapter = import_module("haystack_integrations.components.embedders.fse").FseTextEmbedder

        assert adapter(encoder=encoder).run("hello")["embedding"] == expected
    else:
        raise ValueError("Unknown adapter package")


if __name__ == "__main__":
    main(sys.argv[1])
