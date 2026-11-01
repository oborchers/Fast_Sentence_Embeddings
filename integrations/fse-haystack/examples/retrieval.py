"""Real POTION indexing and a connected Haystack retrieval pipeline."""

import os
from importlib.metadata import version

from haystack import Document, Pipeline
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.embedders.fse import FseDocumentEmbedder, FseTextEmbedder

os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "false"

texts = [
    "Cats are small pets that purr and chase mice.",
    "Baking bread requires flour, water and yeast.",
    "Planets orbit the sun in the solar system.",
]
embedder = FseDocumentEmbedder(model="minishlab/potion-base-8M", workers=1, batch_size=2)
embedder.warm_up()
documents = embedder.run([Document(content=text) for text in texts])["documents"]
assert all(len(doc.embedding) == 256 for doc in documents)
store = InMemoryDocumentStore(embedding_similarity_function="cosine")
store.write_documents(documents)
pipeline = Pipeline()
pipeline.add_component("embed", FseTextEmbedder(model="minishlab/potion-base-8M", workers=1))
pipeline.add_component("retrieve", InMemoryEmbeddingRetriever(document_store=store, top_k=1))
pipeline.connect("embed.embedding", "retrieve.query_embedding")
result = pipeline.run({"embed": {"text": "Which pets purr?"}})["retrieve"]["documents"]
assert len(result) == 1
assert result[0].content == texts[0]
print(f"haystack-ai={version('haystack-ai')}; model=minishlab/potion-base-8M")
print(f"documents={len(texts)}; dimensions=256; top={result[0].content}")
print(f"cosine={result[0].score:.6f}; retrieval passed")
