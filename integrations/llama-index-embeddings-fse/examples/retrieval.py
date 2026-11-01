"""Real POTION index and retrieval, without an LLM or API key."""

from importlib.metadata import version

from llama_index.core import Document, VectorStoreIndex
from llama_index.embeddings.fse import FseEmbedding

texts = [
    "Cats are small pets that purr and chase mice.",
    "Baking bread requires flour, water and yeast.",
    "Planets orbit the sun in the solar system.",
]
embeddings = FseEmbedding(model_name="minishlab/potion-base-8M", workers=1, embed_batch_size=2)
index = VectorStoreIndex.from_documents(
    [Document(text=text) for text in texts], embed_model=embeddings
)
result = index.as_retriever(similarity_top_k=1).retrieve("Which pets purr?")
assert len(result) == 1
assert result[0].node.text == texts[0]
assert len(embeddings.get_query_embedding("Which pets purr?")) == 256
print(f"llama-index-core={version('llama-index-core')}; model=minishlab/potion-base-8M")
print(f"documents={len(texts)}; dimensions=256; top={result[0].node.text}")
print(f"cosine={result[0].score:.6f}; retrieval passed")
