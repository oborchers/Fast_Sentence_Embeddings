"""Real POTION similarity search. Downloads a public model on first run."""

from importlib.metadata import version

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_fse import FseEmbeddings

texts = [
    "Cats are small pets that purr and chase mice.",
    "Baking bread requires flour, water and yeast.",
    "Planets orbit the sun in the solar system.",
]
embeddings = FseEmbeddings(model="minishlab/potion-base-8M", workers=1)
store = InMemoryVectorStore(embeddings)
store.add_documents([Document(page_content=text) for text in texts])
result = store.similarity_search_with_score("Which pets purr?", k=1)
assert len(result) == 1
assert result[0][0].page_content == texts[0]
assert len(embeddings.embed_query("Which pets purr?")) == 256
print(f"langchain-core={version('langchain-core')}; model=minishlab/potion-base-8M")
print(f"documents={len(texts)}; dimensions=256; top={result[0][0].page_content}")
print(f"cosine={float(result[0][1]):.6f}; retrieval passed")
