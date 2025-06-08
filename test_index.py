# test_index.py
from rag.embedder import embed_and_store

DOC_DIR = "data/raw_docs"
INDEX_PATH = "data/faiss_index"

embed_and_store(DOC_DIR, INDEX_PATH)
