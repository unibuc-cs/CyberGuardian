"""Utilities for creating and using vector indexes."""
import os
from pathlib import Path

from utils import pretty_log

INDEX_NAME = "knowledgedb"
VECTOR_DIR = Path(os.environ["VECTOR_DIR"])


def connect_to_vector_index(index_name, embedding_engine):
    """Adds the texts and metadatas to the vector index."""
    from langchain_community.vectorstores import FAISS

    vector_index = FAISS.load_local(VECTOR_DIR, embedding_engine, index_name)

    return vector_index


def get_embedding_engine(**kwargs):
    """Retrieves the embedding engine."""
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embed_model



def create_vector_index(index_name, embedding_engine, documents, metadatas):
    """Creates a vector index that offers similarity search."""
    from langchain_community.vectorstores import FAISS

    files = VECTOR_DIR.glob(f"{index_name}.*")
    if files:
        for file in files:
            file.unlink()
        pretty_log("existing index wiped")

    index = FAISS.from_texts(
        texts=documents, embedding=embedding_engine, metadatas=metadatas
    )

    return index
