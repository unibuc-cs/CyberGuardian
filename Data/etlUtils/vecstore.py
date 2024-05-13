"""Utilities for creating and using vector indexes."""
import os
from pathlib import Path

from Data.utils import pretty_log

INDEX_NAME_MAIN = "knowledgedb"
INDEX_NAME_RAG = "knowledgerag"
VECTOR_DIR_MAIN = Path(os.environ["VECTOR_DIR_MAIN"])
VECTOR_DIR_RAG= Path(os.environ["VECTOR_DIR_RAG"])


def connect_to_vector_index(index_path, index_name, embedding_engine):
    """Adds the texts and metadatas to the vector index."""
    from langchain_community.vectorstores import FAISS

    vector_index = FAISS.load_local(index_path, embedding_engine, index_name,
                                    allow_dangerous_deserialization= True)

    return vector_index


def get_embedding_engine(**kwargs):
    """Retrieves the embedding engine."""
    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return embed_model



def create_vector_index(vectorIndexPath, index_name, embedding_engine, documents, metadatas):
    """Creates a vector index that offers similarity search."""
    from langchain_community.vectorstores import FAISS

    files = vectorIndexPath.glob(f"{index_name}.*")
    if files:
        for file in files:
            file.unlink()
        pretty_log("existing index wiped")

    if len(documents) > 0:
        index = FAISS.from_texts(texts=documents, embedding=embedding_engine, metadatas=[{'title':'dummy'}])
    else:
        # TODO: the use case for rag
        texts = ["FAISS is an important library", "LangChain supports FAISS"]

        # Fake metadatas if nothing is there
        metadata = {'title': 'dummy',
                    'source' : 'my',
                    'page': '0',
                    'Data' : '2024',
                    'sha256' : '0',
                    'is_endmatter' : 'False',
                    'ignore' : 'False'}



        index = FAISS.from_texts(texts, embedding_engine, metadatas=[metadata]*len(texts))

    return index
