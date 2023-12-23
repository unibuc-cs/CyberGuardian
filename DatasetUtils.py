"""Drops a collection from the document storage."""
import os
import projsecrets

import json
from pathlib import Path
import pprint
import pdb

from etl import markdown, pdfs, shared, videos

import docstore
import vecstore

from utils import pretty_log

pp = pprint.PrettyPrinter(indent=2)


def prep_documents_for_vector_storage(documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100, allowed_special="all"
    )
    ids, texts, metadatas = [], [], []
    for document in documents:
        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas

    return ids, texts, metadatas



# TODO: use 8 CPUs here use parallelism !!!
def create_vector_index(collection: str = None, db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    pretty_log(f"collecting documents from {collection.name}")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs)

    pretty_log(f"sending to vector index {vecstore.INDEX_NAME}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(
        vecstore.INDEX_NAME, embedding_engine, texts, metadatas
    )
    vector_index.save_local(folder_path=vecstore.VECTOR_DIR, index_name=vecstore.INDEX_NAME)
    pretty_log(f"vector index {vecstore.INDEX_NAME} created")

def transform_papers_to_json():
    papers_path = Path("data") / "llm-papers.json"

    with open(papers_path) as f:
        pdf_infos = json.load(f)

    # print(pdf_infos[:100:20])

    # E nrich the paper data by finding direct PDF URLs where we can
    paper_data = map(pdfs.get_pdf_url, pdf_infos[::25])

    # turn the PDFs into JSON documents
    it = map(pdfs.extract_pdf, paper_data)
    documents = shared.unchunk(it)

    # Store the collection of docs on the server
    docstore.drop(os.environ["MONGODB_COLLECTION"], os.environ["MONGODB_DATABASE"], os.environ["MONGODB_CLIENT"])

    # Test out debug
    #pp.pprint(documents[0]["metadata"])

    # Split document list into 10 pieces
    chunked_documents = shared.chunk_into(documents, 10)
    results = list(map(shared.add_to_document_db, chunked_documents))

    # Pull only arxiv papers
    query = {"metadata.source": {"$regex": "arxiv\.org", "$options": "i"}}
    # Project out the text field, it can get large
    projection = {"text": 0}
    # get just one result to show it worked
    result = docstore.query_one(query, projection)


    pp.pprint(result)

def solve_vector_storage():
    VECTOR_DIR = vecstore.VECTOR_DIR
    vector_storage = "vector-vol"

    create_vector_index(os.environ["MONGODB_COLLECTION"], os.environ["MONGODB_DATABASE"])


def __main__():
    transform_papers_to_json()
    solve_vector_storage()

if __name__ == "__main__":
    __main__()
