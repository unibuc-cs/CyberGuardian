"The file used to create datasets from videos, markdowns and PDFs"
import math
import os
import argparse
import Data.etlUtils.etl_pdfs
import Data.dataSettings
from pathlib import Path
import pprint

from Data.etlUtils import etl_videos

import docstore
from Data.etlUtils import vecstore
from Data.utils import pretty_log
from langchain_text_splitters import RecursiveCharacterTextSplitter

pp = pprint.PrettyPrinter(indent=2)

import logging
# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger("main")

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

def prep_documents_for_vector_storage(documents, num_documents):
    """Prepare documents from document store for embedding and vector storage.

    Documents are split into chunks so that they can be used with sourced Q&A.

    Arguments:
        documents: A list of LangChain.Documents with text, metadata, and a hash ID.
    """

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=100, allowed_special="all"
    )
    ids, texts, metadatas = [], [], []
    counter = 0
    log_at_each_num_steps = max(int(math.floor(num_documents*0.1)), 1)
    for document in documents: #enumerate(tqdm(documents, total=num_documents, desc="Preparing documents for storage",leave=True, position=0)):
        if counter % log_at_each_num_steps == 0:
            pretty_log(f"Processed {counter}/{num_documents}")

        text, metadata = document["text"], document["metadata"]
        doc_texts = text_splitter.split_text(text)
        doc_metadatas = [metadata] * len(doc_texts)
        ids += [metadata.get("sha256")] * len(doc_texts)
        texts += doc_texts
        metadatas += doc_metadatas
        counter += 1

    return ids, texts, metadatas



def create_vector_index(vectorIndexPath:str,
                        vectorIndexName: str,
                        collection: str = None,
                        db: str = None):
    """Creates a vector index for a collection in the document database."""
    import docstore

    pretty_log("connecting to document store")
    db = docstore.get_database(db)
    pretty_log(f"connected to database {db.name}")

    collection = docstore.get_collection(collection, db)
    num_documents = collection.count_documents({})
    pretty_log(f"collecting documents from {collection.name}. There are {num_documents} documents")
    docs = docstore.get_documents(collection, db)

    pretty_log("splitting into bite-size chunks")
    ids, texts, metadatas = prep_documents_for_vector_storage(docs, num_documents)

    pretty_log(f"sending to vector index {vectorIndexName}")
    embedding_engine = vecstore.get_embedding_engine(disallowed_special=())
    vector_index = vecstore.create_vector_index(vectorIndexPath=vectorIndexPath,
                                                index_name=vectorIndexName,
                                                embedding_engine=embedding_engine,
                                                documents=texts,
                                                metadatas=metadatas)

    vector_index.save_local(folder_path=vectorIndexPath, index_name=vectorIndexName)
    pretty_log(f"vector index {vectorIndexName} created at path {vectorIndexPath}")

def solve_vector_storage(maindb: bool):
    vectorPath = vecstore.VECTOR_DIR_MAIN if maindb is True else vecstore.VECTOR_DIR_RAG
    vector_storage_name = vecstore.INDEX_NAME_MAIN if maindb else vecstore.INDEX_NAME_RAG
    collectionToUse = os.environ["MONGODB_COLLECTION_MAIN"] if maindb else os.environ["MONGODB_COLLECTION_RAG"]

    create_vector_index(vectorIndexPath=vectorPath,
                        vectorIndexName=vector_storage_name,
                        collection=collectionToUse,
                        db=os.environ["MONGODB_DATABASE"])


def drop_collection():
    docstore.drop(os.environ["MONGODB_COLLECTION"], os.environ["MONGODB_DATABASE"], os.environ["MONGODB_CLIENT"])


def create_knowledge_database(drop: bool = False,
                                etl_pdfs: bool = False,
                                etl_markdown: bool = False,
                                etl_videos: bool = False,
                                dovectorstorage_main: bool = False,
                                dovectorstorage_rag: bool = False,
                              demoMode: bool = False):
    if drop:
        drop_collection()

    if etl_markdown:
        Data.etlUtils.etl_pdfs.transform_papers_to_json(Path("Data") / ("dataForTraining") / "webcontent.json", demoMode=demoMode)

    if etl_pdfs:
        Data.etlUtils.etl_pdfs.transform_papers_to_json(Path("Data") / ("dataForTraining") / "pdfpapers.json", demoMode=demoMode)

    if etl_videos:
        Data.etlUtils.etl_videos.main(Path("Data") / ("dataForTraining") / "videos.json", demoMode=demoMode)


    if dovectorstorage_main:
        solve_vector_storage(maindb=True)

    if dovectorstorage_rag:
        solve_vector_storage(maindb=False) # RAG content custom for a particular project example

def __main__():
    parser = argparse.ArgumentParser(description="Dataset building and cleaning utils")
    parser.add_argument("--drop_existing", type=int, default=False)
    parser.add_argument("--etl_pdfs", type=int, default=False)
    parser.add_argument("--etl_markdown", type=int, default=False)
    parser.add_argument("--etl_videos", type=int, default=False)
    parser.add_argument("--dovectorstorage_main", type=int, default=False) # CREATES WHOLE DB , the MAIN one
    parser.add_argument("--dovectorstorage_rag", type=int, default=False) # CREATES just the RAG db
    parser.add_argument("--demoMode", type=int, default=False)

    args = parser.parse_args()

    create_knowledge_database(drop=bool(int(args.drop_existing)),
                              etl_pdfs=bool(int(args.etl_pdfs)),
                              etl_markdown=bool(int(args.etl_markdown)),
                              etl_videos=bool(int(args.etl_videos)),
                            dovectorstorage_main=bool(int(args.dovectorstorage_main)),
                              dovectorstorage_rag=bool(int(args.dovectorstorage_rag)),
                              demoMode=bool(int(args.demoMode)))

if __name__ == "__main__":
    __main__()
