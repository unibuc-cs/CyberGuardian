import os

def add_to_document_db(documents_json, collection=None, db=None):
    """Adds a collection of json documents to a database."""
    from pymongo import InsertOne

    import docstore

    res = None

    db = db or os.environ["MONGODB_DATABASE"]
    collection = collection or os.environ["MONGODB_COLLECTION"]
    collection = docstore.get_collection(collection, db)

    requesting, CHUNK_SIZE = [], 250

    for document in documents_json:
        if isinstance(document, list) and len(document) == 0:
            continue
        elif document is None:
            continue

        requesting.append(InsertOne(document))

        if len(requesting) >= CHUNK_SIZE:
            collection.bulk_write(requesting)
            requesting = []

    if requesting:
        res = collection.bulk_write(requesting)

    assert res is not None or len(documents_json) == 0
    return res

def enrich_metadata(pages):
    """Add our metadata: sha256 hash and ignore flag."""
    import hashlib

    for page in pages:
        m = hashlib.sha256()
        m.update(page["text"].encode("utf-8", "replace"))
        page["metadata"]["sha256"] = m.hexdigest()
        if page["metadata"].get("is_endmatter"):
            page["metadata"]["ignore"] = True # Ignore the pages that are ends (bibliography, references, etc).
        else:
            page["metadata"]["ignore"] = False
    return pages


def chunk_into(list, n_chunks):
    """Splits list into n_chunks pieces, non-contiguously."""
    for ii in range(0, n_chunks):
        yield list[ii::n_chunks]


def unchunk(list_of_lists):
    """Recombines a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]

