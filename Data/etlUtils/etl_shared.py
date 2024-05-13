import os

from tqdm.auto import tqdm
import random
from typing import List

#DEMO_MODE = False
DEMO_VIDEO_SAMPLES = 300
DEMO_PDF_SAMPLES = 50

PARALLEL_NUM_PROCESSES = 16
PARALLEL_USE = False
PARALLEL_CHUNK_SIZE = 8

# Check if string contains only ascii characters
def is_ascii(s):
    try:
        s.encode('ascii')
        return True
    except UnicodeDecodeError:
        return False


# Subsample numSamples from the given raw list
def subsampleForDemo(inputRaw: List, numSamples: int):
    input_samples = random.choices(inputRaw, k=numSamples)
    return input_samples


# Parallel processing with tqdm bar of FUNC using an iterable given by args
def imap_unordered_bar(func, args, n_processes=2, desc: str = None):
    from multiprocessing import Pool
    p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args), desc=desc, position=0, leave=True) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args, chunksize=PARALLEL_CHUNK_SIZE)), nrows=1):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list




def add_to_mongo_db(documents_json, collection=None, db=None):
    """Adds a collection of json documents to a database."""
    from pymongo import InsertOne

    from Data.etlUtils import docstore

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

def add_metadata(pages):
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

# Splits list into n_chunks pieces, non-contiguously.
def chunk_into(list, n_chunks):

    for ii in range(0, n_chunks):
        yield list[ii::n_chunks]

# Recombines a list of lists into a single list
def unchunk(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

