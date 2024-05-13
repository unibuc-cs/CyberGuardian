import os
import projsecrets


def get_documents(collection=None, db=None, client=None, query=None):
    """Fetches a collection of documents from a document database."""
    collection = get_collection(collection, db, client)

    query = query or {"metadata.ignore": False}
    docs = collection.find(query)

    return docs


def drop(collection=None, db=None, client=None):
    """Drops a collection from the database."""
    collection = get_collection(collection, db, client)
    collection.drop()


def query(query, projection=None, collection=None, db=None):
    """Runs a query against the document db and returns a list of results."""
    import docstore

    collection = docstore.get_collection(collection, db)

    return list(collection.find(query, projection))


def query_one(query, projection=None, collection=None, db=None):
    """Runs a query against the document db and returns the first result."""
    import docstore

    db = db or os.environ["MONGODB_DATABASE"]
    collection = collection or os.environ["MONGODB_COLLECTION_MAIN"]

    collection = docstore.get_collection(collection, db)

    return collection.find_one(query, projection)


def get_collection(collection=None, db=None, client=None):
    """Accesses a specific collection in the document store."""
    import pymongo

    if projsecrets.cached_collection is not None:
        return projsecrets.cached_collection

    db = db if db is not None else os.environ["MONGODB_DATABASE"]
    db = get_database(db, client)


    collection = collection or os.environ["MONGODB_COLLECTION_MAIN"]

    if isinstance(collection, pymongo.collection.Collection):
        return collection
    else:
        collection = db.get_collection(collection)
        if collection is not None:
            projsecrets.cached_collection = collection # CACHE FIX

        return collection


def get_database(db=None, client=None):
    """Accesses a specific database in the document store."""
    import pymongo

    if projsecrets.cached_database is not None:
        return projsecrets.cached_database

    client = client or connect()

    db = db or os.environ["MONGODB_COLLECTION_MAIN"]
    if isinstance(db, pymongo.database.Database):
        return db
    else:
        db = client.get_database(db)
        if db is not None:
            projsecrets.cached_database = db  # CACHE FIX
        return db


def connect(user=None, password=None, uri=None):
    import os
    import pymongo
    import urllib

    """Connects to the document store, here MongoDB."""
    if projsecrets.cached_client is not None:
        return projsecrets.cached_client

    mongodb_user = user or os.environ["MONGODB_USER"]
    mongodb_user = urllib.parse.quote_plus(mongodb_user)

    mongodb_password = password or os.environ["MONGODB_PASSWORD"]
    mongodb_password = urllib.parse.quote_plus(mongodb_password)

    mongodb_host = uri or os.environ["MONGODB_HOST"]

    connection_string = f"mongodb+srv://{mongodb_user}:{mongodb_password}@{mongodb_host}/?retryWrites=true&w=majority"
    client = pymongo.MongoClient(connection_string, connect=True, appname="dynabicChatbot")

    if client is not None:
        projsecrets.cached_client = client # CACHE FIX

    return client

def main():
    connect()
    query = '{"type":"Document"}'
    docs = get_documents(query)

    collection = get_collection()
    query2 = '"filter": {"type":"Document"}'
    res = collection.find({"type":"Document"})
    res2 = collection.find({"type": {"not": {"in": ["Document"]}}})
    #collection.count_documents(filter=query)

    for doc in res:
        b = 3
        print(doc)
    a = 3

if __name__ == "__main__":
    main()