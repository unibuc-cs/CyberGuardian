import os

os.environ["MONGODB_HOST"] = "dynabicchatbot.n5fwe4p.mongodb.net"
os.environ["MONGODB_PASSWORD"] = "Arbori2009"
os.environ["MONGODB_USER"] = "paduraru2009"
os.environ["MONGODB_DATABASE"] = "dynabicChatbot"
os.environ["PDF_LOCAL_JSON_DB"] = os.path.join("data", "dataForTraining", "db_pdfs.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB"] = os.path.join("data", "dataForTraining", "db_videos.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB_CLEANED"] = os.path.join("data", "dataForTraining", "db_videos_clean.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB_FAILED"] = os.path.join("data", "dataForTraining", "db_videos_failed.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB_TRANSLATED"] = os.path.join("data", "dataForTraining", "db_videos_translated.jsonl")
os.environ["MONGODB_COLLECTION_MAIN"] = "knowledgedb"
os.environ["MONGODB_COLLECTION_RAG"] = "knowledgerag"
os.environ["VECTOR_DIR_MAIN"] = "./vectors"
os.environ["VECTOR_DIR_RAG"] = "./vectors_rag"
os.environ["MONGODB_CLIENT"] = ""

# WHICH DB to use for RAG ? IF both are false, it will use ONLY the data for rag
os.environ["USE_MAIN_KNOWLEDGE_FOR_RAG"] = "False"
os.environ["USE_ALL_KNOWLEDGE_FOR_RAG"] = "False"
os.environ["USE_RAG_KNOWLEDGE_FOR_RAG"] = "True"

cached_client = None
cached_database = None
cached_collection = None
