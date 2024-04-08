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
os.environ["MONGODB_COLLECTION"] = "knowledgedb"
os.environ["VECTOR_DIR"] = "./vectors"
os.environ["MONGODB_CLIENT"] = ""

cached_client = None
cached_database = None
cached_collection = None