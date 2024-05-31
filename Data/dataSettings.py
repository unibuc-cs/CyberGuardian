# General environments variales to help paths and avoid hardcoding values

import os

os.environ["PDF_LOCAL_JSON_DB"] = os.path.join("Data", "dataForTraining", "db_pdfs.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB"] = os.path.join("Data", "dataForTraining", "db_videos.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB_CLEANED"] = os.path.join("Data", "dataForTraining", "db_videos_clean.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB_FAILED"] = os.path.join("Data", "dataForTraining", "db_videos_failed.jsonl")
os.environ["VIDEOS_LOCAL_JSON_DB_TRANSLATED"] = os.path.join("Data", "dataForTraining", "db_videos_translated.jsonl")
os.environ["MONGODB_COLLECTION_MAIN"] = "knowledgedb"
os.environ["MONGODB_COLLECTION_RAG"] = "knowledgerag"
os.environ["VECTOR_DIR_MAIN"] = "./RAGSupport/vectors"
os.environ["VECTOR_DIR_RAG"] = "./RAGSupport/vectors_rag"

os.environ["LLM_PARAMS_PATH_INFERENCE"] = os.path.join("LLM", "llm_params_inference.json")
os.environ["LLM_PARAMS_PATH_TRAINING"] = os.path.join("LLM", "llm_params_training.json")


# WHICH DB to use for RAG ? IF both are false, it will use ONLY the Data for rag
os.environ["USE_MAIN_KNOWLEDGE_FOR_RAG"] = "False"
os.environ["USE_ALL_KNOWLEDGE_FOR_RAG"] = "False"
os.environ["USE_RAG_KNOWLEDGE_FOR_RAG"] = "True"

cached_client = None
cached_database = None
cached_collection = None