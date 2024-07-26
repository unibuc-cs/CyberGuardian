import logging
from typing import Dict, Any, List
from pydantic import BaseModel
from langchain_core.documents import Document

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("main")



def convert_score_to_float(score: Dict[str, Any]) -> float:
    """
    Convert a score dictionary to a float
    Args:
        score: Dict[str, Any]
    Returns:
        float: The score as a float
    """
    res = None 
    raw_score = score["score"]
    try:
        res = float(raw_score)
    except:
        try:
            res = 1.0 if type(raw_score) == str and raw_score.lower() in ["yes", "true"] else 0.0
        except:
            res = 0.0
            logger.error(f"Could not convert score to float: {score}")
    return res


# Given a list of documents (Document in the langchain RAG API), return a string with all the documents' content
def format_docs_to_str(docs: List[Document]) -> str:
    assert type(docs) is List[Document]
    return "\n\n".join(doc.page_content for doc in docs)
