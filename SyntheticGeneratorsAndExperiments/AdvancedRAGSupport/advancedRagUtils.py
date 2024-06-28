import logging
from typing import Dict, Any

logger = logging.getLogger("AdvancedRagSupport") #get_logger(__name__)
logger.setLevel(logging.INFO)


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
        except e:
            res = 0.0
            logger.error(f"Could not convert score to float: {score}")
    return res
