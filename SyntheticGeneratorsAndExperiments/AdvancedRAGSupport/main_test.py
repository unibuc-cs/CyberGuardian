from AdaptiveRAGSupport import AdaptiveRAGSupport, logger

def demo_test_adaptive_rag():
    # Test the AdaptiveRAGSupport
    model = None
    tokenizer = None
    retriever = None
    support = AdaptiveRAGSupport(model, tokenizer, retriever)

    question = "agent memory" #"What is the best way to train a model?"
    docs_content = support.get_docs_content_by_query(question, support.DEFAULT_MAX_DOCS_RETURNED, verbose=False)
    answer = support.answer_with_rag(question, docs_content, skip_prompt=True)
    logger.warning(answer)
    logger.warning(f"answer is good for the question? {support.answer_grader(question, answer, skip_prompt=True)}")
    logger.warning(f"halucination grader, i.e., is the answer sustained by documents? {support.hallucination_grader(answer, docs_content, skip_prompt=True)}")
    logger.warning(f"question category: {support.question_router(question, answer, docs_content, skip_prompt=True)})")


if __name__ == "__main__":
    demo_test_adaptive_rag()
    
    
import torch
    
MLP 1:

import torch.nn as nn
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1, bias=False),
)


MLP 2:

import torch.nn as nn
model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1),
)
