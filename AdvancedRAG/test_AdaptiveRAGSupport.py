from AdaptiveRAGSupport import AdaptiveRAGSupport, logger

# This is a test for the AdaptiveRAGSupport class
def demo_test_adaptive_rag():
    # Test the AdaptiveRAGSupport
    model = None
    tokenizer = None
    retriever = None
    support = AdaptiveRAGSupport(model, tokenizer, retriever)

    question = "agent memory" #"What is the best way to train a model?"
    docs_content = support.get_docs_content_by_query(question, support.DEFAULT_MAX_DOCS_RETURNED,
                                                     format_as_single_text= True, verbose=False)
    answer = support.answer_with_rag(question, docs_content, skip_prompt=True)
    logger.warning(answer)
    logger.warning(f"answer is good for the question? {support.answer_grader(question, answer, skip_prompt=True)}")
    logger.warning(f"hallucination grader, i.e., is the answer sustained by documents? "
                   f"{support.hallucination_grader(answer, docs_content, skip_prompt=True)}")
    logger.warning(f"question category: {support.question_router(question, skip_prompt=True)})")


if __name__ == "__main__":
    demo_test_adaptive_rag()
    
