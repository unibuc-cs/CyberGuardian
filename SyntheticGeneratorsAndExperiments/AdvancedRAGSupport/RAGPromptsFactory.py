from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class RAGPromptsFactory:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.answer_grader_prompt = self._create_answer_grader_prompt()
        self.rag_prompt = self._create_rag_prompt()
        self.hallucination_grader_prompt = self._create_hallucination_prompt()
        self.question_router_prompt = self._create_question_router_prompt()

    def _create_answer_grader_prompt(self):
        """
        # Setup the grader input template
        """
        answer_grader_chat_messages = [
            {"role": "system",
             "content": """You are a grader assessing whether an 
            answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
            useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
             },

            {
                "role": "user",
                "content": """Here is the answer:
            \n ------- \n
            {generation} 
            \n ------- \n
            Here is the question: {question}"""
            }
        ]

        input_txt = self.tokenizer.apply_chat_template(
            answer_grader_chat_messages,
            add_generation_prompt=True,
            tokenize=False)

        # Langachain prompt template, partially filled with the format instructions above
        prompt = PromptTemplate(
            template=input_txt,
            input_variables=["generation", "question"],
        )

        return prompt

    def _create_rag_prompt(self):
        """
        # Setup the RAG prompt for retriving docs from a collection given a question and answer to a user query
        """
        answer_grader_chat_messages = [
            {"role": "system",
             "content": """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.""",
             },

            {
                "role": "user",
                "content": """Question: {question}
            Context: {context}"""
            }
        ]

        input_txt = self.tokenizer.apply_chat_template(
            answer_grader_chat_messages,
            add_generation_prompt=True,
            tokenize=False)

        # Langachain prompt template, partially filled with the format instructions above
        prompt = PromptTemplate(
            template=input_txt,
            input_variables=["question", "document"],
        )

        return prompt

    def _create_hallucination_prompt(self):
        """
        # Setup the hallucination prompt for retriving docs from a collection given a question and answer to a user query
        """
        answer_grader_chat_messages = [
            {"role": "system",
             "content": """You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation."""
             },

            {
                "role": "user",
                "content": """Here are the facts:
            \n ------- \n
            {documents} 
            \n ------- \n
            Here is the answer: {generation}"""
            }
        ]

        input_txt = self.tokenizer.apply_chat_template(
            answer_grader_chat_messages,
            add_generation_prompt=True,
            tokenize=False)

        # Langachain prompt template, partially filled with the format instructions above
        prompt = PromptTemplate(
            template=input_txt,
            input_variables=["generation", "documents"],
        )

        return prompt

    def _create_question_router_prompt(self):
        """
        # Setup a prompt to route the answer. Returns a JSON with the datasource to use for the question
        """
        ### Router prompt
        chat_messages = [
            {"role": "system",
             "content": """You are an expert at routing a 
            user question to a vectorstore or web search. Use the vectorstore for questions on {expert_domains_text}. You do not need to be stringent with the keywords 
            in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
            or 'vectorstore' based on the question. Return a JSON with a single key 'datasource' and 
            no premable or explanation. Question to route: {question}"""
             },
        ]

        input_txt = self.tokenizer.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=False)

        prompt = PromptTemplate(
            template=input_txt,
            input_variables=["question", "expert_domains_text"],
        )

        return prompt

