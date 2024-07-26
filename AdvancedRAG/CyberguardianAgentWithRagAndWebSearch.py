# This implements the Demo picture in this folder where the agent uses first RAG, and if the information
# is irrelevant for the question or the answer is not good enough, or it hallucinates, it uses the web search tool.

# Load a Llama-3-8B instruct
import transformers
import sys
from torch import cuda, bfloat16
import torch
import transformers
from transformers import AutoTokenizer
from time import time
#import chromadb
#from chromadb.config import Settings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import torch
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Union, List, Dict, Any, NoReturn
from advancedRagUtils import convert_score_to_float, format_docs_to_str
from AdaptiveRAGSupport import AdaptiveRAGSupport, logger

from typing import TypedDict
from langchain_core.documents import Document
from typing import List
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph

import projsecrets
from langchain_community.tools.tavily_search import TavilySearchResults

import os

from pprint import pprint


class CyberguardianAgentWithRagAndWebSearch:
    model = None
    tokenizer = None
    retriever = None
    support = None
    MAX_WEB_SEARCH_RESULTS = 3
    web_search_tool = TavilySearchResults(max_results=MAX_WEB_SEARCH_RESULTS)
    workflow = None
    graph_agent_app: CompiledGraph = None

    # This is the state representation of the agent
    class AgentMemory(TypedDict):
        """
        Agent Memory to store the state of the agent and the conversation history
        """

        question: str  # The last question asked by user
        generation: str  # The last generated prompt from assistant
        web_search: bool  # Should do web search ?
        documents: List[Document]  # The list of documents extracted

    def __init__(self, model, tokenizer, retriever):
        if model is None or tokenizer is None or retriever is None:
            logger.error("Model, tokenizer and retriever should not be None, "
                         "we'll create a new instance of them as demo")
        self.support = AdaptiveRAGSupport(model, tokenizer, retriever)
        self.build_graph()


    # This is the main function that will be called by the agent to create the graph
    # This corresponds to the png image in this folder
    def build_graph(self):

        self.workflow = StateGraph(CyberguardianAgentWithRagAndWebSearch.AgentMemory)
        self.workflow.add_node("websearch", self.web_search)  # web search node
        self.workflow.add_node("retrieve", self.retrieve)  # retrieve node
        self.workflow.add_node("generate", self.generate)  # generate node
        self.workflow.add_node("grade_documents", self.grade_documents)  # grade documents node

        # Check the result
        self.workflow.set_conditional_entry_point(
            self.route_question,
            {
                "websearch": "websearch",
                "vectorstore": "retrieve",
            },
        )  # Dict from result of the function to next node

        self.workflow.add_edge("retrieve", "grade_documents")
        self.workflow.add_conditional_edges(
            "grade_documents",
            self.decide_if_websearch_is_needed,
            {
                "websearch": "websearch",
                "generate": "generate",
            }
        )

        self.workflow.add_edge("websearch", "generate")
        self.workflow.add_conditional_edges(
            "generate",
            self.grade_generation_vs_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "websearch",
            },
        )

        self.graph_agent_app = self.workflow.compile()

    def retrieve(self, state: AgentMemory):
        """
        Retrieve documents from vectorstore
        Args:
            state: AgentMemory
        Returns:
            state (dict): New keys to be added to the state/update the state, containing the documents
        """
        question = state["question"]
        logger.warning(f"Retrieving documents for question: {question}")

        documents = self.support.get_docs_content_by_query(question, self.support.DEFAULT_MAX_DOCS_RETURNED,
                                                           format_as_single_text=False, verbose=False)
        assert type(documents) is List[Document], "Documents should be a list of Document objects"

        return {"documents": documents, "question": question}

    def generate(self, state: AgentMemory):
        """
        Generate an answer using RAG model
        Args:
            state: AgentMemory
        Returns:
            state (dict): New keys to be added to the state/update the state, containing the generation
        """
        question = state["question"]
        documents = state["documents"]
        logger.warning(f"Generating answer for question: {question}")

        answer = self.support.answer_with_rag(question, documents, skip_prompt=True)
        return {"documents": documents, "question": question, "generation": answer}

    def grade_documents(self, state: AgentMemory):
        """
        Checks if the response is sustained by the documents. If any doc is not relevant, we set a flag to run a web search
        Args:
            state: AgentMemory
        Returns:
            state (dict): Filtered out irrelevant documents and updates the web_search flag
        """
        question = state["question"]
        documents = state["documents"]
        assert type(documents) is List[Document], "Documents should be a list of Document objects"
        generation = state["generation"]
        logger.warning(f"Grading documents for question: {question}")

        web_search = False

        # Check if the answer is sustained by the documents
        # Score each individually
        filtered_docs = []
        for doc in documents:
            score = self.support.hallucination_grader(generation, doc.page_content, skip_prompt=True)

            grade = convert_score_to_float(score)

            # Document is relevant if the grade is above 0.5
            if grade > 0.5:
                filtered_docs.append(doc)
            # If not, we set a flag to run a web search
            else:
                web_search = True
                continue  # IS THIS NEEDED ?????

        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def web_search(self, state: AgentMemory):
        """
        Run a web search based on the question

        Args:
            state: AgentMemory
        Returns:
            state (dict): Added the web search results to the state

        """
        # We run a web search
        question = state["question"]
        documents = state["documents"]
        assert type(documents) is List[Document], "Documents should be a list of Document objects"
        # Web search
        logger.warning(f"Running web search for question: {question}")
        web_search_docs = self.web_search_tool.invoke({"query": question, 'k': self.support.NUM_WEB_SEARCH_RESULTS})
        web_results = "\n".join(doc["content"] for doc in web_search_docs)
        web_results = Document(page_content=web_results)
        if documents is not None:
            documents.append(web_results)
        else:
            documents = [web_results]

        return {"documents": documents, "question": question}

    # Conditional edges
    def route_question(self, state: AgentMemory) -> str:
        """
        Route the question to the right category, RAG or websearch
        Args:
            state: AgentMemory
        Returns:
            str: name of the next node to be executed
        """
        question = state["question"]

        logger.warning(f"Routing question: {question}")

        source = self.support.question_router(question, skip_prompt=True)

        logger.warning(f"Question category: {source['datasource']}")
        if source["datasource"] == "websearch":
            return "websearch"
        else:
            return "vectorstore"

    def decide_if_websearch_is_needed(self, state: AgentMemory):
        """
        Check if we need to run the web search after a collection of documents were retrieved based on the state of the agent
        Args:
            state: AgentMemory
        Returns:
            str: name of the next node to be executed
        """
        web_search = state["web_search"]
        logger.warning(f"Assessing the status of web search: {web_search}")
        assert type(web_search) == bool, "web_search should be a boolean"
        if web_search:
            logger.warning(f"---DECISION: Not all documents are relevant, running web search")
            return "websearch"
        else:
            logger.warning(f"---DECISION: ALL documents are relevant, go to generation")
            return "vectorstore"

    def grade_generation_vs_documents_and_question(self, state: AgentMemory):
        """
        Determines if the generation is actually grounded in the documents retrieved from the vectorstore according and answering the question
        Args:
            state: AgentMemory
        Returns:
            str: Decision for next node to call.
        """
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        logger.warning(
            f"Hallucination check and then if the answer is useful: Grading generation vs documents and question")

        # Check if answer is sustained by the documents, hallucination check
        score_hallucination = self.support.hallucination_grader(question, documents, skip_prompt=True)
        grade_hallucination = convert_score_to_float(score_hallucination)
        if grade_hallucination > 0.5:
            # Answer is sustained by the documents, now check if the answer is useful
            logger.warning(f"---DECISION: Generation is grounded in the documents")

            score_useful = self.support.answer_grader(question, generation, skip_prompt=True)
            grade_useful = convert_score_to_float(score_useful)
            if grade_useful > 0.5:
                logger.warning(f"---DECISION: Generation is sustained by the documents and the answer is useful")
                return "useful"
            else:
                logger.warning(f"---DECISION: Generation does not address question or is not useful")
                return "not useful"
        else:
            logger.warning(f"---DECISION: Generation is NOT grounded in the documents, RETRYING")
            return "not supported"

    # This is the main function that will be called by the agent to run the agent with RAG + Web search tool if needed
    def run_agent(self, question: str):
        """
        Run the agent with a question
        Args:
            question: str
        Returns:
            str: The answer to the question
        """
        # Initialize the agent memory and question to be asked
        inputs = {"question": "What are the types of agent memory?"}
        agent_memory = CyberguardianAgentWithRagAndWebSearch.AgentMemory(question=question,
                                                                         generation="",
                                                                         web_search=False,
                                                                         documents=[])

        for output in self.graph_agent_app.stream(inputs):
            for key, value in output.items():
                logger.info(f"Finished Running: {key}:")

        # Return the answer
        return self.graph_agent_app.get_state()["generation"]
