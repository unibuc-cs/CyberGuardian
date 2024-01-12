"""Drops a collection from the document storage."""
import os

import json
from pathlib import Path
import pprint
import pdb
from typing import Any

from etl import markdown, pdfs, shared, videos

import docstore
import vecstore
from utils import pretty_log

pp = pprint.PrettyPrinter(indent=2)

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, TextStreamer
import json
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import time

class QuestionAndAnsweringCustomLlama2():
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embedding_engine = None
        self.base_llm = None
        self.llm_chain = None

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    # Function to format a systemn prompt + user prompt (instruction) to LLama 2 format
    @staticmethod
    def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
        SYSTEM_PROMPT = QuestionAndAnsweringCustomLlama2.B_SYS + new_system_prompt + QuestionAndAnsweringCustomLlama2.E_SYS
        prompt_template = QuestionAndAnsweringCustomLlama2.B_INST + SYSTEM_PROMPT + instruction + QuestionAndAnsweringCustomLlama2.E_INST
        return prompt_template


    def initializeLLMTokenizerandEmbedder(self):
        # Get the embeddings, tokenizer and model
        self.embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                  token=True)

        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                     device_map='auto',
                                                     torch_dtype=torch.bfloat16,
                                                     token=True,
                                                     #  load_in_8bit=True,
                                                     #  load_in_4bit=True,

                                                     )

        # Create a streamer and a text generation pipeline
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)

        pipe = pipeline("text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        max_new_tokens=4096,
                        do_sample=True,
                        temperature=0.1,
                        top_p=0.95,
                        num_return_sequences=1,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.eos_token_id,
                        streamer=streamer,
                        )

        # Create the llm here
        self.llm = HuggingFacePipeline(pipeline=pipe)

        assert self.llm, "model was not initialized properly"

    def initializeQuestionAndAnswering_withRAG_andMemory(self):
        # THE OTHER ONE
        securityOfficer_instruction_rag_nosources = """\
        Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
        {context}
        Question: {question}"""

        securityOfficer_system_prompt = """\
        ""Consider that I'm a beginner in networking and security things. \n
        Give me a concise answer with with a single step at a time. \n
        Limit your response to maximum 2000 words.
        Do not provide any additional text or presentation. Only steps and actions.
        If possible use concrete names of software or tools that could help on each step."""


        securityOfficer_instruction_rag_withsources = """
        Given the following extracted parts of a long document and a question, create a final answer with "SOURCES" that represent exactly the Source name and link given.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.

        QUESTION: {question}

        {summaries}

        FINAL ANSWER:
        """

        llama_condense_template = """
                [INST]Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
                Chat History:
                {chat_history}
                Follow Up Input: {question}
                Standalone question: [/INST]"""


        # Create the question generator chain which takes history + new question and transform to a new standalone question
        llama_condense_prompt = PromptTemplate(template=llama_condense_template,
                                               input_variables=["chat_history", "question"])
        llama_question_generator_chain = LLMChain(llm=llm, prompt=llama_condense_prompt, verbose=False)


        # Create the response chain based on a question and context (i.e., rag in our case)
        llama_docs_prompt = PromptTemplate(template=llama_docs_template, input_variables=["context", "question"])
        llama_doc_chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=llama_docs_prompt,
                                                     document_variable_name="context", verbose=False)


        # Initialize the memory(i.e., the chat history)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        ########### Connecting to the vector storage and load it #############
        #pretty_log("connecting to vector storage")
        vector_index = vecstore.connect_to_vector_index(vecstore.INDEX_NAME, embedding_engine)
        #pretty_log("connected to vector storage")
        pretty_log(f"found {vector_index.index.ntotal} vectors to search over in the database")

        # Create the final retrieval chain which
        llama_v2_chain = ConversationalRetrievalChain(
            retriever=vector_index.as_retriever(search_kwargs={'k': 6}),
            question_generator=llama_question_generator_chain,
            combine_docs_chain=llama_doc_chain,
            memory=memory
        )

def __main__():
    pass

if __name__ == "__main__":
    __main__()
