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
import langchain
from utils import pretty_log

pp = pprint.PrettyPrinter(indent=2)

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, TextStreamer, TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer
import json
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from enum import Enum
from typing import Union, List, Dict

import time

class QuestionAndAnsweringCustomLlama2():

    class SECURITY_PROMPT_TYPE(Enum):
        PROMPT_TYPE_DEFAULT=0,
        PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES=1,
        PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_WITHSOURCES=2,

    class QUESTION_REWRITING_TYPE(Enum):
        QUESTION_REWRITING_DEFAULT=0,

    class LLAMA2_VERSION_TYPE(Enum):
        LLAMA2_7B_chat=0,
        LLAMA2_13B_chat=1,
        LLAMA2_70B_chat=2,

    def __init__(self, QuestionRewritingPrompt: QUESTION_REWRITING_TYPE,
                 QuestionAnsweringPrompt: SECURITY_PROMPT_TYPE,
                 ModelType: LLAMA2_VERSION_TYPE,
                 debug: bool,
                 streamingOnAnotherThread: bool):

        self.tokenizer = None
        self.model = None
        self.embedding_engine = None
        self.base_llm = None
        self.llm = None
        self.llm_conversational_chain = None # The full conversational chain
        self.llama_doc_chain = None # The question answering on a given context (rag) chain
        self.llama_question_generator_chain = None # The history+newquestion => standalone question generation chain
        self.vector_index = None # The index vector store for RAG
        self.streamingOnAnotherThread = streamingOnAnotherThread

        self.debug = debug

        # See the initializePromptTemplates function and enum above
        self.templateprompt_for_question_answering: str = ""
        self.templateprompt_for_standalone_question_generation: str = ""

        self.QuestionRewritingPromptType = QuestionRewritingPrompt
        self.QuestionAnsweringPromptType = QuestionAnsweringPrompt

        self.modelName = None
        if ModelType is self.LLAMA2_VERSION_TYPE.LLAMA2_7B_chat:
            self.modelName = "meta-llama/Llama-2-7b-chat-hf"
        elif ModelType is self.LLAMA2_VERSION_TYPE.LLAMA2_70B_chat:
            self.modelName = "meta-llama/Llama-2-70b-chat-hf"
        if ModelType is self.LLAMA2_VERSION_TYPE.LLAMA2_13B_chat:
            self.modelName = "meta-llama/Llama-2-13b-chat-hf"


        self.initilizeEverything()

    def initilizeEverything(self):
        self.initializePromptTemplates()
        self.initializeLLMTokenizerandEmbedder()
        self.initializeQuestionAndAnswering_withRAG_andMemory()

        langchain.debug = self.debug

    def initializePromptTemplates(self):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

        DEFAULT_SYSTEM_PROMPT = """\
                You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
                If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
                """

        DEFAULT_QUESTION_PROMPT = "Question: {question}"

        # Function to format a system prompt + user prompt (instruction) to LLama 2 format
        def get_prompt(instruction = DEFAULT_QUESTION_PROMPT , new_system_prompt=DEFAULT_SYSTEM_PROMPT):
            SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
            prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
            return prompt_template



        template_securityOfficer_system_prompt = """\
                ""Consider that I'm a beginner in networking and security things. \n
                Give me a concise answer with with a single step at a time. \n
                Limit your response to maximum 2000 words.
                Do not provide any additional text or presentation. Only steps and actions.
                If possible use concrete names of software or tools that could help on each step."""

        template_securityOfficer_instruction_rag_nosources = """\
                Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
                {context}
                Question: {question}"""


        template_securityOfficer_instruction_rag_withsources = """
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


        if self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama2.SECURITY_PROMPT_TYPE.PROMPT_TYPE_DEFAULT:
            self.templateprompt_for_question_answering = get_prompt(instruction=DEFAULT_QUESTION_PROMPT,
                                                                    new_system_prompt=DEFAULT_SYSTEM_PROMPT)
        elif self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama2.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES:
            self.templateprompt_for_question_answering = get_prompt(instruction=template_securityOfficer_instruction_rag_nosources,
                                                                    new_system_prompt=template_securityOfficer_system_prompt)
        elif self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama2.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_WITHSOURCES:
            self.templateprompt_for_question_answering = get_prompt(
                instruction=template_securityOfficer_instruction_rag_withsources,
                new_system_prompt=template_securityOfficer_system_prompt)
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"

        if self.QuestionRewritingPromptType == QuestionAndAnsweringCustomLlama2.QUESTION_REWRITING_TYPE.QUESTION_REWRITING_DEFAULT:
            self.templateprompt_for_standalone_question_generation = llama_condense_template
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"


    def initializeLLMTokenizerandEmbedder(self):
        # Get the embeddings, tokenizer and model
        self.embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName,
                                                  token=True)

        self.model = AutoModelForCausalLM.from_pretrained(self.modelName,
                                                     device_map='auto',
                                                     torch_dtype=torch.bfloat16,
                                                     token=True,
                                                     #  load_in_8bit=True,
                                                     #  load_in_4bit=True,

                                                     )

        # Create a streamer and a text generation pipeline
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True) if self.streamingOnAnotherThread is False \
            else TextIteratorStreamer(self.tokenizer, skip_prompt=True)

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
                        streamer=self.streamer,
                        )

        # Create the llm here
        self.llm = HuggingFacePipeline(pipeline=pipe)

        assert self.llm, "model was not initialized properly"

    def initializeQuestionAndAnswering_withRAG_andMemory(self):
        # Create the question generator chain which takes history + new question and transform to a new standalone question
        llama_condense_prompt = PromptTemplate(template=self.templateprompt_for_standalone_question_generation,
                                               input_variables=["chat_history", "question"])
        self.llama_question_generator_chain = LLMChain(llm=self.llm,
                                                  prompt=llama_condense_prompt,
                                                  verbose=False)


        # Create the response chain based on a question and context (i.e., rag in our case)
        llama_docs_prompt = PromptTemplate(template=self.templateprompt_for_question_answering, input_variables=["context", "question"])
        self.llama_doc_chain = load_qa_with_sources_chain(self.llm, chain_type="stuff", prompt=llama_docs_prompt,
                                                     document_variable_name="context", verbose=False)


        # Initialize the memory(i.e., the chat history)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        ########### Connecting to the vector storage and load it #############
        #pretty_log("connecting to vector storage")
        self.vector_index = vecstore.connect_to_vector_index(vecstore.INDEX_NAME, self.embedding_engine)
        #pretty_log("connected to vector storage")
        pretty_log(f"found {self.vector_index.index.ntotal} vectors to search over in the database")

        # Create the final retrieval chain which
        self.llm_conversational_chain = ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain,
            memory=memory
        )


    def ask_question(self, question: str) -> Union[BaseStreamer, None]:
        if self.streamingOnAnotherThread:
            from threading import Thread

            self.temp_modelevaluate_thread = Thread(target=self.llm_conversational_chain, args=({"question": question},))
            self.temp_modelevaluate_thread.start()

            return self.streamer
        else:
            return self.llm_conversational_chain({"question": question})

    def ask_question_and_streamtoconsole(self, question: str)->str:
        if self.streamingOnAnotherThread:
            streamer: BaseStreamer = self.ask_question(question)

            generated_text = ""
            for new_text in streamer:
                generated_text += new_text
                print(new_text, end='')
            # print(f"Full generated text {generated_text}")
            return generated_text
        else:
            return self.temp_modelevaluate_thread.join()


    # VERY USEFULLY FOR checking the sources and context
    ######################################################
    def simulate_raq_question(self, query: str, run_llm_chain: bool):
        pretty_log("selecting sources by similarity to query")
        sources_and_scores = self.vector_index.similarity_search_with_score(query, k=3)

        sources, scores = zip(*sources_and_scores)
        print(sources_and_scores)

        # Ask only the question on docs provided as context to see how it works without any additional context
        if run_llm_chain:
            result = self.llama_doc_chain(
                {"input_documents": sources, "question": query}, return_only_outputs=True
            )

            answer = result["output_text"]
            print(answer)

    def test_vectorDatasets_similarityScores_and_responses_no_memory(self, run_llm_chain: bool):

        query1 = "What models use human instructions?"
        self.simulate_raq_question(query1, run_llm_chain=False)

        query2 = "Are there any model trained on medical knowledge?"
        self.simulate_raq_question(query2, run_llm_chain=False)

    ######################################################



def __main__():

    securityChatbot = QuestionAndAnsweringCustomLlama2(QuestionRewritingPrompt=QuestionAndAnsweringCustomLlama2.QUESTION_REWRITING_TYPE.QUESTION_REWRITING_DEFAULT,
                                     QuestionAnsweringPrompt=QuestionAndAnsweringCustomLlama2.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES,
                                     ModelType=QuestionAndAnsweringCustomLlama2.LLAMA2_VERSION_TYPE.LLAMA2_7B_chat,
                                    debug=False, streamingOnAnotherThread=True)

    securityChatbot.test_vectorDatasets_similarityScores_and_responses_no_memory(run_llm_chain=False)

    securityChatbot.ask_question_and_streamtoconsole("What models use human instructions?")
    securityChatbot.ask_question_and_streamtoconsole("Which are the advantage of each of these models?")
    securityChatbot.ask_question_and_streamtoconsole("What are the downsides of your last model suggested above ?")

if __name__ == "__main__":
    __main__()
