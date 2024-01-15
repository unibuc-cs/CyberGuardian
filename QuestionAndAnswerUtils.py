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
from ast import literal_eval
pp = pprint.PrettyPrinter(indent=2)
import importlib

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
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from enum import Enum
from typing import Union, List, Dict
from DynabicPrompts import *

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

    class TOOL_TYPE(Enum):
        TOOL_NONE = 0,
        TOOL_RESOURCE_UTILIZATION=1,
        TOOL_DEVICES_LOGS_BY_IP=2,
        TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP=3,
        TOOL_MAP_OF_REQUESTS_COMPARE=4,


    LLAMA2_DEFAULT_END_LLM_RESPONSE ="</s>"

    def __init__(self, QuestionRewritingPrompt: QUESTION_REWRITING_TYPE,
                 QuestionAnsweringPrompt: SECURITY_PROMPT_TYPE,
                 ModelType: LLAMA2_VERSION_TYPE,
                 debug: bool,
                 streamingOnAnotherThread: bool,
                 demoMode:bool):

        self.tokenizer = None
        self.model = None
        self.embedding_engine = None
        self.base_llm = None
        self.llm = None
        self.llm_conversational_chain_default = None # The full conversational chain

        self.llama_doc_chain = None # The question answering on a given context (rag) chain
        self.llama_question_generator_chain = None # The history+newquestion => standalone question generation chain
        self.vector_index = None # The index vector store for RAG
        self.streamingOnAnotherThread = streamingOnAnotherThread

        # Func calls chain
        self.llm_conversational_chain_funccalls_resourceUtilization = None # Func calls conversation
        self.llama_doc_chain_funccalls_resourceUtilization = None #

        self.debug = debug
        self.demoMode = demoMode

        # See the initializePromptTemplates function and enum above
        self.templateprompt_for_question_answering_default: str = ""
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

        # TODO: implement more
        self.initializeQuestionAndAnswering_withRAG_andMemory()

        langchain.debug = self.debug

    def initializePromptTemplates(self):
        if self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama2.SECURITY_PROMPT_TYPE.PROMPT_TYPE_DEFAULT:
            self.templateprompt_for_question_answering_default = get_prompt(instruction=DEFAULT_QUESTION_PROMPT,
                                                                            new_system_prompt=DEFAULT_SYSTEM_PROMPT)
        elif self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama2.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES:
            self.templateprompt_for_question_answering_default = get_prompt(instruction=template_securityOfficer_instruction_rag_nosources_default,
                                                                            new_system_prompt=template_securityOfficer_system_prompt)
        elif self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama2.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_WITHSOURCES:
            self.templateprompt_for_question_answering_default = get_prompt(
                instruction=template_securityOfficer_instruction_rag_withsources_default,
                new_system_prompt=template_securityOfficer_system_prompt)
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"

        ######## FUNCTION CALLS and CUSTOM PROMPTS other than default
        self.templateprompt_for_question_answering_funccall_resourceUtilization = get_prompt(instruction=template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization,
                                                                                             new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)
        self.templateprompt_for_question_answering_funccall_devicesByIPLogs = get_prompt(instruction=template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs,
                                                                                             new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)


        self.templateprompt_for_question_answering_funccall_topDemandingIPS = get_prompt(instruction=template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS,
                                                                                         new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)

        self.templateprompt_for_question_answering_funccall_comparisonMapRequests = get_prompt(instruction=template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests,
                                                                                         new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)

        ################# CUSTOM PROMPTS END

        if self.QuestionRewritingPromptType == QuestionAndAnsweringCustomLlama2.QUESTION_REWRITING_TYPE.QUESTION_REWRITING_DEFAULT:
            self.templateprompt_for_standalone_question_generation = llama_condense_template
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"

    def solveFunctionCalls(self, full_response: str) -> bool:
        full_response = full_response.removesuffix(
            QuestionAndAnsweringCustomLlama2.LLAMA2_DEFAULT_END_LLM_RESPONSE)  # Removing the last </s> ending character specific to llama endint of a response
        full_response.strip()
        if (full_response[0] == '\"' and full_response[-1] == '\"') \
                or (full_response[0] == "\'" and full_response[-1] == "\'"):
            full_response = full_response[1:-1]

        if 'FUNC_CALL' not in full_response:
            return False

        # Identify which function call it is being asked
        # TODO: allow user to inject his own tools
        # TODO: make exception and fail method
        # Parse the parameters in function call
        words_in_func_call = list(full_response.split())
        words_in_func_call = [w.strip() for w in words_in_func_call]

        # Remove potential " in beginning and end
        if (words_in_func_call[0][0] == '"' and words_in_func_call[-1][-1] == '"') or \
                (words_in_func_call[0][0] == "'" and words_in_func_call[-1][-1] == "'"):
            words_in_func_call[0] = words_in_func_call[0][1:]
            words_in_func_call[-1] = words_in_func_call[-1][:-1]

        indexOfFunccall = words_in_func_call.index("FUNC_CALL")
        words_in_func_call = words_in_func_call[indexOfFunccall:]

        assert words_in_func_call[0] == 'FUNC_CALL', "First argument should be FUNC_CALL token"
        assert words_in_func_call[2] == 'Params', "Third argument needs to be Params token"
        # assert "</s>" in words_in_func_call[-1]
        # words_in_func_call[-1] = words_in_func_call[-1].replace("</s>", "")

        # Remove double quotes stuff
        words_in_func_call = [w if w[0] not in ["'", '"'] else w[1:len(w) - 1] for w in words_in_func_call]

        # Second expected as module.func
        mod_name, func_name = words_in_func_call[1].rsplit('.', 1)
        func_params = words_in_func_call[3:]

        if func_params[0][0] =='[':
            assert func_params[-1][-1]==']', "Unclosed parameters list"
            func_params[0] = func_params[0][1:]
            func_params[-1] = func_params[-1][:-1]

            for idx, paramStr in enumerate(func_params):
                if paramStr[-1] == ',':
                    func_params[idx] = func_params[idx][:-1]
                if paramStr[0] == ',':
                    func_params[idx] = func_params[idx[1:]]

        # import the module where function is
        try:
            print(f"Trying to call function {mod_name}.{func_name}, params: {func_params}")
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            return

        # Get the function
        func = getattr(mod, func_name)

        # Call it
        result = func(*func_params)

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
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True) if self.streamingOnAnotherThread is False else TextIteratorStreamer(self.tokenizer, skip_prompt=True)

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
        llama_docs_prompt_default = PromptTemplate(template=self.templateprompt_for_question_answering_default, input_variables=["context", "question"])
        self.llama_doc_chain = load_qa_with_sources_chain(self.llm, chain_type="stuff", prompt=llama_docs_prompt_default,
                                                     document_variable_name="context", verbose=False)


        ##################### FUNCTION DOC_CHAIN STUFF ####################
        llama_docs_prompt_funccall_resourceUtilization = PromptTemplate(template=self.templateprompt_for_question_answering_funccall_resourceUtilization, input_variables=["context", "question"])
        self.llama_doc_chain_funccalls_resourceUtilization = load_qa_with_sources_chain(self.llm, chain_type="stuff",
                                                                                        prompt=llama_docs_prompt_funccall_resourceUtilization,
                                                                                        document_variable_name="context", verbose=False)

        llama_docs_prompt_funccall_devicesByIPLogs = PromptTemplate(template=self.templateprompt_for_question_answering_funccall_devicesByIPLogs, input_variables=["context", "question"])
        self.llama_doc_chain_funccalls_devicesByIPLogs = load_qa_with_sources_chain(self.llm, chain_type="stuff",
                                                                                        prompt=llama_docs_prompt_funccall_devicesByIPLogs,
                                                                                        document_variable_name="context", verbose=False)

        llama_docs_prompt_funccall_topDemandingIPS = PromptTemplate(template=self.templateprompt_for_question_answering_funccall_topDemandingIPS, input_variables=["context", "question", "param1", "param2"])
        self.llama_doc_chain_funccalls_topDemandingIPS = load_qa_with_sources_chain(self.llm, chain_type="stuff",
                                                                                        prompt=llama_docs_prompt_funccall_topDemandingIPS,
                                                                                        document_variable_name="context", verbose=False)

        llama_docs_prompt_funccall_comparisonMapRequests = PromptTemplate(template=self.templateprompt_for_question_answering_funccall_comparisonMapRequests, input_variables=["context", "question"])
        self.llama_doc_chain_funccalls_comparisonMapRequests = load_qa_with_sources_chain(self.llm, chain_type="stuff",
                                                                                        prompt=llama_docs_prompt_funccall_comparisonMapRequests,
                                                                                        document_variable_name="context", verbose=False)
        #################### END FUNCTION DOC_CHAIN STUFF ###################


        # Initialize the memory(i.e., the chat history)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        ########### Connecting to the vector storage and load it #############
        #pretty_log("connecting to vector storage")
        self.vector_index = vecstore.connect_to_vector_index(vecstore.INDEX_NAME, self.embedding_engine)
        #pretty_log("connected to vector storage")
        pretty_log(f"found {self.vector_index.index.ntotal} vectors to search over in the database")

        # Create the final retrieval chain which
        self.llm_conversational_chain_default = ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain,
            return_generated_question=False,
            memory=self.memory, verbose=False)


        ##################### FUNCTION CONV CHAIN STUFF ####################
        self.llm_conversational_chain_funccalls_resourceUtilization = ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain_funccalls_resourceUtilization,
            return_generated_question=False,
            memory=self.memory
        )


        self.llm_conversational_chain_funccalls_devicesByIPLogs = ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain_funccalls_devicesByIPLogs,
            return_generated_question=False,
            memory=self.memory
        )

        """
        self.llm_conversational_chain_funccalls_topDemandingIPS = ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain_funccalls_topDemandingIPS,
            return_generated_question=False,
            memory=self.memory
        )
        """

        self.llm_conversational_chain_funccalls_comparisonMapRequests = ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain_funccalls_comparisonMapRequests,
            return_generated_question=False,
            memory=self.memory
        )
        #################### END FUNCTION DOC_CHAIN STUFF ###################

    # Computes the similarity of request to one of the tools
    # Extract parameters through classic NLP techniques at the moment...well, for the demo just basic stuff..
    # TODO: The similarity operation is currently very simple token based, need to FIX it after demo
    def similarityToTool(self, question: str):
        # Split in lowercase tokens, remove punctuation
        question_lower_words = question.lower().split()
        for index, word in enumerate(question_lower_words):
            if word[-1] in [',','!','?','.']:
                question_lower_words[index] = question_lower_words[index][:-1]
        question_small_words = set(question_lower_words)

        params = []
        toolType: QuestionAndAnsweringCustomLlama2.TOOL_TYPE = QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_NONE

        if set(["resource", "utilization"]).issubset(question_small_words):
            toolType = QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_RESOURCE_UTILIZATION
        elif set("devices grouped by ip".split()).issubset(question_small_words):
            toolType = QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_DEVICES_LOGS_BY_IP
        elif set("requests from the top ips".split()).issubset(question_small_words):
            params = [int(x) for x in question_lower_words if x.isdigit()]
            toolType = QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP
        elif set("world map requests comparing".split()).issubset(question_small_words):
            toolType = QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_MAP_OF_REQUESTS_COMPARE

        return toolType, params

    def getConversationChainByQuestion(self, question: str):
        conv_chain_res = self.llm_conversational_chain_default
        params_res = []
        # This should use classic NLP techniques to compare similarity between question and a pair of functionalities,
        # Such that we use some smaller and more focused prompts and chains for cases.
        # This is mainly needed since we use a small model as foundation, a 7B, which can't hold too much context
        # and adapt to ALL use cases, functions etc. So this is like a task decomposition technique used in SE

        toolTypeSimilarity, params = self.similarityToTool(question)
        if toolTypeSimilarity == QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_RESOURCE_UTILIZATION:
            conv_chain_res = self.llama_doc_chain_funccalls_resourceUtilization #self.llm_conversational_chain_funccalls_resourceUtilization
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_DEVICES_LOGS_BY_IP:
            conv_chain_res = self.llama_doc_chain_funccalls_devicesByIPLogs  #self.llm_conversational_chain_funccalls_devicesByIPLogs
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP:
            conv_chain_res = self.llama_doc_chain_funccalls_topDemandingIPS #self.llm_conversational_chain_funccalls_topDemandingIPS
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama2.TOOL_TYPE.TOOL_MAP_OF_REQUESTS_COMPARE:
            conv_chain_res = self.llama_doc_chain_funccalls_comparisonMapRequests #self.llm_conversational_chain_funccalls_comparisonMapRequests

        refactored_question = question # For now, let it as original
        params_res = params
        return conv_chain_res, refactored_question, params

    # Check if the (shared) memory of all chains has messages in it
    def hasHistoryMessages(self) -> bool:
        return len(self.memory.chat_memory.messages) > 0

    def ask_question(self, question_original: str) -> Union[BaseStreamer, None]:
        chainToUse, question, params = self.getConversationChainByQuestion(question_original)

        isfullConversationalType = True

        if self.streamingOnAnotherThread:
            from threading import Thread

            if isinstance(chainToUse, ConversationalRetrievalChain):
                self.temp_modelevaluate_thread = Thread(target=chainToUse, args=({"question": question}))
            elif isinstance(chainToUse, StuffDocumentsChain):
                self.temp_modelevaluate_thread = Thread(target=chainToUse, args=({"input_documents":[],
                                                                                              "question":question,
                                                                                              "params": params},))
                isfullConversationalType = False
            self.temp_modelevaluate_thread.start()

            return self.streamer, isfullConversationalType
        else:
            return chainToUse({"question": question}) #, "params": params},)

    def ask_question_and_streamtoconsole(self, question: str)->str:
        if self.streamingOnAnotherThread:
            # This is needed since when it has some memory and prev chat history it will FIRST output the standalone question
            # Then respond to the new question
            need_to_ignore_standalone_question_chain = self.hasHistoryMessages()

            streamer, isfullConversationalType  = self.ask_question(question)
            if not isfullConversationalType:
                need_to_ignore_standalone_question_chain = False

            generated_text = ""



            if need_to_ignore_standalone_question_chain:
                for new_text in streamer:
                    generated_text += new_text
                    # print(new_text, end='')
                print("\n")

            for new_text in streamer:
                generated_text += new_text
                print(new_text, end='')
            print("\n")

            # print(f"Full generated text {generated_text}")

            self.temp_modelevaluate_thread.join()
            return generated_text
        else:
            self.ask_question(question)


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
                                    debug=False, streamingOnAnotherThread=True, demoMode=False)

    #securityChatbot.test_vectorDatasets_similarityScores_and_responses_no_memory(run_llm_chain=False)


    securityChatbot.ask_question_and_streamtoconsole("Ok. I'm on it, can you show me a resource utilization graph comparison between a normal session and current situation")

    #print("#############################################\n"*3)

    securityChatbot.ask_question_and_streamtoconsole("Show me the logs of the devices grouped by IP which have more than 25% requests over the median of a normal session per. Sort them by count")

    #print("#############################################\n"*3)

    fullGenText = securityChatbot.ask_question_and_streamtoconsole("Can you show a sample of GET requests from the top 3 demanding IPs, including their start time, end time? Only show the last 10 logs.")

    fullGenText =  securityChatbot.ask_question_and_streamtoconsole("Give me a world map of requests by comparing the current data and a known snapshot with bars")
    securityChatbot.solveFunctionCalls(fullGenText)


    #query = "Show me the logs of the devices grouped by IP which have more than 25% requests over the median of a normal session per. Sort them by count"
    #securityChatbot.llama_doc_chain_funccalls_devicesByIPLogs({"input_documents": [], "question": query}, return_only_outputs=True)

    #securityChatbot.ask_question_and_streamtoconsole("What is a DDoS attack?")

    #securityChatbot.ask_question_and_streamtoconsole("Ok. I'm on it, can you show me a resource utilization graph comparison between a normal session and current situation")
    #securityChatbot.ask_question_and_streamtoconsole(
    #    "Ok. I'm on it, can you show me a resource utilization graph comparison between a normal session and current situation")

    #securityChatbot.ask_question_and_streamtoconsole("Which are the advantage of each of these models?")
    #securityChatbot.ask_question_and_streamtoconsole("What are the downsides of your last model suggested above ?")

if __name__ == "__main__":
    __main__()
