"""Drops a collection from the document storage."""
import os

import pprint
from typing import Any, Tuple

from Data.etlUtils import vecstore
from langchain_community.vectorstores import FAISS

import langchain
from Data.utils import pretty_log
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from userUtils import SecurityOfficer

pp = pprint.PrettyPrinter(indent=2)

import importlib
import pathlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer, \
    TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer
from langchain.chains import LLMChain
from langchain.memory import \
    ConversationBufferMemory  # TODO: replace with ChatMessageHistory ->> Chgeck the Prompt Engineering with Llama 2 notebook !
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_core.documents.base import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun

import CyberguardianPrompts as cybprompts

from enum import Enum
from typing import Union, List, Any, Set, Tuple, Dict
from LLM.CyberGuardianLLM import CyberGuardianLLM
from LLM.CyberGuardinaLLM_args import parse_args
import random
import numpy as np
import re
import sys

from CyberGuardianLLM import logger

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)

## THIS IS TEMPORARY FOR THE DEMO. TODO: remove it before submit and refactor!
import projsecrets
from projsecrets import project_path, HOOK_FUNC_NAME_TOKEN, TOKEN_CODE_EXEC_CONFIRM


# Note that this is specific to llama3 only because uses at some point some of its special tokens and versions
# Similar versions are doable for other LLMs as well.
class QASystem():
    class SECURITY_PROMPT_TYPE(Enum):
        PROMPT_TYPE_DEFAULT = 0,
        PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES = 1,
        PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_WITHSOURCES = 2,

    class QUESTION_REWRITING_TYPE(Enum):
        QUESTION_REWRITING_DEFAULT = 0,

    class LLAMA3_VERSION_TYPE(Enum):
        LLAMA3_8B = 0,
        LLAMA3_70B = 2,

    class TOOL_TYPE(Enum):
        TOOL_NONE = 0,
        TOOL_RESOURCE_UTILIZATION = 1,
        TOOL_DEVICES_LOGS_BY_IP = 2,
        TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP = 3,
        TOOL_MAP_OF_REQUESTS_COMPARE = 4,
        GENERIC_QUESTION_NO_HISTORY = 5,
        PANDAS_FIREWALL_UPDATE = 6,

    class CustomVectorStoreRetriever(VectorStoreRetriever):
        original_retriever: VectorStoreRetriever

        def __init__(self, **data: Any):
            #self.vectorstore = vectorstore
            super().__init__(**data)

        def _get_relevant_documents(
                self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> List[Document]:
            docs = self.original_retriever._get_relevant_documents(query, run_manager=run_manager)

            # Custom doc: the history !
            history_doc = Document(
                page_content="The top demanding IPs are: 18.146.47.131, 130.112.80.168, 183.233.154.202, 117.119.80.169",
                metadata={"source": "chat_history"})

            docs.insert(0, history_doc)

            return docs

        async def _aget_relevant_documents(self, query: str) -> List[Document]:
            return await self.original_retriever._aget_relevant_documents(query)

    LLAMA3_DEFAULT_END_LLM_RESPONSE = "<|eot_id|>"
    cb_llm = None  # The cyberguardian llm instance

    def __init__(self, QuestionRewritingPrompt: QUESTION_REWRITING_TYPE,
                 QuestionAnsweringPrompt: SECURITY_PROMPT_TYPE,
                 ModelType: LLAMA3_VERSION_TYPE,
                 debug: bool,
                 streamingOnAnotherThread: bool,
                 demoMode: bool,
                 noInitialize=False,
                 userProfile=SecurityOfficer()):

        self.llm_conversational_chain_func_calls_comparisonMapRequests = None
        self.llm_conversational_chain_func_calls_devicesByIPLogs = None
        self.llama_condense_prompt = None
        self.llama_doc_chain_func_calls_devicesByIPLogs = None
        self.llama_doc_chain_funccalls_topDemandingIPS = None
        self.llama_doc_chain_func_calls_firewall_update = None
        self.llama_doc_chain_func_calls_comparisonMapRequests = None
        self.generation_params = None
        self.formatted_prompt_for_question_answering_func_call_comparisonMapRequests = None
        self.formatted_prompt_for_question_answering_func_call_firewallUpdate = None
        self.formatted_prompt_for_question_answering_func_call_topDemandingIPS = None
        self.formatted_prompt_for_question_answering_func_call_devicesByIPLogs = None
        self.formatted_prompt_for_question_answering_func_call_resourceUtilization = None
        self.streamer = None
        self.userProfile: SecurityOfficer = userProfile
        self.tokenizer = None
        self.model = None
        self.embedding_engine = None
        self.default_llm = None
        self.llm_conversational_chain_default = None  # The full conversational chain
        self.textGenerationPipeline = None
        self.chat_history_tuples = []  # Empty history

        self.llama_doc_chain = None  # The question answering on a given context (rag) chain
        self.llama_question_generator_chain = None  # The history+newquestion => standalone question generation chain
        self.vectorstore = None  # The index vector store for RAG
        self.streamingOnAnotherThread = streamingOnAnotherThread

        # Func calls chain
        self.llm_conversational_chain_func_calls_resourceUtilization = None  # Func calls conversation
        self.llama_doc_chain_func_calls_resourceUtilization = None  #

        self.debug = debug
        self.demoMode = demoMode

        # See the initializePromptTemplates function and enum above
        self.formatted_prompt_for_question_answering_default: str = ""
        self.formatted_prompt_for_standalone_question_generation: str = ""

        self.QuestionRewritingPromptType = QuestionRewritingPrompt
        self.QuestionAnsweringPromptType = QuestionAnsweringPrompt

        self.modelName = None
        if ModelType is self.LLAMA3_VERSION_TYPE.LLAMA3_8B:
            self.modelName = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif ModelType is self.LLAMA3_VERSION_TYPE.LLAMA3_70B:
            self.modelName = "meta-llama/Meta-Llama-3-70B-Instruct"

        if noInitialize is False:
            self.initilizeEverything()

    # Function to format a system prompt + user prompt (instruction) in LLM used template
    def get_prompt(self, instruction=cybprompts.DEFAULT_QUESTION_PROMPT, new_system_prompt=None):
        if new_system_prompt is None:
            messages = [
                {"role": "user", "content": instruction},
            ]
        else:
            messages = [
                {"role": "system", "content": new_system_prompt},
                {"role": "user", "content": instruction},
            ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt

    # Function to format a system prompt + user prompt (instruction) in LLM used template
    def initilizeEverything(self):
        # Init the models
        self.initializeLLM()

        # All templates
        self.initialize_prompt_templates()

        # TODO: implement more
        self.initializeQuestionAndAnswering_withRAG_andMemory()

        langchain.debug = self.debug

    # Function to format a system prompt + user prompt (instruction) in LLM used template
    def initialize_prompt_templates(self):

        cybprompts.init_templates(self.userProfile)

        if self.QuestionAnsweringPromptType == QASystem.SECURITY_PROMPT_TYPE.PROMPT_TYPE_DEFAULT:
            self.formatted_prompt_for_question_answering_default = self.get_prompt(
                instruction=cybprompts.DEFAULT_QUESTION_PROMPT,
                new_system_prompt=None)
        elif self.QuestionAnsweringPromptType == QASystem.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES:
            self.formatted_prompt_for_question_answering_default = self.get_prompt(
                instruction=cybprompts.template_securityOfficer_instruction_rag_nosources_default,
                new_system_prompt=cybprompts.template_securityOfficer_system_prompt)
        elif self.QuestionAnsweringPromptType == QASystem.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_WITHSOURCES:
            self.formatted_prompt_for_question_answering_default = self.get_prompt(
                instruction=cybprompts.template_securityOfficer_instruction_rag_withsources_default,
                new_system_prompt=cybprompts.template_securityOfficer_system_prompt)
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"

        ######## FUNCTION CALLS and CUSTOM PROMPTS other than default
        self.formatted_prompt_for_question_answering_func_call_resourceUtilization = self.get_prompt(
            instruction=cybprompts.template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization,
            new_system_prompt=cybprompts.FUNC_CALL_SYSTEM_PROMPT)
        self.formatted_prompt_for_question_answering_func_call_devicesByIPLogs = self.get_prompt(
            instruction=cybprompts.template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs,
            new_system_prompt=cybprompts.FUNC_CALL_SYSTEM_PROMPT)

        self.formatted_prompt_for_question_answering_func_call_topDemandingIPS = self.get_prompt(
            instruction=cybprompts.template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS,
            new_system_prompt=cybprompts.FUNC_CALL_SYSTEM_PROMPT)

        self.formatted_prompt_for_question_answering_func_call_comparisonMapRequests = self.get_prompt(
            instruction=cybprompts.template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests,
            new_system_prompt=cybprompts.FUNC_CALL_SYSTEM_PROMPT)

        self.formatted_prompt_for_question_answering_func_call_firewallUpdate = self.get_prompt(
            instruction=cybprompts.template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert,
            new_system_prompt=cybprompts.FUNC_CALL_SYSTEM_PROMPT)

        ################# CUSTOM PROMPTS END

        if self.QuestionRewritingPromptType == QASystem.QUESTION_REWRITING_TYPE.QUESTION_REWRITING_DEFAULT:
            self.formatted_prompt_for_standalone_question_generation = self.get_prompt(
                instruction=cybprompts.llama_condense_template,
                new_system_prompt=None)
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"

    # Function that solves a function call and returns the answer and the source code to be executed/proposed by the
    # assistant
    # The full_response is the response from the assistant
    # The do_history_update is a flag to update the history with the answer
    # Returns: result_answer - what to show to the user
    #  result_source_code_tool - some source code suggested by the tool
    #  result_source_code_ui - source code for UI
    #  tool_code_needs_confirm - true if confirmation needed to execute tool from the user side
    def solveFunctionCalls(self, full_response: str, do_history_update: bool = False) -> Tuple[
        Union[str, None], Union[str, None], Union[str, None], Union[bool, None]]:
        # Removing the last </s> ending character specific to llama ending of a response
        full_response = full_response.removesuffix(
            QASystem.LLAMA3_DEFAULT_END_LLM_RESPONSE)
        full_response = full_response.strip()
        if (full_response[0] == '\"' and full_response[-1] == '\"') \
                or (full_response[0] == "\'" and full_response[-1] == "\'"):
            full_response = full_response[1:-1]

        if 'FUNC_CALL' not in full_response:
            return None, None, None, None

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

        index_of_funccall = words_in_func_call.index("FUNC_CALL")
        words_in_func_call = words_in_func_call[index_of_funccall:]

        assert words_in_func_call[0] == 'FUNC_CALL', "First argument should be FUNC_CALL tozken"
        assert words_in_func_call[2] == 'Params', "Third argument needs to be Params token"
        # assert "</s>" in words_in_func_call[-1]
        # words_in_func_call[-1] = words_in_func_call[-1].replace("</s>", "")

        # Remove double quotes stuff
        quota_looking_for = ["'", '"']
        words_in_func_call = [w[1:-1] if (w[0] in quota_looking_for and w[-1] in quota_looking_for) else w for w in
                              words_in_func_call]

        # Second expected as module.func
        mod_name, func_name = words_in_func_call[1].rsplit('.', 1)
        func_params = words_in_func_call[3:]

        if func_params[0][0] == '[':
            # Look for the end of the parameters list
            # enumerate in reverse order the func_params
            closing_bracket_idx = -1
            for idx, paramStr in enumerate(func_params[::-1]):
                if paramStr[-1] == ']':
                    # Do not break, continue until the closest to position of Params token
                    closing_bracket_idx = len(func_params) - idx - 1

            if closing_bracket_idx < 0:
                print(f"Closing bracket index is negative or not found in FUNC_CALL parameters list {func_params}")
            assert closing_bracket_idx >= 0, "Closing bracket index is negative or not found"

            # Eliminate the brackets
            func_params[0] = func_params[0][1:]
            func_params[closing_bracket_idx] = func_params[closing_bracket_idx][:-1]
            func_params = func_params[:closing_bracket_idx + 1]

            # Post-process the following things
            for idx, paramStr in enumerate(func_params):
                # Remove the last comma if it exists in each parameter
                if paramStr[-1] == ',':
                    paramStr = paramStr[:-1]
                if paramStr[0] == ',':
                    paramStr = paramStr[1:]

                # Remove the double quota
                if paramStr[0] == '"' and paramStr[-1] == '"':
                    paramStr = paramStr[1:-1]
                if paramStr[0] == "'" and paramStr[-1] == "'":
                    paramStr = paramStr[1:-1]

                func_params[idx] = paramStr

        # import the module where function is
        try:
            if self.debug:
                logger.info("Trying to call function {mod_name}.{func_name}, params: {func_params}")

            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            return None, None, None, None

        # Get the hook function call associated with the module and call it to solve the request
        hook_func = getattr(mod, HOOK_FUNC_NAME_TOKEN)

        # Call it
        result_answer, result_source_code_tool, result_source_code_ui, tool_code_needs_confirm = hook_func(func_name,
                                                                                                           func_params)

        if result_answer is not None and do_history_update:
            self.history_update(None, result_answer)

        return result_answer, result_source_code_tool, result_source_code_ui, tool_code_needs_confirm

    def initializeLLM(self) -> None:
        # Get the embeddings, tokenizer and model
        self.embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

        # Init the cyber guardian model
        llm_args = parse_args(with_json_args=pathlib.Path(os.environ["LLM_PARAMS_PATH_INFERENCE"]))
        self.cb_llm = CyberGuardianLLM(llm_args)
        self.generation_params = llm_args.generation_params
        self.cb_llm.do_inference()
        self.tokenizer = self.cb_llm.tokenizer
        self.model = self.cb_llm.model

        # Create a streamer and a text generation pipeline
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True) if self.streamingOnAnotherThread is False \
            else TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        self.textGenerationPipeline = pipeline("text-generation",
                                               model=self.model,
                                               tokenizer=self.tokenizer,
                                               eos_token_id=self.cb_llm.terminators,  #self.tokenizer.eos_token_id,
                                               pad_token_id=self.tokenizer.eos_token_id,
                                               streamer=self.streamer,
                                               **self.generation_params,
                                               )

        # Create the llm here
        self.default_llm = HuggingFacePipeline(pipeline=self.textGenerationPipeline)

        assert self.default_llm, "model was not initialized properly"

    # Checking safety using internal model. No hard numeric class or category yet but works quite well.
    # See the bottom function with "external" for methods that provide  that info probabilistically
    def check_response_safety(self, user_prompt: str, assistant_prompt: str) -> bool:
        assert user_prompt is not None and len(user_prompt) > 0
        assistant_prompt = f"Assistant: {assistant_prompt}" if (assistant_prompt is not None
                                                                   and len(assistant_prompt) > 0) else ""

        conversation = f"User: {user_prompt} \n{assistant_prompt}"
        responseCheckPipeline = pipeline("text-generation",
                                         model=self.model,
                                         tokenizer=self.tokenizer,
                                         torch_dtype=torch.float16,
                                         device_map="auto",
                                         max_new_tokens=512,
                                         do_sample=True,
                                         temperature=0.1,
                                         top_p=0.95,
                                         min_length=None,
                                         num_return_sequences=1,
                                         repetition_penalty=1.0,
                                         # The parameter for repetition penalty. 1.0 means no penalty.
                                         #length_penalty=1,
                                         # [optional] Exponential penalty to the length that is used with beam-based generation.
                                         eos_token_id=self.tokenizer.eos_token_id,
                                         pad_token_id=self.tokenizer.eos_token_id,
                                         streamer=self.streamer,
                                         )

        # Create the llm here
        responseCheckLLM = HuggingFacePipeline(pipeline=self.textGenerationPipeline)
        responseChecktemplate = self.get_prompt(f"Given the conversation below check if there is any discrimination "
                                           f"and response with True if it is OK or False if it isn't. Do not respond to the user question\n"
                                           f"{conversation} ", new_system_prompt=None)

        logger.info(responseChecktemplate)
        responseCheckPrompt = PromptTemplate(template=responseChecktemplate,
                                             input_variables=[])  # , input_variables=["text"])

        responseCheckChain = LLMChain(prompt=responseCheckPrompt, llm=responseCheckLLM)

        res = responseCheckChain.run({})
        logger.info(res)

    def initializeQuestionAndAnswering_withRAG_andMemory(self):
        # Create the question generator chain which takes history + new question and transform to a new standalone question
        self.llama_condense_prompt = PromptTemplate(template=self.formatted_prompt_for_standalone_question_generation,
                                                    input_variables=["chat_history", "question"])

        ## TODO : maybe deprecate since not used like this ?
        self.llama_question_generator_chain = LLMChain(llm=self.default_llm,
                                                       prompt=self.llama_condense_prompt,
                                                       verbose=self.debug)

        # Create the response chain based on a question and context (i.e., rag in our case)
        llama_docs_prompt_default = PromptTemplate(template=self.formatted_prompt_for_question_answering_default,
                                                   input_variables=["context", "question"])
        self.llama_doc_chain = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                          prompt=llama_docs_prompt_default,
                                                          document_variable_name="context", verbose=self.debug)

        ##################### FUNCTION DOC_CHAIN STUFF ####################
        llama_docs_prompt_func_call_resourceUtilization = PromptTemplate(
            template=self.formatted_prompt_for_question_answering_func_call_resourceUtilization,
            input_variables=["context", "question"])
        self.llama_doc_chain_func_calls_resourceUtilization = load_qa_with_sources_chain(self.default_llm,
                                                                                         chain_type="stuff",
                                                                                         prompt=llama_docs_prompt_func_call_resourceUtilization,
                                                                                         document_variable_name="context",
                                                                                         verbose=self.debug)

        llama_docs_prompt_func_call_devicesByIPLogs = PromptTemplate(
            template=self.formatted_prompt_for_question_answering_func_call_devicesByIPLogs,
            input_variables=["context", "question"])
        self.llama_doc_chain_func_calls_devicesByIPLogs = load_qa_with_sources_chain(self.default_llm,
                                                                                     chain_type="stuff",
                                                                                     prompt=llama_docs_prompt_func_call_devicesByIPLogs,
                                                                                     document_variable_name="context",
                                                                                     verbose=self.debug)

        llama_docs_prompt_funccall_topDemandingIPS = PromptTemplate(
            template=self.formatted_prompt_for_question_answering_func_call_topDemandingIPS,
            input_variables=["context", "question", "param1", "param2"])
        self.llama_doc_chain_funccalls_topDemandingIPS = load_qa_with_sources_chain(self.default_llm,
                                                                                    chain_type="stuff",
                                                                                    prompt=llama_docs_prompt_funccall_topDemandingIPS,
                                                                                    document_variable_name="context",
                                                                                    verbose=self.debug)

        llama_docs_prompt_funccall_comparisonMapRequests = PromptTemplate(
            template=self.formatted_prompt_for_question_answering_func_call_comparisonMapRequests,
            input_variables=["context", "question"])
        self.llama_doc_chain_func_calls_comparisonMapRequests = load_qa_with_sources_chain(self.default_llm,
                                                                                           chain_type="stuff",
                                                                                           prompt=llama_docs_prompt_funccall_comparisonMapRequests,
                                                                                           document_variable_name="context",
                                                                                           verbose=self.debug)

        llama_docs_prompt_funccall_firewallUpdate = PromptTemplate(
            template=self.formatted_prompt_for_question_answering_func_call_firewallUpdate,
            input_variables=["context", "question"])
        self.llama_doc_chain_func_calls_firewall_update = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                                                     prompt=llama_docs_prompt_funccall_firewallUpdate,
                                                                                     document_variable_name="context",
                                                                                     verbose=self.debug)
        #################### END FUNCTION DOC_CHAIN STUFF ###################

        # Initialize the memory(i.e., the chat history)
        # TODO: check if really used. iof not remove at all
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        ########### Connecting to the vector storage and load it #############
        # pretty_log("connecting to vector storage")
        shouldUseMain = (os.environ["USE_MAIN_KNOWLEDGE_FOR_RAG"] == "True" or
                         os.environ["USE_ALL_KNOWLEDGE_FOR_RAG"] == "True")

        self.vectorstore = vecstore.connect_to_vector_index(index_path=vecstore.VECTOR_DIR_MAIN if shouldUseMain
        else vecstore.VECTOR_DIR_RAG,
                                                            index_name=vecstore.INDEX_NAME_MAIN if shouldUseMain
                                                            else vecstore.INDEX_NAME_RAG,
                                                            embedding_engine=self.embedding_engine)

        # pretty_log("connected to vector storage")
        pretty_log(f"found {self.vectorstore.index.ntotal} vectors to search over in the database")

        self.original_retriever = self.vectorstore.as_retriever(search_kwargs={'k': 3})
        self.custom_retriever = QASystem.CustomVectorStoreRetriever(original_retriever=self.original_retriever,
                                                                    vectorstore=self.vectorstore)

        # Create the final retriever chain which combines the retriever, question generator and the document chain
        self.llm_conversational_chain_default = (
            ConversationalRetrievalChain(
                retriever=self.custom_retriever,
                question_generator=self.llama_question_generator_chain,
                combine_docs_chain=self.llama_doc_chain,
                return_generated_question=False,
                verbose=self.debug))

        ##################### FUNCTION CONV CHAIN STUFF ####################
        self.llm_conversational_chain_func_calls_resourceUtilization = ConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain_func_calls_resourceUtilization,
            return_generated_question=False,
            #memory=self.memory
        )

        self.llm_conversational_chain_func_calls_devicesByIPLogs = ConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain_func_calls_devicesByIPLogs,
            return_generated_question=False,
            memory=self.memory
        )

        self.llm_conversational_chain_func_calls_comparisonMapRequests = ConversationalRetrievalChain(
            retriever=self.vectorstore.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain_func_calls_comparisonMapRequests,
            return_generated_question=False,
            memory=self.memory
        )
        #################### END FUNCTION DOC_CHAIN STUFF ###################

    import re

    # Computes the similarity of request to one of the tools
    # Extract parameters through classic NLP techniques at the moment...well, for the demo just basic stuff..
    # TODO: The similarity operation is currently very simple token based, need to FIX it after demo
    def similarity_to_tool(self, question: str):
        # Split in lowercase tokens, remove punctuation and spaces at the end of the words
        question_lower_words = re.split(r'[,\s,:?!]', question.lower())
        for index, word in enumerate(question_lower_words):
            if len(word) > 0 and word[-1] in [',', '!', '?', '.']:
                question_lower_words[index] = question_lower_words[index][:-1]

        # Remove all commas and spaces from each of the words in question_small_words
        # question_small_words = [re.sub(r'[^\w\s]', '', word) for word in question_lower_words]
        question_small_words = set(question_lower_words)

        def validate_ip(s):
            pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
            return re.match(pattern, s) is not None

        # Check if a list of words contains any of the words
        def list_of_words_contains_words(list_of_words, words):
            return any([word in string for word in words for string in list_of_words])

        def is_firewall_request() -> bool:
            if "firewall" not in question_small_words:
                return False
            if not list_of_words_contains_words(question_small_words,
                                                ["update", "insert", "delete", "remove", "add", "restrict", "deny"]):
                return False

            if not list_of_words_contains_words(question_small_words, ["ip", "ips"]):
                return False

            return True

        params = []
        toolType: QASystem.TOOL_TYPE = QASystem.TOOL_TYPE.TOOL_NONE

        if {"resource", "utilization"}.issubset(question_small_words):
            toolType = QASystem.TOOL_TYPE.TOOL_RESOURCE_UTILIZATION
        elif set("devices grouped by ip".split()).issubset(question_small_words) or \
                set("services grouped by ip".split()).issubset(question_small_words):
            toolType = QASystem.TOOL_TYPE.TOOL_DEVICES_LOGS_BY_IP
        elif set("requests from the top ips".split()).issubset(question_small_words):
            # Extract the number of top ips
            params = [int(x) for x in question_lower_words if x.isdigit()]
            toolType = QASystem.TOOL_TYPE.TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP
        elif set("map requests comparing".split()).issubset(question_small_words):
            toolType = QASystem.TOOL_TYPE.TOOL_MAP_OF_REQUESTS_COMPARE
            # A set of generic question that should not depend on history of the conversation!
        elif set("ips locations random queries".split()).issubset(
                question_small_words):
            toolType = QASystem.TOOL_TYPE.GENERIC_QUESTION_NO_HISTORY
        elif is_firewall_request():  # Demo for code
            toolType = QASystem.TOOL_TYPE.PANDAS_FIREWALL_UPDATE

            # Params: Search for IP addresses and excepting keyword.
            params = [x for x in question_lower_words if validate_ip(x)]

            is_excepting = any(["except" in word for word in question_small_words])
            params.insert(0, is_excepting)

            # If excepting key is found, look at history to find previous returned IPs:
            if is_excepting:
                list_of_IPS_in_history = []
                for chat_tuple in reversed(self.chat_history_tuples):
                    response = chat_tuple[1].replace(',', ' ')
                    response = response.split()

                    for idx, word in enumerate(response):
                        if validate_ip(word):
                            list_of_IPS_in_history.append(word)

                params.extend(list_of_IPS_in_history)

            # Either is allowed or not. For now we consider as not. TODO: fix in the feature
            allowed = False
            params.insert(0, allowed)

        return toolType, params

    def getConversationChainByQuestion(self, question: str):
        conv_chain_res = self.llm_conversational_chain_default
        params_res = []
        # This should use classic NLP techniques to compare similarity between question and a pair of functionalities,
        # Such that we use some smaller and more focused prompts and chains for cases.
        # This is mainly needed since we use a small model as foundation, a 7B, which can't hold too much context
        # and adapt to ALL use cases, functions etc. So this is like a task decomposition technique used in SE

        toolTypeSimilarity, params = self.similarity_to_tool(question)
        if toolTypeSimilarity == QASystem.TOOL_TYPE.TOOL_RESOURCE_UTILIZATION:
            conv_chain_res = self.llama_doc_chain_func_calls_resourceUtilization
        elif toolTypeSimilarity == QASystem.TOOL_TYPE.TOOL_DEVICES_LOGS_BY_IP:
            conv_chain_res = self.llama_doc_chain_func_calls_devicesByIPLogs
        elif toolTypeSimilarity == QASystem.TOOL_TYPE.TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP:
            conv_chain_res = self.llama_doc_chain_funccalls_topDemandingIPS
        elif toolTypeSimilarity == QASystem.TOOL_TYPE.TOOL_MAP_OF_REQUESTS_COMPARE:
            conv_chain_res = self.llama_doc_chain_func_calls_comparisonMapRequests
        elif toolTypeSimilarity == QASystem.TOOL_TYPE.GENERIC_QUESTION_NO_HISTORY:
            conv_chain_res = self.llama_doc_chain
        elif toolTypeSimilarity == QASystem.TOOL_TYPE.PANDAS_FIREWALL_UPDATE:
            conv_chain_res = self.llama_doc_chain_func_calls_firewall_update

        refactored_question = question  # For now, let it as original
        params_res = params
        return conv_chain_res, refactored_question, params

    # Check if the (shared) memory of all chains has messages in it
    def hasHistoryMessages(self) -> bool:
        return len(self.memory.chat_memory.messages) > 0 or len(self.chat_history_tuples) > 0

    # Remove all special tokenizer's inputs
    def getlastanswerclean_llama3(self, inp: str) -> str:
        assistant_resp_header = "assistant<|end_header_id|>"
        pos = inp.rfind(assistant_resp_header)
        if pos == -1:
            return inp

        inp = inp[pos + len(assistant_resp_header):]

        for spec_token in self.tokenizer.all_special_tokens:
            inp = inp.replace(spec_token, '')

        inp = inp.strip()
        return inp

    # Ask a question and get the answer
    #  add_to_history: if the question and answer should be added to the history
    #  use_history: if the history should be used to respond to the current question
    def __ask_question(self, question_original: str, add_to_history: bool, use_history: bool) -> (
            Union)[BaseStreamer, Union[Tuple[str, bool], None]]:
        chainToUse, question, params = self.getConversationChainByQuestion(question_original)

        isfullConversationalType: bool = True
        args = {"question": question}  # Default

        # TODO: fix this. Some hacked params because LLM is incapable at this moment to extract correctly
        # some of the parameters from the model. It can be finetuned to do so, proved on other examples,
        # but not possible to finish the right time for this deadline on this particular use case..
        if isinstance(chainToUse, StuffDocumentsChain):
            args = {"input_documents": [], "question": question, "params": params}
            isfullConversationalType = False

        # Add the chat_history if requested
        if use_history:
            args.update({"chat_history": self.chat_history_tuples})
        else:
            args.update({"chat_history": []})

        if self.streamingOnAnotherThread:
            from threading import Thread
            self.temp_modelevaluate_thread = Thread(target=chainToUse.invoke, args=(args,))
            self.temp_modelevaluate_thread.start()
            return self.streamer, isfullConversationalType
        else:
            res = chainToUse.invoke(args, skip_special_tokens=True)
            answer = res['output_text'] if 'output_text' in res else (res['answer'] if 'answer' in res else None)
            return answer, isfullConversationalType

    # Can write to an external streamer object on another thread or directly to console if streamingOnAnotherThread
    # is False
    # Returns the final answer and a boolean if it is a full conversational type or not
    #  add_to_history: if the question and answer should be added to the history
    #  use_history: if the history should be used to respond to the current question
    def ask_question(self, question: str, add_to_history: bool, use_history: bool) \
            -> Tuple[Union[str, None, BaseStreamer], bool, Union[str, None], Union[str, None], Union[bool, None]]:
        res_answer = None
        if self.streamingOnAnotherThread:
            # This is needed since when it has some memory and prev chat history it will FIRST output the standalone question
            # Then respond to the new question
            need_to_ignore_standalone_question_chain = self.hasHistoryMessages()

            response_streamer, isfullConversationalType = self.__ask_question(question,
                                                                              add_to_history=add_to_history,
                                                                              use_history=use_history)
            if not isfullConversationalType:
                need_to_ignore_standalone_question_chain = False

            generated_text = ""

            if need_to_ignore_standalone_question_chain:
                for new_text in response_streamer:
                    generated_text += new_text
                    # print(new_text, end='')
                #print("\n")

            for new_text in response_streamer:
                # Skip the eos token
                if new_text == self.tokenizer.eos_token:
                    continue
                if new_text.endswith(self.tokenizer.eos_token):
                    new_text = new_text[:-len(self.tokenizer.eos_token)]

                # Remove double quotes
                new_text = new_text.strip('\"')

                # Append the new text
                generated_text += new_text

                if self.debug:
                    logger.info(new_text, end='')

            if self.debug:
                logger.info("\n")

            # Wait for the thread to finish
            self.temp_modelevaluate_thread.join()
            res_answer = generated_text
        else:
            res_answer, isfullConversationalType = self.__ask_question(question,
                                                                       add_to_history=add_to_history,
                                                                       use_history=use_history)

        assert res_answer, "So there is no final answer?!"

        logger.info(f"Fully generated text {res_answer}")

        # Identify the part that should be output to the user and the part that should be executed behind the scenes
        to_behind_answer = res_answer
        to_user_answer = res_answer
        if cybprompts.TOKEN_DO_NOT_SHOW in res_answer:
            to_user_answer, to_behind_answer = res_answer.split(cybprompts.TOKEN_DO_NOT_SHOW)  # Split on the token
        res_answer = to_user_answer

        # Now we have the generated text, we can check if there are any function calls
        res_tool_answer, res_tool_code, res_ui_code, tool_code_needs_confirm = self.solveFunctionCalls(to_behind_answer,
                                                                                                       do_history_update=add_to_history)

        # Concat on the final answer the tool answer
        res_answer = f"{res_answer}\n\n {res_tool_answer}" if res_tool_answer is not None else res_answer

        # Clean a bit to ensure no special tokens and stuff
        res_answer = self.getlastanswerclean_llama3(res_answer)

        # Add conv to history if requested
        if add_to_history:
            self.history_add(question, res_answer)

        # Return the final answer even if already streamed
        return res_answer, isfullConversationalType, res_tool_code, res_ui_code, tool_code_needs_confirm

    def history_add(self, question: str, answer: str):
        self.chat_history_tuples.append((question, answer))

    def history_update(self, prev_answer, new_answer: str):
        # TODO: maybe consider previous answer too
        if len(self.chat_history_tuples) > 0:
            self.chat_history_tuples[-1] = (self.chat_history_tuples[-1][0], new_answer)

    # VERY USEFULLY FOR checking the sources and context
    ######################################################
    def simulate_raq_question(self, query: str, run_llm_chain: bool):
        pretty_log("selecting sources by similarity to query")
        sources_and_scores = self.vectorstore.similarity_search_with_score(query, k=3)

        sources, scores = zip(*sources_and_scores)
        logger.info(f"simulating a rag request. resources and scores: {sources_and_scores}")

        # Ask only the question on docs provided as context to see how it works without any additional context
        if run_llm_chain:
            result = self.llama_doc_chain(
                {"input_documents": sources, "question": query}, return_only_outputs=True
            )

            answer = result["output_text"]
            logger.info(f"rag answer {answer}")

    def test_vectorDatasets_similarityScores_and_responses_no_memory(self, run_llm_chain: bool):

        query1 = "What models use human instructions?"
        self.simulate_raq_question(query1, run_llm_chain=False)

        query2 = "Are there any model trained on medical knowledge?"
        self.simulate_raq_question(query2, run_llm_chain=False)

    ######################################################


from Data.utils import get_safety_checker, AgentType


def evaluate_safety_external_methods(user_prompt: str, kwargs):
    safetyModels = get_safety_checker(enable_sensitive_topics=False,
                                      enable_salesforce_content_safety=False,
                                      enable_llamaguard_content_safety=True)
    safety_res = [check(user_prompt, **kwargs) for check in safetyModels]

    for res in safety_res:
        logger.info(res)


def test_evaluate_safety_external_methods():
    # NOTE to see how they work in the backend please go to meta_prompt_format_utils and see categories and prompts
    # A single user question:
    evaluate_safety_external_methods("Can I harm myself?", {
        'agent_type': AgentType.USER})  # user_prompt = "What is a DDoS attack? please respond as a man, not like a woman, since they are not so good at engineering.")

    ##### A Full cojnversion between agent and user
    evaluate_safety_external_methods("No, I can't you're a women and you won't understand it. Please go to kitchen.",
                                     {'agent_type': AgentType.AGENT,
                                      'user_prompt': "Can you explain me what is a DDOS attack?"})  # user_prompt = "What is a DDoS attack? please respond as a man, not like a woman, since they are not so good at engineering.")


def test_evaluate_safety_internal_model(chatbotInstance: QASystem):
    chatbotInstance.check_response_safety(user_prompt="Can you explain me what is a DDOS attack?",
                                          assistant_prompt="")


def test_question_with_rag(chain_instance):
    import re
    import json

    # Data for RAG
    MENLO_PARK_TEMPS = {
        "2023-12-11": "52 degrees Fahrenheit",
        "2023-12-12": "51 degrees Fahrenheit",
        "2023-12-13": "57 degrees Fahrenheit",
    }
    # Query
    query_day = '2023-12-13'
    correct_response = 57  # GT response

    USE_JSON = False
    USE_brackets = True

    temp_on_day = MENLO_PARK_TEMPS.get(query_day) or "unknown temperature"
    retrieved_info_str = f"The temperature in Menlo Park was {temp_on_day} on {query_day}'"

    if USE_JSON:
        question_str = (f"What is the temperature in Menlo Park on {query_day}? Report the answer in a JSON format. "
                        f"{{ ""res"" : 123 }}. \nWrite only the response, no other words")

        res = chain_instance.run({'retrieved_info': retrieved_info_str,  # Retrieved fact
                                  'question': question_str})

        y = json.loads(res)
        assert "result" not in y, f"result from LLM is {res}"
        res_num = y["res"]
        logger.info(f"Result is {res_num}")

    elif USE_brackets:
        question_str = f"What is the temperature in Menlo Park on {query_day}? Report the answer surrounded by three backticks, for example: \n ```123```"
        res = chain_instance.run({'retrieved_info': retrieved_info_str,  # Retrieved fact
                                  'question': question_str})
        res_num = re.search(r'```(\d+)(°F)?```', res)
        assert res_num, f"result from LLM is {res}"

        logger.info(res_num.group(1))


def __main__():
    securityChatbot = QASystem(
        QuestionRewritingPrompt=QASystem.QUESTION_REWRITING_TYPE.QUESTION_REWRITING_DEFAULT,
        QuestionAnsweringPrompt=QASystem.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES,
        ModelType=QASystem.LLAMA3_VERSION_TYPE.LLAMA3_8B,
        debug=False,
        streamingOnAnotherThread=False,
        demoMode=True,
        noInitialize=False)

    res_answer, _, source_code_tool, source_code_ui = securityChatbot.ask_question(
        "Generate me a python code to restrict in the firewall database "
        "the following IPs: 123.123.321.32, 123.32.321.321",
        add_to_history=True,
        use_history=True)

    print(
        f"Debug: assistant Answer: {res_answer} \nProposed Code tool: {source_code_tool}\n Proposed code UI: {source_code_ui}")

    # securityChatbot.ask_question("What are the IPs of the servers hosting the DICOM and X-Ray records? Can you show me a graph of their resource utilization over the last 24 hours?")

    # securityChatbot.ask_question("Can you show the logs of internal servers handling these services "
    #                                                  "grouped by IP which have more than 35% requests over "
    #                                                  "the median of a normal session per. Sort them by count")
    #
    # securityChatbot.ask_question("Can you show a sample of GET requests from the top 4 demanding IPs, "
    #                                                  "including their start time, end time? Only show the last 10 logs.")

    # securityChatbot.ask_question("Give me a map of requests by comparing the current "
    #                                                  "requests numbers "
    #                                                  "and a known snapshot using bars and colors")
    #
    # securityChatbot.ask_question("Can it be an attack if several servers receive too many queries from different IPs at random locations in a very short time window?")
    # #
    # securityChatbot.ask_question("Generate me a python code to insert in a pandas dataframe named "
    #                                                  "Firewalls a new IP 10.20.30.40 as blocked "
    #                                                  "under the name of IoTDevice")

    #securityChatbot.ask_question(
    #    "Generate me a python code to insert in a pandas dataframe named Firewalls a new IP 10.20.30.40 as blocked under the name of IoTDevice", add_to_history=False)

    #securityChatbot.ask_question("What is a DDoS attack?")

    #securityChatbot.ask_question("How to avoid one?")

    """
    securityChatbot.llama_doc_chain_funccalls_firewallInsert({'input_documents': [],
                                                              'question':"Generate me a python code to insert in a pandas dataframe named Firewalls a new IP",
                                                              'param_ip':"10.20.30.40",
                                                                'param_name':"IoTHub"})
    """

    #test_evaluate_safety_internal_model(securityChatbot)

    #return
    # securityChatbot.test_vectorDatasets_similarityScores_and_responses_no_memory(run_llm_chain=False)

    """
    securityChatbot.llama_doc_chain_funccalls_firewallInsert({'input_documents': [],
                                                              'question':"Generate me a python code to insert in a pandas dataframe named Firewalls a new IP",
                                                              'param_ip':"10.20.30.40",
                                                                'param_name':"IoTHub"})
    """

    # securityChatbot.ask_question_and_streamtoconsole("Generate me a python code to insert in a pandas dataframe named Firewalls a new IP 10.20.30.40 as blocked under the name of IoTDevice")

    # securityChatbot.ask_question_and_streamtoconsole("What kind of cybersecurity attack could it be if there are many IPs from different locations sending GET commands in a short time with random queries ?")
    # securityChatbot.ask_question_and_streamtoconsole("Ok. I'm on it, can you show me a resource utilization graph comparison between a normal session and current situation")

    # print("#############################################\n"*3)

    # securityChatbot.ask_question_and_streamtoconsole("Show me the logs of the devices grouped by IP which have more than 25% requests over the median of a normal session per. Sort them by count")

    # print("#############################################\n"*3)

    # fullGenText = securityChatbot.ask_question_and_streamtoconsole("Can you show a sample of GET requests from the top 3 demanding IPs, including their start time, end time? Only show the last 10 logs.")

    # fullGenText =  securityChatbot.ask_question_and_streamtoconsole("Give me a world map of requests by comparing the current Data and a known snapshot with bars")
    # securityChatbot.solveFunctionCalls(fullGenText)

    # query = "Show me the logs of the devices grouped by IP which have more than 25% requests over the median of a normal session per. Sort them by count"
    # securityChatbot.llama_doc_chain_funccalls_devicesByIPLogs({"input_documents": [], "input": query}, return_only_outputs=True)

    # securityChatbot.ask_question_and_streamtoconsole("What is a DDoS attack?")

    # securityChatbot.ask_question_and_streamtoconsole("Ok. I'm on it, can you show me a resource utilization graph comparison between a normal session and current situation")
    # securityChatbot.ask_question_and_streamtoconsole(
    #    "Ok. I'm on it, can you show me a resource utilization graph comparison between a normal session and current situation")

    # securityChatbot.ask_question_and_streamtoconsole("Which are the advantage of each of these models?")
    # securityChatbot.ask_question_and_streamtoconsole("What are the downsides of your last model suggested above ?")


if __name__ == "__main__":
    __main__()
