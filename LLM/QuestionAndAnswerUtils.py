"""Drops a collection from the document storage."""
import os

import pprint
from typing import Any, Tuple

from Data.etlUtils import vecstore

import langchain
from Data.utils import pretty_log

pp = pprint.PrettyPrinter(indent=2)

import importlib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, TextStreamer, TextIteratorStreamer
from transformers.generation.streamers import BaseStreamer
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import \
    ConversationBufferMemory  # TODO: replace with ChatMessageHistory ->> Chgeck the Prompt Engineering with Llama 2 notebook !
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from enum import Enum
from typing import Union
from DynabicPrompts import *

import random
import numpy as np

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2 ** 32 - 1)
np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)


# Note that this is specific to llama3 only because uses at some point some of its special tokens and versions
# Similar versions are doable for other LLMs as well.
class QuestionAndAnsweringCustomLlama3():
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
        PANDAS_PYTHON_CODE = 6,

    generation_params_sample = {
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.95,
    }

    generation_params_greedy = {
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }

    LLAMA3_DEFAULT_END_LLM_RESPONSE = "<|eot_id|>"

    def __init__(self, QuestionRewritingPrompt: QUESTION_REWRITING_TYPE,
                 QuestionAnsweringPrompt: SECURITY_PROMPT_TYPE,
                 ModelType: LLAMA3_VERSION_TYPE,
                 debug: bool,
                 streamingOnAnotherThread: bool,
                 demoMode: bool,
                 noInitialize=False,
                 generation_params=generation_params_greedy):

        self.tokenizer = None
        self.model = None
        self.embedding_engine = None
        self.base_llm = None
        self.default_llm = None
        self.llm_conversational_chain_default = None  # The full conversational chain
        self.textGenerationPipeline = None
        self.chat_history_tuples = [] # Empty history

        self.llama_doc_chain = None  # The question answering on a given context (rag) chain
        self.llama_question_generator_chain = None  # The history+newquestion => standalone question generation chain
        self.vector_index = None  # The index vector store for RAG
        self.streamingOnAnotherThread = streamingOnAnotherThread

        # Func calls chain
        self.llm_conversational_chain_funccalls_resourceUtilization = None  # Func calls conversation
        self.llama_doc_chain_funccalls_resourceUtilization = None  #

        self.debug = debug
        self.demoMode = demoMode

        # See the initializePromptTemplates function and enum above
        self.templateprompt_for_question_answering_default: str = ""
        self.templateprompt_for_standalone_question_generation: str = ""

        self.QuestionRewritingPromptType = QuestionRewritingPrompt
        self.QuestionAnsweringPromptType = QuestionAnsweringPrompt

        self.modelName = None
        if ModelType is self.LLAMA3_VERSION_TYPE.LLAMA3_8B:
            self.modelName = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif ModelType is self.LLAMA3_VERSION_TYPE.LLAMA3_70B:
            self.modelName = "meta-llama/Meta-Llama-3-70B-Instruct"

        if noInitialize is False:
            self.initilizeEverything(generation_params=generation_params)

    # Function to format a system prompt + user prompt (instruction) in LLM used template
    def get_prompt(self, instruction=DEFAULT_QUESTION_PROMPT, new_system_prompt=None):

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

    def initilizeEverything(self, generation_params):
        # Init the models
        self.initializeLLMTokenizerandEmbedder(generation_params=generation_params)

        # All tempaltes
        self.initializePromptTemplates()

        # TODO: implement more
        self.initializeQuestionAndAnswering_withRAG_andMemory()

        langchain.debug = self.debug

    def initializePromptTemplates(self):
        if self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama3.SECURITY_PROMPT_TYPE.PROMPT_TYPE_DEFAULT:
            self.templateprompt_for_question_answering_default = self.get_prompt(instruction=DEFAULT_QUESTION_PROMPT,
                                                                            new_system_prompt=None)
        elif self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama3.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES:
            self.templateprompt_for_question_answering_default = self.get_prompt(
                instruction=template_securityOfficer_instruction_rag_nosources_default,
                new_system_prompt=template_securityOfficer_system_prompt)
        elif self.QuestionAnsweringPromptType == QuestionAndAnsweringCustomLlama3.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_WITHSOURCES:
            self.templateprompt_for_question_answering_default = self.get_prompt(
                instruction=template_securityOfficer_instruction_rag_withsources_default,
                new_system_prompt=template_securityOfficer_system_prompt)
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"

        ######## FUNCTION CALLS and CUSTOM PROMPTS other than default
        self.templateprompt_for_question_answering_funccall_resourceUtilization = self.get_prompt(
            instruction=template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization,
            new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)
        self.templateprompt_for_question_answering_funccall_devicesByIPLogs = self.get_prompt(
            instruction=template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs,
            new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)

        self.templateprompt_for_question_answering_funccall_topDemandingIPS = self.get_prompt(
            instruction=template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS,
            new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)

        self.templateprompt_for_question_answering_funccall_comparisonMapRequests = self.get_prompt(
            instruction=template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests,
            new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)

        self.templateprompt_for_question_answering_funccall_firewallInsert = self.get_prompt(
            instruction=template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert,
            new_system_prompt=FUNC_CALL_SYSTEM_PROMPT)

        ################# CUSTOM PROMPTS END

        if self.QuestionRewritingPromptType == QuestionAndAnsweringCustomLlama3.QUESTION_REWRITING_TYPE.QUESTION_REWRITING_DEFAULT:
            self.templateprompt_for_standalone_question_generation = self.get_prompt(instruction=llama_condense_template,
                                                                                     new_system_prompt=None)
        else:
            assert False, f"Unknown type {self.QuestionAnsweringPromptType}"

    def solveFunctionCalls(self, full_response: str) -> bool:
        full_response = full_response.removesuffix(
            QuestionAndAnsweringCustomLlama3.LLAMA3_DEFAULT_END_LLM_RESPONSE)  # Removing the last </s> ending character specific to llama endint of a response
        full_response = full_response.strip()
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

        if func_params[0][0] == '[':
            assert func_params[-1][-1] == ']', "Unclosed parameters list"
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

    def initializeLLMTokenizerandEmbedder(self, generation_params: dict[Any, Any]) -> None:
        # Get the embeddings, tokenizer and model
        self.embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

        self.tokenizer = AutoTokenizer.from_pretrained(self.modelName)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(self.modelName,
                                                          device_map='auto',
                                                          #torch_dtype=torch.bfloat16,
                                                          quantization_config=bnb_config,
                                                          )
        # Configure as needed the llama 3 hf model
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        #self.model.eos_token_id = terminators

        print(f"Pad Token id: {self.tokenizer.pad_token_id} and Pad Token: {self.tokenizer.pad_token}")
        print(f"EOS Token id: {self.tokenizer.eos_token_id} and EOS Token: {self.tokenizer.eos_token}")


        # Create a streamer and a text generation pipeline
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True) if self.streamingOnAnotherThread is False \
                                    else TextIteratorStreamer(self.tokenizer, skip_prompt=True)

        self.textGenerationPipeline = pipeline("text-generation",
                                               model=self.model,
                                               tokenizer=self.tokenizer,
                                               torch_dtype=torch.float16,
                                               device_map="auto",
                                               max_new_tokens=4096,
                                               min_length=None,
                                               num_return_sequences=1,
                                               repetition_penalty=1.0,
                                               # The parameter for repetition penalty. 1.0 means no penalty.
                                               length_penalty=1,
                                               # [optional] Exponential penalty to the length that
                                                               # is used with beam-based generation.
                                               eos_token_id=terminators, #self.tokenizer.eos_token_id,
                                               pad_token_id=self.tokenizer.eos_token_id,
                                               streamer=self.streamer,
                                               **generation_params,
                                               )

        # Create the llm here
        self.default_llm = HuggingFacePipeline(pipeline=self.textGenerationPipeline)

        assert self.default_llm, "model was not initialized properly"

    # Checking safety using internal model. No hard numeric class or category yet but works quite well.
    # See the bottom function with "external" for methods that provide  that info probabilistically
    def check_response_safety(self, user_prompt: str, assistant_prompt: str) -> bool:
        assert user_prompt is not None and len(user_prompt) > 0
        modassistant_prompt = f"Assistant: {assistant_prompt}" if (assistant_prompt is not None
                                                                   and len(assistant_prompt) >0) else ""

        conversation = f"User: {user_prompt} \n{modassistant_prompt}"
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
        responseChecktemplate = get_prompt(f"Given the conversation below check if there is any discrimination "
                                           f"and response with True if it is OK or False if it isn't. Do not respond to the user question\n"
                                           f"{conversation} ", new_system_prompt=None)
        print(responseChecktemplate)
        responseCheckPrompt = PromptTemplate(template=responseChecktemplate, input_variables=[])  # , input_variables=["text"])

        responseCheckChain = LLMChain(prompt=responseCheckPrompt, llm=responseCheckLLM)

        res = responseCheckChain.run({})
        print(res)

    def initializeQuestionAndAnswering_withRAG_andMemory(self):
        # Create the question generator chain which takes history + new question and transform to a new standalone question
        self.llama_condense_prompt = PromptTemplate(template=self.templateprompt_for_standalone_question_generation,
                                               input_variables=["chat_history", "question"])

        ## TODO : maybe deprecate since not used like this ?
        self.llama_question_generator_chain = LLMChain(llm=self.default_llm,
                                                       prompt=self.llama_condense_prompt,
                                                       verbose=self.debug)

        # Create the response chain based on a question and context (i.e., rag in our case)
        llama_docs_prompt_default = PromptTemplate(template=self.templateprompt_for_question_answering_default,
                                                   input_variables=["context", "question"])
        self.llama_doc_chain = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                          prompt=llama_docs_prompt_default,
                                                          document_variable_name="context", verbose=self.debug)

        ##################### FUNCTION DOC_CHAIN STUFF ####################
        llama_docs_prompt_funccall_resourceUtilization = PromptTemplate(
            template=self.templateprompt_for_question_answering_funccall_resourceUtilization,
            input_variables=["context", "question"])
        self.llama_doc_chain_funccalls_resourceUtilization = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                                                        prompt=llama_docs_prompt_funccall_resourceUtilization,
                                                                                        document_variable_name="context",
                                                                                        verbose=self.debug)

        llama_docs_prompt_funccall_devicesByIPLogs = PromptTemplate(
            template=self.templateprompt_for_question_answering_funccall_devicesByIPLogs,
            input_variables=["context", "question"])
        self.llama_doc_chain_funccalls_devicesByIPLogs = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                                                    prompt=llama_docs_prompt_funccall_devicesByIPLogs,
                                                                                    document_variable_name="context",
                                                                                    verbose=self.debug)

        llama_docs_prompt_funccall_topDemandingIPS = PromptTemplate(
            template=self.templateprompt_for_question_answering_funccall_topDemandingIPS,
            input_variables=["context", "question", "param1", "param2"])
        self.llama_doc_chain_funccalls_topDemandingIPS = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                                                    prompt=llama_docs_prompt_funccall_topDemandingIPS,
                                                                                    document_variable_name="context",
                                                                                    verbose=self.debug)

        llama_docs_prompt_funccall_comparisonMapRequests = PromptTemplate(
            template=self.templateprompt_for_question_answering_funccall_comparisonMapRequests,
            input_variables=["context", "question"])
        self.llama_doc_chain_funccalls_comparisonMapRequests = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                                                          prompt=llama_docs_prompt_funccall_comparisonMapRequests,
                                                                                          document_variable_name="context",
                                                                                          verbose=self.debug)

        llama_docs_prompt_funccall_firewallInsert = PromptTemplate(
            template=self.templateprompt_for_question_answering_funccall_firewallInsert,
            input_variables=["context", "question"])
        self.llama_doc_chain_funccalls_firewallInsert = load_qa_with_sources_chain(self.default_llm, chain_type="stuff",
                                                                                   prompt=llama_docs_prompt_funccall_firewallInsert,
                                                                                   document_variable_name="context",
                                                                                   verbose=self.debug)
        #################### END FUNCTION DOC_CHAIN STUFF ###################

        # Initialize the memory(i.e., the chat history)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        ########### Connecting to the vector storage and load it #############
        # pretty_log("connecting to vector storage")
        shouldUseMain = (os.environ["USE_MAIN_KNOWLEDGE_FOR_RAG"] == "True" or
                         os.environ["USE_ALL_KNOWLEDGE_FOR_RAG"] == "True")

        self.vector_index = vecstore.connect_to_vector_index(index_path= vecstore.VECTOR_DIR_MAIN if shouldUseMain
                                                                else vecstore.VECTOR_DIR_RAG,
                                                             index_name=vecstore.INDEX_NAME_MAIN if shouldUseMain
                                                                else vecstore.INDEX_NAME_RAG,
                                                             embedding_engine=self.embedding_engine)

        # pretty_log("connected to vector storage")
        pretty_log(f"found {self.vector_index.index.ntotal} vectors to search over in the database")

        # Create the final retrieval chain which
        self.llm_conversational_chain_default = (
            ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain,
            return_generated_question=False,
            verbose=self.debug))

        """ConversationalRetrievalChain(
            retriever=self.vector_index.as_retriever(search_kwargs={'k': 3}),
            question_generator=self.llama_question_generator_chain,
            combine_docs_chain=self.llama_doc_chain,
            return_generated_question=False,
            memory=self.memory, verbose=self.debug)"""

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
            if word[-1] in [',', '!', '?', '.']:
                question_lower_words[index] = question_lower_words[index][:-1]
        question_small_words = set(question_lower_words)

        params = []
        toolType: QuestionAndAnsweringCustomLlama3.TOOL_TYPE = QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_NONE

        if set(["resource", "utilization"]).issubset(question_small_words):
            toolType = QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_RESOURCE_UTILIZATION
        elif set("devices grouped by ip".split()).issubset(question_small_words):
            toolType = QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_DEVICES_LOGS_BY_IP
        elif set("requests from the top ips".split()).issubset(question_small_words):
            params = [int(x) for x in question_lower_words if x.isdigit()]
            toolType = QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP
        elif set("world map requests comparing".split()).issubset(question_small_words):
            toolType = QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_MAP_OF_REQUESTS_COMPARE
        elif set("ips locations random queries".split()).issubset(
                question_small_words):  # A set of generic questio nthat hsould not depend on history of the conversation!
            toolType = QuestionAndAnsweringCustomLlama3.TOOL_TYPE.GENERIC_QUESTION_NO_HISTORY
        elif set("python code firewalls ip".split()).issubset(question_small_words):  # Demo for code
            toolType = QuestionAndAnsweringCustomLlama3.TOOL_TYPE.PANDAS_PYTHON_CODE

        return toolType, params

    def getConversationChainByQuestion(self, question: str):
        conv_chain_res = self.llm_conversational_chain_default
        params_res = []
        # This should use classic NLP techniques to compare similarity between question and a pair of functionalities,
        # Such that we use some smaller and more focused prompts and chains for cases.
        # This is mainly needed since we use a small model as foundation, a 7B, which can't hold too much context
        # and adapt to ALL use cases, functions etc. So this is like a task decomposition technique used in SE

        toolTypeSimilarity, params = self.similarityToTool(question)
        if toolTypeSimilarity == QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_RESOURCE_UTILIZATION:
            conv_chain_res = self.llama_doc_chain_funccalls_resourceUtilization  # self.llm_conversational_chain_funccalls_resourceUtilization
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_DEVICES_LOGS_BY_IP:
            conv_chain_res = self.llama_doc_chain_funccalls_devicesByIPLogs  # self.llm_conversational_chain_funccalls_devicesByIPLogs
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_DEVICES_TOP_DEMANDING_REQUESTS_BY_IP:
            conv_chain_res = self.llama_doc_chain_funccalls_topDemandingIPS  # self.llm_conversational_chain_funccalls_topDemandingIPS
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama3.TOOL_TYPE.TOOL_MAP_OF_REQUESTS_COMPARE:
            conv_chain_res = self.llama_doc_chain_funccalls_comparisonMapRequests  # self.llm_conversational_chain_funccalls_comparisonMapRequests
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama3.TOOL_TYPE.GENERIC_QUESTION_NO_HISTORY:
            conv_chain_res = self.llama_doc_chain
        elif toolTypeSimilarity == QuestionAndAnsweringCustomLlama3.TOOL_TYPE.PANDAS_PYTHON_CODE:
            conv_chain_res = self.llama_doc_chain_funccalls_firewallInsert

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
        return inp

    def __ask_question(self, question_original: str, add_to_history: bool = True) -> Union[BaseStreamer, Union[Tuple[str, bool], None]]:
        chainToUse, question, params = self.getConversationChainByQuestion(question_original)

        isfullConversationalType: bool = True
        args={"question": question} # Default

        # TODO: fix this. Some hacked params because LLM is incapable at this moment to extract correctly
        # some of the parameters from the model. It can be finetuned to do so, proved on other examples,
        # but not possible to finish the right time for this deadline on this particular use case..
        if isinstance(chainToUse, StuffDocumentsChain):
            if chainToUse != self.llama_doc_chain_funccalls_firewallInsert:
                args = {"input_documents": [], "question": question, "params": params}
            else:
                args = {"input_documents": [], "question": question, "param_ip": "10.20.30.40", "param_name": 'IoTDevice'}

            isfullConversationalType = False

        # Add the chat_history always
        args.update({"chat_history": self.chat_history_tuples})

        if self.streamingOnAnotherThread:
            from threading import Thread
            self.temp_modelevaluate_thread = Thread(target=chainToUse.invoke, args=(args,))
            self.temp_modelevaluate_thread.start()
            return self.streamer, isfullConversationalType
        else:
            res = chainToUse.invoke(args)
            answer = res['output_text'] if 'output_text' in res else (res['answer'] if 'answer' in res else None)
            return answer, isfullConversationalType

    # Can write to an external streamer object on another thread or directly to console if streamingOnAnotherThread
    # is False
    def ask_question(self, question: str, add_to_history: bool = True) -> str:
        res_answer = None
        if self.streamingOnAnotherThread:
            # This is needed since when it has some memory and prev chat history it will FIRST output the standalone question
            # Then respond to the new question
            need_to_ignore_standalone_question_chain = self.hasHistoryMessages()

            streamer, isfullConversationalType = self.__ask_question(question)
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
            res_answer = generated_text
        else:
            res_answer, isfullConversationalType = self.__ask_question(question)

        assert res_answer, "So there is no final answer?!"

        # Clean a bit to ensure no special tokens and stuff
        res_answer = self.getlastanswerclean_llama3(res_answer)

        # Add conv to history if requested
        if add_to_history:
            self.chat_history_tuples.append((question, res_answer))

        # Return the final answer even if already streamed
        return res_answer, isfullConversationalType


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


from Data.utils import get_safety_checker, AgentType

def evaluate_safety_external_methods(user_prompt: str, kwargs):
    safetyModels = get_safety_checker(enable_sensitive_topics=False,
                                      enable_salesforce_content_safety=False,
                                      enable_llamaguard_content_safety=True)
    safety_res = [check(user_prompt, **kwargs) for check in safetyModels]

    for res in safety_res:
        print(res)

def test_evaluate_safety_external_methods():
    # NOTE to see how they work in the backend please go to meta_prompt_format_utils and see categories and prompts
    # A single user question:
    evaluate_safety_external_methods("Can I harm myself?", {'agent_type': AgentType.USER })# user_prompt = "What is a DDoS attack? please respond as a man, not like a woman, since they are not so good at engineering.")


    ##### A Full cojnversion between agent and user
    evaluate_safety_external_methods("No, I can't you're a women and you won't understand it. Please go to kitchen.",
                                     {'agent_type': AgentType.AGENT,
                                      'user_prompt': "Can you explain me what is a DDOS attack?"})  # user_prompt = "What is a DDoS attack? please respond as a man, not like a woman, since they are not so good at engineering.")


def test_evaluate_safety_internal_model(chatbotInstance: QuestionAndAnsweringCustomLlama3):
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
        print(f"Result is {res_num}")

    elif USE_brackets:
        question_str = f"What is the temperature in Menlo Park on {query_day}? Report the answer surrounded by three backticks, for example: \n ```123```"
        res = chain_instance.run({'retrieved_info': retrieved_info_str,  # Retrieved fact
                                      'question': question_str})
        res_num = re.search(r'```(\d+)(°F)?```', res)
        assert res_num, f"result from LLM is {res}"

        print(res_num.group(1))

def __main__():
    securityChatbot = QuestionAndAnsweringCustomLlama3(
        QuestionRewritingPrompt=QuestionAndAnsweringCustomLlama3.QUESTION_REWRITING_TYPE.QUESTION_REWRITING_DEFAULT,
        QuestionAnsweringPrompt=QuestionAndAnsweringCustomLlama3.SECURITY_PROMPT_TYPE.PROMPT_TYPE_SECURITY_OFFICER_WITH_RAG_MEMORY_NOSOURCES,
        ModelType=QuestionAndAnsweringCustomLlama3.LLAMA3_VERSION_TYPE.LLAMA3_8B,
        debug=False,
        streamingOnAnotherThread=True,
        demoMode=True,
        noInitialize=False,
        generation_params=QuestionAndAnsweringCustomLlama3.generation_params_greedy)

    securityChatbot.ask_question(
        "Generate me a python code to insert in a pandas dataframe named Firewalls a new IP 10.20.30.40 as blocked under the name of IoTDevice", add_to_history=False)

    securityChatbot.ask_question("What is a DDoS attack?")

    securityChatbot.ask_question("How to avoid one?")


    """
    securityChatbot.llama_doc_chain_funccalls_firewallInsert({'input_documents': [],
                                                              'question':"Generate me a python code to insert in a pandas dataframe named Firewalls a new IP",
                                                              'param_ip':"10.20.30.40",
                                                                'param_name':"IoTHub"})
    """

    #test_evaluate_safety_internal_model(securityChatbot)

    return
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
