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
from transformers import pipeline
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

    @staticmethod
    def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT ):
        SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
        prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
        return prompt_template

    def ask_question_llama2_cont(query):
        #result = llm_chain.predict(user_input=query); #({"question": query}, return_only_outputs=True)
        #answer = result["output_text"]

        result1 = llm_chain({"user_input" : "Give me some indications to solve a denial of service attack"})
        print(result1)
        result2 = llm_chain({"user_input": "What did I ask you previously?"})
        print(result2)

        return result2



def qanda_llama2_cont(request_id=None, with_logging: bool = False) -> str:
    """Runs sourced Q&A for a query using LangChain.

    Arguments:
        query: The query to run Q&A on.
        request_id: A unique identifier for the request.
        with_logging: If True, logs the interaction to Gantry.
    """
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    #import prompts
    import vecstore
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline


    global embedding_engine
    embedding_engine = vecstore.get_embedding_engine(allowed_special="all")

    # MODEL LOADING STUFF

    global tokenizer
    global model

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                              token=True)

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",
                                                 device_map='auto',
                                                 torch_dtype=torch.bfloat16,
                                                 token=True,
                                                 #  load_in_8bit=True,
                                                 #  load_in_4bit=True,
                                                 )
    # use class TextIteratorStreamer(TextStreamer): check for gradio example C:\Users\cipri\AppData\Local\JetBrains\PyCharm2023.3\remote_sources\-643487973\-1298038738\transformers\generation\streamers.py
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    pipe = pipeline("text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    max_new_tokens=4096,
                    do_sample=True,
                    #temperature=0.1,
                    top_p=0.95,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id = tokenizer.eos_token_id,
                    streamer=streamer,
                    )

    llm = HuggingFacePipeline(pipeline=pipe)
    
    global base_llm
    base_llm = llm

    instruction = "Chat History:\n\n{chat_history} \n\nUser: {user_input}"
    system_prompt = """\
        ""Consider that I'm a beginner in networking and security things. \n
        Give me a concise answer with with a single step at a time. \n
        Limit your resonse to maximum 128 words.
        Do not provide any additional text or presentation. Only steps and actions.
        If possible use concrete names of software or tools that could help on each step."""
    template = get_prompt(instruction, system_prompt)
    print(template)

    prompt = PromptTemplate(
        input_variables=["chat_history", "user_input"], template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    global llm_chain
    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

    #device=0
    #inputs = tokenizer("Give me some ideas", return_tensors="pt").to(device)
    #streamer = TextStreamer(tokenizer, skip_prompt=True)
    #llm_chain({"user_input": "Give me some indications to solve a denial of service attack"})

    return

    ################# PART 2 ###################
    pretty_log("connecting to vector storage")
    vector_index = vecstore.connect_to_vector_index(
        vecstore.INDEX_NAME, embedding_engine
    )
    pretty_log("connected to vector storage")
    pretty_log(f"found {vector_index.index.ntotal} vectors to search over")

    pretty_log(f"running on query: {query}")
    pretty_log("selecting sources by similarity to query")
    sources_and_scores = vector_index.similarity_search_with_score(query, k=3)

    sources, scores = zip(*sources_and_scores)

    pretty_log("running query against Q&A chain")


    #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256) # gpt4 no longer available for free
    chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        verbose=with_logging,
        prompt=prompts.main,
        document_variable_name="sources",
    )

    result = chain(
        {"input_documents": sources, "question": query}, return_only_outputs=True
    )
    answer = result["output_text"]

    if with_logging:
        print(answer)
        pretty_log("logging results to gantry")
        record_key = log_event(query, sources, answer, request_id=request_id)
        if record_key:
            pretty_log(f"logged to gantry with key {record_key}")

    return answer

def __main__():


if __name__ == "__main__":
    __main__()
