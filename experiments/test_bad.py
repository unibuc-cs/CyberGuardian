import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline, TextStreamer
import json
import torch
import textwrap
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings

import textwrap
def parse_text(text):
    wrapped_text = textwrap.fill(text, width=100)
    print(wrapped_text + '\n\n')
    # return assistant_text


embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

model_name = "meta-llama/Llama-2-7b-chat-hf"  # "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=True)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.bfloat16,
                                             token=True,
                                             #  load_in_8bit=True,
                                             load_in_4bit=True,
                                             )



DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

DEFAULT_QUESTION_PROMPT = "Question: {question}"
TEXT_QUESTION_PROMPT = "Text: {text}"


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


# Function to format a system prompt + user prompt (instruction) to LLama 2 format
def get_prompt(instruction=DEFAULT_QUESTION_PROMPT, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + (SYSTEM_PROMPT if len(new_system_prompt) > 0 else "") + instruction + E_INST
    return prompt_template


customRewritePrompt_System = ("You are a top computer science professor and I ask you to rewrite the given text in the Text keeping the same meaning and "
                       "avoid plagiarism detection.")

used_prompt_template = get_prompt(TEXT_QUESTION_PROMPT, new_system_prompt=customRewritePrompt_System)
print(used_prompt_template)
used_prompt = PromptTemplate(template=used_prompt_template, input_variables=["text"])



streamer = TextStreamer(tokenizer, skip_prompt=True)

pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.1,
                 top_p=0.95,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                 pad_token_id=tokenizer.eos_token_id,
                 streamer=streamer,
                )

llm = HuggingFacePipeline(pipeline = pipe, model_kwargs = {'temperature':0})
#%%
system_prompt = "You are an advanced assistant that excels at translation. "
instruction = "Convert the following text from English to French:\n\n {text}"
template = get_prompt(instruction, system_prompt)
print(template)

prompt = PromptTemplate(template=template, input_variables=["text"])
llm_chain = LLMChain(prompt=prompt, llm=llm)


llm = LLMChain(llm=model, prompt=prompt)

text = "how are you today?"
output = llm_chain.run(text)


parse_text(output)

exit(0)


llm_v2 = ConversationChain(model)
