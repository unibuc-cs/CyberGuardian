B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

FUNC_CALL_SYSTEM_PROMPT = ""

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

DEFAULT_QUESTION_PROMPT = "Question: {question}"


# Function to format a system prompt + user prompt (instruction) to LLama 2 format
def get_prompt(instruction=DEFAULT_QUESTION_PROMPT, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + (SYSTEM_PROMPT if len(new_system_prompt) > 0 else "") + instruction + E_INST
    return prompt_template


template_securityOfficer_system_prompt = """\
Consider that I'm a beginner in networking and security things. \n
Give me a concise answer with with a single step at a time. \n
Limit your response to maximum 2000 words.
Do not provide any additional text or presentation. Only steps and actions.
If possible use concrete names of software or tools that could help on each step.
Use emoticons in your responses.
"""

template_securityOfficer_instruction_rag_nosources_default = """\
Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
{context}
Question: {question}"""

template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization = """\
        Write only the following string and no other words, do not start your response with Sure. 
        "Ok, I will show you two histograms of usage by invoking FUNC_CALL dynabicagenttools.showResourceUtilizationComparison Params '../dynabicChatbot/data/SmartHome_DDoSSnapshot/good_RESOURCES_OCCUPANCY_HACKED_False.csv' '../dynabicChatbot/data/SmartHome_DDoSSnapshot/good_RESOURCES_OCCUPANCY_HACKED_True.csv'"
        {context}
        Question: {question}"""

template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs = """\
        Write only the following string and no other words, do not start your response with Sure.
        "Ok, I will show you the pandas dataset according to your request. I will invoke FUNC_CALL dynabicagenttools.show_outlier_ips_usage Params '../dynabicChatbot/data/SmartHome_DDoSSnapshot/DATASET_LOGS_HACKED_False.csv' '../dynabicChatbot/data/SmartHome_DDoSSnapshot/DATASET_LOGS_HACKED_True.csv'"
        {context}"""

template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS = """\
        Write only the following string and no other words, do not start your response with Sure.
        "Ok, I will show you the logs dataset by invoking FUNC_CALL dynabicagenttools.showLastNGetQueriesFromTopM_demandingIPs Params {params} "
        {context}"""

template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests="""\
        Write only the following string and no other words, do not start your response with Sure.
        "I will invoke FUNC_CALL dynabicagenttools.shopComparativeColumnsDatasets Params '../dynabicChatbot/data/SmartHome_DDoSSnapshot/DATASET_LOGS_HACKED_True.csv' '../dynabicChatbot/data/SmartHome_DDoSSnapshot/DATASET_LOGS_HACKED_False.csv'"
        {context}"""

template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert = """\
        Write only the following string and no other words, do not start your response with Sure. Do not write like I provided you the code.
        '''
        df1 = pd.read_csv("../data/SmartHome_DDoSSnapshot/FIREWALL_PROCESSES.csv")

        new_row = ('IP': {param_ip}, 'NAME': {param_name}, 'DATE': datetime.now(), 'BLOCKED':1)

        df1=pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)

        df1.to_csv("../data/SmartHome_DDoSSnapshot/FIREWALL_PROCESSES.csv")
        '''
        {context}"""

template_securityOfficer_instruction_rag_withsources_default = """
Given the following extracted parts of a long document and a question, create a final answer with "SOURCES" that represent exactly the Source name and link given.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}

{summaries}

FINAL ANSWER:"""

llama_condense_template = """
[INST]Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question: [/INST]"""