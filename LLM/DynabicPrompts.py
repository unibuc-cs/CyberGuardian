import os

FUNC_CALL_SYSTEM_PROMPT = None # ""
DEFAULT_SYSTEM_PROMPT = None
DEFAULT_QUESTION_PROMPT = "Question: {question}"


from enum import Enum
from UI.demoSupport import UseCase, USE_CASE


template_securityOfficer_system_prompt = None
template_securityOfficer_instruction_rag_nosources_default = None
template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization = None
template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs = None
template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS = None
template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests = None
template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert = None
template_securityOfficer_instruction_rag_withsources_default = None
llama_condense_template = None

# Set the templates based on the use case
def set_templates():
    global template_securityOfficer_system_prompt
    global template_securityOfficer_instruction_rag_nosources_default
    global template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization
    global template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs
    global template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS
    global template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests
    global template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert
    global template_securityOfficer_instruction_rag_withsources_default
    global llama_condense_template

    if USE_CASE == UseCase.Default or USE_CASE == UseCase.SmartHome:
        template_securityOfficer_system_prompt = """\
        Consider that I'm a beginner in networking and security things. \n
        Give me a concise answer with a single step at a time.  If there is only one step, do not write Step 1.\n
        Limit your response to maximum 2000 words.
        Do not provide any additional text or presentation. 
        If possible use concrete names of software or tools that could help.
        Use emoticons in your responses.
        """

        template_securityOfficer_instruction_rag_nosources_default = """\
        Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
        {context}
        Question: {question}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization = """\
                Write only the following string and no other words, do not start your response with Sure. 
                "Ok, I will show you two histograms of usage by invoking FUNC_CALL dynabicagenttools.showResourceUtilizationComparison Params 'SmartHome_DDoSSnapshot_Data/good_RESOURCES_OCCUPANCY_HACKED_False.csv' 'SmartHome_DDoSSnapshot_Data/good_RESOURCES_OCCUPANCY_HACKED_True.csv' "
                {context}
                Question: {question}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs = """\
                Write only the following string and no other words, do not start your response with Sure.
                "Ok, I will show you the pandas dataset according to your request. I will invoke FUNC_CALL dynabicagenttools.show_outlier_ips_usage Params 'SmartHome_DDoSSnapshot_Data/DATASET_LOGS_HACKED_False.csv' 'SmartHome_DDoSSnapshot_Data/DATASET_LOGS_HACKED_True.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS = """\
                Write only the following string and no other words, do not start your response with Sure.
                "Ok, I will show you the logs dataset by invoking FUNC_CALL dynabicagenttools.showLastNGetQueriesFromTopM_demandingIPs Params {params} "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests="""\
                Write only the following string and no other words, do not start your response with Sure.
                "I will invoke FUNC_CALL dynabicagenttools.showComparativeColumnsDatasets Params 'RAGSupport/dataForRAG/SmartHome_DDoSSnapshot_Data/DATASET_LOGS_HACKED_True.csv' 'SmartHome_DDoSSnapshot_Data/DATASET_LOGS_HACKED_False.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert = """\
                Write only the following string and no other words, do not start your response with Sure. Do not write like I provided you the code.
                '''
                df1 = pd.read_csv("SmartHome_DDoSSnapshot_Data/FIREWALL_PROCESSES.csv")
        
                new_row = ('IP': {param_ip}, 'NAME': {param_name}, 'DATE': datetime.now(), 'BLOCKED':1)
        
                df1=pd.concat([df1, pd.DataFrame([new_row])], ignore_index=True)
        
                df1.to_csv("SmartHome_DDoSSnapshot_Data/FIREWALL_PROCESSES.csv")
                '''
                {context}"""

        template_securityOfficer_instruction_rag_withsources_default = """
        Given the following extracted parts of a long document and a question, create a final answer with "SOURCES" that represent exactly the Source name and link given.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.
        
        QUESTION: {question}
        
        {summaries}
        
        FINAL ANSWER:"""

        llama_condense_template = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone Question:"
    elif USE_CASE == UseCase.Hospital:
        template_securityOfficer_system_prompt = """\
        Consider that I'm a beginner in networking and security things. \n
        Give me a concise answer with a single step at a time.  If there is only one step, do not write Step 1.\n
        Limit your response to maximum 2000 words.
        Do not provide any additional text or presentation. 
        If possible use concrete names of software or tools that could help.
        Use emoticons in your responses.
        """

        template_securityOfficer_instruction_rag_nosources_default = """\
        Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
        {context}
        Question: {question}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization = """\
                Write only the following string and no other words, do not start your response with Sure. 
                The DICOM and X-Ray are handled by the PACS Server at 192.168.61.0 port 24, I will show you two histograms of usage by invoking the following code\n\n FUNC_CALL dynabicagenttools.showResourceUtilizationComparison_v2 Params 'Hospital_DDoSSnapshot_Data/good_RESOURCES_OCCUPANCY_HACKED_False.csv' 'RAGSupport/dataForRAG/Hospital_DDoSSnapshot_Data/good_RESOURCES_OCCUPANCY_HACKED_True.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs = """\
                Write only the following string and no other words, do not start your response with Sure.
                "Ok, I will show you the pandas dataset according to your request. I will invoke FUNC_CALL dynabicagenttools.show_outlier_ips_usage Params 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_False.csv' 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_True.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS = """\
                Write only the following string and no other words, do not start your response with Sure.
                "Ok, I will show you the logs dataset by invoking FUNC_CALL dynabicagenttools.showLastNGetQueriesFromTopM_demandingIPs Params {params} "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests = """\
                Write only the following string and no other words, do not start your response with Sure.
                "I will invoke FUNC_CALL dynabicagenttools.showComparativeColumnsDatasets Params 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_True.csv' 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_False.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert = """\
                Write only the following string and no other words, do not start your response with sure.
                "Ok, I will invoke first FUNC_CALL dynabicagenttools.firewallUpdate Params {params} to analyze the request first".
                {context}"""

        template_securityOfficer_instruction_rag_withsources_default = """
        Given the following extracted parts of a long document and a question, create a final answer with "SOURCES" that represent exactly the Source name and link given.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.

        QUESTION: {question}

        {summaries}

        FINAL ANSWER:"""

        llama_condense_template = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone Question:"
    else:
        assert False, "Unknown use case"

# Set the system prompts based on the use case
set_templates()
