import os

FUNC_CALL_SYSTEM_PROMPT = None # ""
DEFAULT_SYSTEM_PROMPT = None
DEFAULT_QUESTION_PROMPT = "Question: {question}"


from enum import Enum
from UI.demoSupport import UseCase, get_demo_usecase
from userUtils import (SecurityOfficerExpertise, getUserExpertiseStr,
                       Response_Preferences, Preference_Politely, Preference_Emojis, SecurityOfficer)

template_securityOfficer_system_prompt = None
template_securityOfficer_instruction_rag_nosources_default = None
template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization = None
template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs = None
template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS = None
template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests = None
template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert = None
template_securityOfficer_instruction_rag_withsources_default = None
llama_condense_template = None

TOKEN_DO_NOT_SHOW = "TOKEN_DO_NOT_SHOW"

# Set the templates based on the use case
def init_templates(userProfile: SecurityOfficer):
    global template_securityOfficer_system_prompt
    global template_securityOfficer_instruction_rag_nosources_default
    global template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization
    global template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs
    global template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS
    global template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests
    global template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert
    global template_securityOfficer_instruction_rag_withsources_default
    global llama_condense_template



    ###### The officer system prompt is defined by considering the user's expertise and preferences #####
    assert userProfile is not None, "User profile is not defined"

    str_prompt_SecurityOfficerExpertise = None
    if userProfile.expertise == SecurityOfficerExpertise.BEGINNER:
        str_prompt_SecurityOfficerExpertise = "Consider that I'm a beginner in networking and security things."
    elif userProfile.expertise == SecurityOfficerExpertise.MIDDLE:
        str_prompt_SecurityOfficerExpertise = "Consider that I'm an intermediate in networking and security things. Explain the steps in detail, but keep it simple."
    elif userProfile.expertise == SecurityOfficerExpertise.ADVANCED:
        str_prompt_SecurityOfficerExpertise = "Consider that I'm an advanced in networking and security things. Explain in very short details, keep it simple and to the point."
    else:
        assert False, "Unexpected user expertise"

    str_prompt_Response_Preferences = None
    if userProfile.preference == Response_Preferences.DETAILED:
        str_prompt_Response_Preferences = """Prefer long and detailed explanations. 
        If possible use concrete names of software or tools that could help."""
    elif userProfile.preference == Response_Preferences.CONCISE:
        str_prompt_Response_Preferences = "Give me a concise answer, step by step within 2000 words with a single step at a time.  If there is only one step, do not write Step 1."


    str_prompt_Preference_Emojis = None
    if userProfile.emojis == Preference_Emojis.USE_EMOJIS:
        str_prompt_Preference_Emojis = "Use emoticons in your response."
    elif userProfile.emojis == Preference_Emojis.NO_EMOJIS:
        str_prompt_Preference_Emojis = "Do not use emoticons in your response."

    str_prompt_Preference_Politely = None
    if userProfile.politely == Preference_Politely.POLITE_PRESENTATION:
        str_prompt_Preference_Politely = "Be polite and professional."
    elif userProfile.politely == Preference_Politely.FRIENDLY_PRESENTATION:
        str_prompt_Preference_Politely = "Have a friendly tone and be patient."

    template_securityOfficer_system_prompt = f"""\
        {str_prompt_SecurityOfficerExpertise}
        {str_prompt_Response_Preferences}
        {str_prompt_Preference_Emojis}
        {str_prompt_Preference_Politely}
        """

    demo_usecase: UseCase = get_demo_usecase()
    if demo_usecase == UseCase.Default or demo_usecase == UseCase.SmartHome:
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
                "Ok, I will show you the logs dataset by invoking dynabicagenttools.showLastNGetQueriesFromTopM_demandingIPs. TOKEN_DO_NOT_SHOW FUNC_CALL dynabicagenttools.showLastNGetQueriesFromTopM_demandingIPs Params {params}"
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests="""\
                Write only the following string and no other words, do not start your response with Sure.
                "I will invoke dynabicagenttools.showComparativeColumnsDatasets. TOKEN_DO_NOT_SHOW FUNC_CALL dynabicagenttools.showComparativeColumnsDatasets Params 'SmartHome_DDoSSnapshot_Data/DATASET_LOGS_HACKED_True.csv' 'SmartHome_DDoSSnapshot_Data/DATASET_LOGS_HACKED_False.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert = """\
                Write only the following string and no other words, do not start your response with sure.
                "Ok, I will call dynabicagenttools.firewallUpdate to analyze the request first. TOKEN_DO_NOT_SHOW: FUNC_CALL dynabicagenttools.firewallUpdate Params {params}".
                {context}"""

        template_securityOfficer_instruction_rag_withsources_default = """
        Given the following extracted parts of a long document and a question, create a final answer with "SOURCES" that represent exactly the Source name and link given.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.
        
        QUESTION: {question}
        
        {summaries}
        
        FINAL ANSWER:"""

        llama_condense_template = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.\n\nChat History:\n{chat_history}\nFollow Up Input: {question}\nStandalone Question:"
    elif demo_usecase == UseCase.Hospital:
        template_securityOfficer_instruction_rag_nosources_default = """\
        Use the following pieces of context to answer the question. If no context provided, answer like a AI assistant.
        {context}
        Question: {question}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_resourceUtilization = """\
                Write only the following string and no other words, do not start your response with Sure. 
                The DICOM and X-Ray are handled by the PACS Server at 192.168.61.0 port 24, I will show you two histograms of usage by invoking dynabicagenttools.showResourceUtilizationComparison_v2. TOKEN_DO_NOT_SHOW FUNC_CALL dynabicagenttools.showResourceUtilizationComparison_v2 Params 'Hospital_DDoSSnapshot_Data/good_RESOURCES_OCCUPANCY_HACKED_False.csv' 'Hospital_DDoSSnapshot_Data/good_RESOURCES_OCCUPANCY_HACKED_True.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_devicesByIPLogs = """\
                Write only the following string and no other words, do not start your response with Sure.
                "Ok, I will show you the pandas dataset according to your request. I will invoke FUNC_CALL dynabicagenttools.show_outlier_ips_usage Params 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_False.csv' 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_True.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_topDemandingIPS = """\
                Write only the following string and no other words, do not start your response with Sure.
                "Ok, I will show you the logs dataset by invoking dynabicagenttools.showLastNGetQueriesFromTopM_demandingIPs. TOKEN_DO_NOT_SHOW FUNC_CALL dynabicagenttools.showLastNGetQueriesFromTopM_demandingIPs Params {params} "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_comparisonMapRequests = """\
                Write only the following string and no other words, do not start your response with Sure.
                "I will invoke dynabicagenttools.showComparativeColumnsDatasets TOKEN_DO_NOT_SHOW FUNC_CALL dynabicagenttools.showComparativeColumnsDatasets Params 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_True.csv' 'Hospital_DDoSSnapshot_Data/DATASET_LOGS_HACKED_False.csv' "
                {context}"""

        template_securityOfficer_instruction_rag_nosources_funccalls_firewallInsert = """\
                Write only the following string and no other words, do not start your response with sure.
                "Ok, I will call dynabicagenttools.firewallUpdate to analyze the request first. TOKEN_DO_NOT_SHOW: FUNC_CALL dynabicagenttools.firewallUpdate Params {params}".
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
