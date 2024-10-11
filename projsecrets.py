import os

project_path = os.environ.get('CYBERGUARDIAN_PATH', None)
project_path_UI_folder = os.path.join(project_path, "UI")
project_path_LLM_folder = os.path.join(project_path, "LLM")

import sys
# Append some local paths to the sys.path
sys.path.append(os.path.join(project_path, project_path_UI_folder))
sys.path.append(os.path.join(project_path, project_path_LLM_folder))

TOKEN_CODE_EXEC_CONFIRM = 'exec_code_confirm'
HOOK_FUNC_NAME_TOKEN = "hook_call"

assert project_path is not None, ("The project path is not set in the environment variables. Define it with "
                                  "CYBERGUARDIAN_PATH name and the path to the project as value "
                                  "(e.g. where you cloned the project from git.")

cached_client = None
cached_collection = None
cached_database = None

os.environ["MONGODB_HOST"] = "dynabicchatbot.n5fwe4p.mongodb.net"
os.environ["MONGODB_PASSWORD"] = "Arbori2009"
os.environ["MONGODB_USER"] = "paduraru2009"
os.environ["MONGODB_DATABASE"] = "dynabicChatbot"
os.environ["MONGODB_CLIENT"] = ""
os.environ["MONGODB_COLLECTION"] = "dynabicChatbot"
os.environ["TAVILY_API_KEY"] = "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"
