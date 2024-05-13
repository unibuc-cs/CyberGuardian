import os

os.environ["MONGODB_HOST"] = "dynabicchatbot.n5fwe4p.mongodb.net"
os.environ["MONGODB_PASSWORD"] = "Arbori2009"
os.environ["MONGODB_USER"] = "paduraru2009"
os.environ["MONGODB_DATABASE"] = "dynabicChatbot"
os.environ["MONGODB_CLIENT"] = ""

cached_client = None
cached_database = None
cached_collection = None
