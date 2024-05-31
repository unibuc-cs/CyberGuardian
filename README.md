# DynabicChatbot

## TODO: 
 - Write a proper documentat!
 - Difference in responses between EXPERT and begginer expertise + how risks from user evaluation are taken into consideration by the LLM  etc.
 - Sync https://huggingface.co/datasets/unibuc-cs/CyberGuardianDataset as a submodule , cleanup things here and use that subfolder!
 - Clarify the FinetuningModel.py and if LoRA weights can be uploaded on hugging face.
 - Change the names.
 - Maybe put the UI Interface here too ?
 - toolTypeSimilarity, params = self.similarityToTool(question)
 - Dockerize.
 - ETL for RAG for custom user data!
 - Fix belows logic:
``` 
        self.vector_index = vecstore.connect_to_vector_index
```

- FIX the python code added to history and not working properly afterwards! History is messing with LLM after sending a python code . Start from this:
```
    securityChatbot.ask_question(
        "Generate me a python code to insert in a pandas dataframe named Firewalls a new IP 10.20.30.40 as blocked under the name of IoTDevice", add_to_history=False)
```

