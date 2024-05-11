# DynabicChatbot

## TODO:
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