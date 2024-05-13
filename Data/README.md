---
configs:
- config_name: docs
  data_files:
  - split: train
    path:
    - dataForTraining/db_pdfs.jsonl
- config_name: videos
  data_files:
  - split: train
    path:
    - dataForTraining/db_videos_clean.jsonl
- config_name: raw_pdfpapers
  data_files:
  - split: train
    path:
    - sources/pdfpapers.json
- config_name: raw_videos
  data_files:
  - split: train
    path:
    - sources/videos.json
- config_name: raw_markdown
  data_files:
  - split: train
    path: 
    - sources/webcontent.json

license: mit
language:
- en
tags:
- cybersecurity
- video transcripts
- arxiv papers
pretty_name: >-
  A cybersecurity LLM training dataset with latest research extracted from a collection of docs and youtube transcripts.
---

# A dataset for training LLM agents on latest research in the cybersecurity domain.

### 1. Purpose: fine-tune existing LLMs to incorporate novelties and general language understanding of the field.
### 2. Check our repository containing the CyberGuarding - an interactive chatbot for cybersecurity help, https://github.com/unibuc-cs/CyberGuardian 
This repository serves as a submodule for this one.

### 3. How to use the data in training your models?

There are three configs, one for each supported data type.
```
data_from_documents = load_dataset("unibuc-cs/CyberGuardian-dataset", 'docs')
data_from_videos = load_dataset("unibuc-cs/CyberGuardian-dataset", 'video')
data_from_markdown = load_dataset("unibuc-cs/CyberGuardian-dataset", 'markdown')
```

Note that there is only a split, the 'train' one. Its up to you to split as you wish the data.

### 4. Community: we hope that this will be an updated dataset along time.

### 4. Raw sources of data
The folder sources contain the JSON with resources for each type: video, docs, markdown content.
Custom folders can also be added as shown.

### 5. Export/Update from from Raw sources
Input your credentials in projsecrets.py then run source/DatasetUtils.py. Please read the script's parameters description for more de tails.
Optionally it creates a FAISS index for RAG purposes for instance.

This is how the dataForTraining was obtained, and how you can add your own raw data files using links to papers, youtube courses or presentations, etc., then just run the script.
A mongoDB is used internally (see secrets.py) to keep the results on a storage, if you want.

### TEch explanation or using scripts.

- You should run the git cloned version of this repo from an upper folder as current directory, e.g., **Project/Data**. Where **Data** is the name of the cloned github folder including all its files and folders directly.
- In your top **Project** directory add a **projsecrets.py** file where you need to fill your MongDB user details as below. Note that a free database is more than enough for current purposes..

```
os.environ["MONGODB_HOST"] = "" # FILL in Your server. E.g., dynabicchatbot.n5fwe4p.mongodb.net
os.environ["MONGODB_PASSWORD"] = "" # Your MongoDB password.
os.environ["MONGODB_USER"] = "" # Your MongoDB user.
os.environ["MONGODB_DATABASE"] = ""  # You create a database, to work with the rest of the project correctly we recommend you use dynabicChatbot name
os.environ["MONGODB_CLIENT"] = "" # Leave empty
```

# TODO before making it pubblic:
## ADD OUR CYBERGUARDIAN PAPER FOR CITATION !!
## (optional) IT WOULD BE NICE TO MAKE FAISS AND MONGODB support REALLY optional. But users can still handle this..

