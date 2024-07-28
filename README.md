# CyberGuardian assistant

## Model loading, usage, demo
* Step 1: Check our huggingface repository and download the model
  https://huggingface.co/unibuc-cs/CyberGuardian


* Step 2: Use Ollama or llama.cpp to do local inference of the model, indepedently of the architecture you run on (e.g., you can use MacOS, CPU only, GPUs, etc.).

* Step 3: Install the UI/requirements.txt packages to run the project.

* Step 4: While the UI interface is currently built with Streamlit library, use **streamlit run UI/Main_Page.py** to start.

* Notes:
   - Check our video demos, create a profile, and test your skills.
   - To run our demo in the presentations, define a OS env variable: DEMO_USE_CASE=hospital
 
## Dataset
   - Our dataset is uploaded to https://huggingface.co/datasets/unibuc-cs/CyberGuardianDataset
   - Note that the repository contains scripts to inject your own data, update the dataset and even include your data in RAG.
   - Check the documentation in the Readme.

## Development 
We created a set of Pycharm confirmation to sustain development. If you load the project you will see various setups to speed to things.
For example:
  - Use **UITest_NO_LOGIN** to test interaction with the model in UI, skipping all the account creation or login.
  - Use **CyberGuardianLLM_Training** to start fine-tuning.
  - Dataset handling can be done using **DatasetUtils** config
  - To interact with the model and ask questions in a reproduceable way, use **main_qa**.

## Please cite our work properly if you find it useful in any way:
```
@conference{CyberGuardian,
author={Ciprian Paduraru. and Catalina Patilea. and Alin Stefanescu.},
title={CyberGuardian: An Interactive Assistant for Cybersecurity Specialists Using Large Language Models},
booktitle={Proceedings of the 19th International Conference on Software Technologies - ICSOFT},
year={2024},
pages={442-449},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0012811700003753},
isbn={978-989-758-706-1},
issn={2184-2833},
}
```

Acknowledgements: This research was supported by European Union’s Horizon Europe research and innovation programme under grant agreement no. 101070455, [project DYNABIC](https://dynabic.eu), where we use the code for the Chat4Operator component.

