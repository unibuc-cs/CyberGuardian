# CyberGuardian chatbot

## Model loading, usage, demo
* Step 1: Check our huggingface repository and download the model
https://huggingface.co/datasets/unibuc-cs/CyberGuardianDataset

* Step 2: Use Ollama or llama.cpp to do local inference of the model, indepedently of the architecture you run on (e.g., you can use MacOS, CPU only, GPUs, etc.).

* Step 3: Install the UI/requirements.txt packages to run the project.

* Step 4: While the UI interface is currently built with Streamlit library, use **streamlit run UI/Main_Page.py** to start.

* Notes:
   - Check our video demos, create a profile, and test your skills.
   - To run our demo in the presentations, define a OS env variable: DEMO_USE_CASE=hospital

## Development 
We created a set of Pycharm confirmation to sustain development. If you load the project you will see various setups to speed to things.
For example:
  - Use **UITest_NO_LOGIN** to test interaction with the model in UI, skipping all the account creation or login.
  - Use **CyberGuardianLLM_Training** to start fine-tuning.
  - Dataset handling can be done using **DatasetUtils** config
  - To interact with the model and ask questions in a reproduceable way, use **main_qa**.

## TODO: 
 - Final paper ICSOFT cite 
 - Upload documentation folder with demos and paper, presentations.
 - Make both repositories public

Acknowledgements: This research was supported by European Union’s Horizon Europe research and innovation programme under grant agreement no. 101070455, [project DYNABIC](https://dynabic.eu), where we use the code for the Chat4Operator component.

