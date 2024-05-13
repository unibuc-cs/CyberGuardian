import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)#, token=os.environ['HF_TOKEN'])
tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({"pad_token" : "<|reserved_special_token_0|>"})
#tokenizer.pad_token_id = tokenizer.eos_token_id
#tokenizer.padding = 'longest'
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})#, token=os.environ['HF_TOKEN'])
#model.config.pad_token_id = tokenizer.pad_token_id

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

model.config.pad_token_id = tokenizer.pad_token_id

print(f"Pad Token id: {tokenizer.pad_token_id} and Pad Token: {tokenizer.pad_token}")
print(f"EOS Token id: {tokenizer.eos_token_id} and EOS Token: {tokenizer.eos_token}")

from peft import LoraConfig
import os
os.environ["WANDB_DISABLED"] = "true"

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

#print(model)
from datasets import load_dataset
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}{EOS_TOKEN}"
    return [text]


data = load_dataset("unibuc-cs/CyberGuardian-dataset", 'docs')
#Data = Data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

def formatting_func_cyber(example):
    formatted_text = []
    for textStuff in example['text']:
        formatted_text.append(textStuff)
    return formatted_text


print(data)
batch_size=4
batch = data['train'].select(range(4))
print(formatting_func_cyber(batch))

import transformers
from trl import SFTTrainer

#print(model)
from datasets import load_dataset
EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func_quotes(example):
    # Create a list to store the formatted texts for each item in the example
    formatted_texts = []

    # Iterate through each example in the batch
    for quote, author in zip(example['quote'], example['author']):
        # Format each example as a prompt-response pair
        formatted_text = f"Quote: {quote}\nAuthor: {author}{EOS_TOKEN}"
        formatted_texts.append(formatted_text)

    # Return the list of formatted texts
    return formatted_texts


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    tokenizer=tokenizer,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=300,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        #padding='max_length',
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func_cyber,# formatting_prompts_func_quotes,
)
trainer.train()