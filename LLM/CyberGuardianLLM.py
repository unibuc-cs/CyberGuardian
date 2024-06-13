# To the contribution of this development, the following sources were used: Hugging Face, Accelerate, and the Transformers library
# The documentation and tutorials helped me to understand the code and adapt it to my needs.

import json
import logging
import math
import os
import random
import argparse
from itertools import chain
from pathlib import Path
from typing import Tuple, Dict
from peft import LoraConfig, PeftModel

import datasets
import torch
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import set_seed, load_and_quantize_model
from datasets import load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig,
)

logger = logging.getLogger("CYBERGUARDIAN_LOGGER") #get_logger(__name__) # The corret way would be to use the get_logger function from accelerate.logging, but it is not available in this environment

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class CyberGuardianLLM:

    args: argparse.Namespace = None
    accelerator: Accelerator = None
    model: AutoModelForCausalLM = None
    tokenizer: AutoTokenizer = None
    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    train_dataset: datasets.Dataset = None
    eval_dataset: datasets.Dataset = None
    lr_scheduler: torch.optim.lr_scheduler = None
    optimizer: torch.optim.Optimizer = None
    config: AutoConfig = None
    total_batch_size: int = None
    checkpointing_steps: int = None
    terminators: list = None
    torch_dtype = None
    attn_implementation = None


    def __init__(self, args: argparse.Namespace):
        self.args = args

    def do_training(self):
        self.prepare_accelerator()
        self.load_model_and_tokenizer()
        self.prepare_data()
        self.prepare_training()
        self._do_training()

    # Do inference on the model
    def do_inference(self, push_to_hub=False):
        self.load_model_and_tokenizer()
        self.prepare_inference(push_to_hub)

    def prepare_accelerator(self):
        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
        # in the environment
        accelerator_log_kwargs = {}

        if self.args.with_tracking:
            accelerator_log_kwargs["log_with"] = self.args.report_to
            accelerator_log_kwargs["project_dir"] = self.args.output_dir

        self.accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                                  **accelerator_log_kwargs)

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Create output dir
        if self.accelerator.is_main_process:
            if self.args.output_dir is not None:
                os.makedirs(self.args.output_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()

    def load_model_and_tokenizer(self):
        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        if self.args.config_name:
            self.config = AutoConfig.from_pretrained(
                self.args.config_name,
                trust_remote_code=self.args.trust_remote_code,
            )
        elif self.args.model_name_or_path:
            self.config = AutoConfig.from_pretrained(
                self.args.model_name_or_path,
                trust_remote_code=self.args.trust_remote_code,
            )
        else:
            self.config = CONFIG_MAPPING[self.args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, use_fast=not self.args.use_slow_tokenizer, trust_remote_code=self.args.trust_remote_code
            )
        elif self.args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name_or_path, use_fast=not self.args.use_slow_tokenizer, trust_remote_code=self.args.trust_remote_code
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        # Set the terminators
        self.terminators = [
        self.tokenizer.eos_token_id,
        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        if torch.cuda.get_device_capability()[0] >= 8:
            self.attn_implementation = "flash_attention_2"
            self.torch_dtype = torch.bfloat16
        else:
            self.attn_implementation = "eager"
            self.torch_dtype = torch.float16

        # TODO: add to args
        if self.args.model_name_or_path:

            # Solve quantization
            bnb_config = None
            if self.args.use_4bit_double_quant or self.args.use_4bit_quant:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.torch_dtype,
                    bnb_4bit_use_double_quant=False if self.args.use_4bit_double_quant is None
                    else self.args.use_4bit_double_quant
                )
            elif self.args.use_8bit_quant:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=self.torch_dtype,
                )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                quantization_config=bnb_config,
                from_tf=bool(".ckpt" in self.args.model_name_or_path),
                config=self.config,
                device_map="auto",
                #low_cpu_mem_usage=self.args.low_cpu_mem_usage,
                torch_dtype=self.torch_dtype,
                trust_remote_code=self.args.trust_remote_code,
                attn_implementation=self.attn_implementation
            )
        else:
            logger.info("Training new model from scratch")
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=self.args.trust_remote_code)


        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = self.model.get_input_embeddings().weight.shape[0]
        if len(self.tokenizer) > embedding_size:
            self.model.resize_token_embeddings(len(self.tokenizer))


    # Prepares  and loads the data for training and evaluation.
    # Tokenizes the data and generates chunks of block_size for causal model training.
    def prepare_data(self):
        # Get the datasets:  either provide your own CSV/JSON/TXT training and evaluation files (see below)
        # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
        # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
        # 'text' is found.
        # In distributed training, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.
        if self.args.dataset_useonline:
            # Downloading and loading a dataset from the hub.
            dataset_name = "unibuc-cs/CyberGuardian-dataset"
            dataset_config_name = "docs"
            raw_datasets = load_dataset(dataset_name, dataset_config_name)
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(dataset_name,
                                                          dataset_config_name,
                                                          split=f"train[:{self.args.validation_split_percentage}%]",
                                                          )
                raw_datasets["train"] = load_dataset(
                    dataset_name,
                    dataset_config_name,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                )
        else:
            data_files = {}
            dataset_args = {}
            extension = None
            if self.args.train_file is not None:
                data_files["train"] = self.args.train_file
                extension = self.args.train_file.split(".")[-1]
            if self.args.validation_file is not None:
                data_files["validation"] = self.args.validation_file
                extension = self.args.validation_file.split(".")[-1]
            if extension == "txt":
                extension = "text"
                dataset_args["keep_linebreaks"] = not self.args.no_keep_linebreaks
            raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{self.args.validation_split_percentage}%]",
                    **dataset_args,
                )
                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{self.args.validation_split_percentage}%:]",
                    **dataset_args,
                )

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        tokenizer = self.tokenizer
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with self.accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not self.args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if self.args.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > self.config.max_position_embeddings:
                logger.error(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({self.tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, self.config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, self.config.max_position_embeddings)
        else:
            if self.args.block_size > self.tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({self.args.block_size}) is larger than the maximum length for the model "
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.args.block_size, self.tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with self.accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=self.args.preprocessing_num_workers,
                load_from_cache_file=not self.args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

        self.train_dataset = lm_datasets["train"]
        self.eval_dataset = lm_datasets["validation"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(self.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {self.train_dataset[index]}.")

        # DataLoaders creation:
        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, collate_fn=default_data_collator,
            batch_size=self.args.per_device_train_batch_size
        )
        self.eval_dataloader = DataLoader(
            self.eval_dataset, collate_fn=default_data_collator,
            batch_size=self.args.per_device_eval_batch_size
        )

    # Prepare the optimizers, number of steps, parameters and everything else needed to train the model.
    def prepare_training(self):
        # Add the Lora adapter to the model
        peft_config = peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            #target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
        self.model.add_adapter(peft_config)

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)

        # Scheduler and math around the number of training steps.
        override_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            override_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.num_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.args.max_train_steps
            if override_max_train_steps
            else self.args.max_train_steps * self.accelerator.num_processes,
        )

        # Prepare everything with our `accelerator`.
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler \
            = self.accelerator.prepare(self.model, self.optimizer, self.train_dataloader,
                                       self.eval_dataloader, self.lr_scheduler)

        # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
        if self.accelerator.distributed_type == DistributedType.TPU:
            self.model.tie_weights()

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if override_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
        # Afterward we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        # Figure out how many steps we should save the Accelerator states
        self.checkpointing_steps = self.args.checkpointing_steps
        if self.checkpointing_steps is not None:
                if self.checkpointing_steps.isdigit():
                    self.checkpointing_steps = int(self.checkpointing_steps)
                elif "percent" in self.checkpointing_steps:
                    checkpointing_steps = self.checkpointing_steps.replace("percent", "")
                    checkpointing_steps = int(checkpointing_steps)*self.args.max_train_steps/100

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initialize automatically on the main process.
        if self.args.with_tracking:
            experiment_config = vars(self.args)
            # TensorBoard cannot log Enums, need the raw value
            experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
            self.accelerator.init_trackers("CyberGuardianLLMTracking", experiment_config)

        # Train!
        self.total_batch_size = (self.args.per_device_train_batch_size *
                            self.accelerator.num_processes *
                            self.args.gradient_accumulation_steps)

    def _do_training(self):
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(self.train_dataset)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")


        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(self.args.max_train_steps), disable=not self.accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        checkpoint_path = None
        if self.args.resume_from_checkpoint is not None:
            if self.args.resume_from_checkpoint.strip() != "last": # Exact path specified
                checkpoint_path = self.args.resume_from_checkpoint
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint from output_dir parameter
                dirs = [f.path for f in os.scandir(self.args.output_dir) if f.is_dir() and "checkpoint_" in f.name]
                dirs.sort(key=os.path.getctime)
                if len(dirs) == 0:
                    logger.error("!!!!!!!!!! No checkpoints found in output_dir")
                else:
                    path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                    checkpoint_path = path
                    path = os.path.basename(checkpoint_path)

            if checkpoint_path is not None:
                self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
                self.accelerator.load_state(checkpoint_path)
                # Extract `epoch_{i}` or `step_{i}`
                training_difference = os.path.splitext(path)[0]

                if "epoch" in training_difference:
                    starting_epoch = int(training_difference.replace("checkpoint_epoch_", "")) + 1
                    resume_step = None
                    completed_steps = starting_epoch * self.num_update_steps_per_epoch
                else:
                    # need to multiply `gradient_accumulation_steps` to reflect real steps
                    resume_step = int(training_difference.replace("checkpoint_step_", "")) * self.args.gradient_accumulation_steps
                    starting_epoch = resume_step // len(self.train_dataloader)
                    completed_steps = resume_step // self.args.gradient_accumulation_steps
                    resume_step -= starting_epoch * len(self.train_dataloader)
            else:
                self.args.resume_from_checkpoint = None

        # update the progress_bar if load from checkpoint
        progress_bar.update(completed_steps)

        # This is used to track the number of steps in this session including loaded checkpoint
        checkpointing_steps = None if not isinstance(self.checkpointing_steps, int) else self.checkpointing_steps

        # This is used to track the number of steps in this session only
        completed_steps_this_session = 0

        # Save the accelerator state with everything at each `checkpointing_steps` steps, and at the end
        def save_checkpoint():
            output_dir = f"checkpoint_step_{completed_steps}"
            if self.args.output_dir is not None:
                output_dir = os.path.join(self.args.output_dir, output_dir)
            self.accelerator.save_state(output_dir)

        # Iterate over the epochs
        for epoch in range(starting_epoch, self.args.num_train_epochs):
            # Set the model to training mode
            self.model.train()

            # We keep track of the loss at each epoch
            if self.args.with_tracking:
                total_loss = 0

            # Skip the first `n` batches in the dataloader when resuming from a checkpoint
            if self.args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                active_dataloader = self.accelerator.skip_first_batches(self.train_dataloader, resume_step)
            else:
                active_dataloader = self.train_dataloader

            # Iterate over the batches
            for step, batch in enumerate(active_dataloader):
                with self.accelerator.accumulate(self.model):
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss

                    # We keep track of the loss at each epoch
                    if self.args.with_tracking:
                        total_loss += loss.detach().float()

                    # Backward pass
                    self.accelerator.backward(loss)
                    # Update the weights
                    self.optimizer.step()
                    # Update the learning rate
                    self.lr_scheduler.step()
                    # Zero the gradients
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    completed_steps_this_session += 1


                # Save the model and the tokenizer every `checkpointing_steps` steps
                if checkpointing_steps is not None:
                    if completed_steps % self.checkpointing_steps == 0:
                        save_checkpoint()

                # If we have reached the maximum number of training steps, we stop the training
                if completed_steps_this_session >= self.args.max_train_steps:
                    break

            # Set the model to evaluation mode
            self.model.eval()
            losses = []
            # Iterate over the evaluation batches
            completed_eval_steps = 0
            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(**batch)

                # Gather the losses
                loss = outputs.loss
                losses.append(self.accelerator.gather_for_metrics(loss.repeat(self.args.per_device_eval_batch_size)))
                completed_eval_steps += 1

                if self.args.max_eval_steps is not None and completed_eval_steps >= self.args.max_eval_steps:
                    break

            losses = torch.cat(losses)
            # Calculate the perplexity
            try:
                eval_loss = torch.mean(losses)
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")

            logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

            if self.args.with_tracking:
                self.accelerator.log(
                    {
                        "perplexity": perplexity,
                        "eval_loss": eval_loss,
                        "train_loss": total_loss.item() / len(self.train_dataloader),
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

            # Save the state every epoch
            if self.args.checkpointing_steps == "epoch":
                output_dir = f"checkpoint_epoch_{epoch}"
                if self.args.output_dir is not None:
                    output_dir = os.path.join(self.args.output_dir, output_dir)
                self.accelerator.save_state(output_dir)

        # End of training, log
        if self.args.with_tracking:
            self.accelerator.end_training()

        # Save the model and the tokenizer to the output directory
        if self.args.output_dir is not None:
            # Wait for all processes to finish
            self.accelerator.wait_for_everyone()

            # Save the model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                self.args.output_dir, is_main_process=self.accelerator.is_main_process, save_function=self.accelerator.save
            )

            if self.accelerator.is_main_process:
                # Save the tokenizer
                self.tokenizer.save_pretrained(self.args.output_dir)

                save_checkpoint()

                # Save the perplexity statistics
                with open(os.path.join(self.args.output_dir, "all_results.json"), "w") as f:
                    json.dump({"perplexity": perplexity}, f)

    def prepare_inference(self, push_to_hub=False):
        assert self.args.pretrained_peft_adapter_dir is not None, "You must provide a pretrained (PEFT) model path for inference"

        # Merge adapter with base model
        self.model.load_adapter(self.args.pretrained_peft_adapter_dir)

        if push_to_hub:
            self.model.push_to_hub("unibuc-cs/CyberGuardian")

    def test_model(self, messages: list):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        self.tokenizer.chat_template = self.tokenizer.default_chat_template

        outputs = self.model.generate(**inputs, max_new_tokens=4096,
                             eos_token_id=self.terminators,
                             do_sample=True,
                             temperature=0.1,
                             top_p=0.9, )

        for i in range(1):
            print(self.tokenizer.decode(outputs[i], skip_special_tokens=False))

        a = 3
