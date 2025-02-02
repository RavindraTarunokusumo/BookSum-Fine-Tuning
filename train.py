import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm
from zeus.monitor import ZeusMonitor  # import Zeus for energy measurement


# Set the seed for reproducibility.
np.random.seed(0)
torch.manual_seed(0)

# Define the function to load and preprocess the dataset.
def load_and_preprocess_dataset(dataset, tokenizer):

    # Define the preprocessing function
    def preprocess_function(examples):
        inputs = []
        labels = []
        for text, summ_list in zip(examples["text"], examples["summary"]):
            # Use the first available summary (if exists)
            summary_text = summ_list[0]["text"] if (isinstance(summ_list, list) and len(summ_list) > 0 and "text" in summ_list[0]) else ""
            prompt = f"Summarize the following text:\n\n{text}\n\nSummary: "
            full_text = prompt + summary_text
            tokenized_full = tokenizer(full_text, truncation=True, max_length=1024)
            tokenized_prompt = tokenizer(prompt, truncation=True, max_length=1024)
            input_ids = tokenized_full["input_ids"]
            labels_ids = input_ids.copy()
            prompt_length = len(tokenized_prompt["input_ids"])
            # Mask the prompt part in the labels so loss is computed only on the summary.
            labels_ids[:prompt_length] = [-100] * prompt_length
            inputs.append(input_ids)
            labels.append(labels_ids)
        return {"input_ids": inputs, "labels": labels}

    # Map preprocessing over the dataset splits.
    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )   

    return tokenized_datasets

def main():
    # 1. Model and tokenizer names
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"  # HF repo for the model

    # 2. Load pretrained tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 3. Load the BookSum dataset (using the chapter-level variant from "ubaada/booksum-complete-cleaned").
    dataset = load_dataset("ubaada/booksum-complete-cleaned", "chapters")

    # 4. Tokenize and preprocess the dataset
    tokenized_dataset = load_and_preprocess_dataset(dataset, tokenizer)

    # 5. Define the data collator
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_config)
    print("Trainable parameters:")
    model.print_trainable_parameters()

    # 6. Training arguments with FP16 and DDP
    training_args = TrainingArguments(
        output_dir="./deepseek_llama8b_booksum",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True,                   # use FP16 if supported
        logging_steps=100,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        seed=0,
        strategy="ddp",              # Use DistributedDataParallel for multi-GPU training
    )

    # 7. Data collator for padding
    def data_collator(features):
        return tokenizer.pad(features, return_tensors="pt")

    # 8. Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
    )
    # 9. Measure energy consumption with Zeus.
    # Only run the monitoring on the main process to avoid duplicate prints.
    # Here we check the environment variable "RANK" (set by Accelerate/torch.distributed).
    monitor = None
    if int(os.environ.get("RANK", "0")) == 0:
        # Monitor all GPUs visible (e.g., indices 0-3) and all CPU packages.
        monitor = ZeusMonitor(gpu_indices=[0, 1, 2, 3], cpu_indices=None, approx_instant_energy=True)
        monitor.begin_window("training")

    # 10. Start fine-tuning.
    trainer.train()

    # 11. End the measurement window (if monitoring) and print results.
    if monitor is not None:
        measurement = monitor.end_window("training")
        print(f"Training took {measurement.time:.2f} seconds and consumed {measurement.total_energy:.2f} Joules.")
        print("Detailed GPU energy consumption:", measurement.gpu_energy)
        if measurement.cpu_energy is not None:
            print("Detailed CPU energy consumption:", measurement.cpu_energy)

    # 12. Save the final model.
    model.save_pretrained("./deepseek_llama8b_booksum_final")

if __name__ == "__main__":
    main()