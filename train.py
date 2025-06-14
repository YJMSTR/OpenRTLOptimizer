import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import torch.nn as nn

# Custom Trainer to handle device placement issues with device_map="auto"
class CustomSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Pop labels from inputs to avoid the model's internal loss calculation.
        labels = inputs.pop("labels")

        # Get model outputs (logits).
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Move labels to the same device as logits.
        labels = labels.to(logits.device)

        # Standard Causal LM loss calculation.
        # Shift so that tokens < n predict n.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens.
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # Ensure shift_labels is on the same device as shift_logits.
        shift_labels = shift_labels.to(shift_logits.device)

        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss

# 1. Model and Tokenizer Configuration
model_name = "Qwen/Qwen3-30B-A3B"
# We will load the model from the local directory where it's being downloaded
local_model_path = "models/Qwen3-30B-A3B"
dataset_path = "./OptimizeRules/finetune_dataset.jsonl"
new_model_name = "models/Verilog-Optimizer-Qwen3-30B"

# 2. QLoRA Configuration (for 4-bit quantization)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# 3. LoRA Configuration
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[ # Target modules can vary based on model architecture
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# 4. Load Base Model
print(f"Loading base model from: {local_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    quantization_config=bnb_config,
    device_map="auto", # Re-enabled for model parallelism
    trust_remote_code=True # Qwen models may require this
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# 5. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 6. Load and Format Dataset
def format_instruction(sample):
    # This function formats the input sample into a single string for training
    # The format should ideally match the model's expected chat or instruction format
    return [f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""]

dataset = load_dataset("json", data_files=dataset_path, split="train")

# 7. Training Arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5, # Small dataset, more epochs
    per_device_train_batch_size=1, # Can fit on a 4090 with QLoRA
    gradient_accumulation_steps=4, # Effective batch size = 1 * 4 GPUs * 4 acc_steps = 16
    gradient_checkpointing=True, # Use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=5,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True, # 4090 supports bf16 for better stability
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none", # Disable wandb integration
)

# 8. Initialize SFTTrainer
trainer = CustomSFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_arguments,
    formatting_func=format_instruction,
)

# 9. Start Training
print("Starting training...")
trainer.train()

# 10. Save the fine-tuned model
print(f"Saving fine-tuned model to ./{new_model_name}")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

print("Training finished!") 