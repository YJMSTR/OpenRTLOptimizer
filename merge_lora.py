import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def merge_lora_model():
    """
    Merges the fine-tuned LoRA adapter with the base model and saves the resulting
    full model to a new directory. This is a necessary step before deploying with
    frameworks like vLLM.
    """
    # --- 1. Configuration ---
    base_model_path = "models/Qwen3-30B-A3B"
    adapter_path = "models/Verilog-Optimizer-Qwen3-30B"
    merged_model_path = "models/Verilog-Optimizer-Qwen3-30B-Merged"

    print("--- Starting LoRA Merge Process ---")
    print(f"Base model: {base_model_path}")
    print(f"LoRA adapter: {adapter_path}")
    print(f"Output directory: {merged_model_path}")

    # --- 2. Load Base Model and Tokenizer ---
    # We load the model in full precision (e.g., bfloat16) for the merge,
    # as quantization is best applied by the inference engine (like vLLM) later.
    print("\nLoading base model in bfloat16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto", # Use all GPUs to load the large model
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # --- 3. Load LoRA Adapter and Merge ---
    print(f"\nLoading LoRA adapter from {adapter_path}...")
    # This dynamically loads the LoRA layers onto the base model
    model_with_lora = PeftModel.from_pretrained(base_model, adapter_path)

    print("\nMerging the adapter weights into the base model...")
    # This performs the actual merge operation
    merged_model = model_with_lora.merge_and_unload()
    print("Merge complete.")

    # --- 4. Save Merged Model ---
    print(f"\nSaving the merged model and tokenizer to {merged_model_path}...")
    os.makedirs(merged_model_path, exist_ok=True)
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)

    print("\n--- Merge Process Finished Successfully! ---")
    print("Your full, standalone model is now ready for deployment at:")
    print(f"{os.path.abspath(merged_model_path)}")

if __name__ == "__main__":
    merge_lora_model() 