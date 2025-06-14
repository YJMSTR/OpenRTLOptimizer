import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer

def main():
    """
    Loads a model and enters an interactive loop to optimize Verilog code queries from the user,
    streaming the output token by token.
    """
    # --- 1. Configuration ---
    base_model_path = "models/Qwen3-30B-A3B"
    adapter_path = "models/Verilog-Optimizer-Qwen3-30B"
    
    print("Initializing model... (This may take a few minutes)")

    # Use 4-bit quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    # --- 2. Load Model and Tokenizer (runs only once) ---
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # --- 3. Initialize Text Streamer for real-time output ---
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    print("\n--- Interactive Verilog Optimizer Ready ---")
    print("Type 'quit' or 'exit' in the instruction prompt to end the session.")

    # --- 4. Interactive Loop ---
    while True:
        print("\n" + "="*50)
        
        # Get user instruction
        instruction_prompt = "Enter optimization instruction (e.g., 'Optimize for area and power'):\n> "
        instruction = input(instruction_prompt)
        if instruction.lower() in ["quit", "exit"]:
            print("Exiting.")
            break
        
        # Get user Verilog code
        print("\nEnter Verilog code to optimize (type 'EOF' on a new line to finish):")
        input_lines = []
        while True:
            line = input()
            if line.strip().upper() == 'EOF':
                break
            input_lines.append(line)
        
        input_verilog = "\n".join(input_lines)
        if not input_verilog:
            print("Warning: No Verilog code provided. Continuing.")
            continue

        # Format the prompt
        prompt = f"""### Instruction:
{instruction}

### Input:
{input_verilog}

### Response:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        print("\n--- Model Output (Streaming) ---")
        
        # Generate and stream the output
        _ = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=2048,  # A safe upper limit for generated code
            do_sample=False
        )
        print("\n" + "="*50)

if __name__ == "__main__":
    main() 