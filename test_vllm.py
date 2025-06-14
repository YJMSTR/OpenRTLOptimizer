import openai

def main():
    """
    An interactive client to send optimization requests to the running vLLM server.
    """
    # Configure the OpenAI client to point to your local vLLM server
    client = openai.OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="vllm"  # The API key can be a dummy string for local vLLM
    )

    print("\n--- vLLM Interactive Client ---")
    print("Your vLLM server is running in the background.")
    print("Type 'quit' or 'exit' in the instruction prompt to end the session.")

    while True:
        print("\n" + "="*50)
        
        # Get user instruction
        instruction_prompt = "Enter optimization instruction:\n> "
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

        # Format the prompt in the new "messages" format for the Chat API
        messages = [
            {
                "role": "user",
                "content": f"""### Instruction:
{instruction}

### Input:
{input_verilog}

### Response:
"""
            }
        ]
        
        print("\n--- Sending request to vLLM server ---")

        try:
            # Send the request to the vLLM server's Chat Completions endpoint
            response = client.chat.completions.create(
                model="verilog-optimizer",
                messages=messages,
                max_tokens=2048,
                temperature=0.0,
                stream=True
            )
            
            print("\n--- vLLM Server Response (Streaming) ---")
            
            full_response = ""
            for chunk in response:
                token = chunk.choices[0].delta.content
                if token:  # Ensure token is not None
                    print(token, end="", flush=True)
                    full_response += token
            
            print("\n" + "="*50)

        except openai.APIConnectionError as e:
            print("\n--- ERROR ---")
            print("Could not connect to the vLLM server.")
            print("Please ensure the vLLM server is running in another terminal.")
            print(f"Error details: {e.__cause__}")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            break


if __name__ == "__main__":
    main() 