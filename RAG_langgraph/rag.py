from transformers import AutoTokenizer, AutoModelForCausalLM

qwen3_model_id = "unsloth/Qwen3-30B-A3B-128K-Q4_K_M.gguf"
qwen3_filename = "models/Qwen3-30B-A3B-128K-Q4_K_M.gguf"

qwen3_tokenizer = AutoTokenizer.from_pretrained(qwen3_model_id)
qwen3_model = AutoModelForCausalLM.from_pretrained(qwen3_model_id)

vl_model_path = "models/BAGEL-7B-MoT"
vl_tokenizer = AutoTokenizer.from_pretrained(vl_model_path)
vl_model = AutoModelForCausalLM.from_pretrained(vl_model_path)

rerank_model_path = "models/MonoQwen2-VL-v0.1"
retrieval_model_path = "models/colqwen2-v1.0"