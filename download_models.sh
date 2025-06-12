git clone https://hf-mirror.com/ByteDance-Seed/BAGEL-7B-MoT models/BAGEL-7B-MoT && cd models/BAGEL-7B-MoT && git lfs pull && cd ../..
git clone https://hf-mirror.com/vidore/colqwen2-v1.0 models/colqwen2-v1.0 && cd models/colqwen2-v1.0 && git lfs pull && cd ../..
git clone https://hf-mirror.com/vidore/colqwen2-base models/colqwen2-base && cd models/colqwen2-base && git lfs pull && cd ../..
git clone https://hf-mirror.com/lightonai/MonoQwen2-VL-v0.1 models/MonoQwen2-VL-v0.1 && cd models/MonoQwen2-VL-v0.1 && git lfs pull && cd ../..

if [ ! -f "models/Qwen3-30B-A3B-128K-Q4_K_M.gguf" ]; then
    wget https://hf-mirror.com/unsloth/Qwen3-30B-A3B-128K-GGUF/resolve/main/Qwen3-30B-A3B-128K-Q4_K_M.gguf -O models/Qwen3-30B-A3B-128K-Q4_K_M.gguf
fi

