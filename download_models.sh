#!/bin/bash
# This script downloads various LLMs from the Hugging Face Hub using the recommended 'huggingface-cli' tool.
# It uses hf_transfer for accelerated downloads.

# Exit immediately if a command exits with a non-zero status.
set -e

# Set HF_HUB_ENABLE_HF_TRANSFER to 1 to enable accelerated downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- Model Download Commands ---

echo "Downloading Qwen/Qwen3-30B-A3B..."
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir models/Qwen3-30B-A3B

echo "Downloading ByteDance-Seed/BAGEL-7B-MoT..."
huggingface-cli download ByteDance-Seed/BAGEL-7B-MoT --local-dir models/BAGEL-7B-MoT

echo "Downloading vidore/colqwen2-v1.0..."
huggingface-cli download vidore/colqwen2-v1.0 --local-dir models/colqwen2-v1.0

echo "Downloading vidore/colqwen2-base..."
huggingface-cli download vidore/colqwen2-base --local-dir models/colqwen2-base

echo "Downloading lightonai/MonoQwen2-VL-v0.1..."
huggingface-cli download lightonai/MonoQwen2-VL-v0.1 --local-dir models/MonoQwen2-VL-v0.1

echo "All models specified in the script have been downloaded successfully."


