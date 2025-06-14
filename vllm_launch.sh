python -m vllm.entrypoints.openai.api_server \
    --model ./models/Verilog-Optimizer-Qwen3-30B-Merged \
    --served-model-name verilog-optimizer \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.75 \
    --max-num-seqs 128 \
    --disable-log-stats 