# Launches sglang server for Qwen2.5-1.5B-Instruct on 4 GPUs.
# bash examples/remote_inference_engine/run_sglang_server.sh
set -x

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --isolated --extra sglang -m \
    skyrl_train.inference_engines.sglang.sglang_server \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --tp 4 \
    --host 127.0.0.1 \
    --port 8001 \
    --context-length 4096 \
    --dtype bfloat16