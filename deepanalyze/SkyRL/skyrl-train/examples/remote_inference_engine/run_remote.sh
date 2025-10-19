set -x

# GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K using a standalone vllm server (at 127.0.0.1:8001)
# First run `uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k` to setup the dataset.
# then to launch the server, first run 
# bash examples/remote_inference_engine/run_vllm_server.sh
# then to start training, run
# bash examples/remote_inference_engine/run_remote.sh

DATA_DIR="$HOME/data/gsm8k"

BACKEND="vllm" # or "sglang"
TP=4

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
    data.train_data="['$DATA_DIR/train.parquet']" \
    data.val_data="['$DATA_DIR/validation.parquet']" \
    trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
    generator.run_engines_locally=False \
    generator.remote_inference_engine_urls="['127.0.0.1:8001']" \
    generator.inference_engine_tensor_parallel_size="$TP" \
    generator.backend="$BACKEND" \
    generator.sampling_params.temperature=0.6 \
    generator.sampling_params.top_p=0.95 \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.placement.colocate_all=False \
    trainer.placement.policy_num_gpus_per_node=4 \
    trainer.placement.ref_num_gpus_per_node=4 \
    trainer.strategy=fsdp2 \
    trainer.train_batch_size=64 \
    trainer.policy_mini_batch_size=64 \
    trainer.micro_forward_batch_size_per_gpu=20 \
    trainer.micro_train_batch_size_per_gpu=20 \
    trainer.logger="wandb" \
    trainer.project_name="skyrl" \
    trainer.run_name="skyrl-remote" \
    trainer.resume_mode=null \
    trainer.ckpt_path="$HOME/ckpts/remote_ckpt" \
    trainer.eval_batch_size=1024 \
    trainer.eval_before_train=true \
    trainer.eval_interval=5 \
    $@
