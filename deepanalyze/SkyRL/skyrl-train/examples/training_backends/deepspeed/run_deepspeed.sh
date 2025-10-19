set -x

# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K with DeepSpeed backend.
# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/training_backends/deepspeed/run_deepspeed.sh

uv run --isolated --extra vllm --extra deepspeed -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data="['$HOME/data/gsm8k/train.parquet']" \
  data.val_data="['$HOME/data/gsm8k/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=deepspeed \
  trainer.placement.policy_num_gpus_per_node=4 \
  trainer.placement.ref_num_gpus_per_node=4 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.train_batch_size=1024 \
  trainer.micro_train_batch_size_per_gpu=40 \
  trainer.micro_forward_batch_size_per_gpu=40 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy_mini_batch_size=256 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_ckpt_deepspeed" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@