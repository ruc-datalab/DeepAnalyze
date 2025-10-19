set -x

# Colocated GRPO training+generation for Qwen2.5-32B-Instruct on GSM8K.
# export WANDB_API_KEY=<your_key_here>
# bash examples/gsm8k/run_32b_gsm8k.sh


DATA_DIR="$HOME/data/gsm8k"

# NOTE: the batch size can likely be increased for more throughput.
uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-32B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  trainer.placement.ref_num_nodes=2 \
  trainer.placement.policy_num_nodes=2 \
  generator.num_inference_engines=4 \
  generator.inference_engine_tensor_parallel_size=4 \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  trainer.critic_mini_batch_size=32 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=false \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=4 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="gsm8k-32b" \
  trainer.run_name="gsm8k_test_32b" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_32B_ckpt" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@