set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-3B-Instruct on SearchR1 data.
# export WANDB_API_KEY=<your_key_here>
# bash examples/livecodebench/run_lcb.sh

DATA_DIR="$HOME/data/lcb"
train_data="['${DATA_DIR}/deepcoder_train.json']"
val_data="['${DATA_DIR}/test_livecodebench.json']"

# NOTE (sumanthrh): micro_train_batch_size and micro_forward_batch_size can be tuned
uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data=$train_data \
  data.val_data=$val_data \
  trainer.policy.model.path="Qwen/Qwen2.5-3B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.placement.policy_num_gpus_per_node=8 \
  trainer.placement.ref_num_gpus_per_node=8 \
  generator.num_inference_engines=2 \
  generator.inference_engine_tensor_parallel_size=4 \
  trainer.policy_mini_batch_size=4 \
  trainer.train_batch_size=16 \
  trainer.micro_forward_batch_size_per_gpu=16 \
  trainer.micro_train_batch_size_per_gpu=2 \
  trainer.max_prompt_length=29000 \
  generator.max_input_length=29000 \
  generator.sampling_params.max_generate_length=3000 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.ckpt_interval=100000 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=lcb \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  trainer.logger="wandb" \
  trainer.project_name="skyrl" \
  trainer.run_name="skyrlcode_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/lcb_3B_ckpt" \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@
