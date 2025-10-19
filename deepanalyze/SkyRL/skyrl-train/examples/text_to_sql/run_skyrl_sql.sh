set -x

# Colocated GRPO training+generation for Qwen2.5-Coder-7B-Instruct on SkyRL-SQL-653 data.
# Uses 1 node with 8 GPUs.
# huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
# export WANDB_API_KEY=<your_key_here>
# bash examples/text_to_sql/run_skyrl_sql.sh

# change these paths to your own
DATA_DIR="$HOME/data/sql"
DB_PATH="$HOME/data/sql/db_files/data"
CKPT_PATH="$HOME/ckpts/skyrl_sql_7B_ckpt"

NUM_GPUS=8
NUM_INFERENCE_ENGINES=2
TP_SIZE=4
MAX_INPUT_LENGTH=29000
MAX_GENERATE_LENGTH=3000
TRAIN_BATCH_SIZE=256

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
  trainer.algorithm.advantage_estimator="grpo" \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.policy.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
  trainer.epochs=30 \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.policy.fsdp_config.cpu_offload=false \
  trainer.ref.fsdp_config.cpu_offload=true \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.sequence_parallel_size=1 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$TP_SIZE \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.max_prompt_length=6000 \
  generator.max_input_length=$MAX_INPUT_LENGTH \
  generator.sampling_params.max_generate_length=$MAX_GENERATE_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy_mini_batch_size=256 \
  trainer.algorithm.use_kl_loss=false \
  trainer.ckpt_interval=60 \
  trainer.hf_save_interval=30 \
  trainer.dump_data_batch=true \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=false \
  environment.env_class=text2sql \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.gpu_memory_utilization=0.7 \
  generator.max_turns=6 \
  generator.use_conversation_multi_turn=false \
  generator.sampling_params.temperature=0.6 \
  generator.sampling_params.top_p=0.95 \
  environment.skyrl_gym.text2sql.db_path=$DB_PATH \
  trainer.logger="wandb" \
  trainer.project_name="skyrlsql" \
  trainer.run_name="skyrlsql_repro" \
  trainer.resume_mode=latest \
  trainer.ckpt_path=$CKPT_PATH \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  $@