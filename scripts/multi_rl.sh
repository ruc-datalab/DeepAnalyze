export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=8
MODEL_COLDSTART_PATH="PATH_TO_SAVE_MODEL"
FINAL_MODEL_PATH="PATH_TO_SAVE_MODEL"
DATA_DIR="PATH_TO_DataScience-Instruct"

INFERENCE_BACKEND="vllm"

# Make sure you are in directory ./deepanalyze/SkyRL/skyrl-train/
python -m examples.deepanalyze.main_deepanalyze \
    trainer.algorithm.advantage_estimator="grpo" \
    trainer.epochs=1 \
    data.train_data="[
        \"${DATA_DIR}/RL/qa.parquet\",
        \"${DATA_DIR}/RL/datatask.parquet\",
        \"${DATA_DIR}/RL/reseach.parquet\"
    ]" \
    trainer.policy.model.path="${MODEL_COLDSTART_PATH}" \
    trainer.placement.colocate_all=true \
    trainer.strategy="fsdp2" \
    trainer.policy.fsdp_config.cpu_offload=true \
    trainer.ref.fsdp_config.cpu_offload=true \
    trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
    trainer.placement.ref_num_gpus_per_node=${NUM_GPUS} \
    generator.num_inference_engines=${NUM_GPUS} \
    generator.inference_engine_tensor_parallel_size=1 \
    trainer.train_batch_size=256 \
    trainer.micro_forward_batch_size_per_gpu=16 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.max_prompt_length=8000 \
    generator.max_input_length=32768 \
    generator.sampling_params.max_generate_length=32768 \
    trainer.policy.optimizer_config.lr=5e-7 \
    trainer.policy_mini_batch_size=256 \
    trainer.algorithm.use_kl_loss=false \
    generator.backend="${INFERENCE_BACKEND}" \
    generator.run_engines_locally=true \
    generator.weight_sync_backend="nccl" \
    generator.async_engine=true \
    generator.batched=false \
    generator.use_conversation_multi_turn=false \
    generator.n_samples_per_prompt=5 \
    generator.gpu_memory_utilization=0.5 \
    generator.max_turns=30 \
    generator.sampling_params.temperature=0.0 \
    generator.sampling_params.top_p=0.95 \
    generator.sampling_params.stop_token_ids="[151676,151645]" \
    environment.env_class="deepanalyze" \
    environment.skyrl_gym.deepanalyze.workspace="${DATA_DIR}/RL/data/" \
    trainer.logger="[\"console\",\"tensorboard\"]" \
    trainer.project_name="deepanalyze" \
    trainer.run_name="deepanalyze_0912" \
    trainer.resume_mode="latest" \
    trainer.ckpt_path="${FINAL_MODEL_PATH}/ckpt" \
    trainer.export_path="${FINAL_MODEL_PATH}/export" \
    trainer.eval_batch_size=8 \
    trainer.eval_before_train=false \
    trainer.eval_interval=-1 \
    trainer.hf_save_interval=1 \
    trainer.ckpt_interval=1