export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_PORT=12345

BASE_MODEL="PATH_TO_MODEL_ADD_VOCAB"
MODEL_SINGLE_ABILITY_PATH="PATH_TO_PREVIOUS_STAGE_MODEL"
MODEL_MULTI_ABILITY_PATH="PATH_TO_SAVE_MODEL"
DATA_DIR="PATH_TO_DataScience-Instruct"


# Make sure you are in directory ./deepanalyze/ms-swift/
swift sft \
    --model "${MODEL_SINGLE_ABILITY_PATH}" \
    --train_type "full" \
    --dataset \
        "${DATA_DIR}/interation/data_pipeline_3601.json" \
        "${DATA_DIR}/interation/data_preparation_3311.json" \
        "${DATA_DIR}/interation/data_cleaning_1616.json" \
        "${DATA_DIR}/interation/data_analysis_3936.json" \
        "${DATA_DIR}/interation/data_insight_1062.json" \
        "${DATA_DIR}/interation/research_database_818.json" \
        "${DATA_DIR}/interation/research_xlsx_848.json" \
        "${DATA_DIR}/interation/research_other_3505.json" \
        "${DATA_DIR}/interation/research_data_preparation_488.json" \
        "${DATA_DIR}/interation/research_data_analysis_1339.json" \
        "${DATA_DIR}/interation/research_data_insight_1351.json" \
        "${DATA_DIR}/interation/research_report_generation_4327.json" \
    --torch_dtype "bfloat16" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 32 \
    --packing true \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 1 \
    --max_length 32768 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 3 \
    --response_prefix "" \
    --save_only_model false \
    --output_dir "${MODEL_MULTI_ABILITY_PATH}" \
    --deepspeed "zero3" \
    --use_liger_kernel true \
    --attn_impl "flash_attn" \
    --model_type "deepseek_r1_distill"