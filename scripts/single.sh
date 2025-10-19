export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MASTER_PORT=12345

BASE_MODEL="PATH_TO_MODEL_ADD_VOCAB"
MODEL_SINGLE_ABILITY_PATH="PATH_TO_SAVE_MODEL"
DATA_DIR="PATH_TO_DataScience-Instruct"

# Make sure you are in directory ./deepanalyze/ms-swift/
swift sft \
    --model "${BASE_MODEL}" \
    --train_type "full" \
    --dataset \
        "${DATA_DIR}/reasoning/SKGInstruct_199989.json" \
        "${DATA_DIR}/reasoning/TableQA_distillation_39301.json" \
        "${DATA_DIR}/reasoning/TableQA_refinement_39301.json" \
        "${DATA_DIR}/reasoning/TableGPT_29448.json" \
        "${DATA_DIR}/reasoning/file_database_3833.json" \
        "${DATA_DIR}/reasoning/file_csv_3007.json" \
        "${DATA_DIR}/reasoning/file_xlsx_3663.json" \
        "${DATA_DIR}/reasoning/file_any_2520.json" \
        "${DATA_DIR}/reasoning/math_20000.json" \
        "${DATA_DIR}/reasoning/code_20000.json" \
        "${DATA_DIR}/reasoning/science_20000.json" \
        "${DATA_DIR}/reasoning/instruction_following_20000.json" \
        "${DATA_DIR}/reasoning/other_19998.json" \
    --torch_dtype "bfloat16" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 4 \
    --packing true \
    --eval_steps 50 \
    --save_steps 50 \
    --logging_steps 1 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 3 \
    --response_prefix "" \
    --save_only_model false \
    --output_dir "${MODEL_SINGLE_ABILITY_PATH}" \
    --deepspeed "zero3" \
    --use_liger_kernel true \
    --attn_impl "flash_attn" \
    --model_type "deepseek_r1_distill"