export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="path_to_DeepAnalyze-8B"
MODEL_NAME="DeepAnalyze-8B"

EVAL_MODEL_PATH="model path or name"  # Evaluation model path (for LLM evaluation)
# TASK_NAME="tablebench"  # Task name, e.g., tatqa, wikitq
TRAIN_TYPE="sft"  # Training type, e.g., grpo, ppo, sft
MODEL_SIZE="8b"    # Model size, e.g., 3b, 7b
TENSOR_PARALLEL_SIZE=1 # Tensor parallel size, to match attention head count
BATCH_SIZE=256  # Batch size
MAX_TOKENS=32000     # Maximum tokens for model generation
EVAL_MODE="standard"   # Evaluation mode: standard(exact match), llm(LLM only), combined(exact match + LLM)
LLM_EVAL_BATCH_SIZE=32  # LLM evaluation batch size

for TASK_NAME in ottqa tatqa finqa  hybridqa  multihiertt   tablebench hitab wikitq fetaqa aitqa feverous totto wikisql tabfact
do
  # ===================== Path Configuration ============================
  BASE_PATH="$(pwd)"  # Use current directory as base path
  INFER_SCRIPT="${BASE_PATH}/tests_our/${TASK_NAME}.py"

  # Select different evaluation script paths based on evaluation mode
  if [ "$EVAL_MODE" = "standard" ]; then
    EVAL_SCRIPT="${BASE_PATH}/tests_our/eval/${TASK_NAME}_eval.py"
  elif [ "$EVAL_MODE" = "llm" ]; then
    EVAL_SCRIPT="${BASE_PATH}/tests_our/llm_eval/${TASK_NAME}_eval.py"
  else  # combined
    EVAL_SCRIPT="${BASE_PATH}/tests_our/llm_eval/${TASK_NAME}_combined_eval.py"
  fi

  # Auto-generate output file paths
  PRED_OUTPUT="${BASE_PATH}/results/${MODEL_NAME}/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}.json"
  INFER_LOG="${BASE_PATH}/results/${MODEL_NAME}/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_infer.log"

  # Evaluation output paths differentiated by evaluation type
  if [ "$EVAL_MODE" = "standard" ]; then
    EVAL_OUTPUT="${BASE_PATH}/results/${MODEL_NAME}/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_eval_results.json"
  elif [ "$EVAL_MODE" = "llm" ]; then
    EVAL_OUTPUT="${BASE_PATH}/results/${MODEL_NAME}/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_llm_eval_results.json"
    EVAL_LOG="${BASE_PATH}/results/${MODEL_NAME}/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_llm_eval.log"
  else  # combined
    EVAL_OUTPUT="${BASE_PATH}/results/${MODEL_NAME}/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_combined_eval_results.json"
    EVAL_LOG="${BASE_PATH}/results/${MODEL_NAME}/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_combined_eval.log"
  fi

  # Ensure output directories exist
  mkdir -p "$(dirname "$PRED_OUTPUT")"
  mkdir -p "$(dirname "$INFER_LOG")"
  mkdir -p "$(dirname "$EVAL_OUTPUT")"

  # ===================== Run Inference ============================
  echo "Running inference with VLLM..."
  python "$INFER_SCRIPT" \
      --model_path "$MODEL_PATH" \
      --output_file "$PRED_OUTPUT" \
      --log_file "$INFER_LOG" \
      --base_path "$BASE_PATH" \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
      --batch_size $BATCH_SIZE \
      --max_tokens $MAX_TOKENS \
      --temperature 0.0

  if [ $? -ne 0 ]; then
    echo "Inference failed"
    exit 1
  fi

  # ===================== Run Evaluation ============================
  echo "Running evaluation with mode: $EVAL_MODE..."

  if [ "$EVAL_MODE" = "standard" ]; then
    # Use standard evaluation
    echo "Using standard evaluation"
    python "$EVAL_SCRIPT" \
        --results_file "$PRED_OUTPUT" \
        --output_file "$EVAL_OUTPUT" \
        --base_path "$BASE_PATH" 
  elif [ "$EVAL_MODE" = "llm" ]; then
    # Use LLM evaluation
    echo "Using LLM-based evaluation with model: $EVAL_MODEL_PATH"
    python "$EVAL_SCRIPT" \
        --results_file "$PRED_OUTPUT" \
        --output_file "$EVAL_OUTPUT" \
        --model_path "$EVAL_MODEL_PATH" \
        --log_file "$EVAL_LOG" \
        --base_path "$BASE_PATH" \
        --batch_size $LLM_EVAL_BATCH_SIZE \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE
  else  # combined
    # Use combined evaluation
    echo "Using combined evaluation (exact match + LLM) with model: $EVAL_MODEL_PATH"
    python "$EVAL_SCRIPT" \
        --results_file "$PRED_OUTPUT" \
        --output_file "$EVAL_OUTPUT" \
        --model_path "$EVAL_MODEL_PATH" \
        --log_file "$EVAL_LOG" \
        --base_path "$BASE_PATH" \
        --batch_size $LLM_EVAL_BATCH_SIZE \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --evaluation_mode "combined"
  fi

  if [ $? -ne 0 ]; then
    echo "Evaluation failed"
    exit 1
  fi

  echo "Testing and evaluation completed successfully!"
done