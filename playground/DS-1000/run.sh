export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="path_to_DeepAnalyze-8B"

python run_deepanalyze.py --model $MODEL_PATH
python test_ds1000.py --model $MODEL_PATH