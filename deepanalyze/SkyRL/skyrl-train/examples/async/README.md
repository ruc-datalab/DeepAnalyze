# Async Training Example

One-step off-policy GRPO for Qwen2.5-1.5B-Instruct on GSM8K.

## Usage

```bash 
# prepare the dataset
uv run -- python examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

export WANDB_API_KEY=<your_key_here>

bash examples/async/async_run_gsm8k.sh
```

For more details, refer to the [documentation](https://skyrl.readthedocs.io/en/latest/tutorials/async.html)
