# Full Context Training

This folder contains scripts for full context training. The idea is to train for a few steps with the given parameters of `train_batch_size`, `policy_mini_batch_size`, `micro_forward_batch_size_per_gpu`, `micro_train_batch_size_per_gpu`, etc with the specified maximum sequence length catch OOMs early.

The maximum sequence length is simply `generator.max_input_length` + `generator.sampling_params.max_generate_length`. Thus, we will perform `trainer.num_dummy_steps` number of iterations with batches of size `(trainer.train_batch_size, generator.max_input_length + generator.sampling_params.max_generate_length)`

## Usage

```bash
bash scripts/full_context/run.sh
```