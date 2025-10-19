PPO on GSM8K
=========================================

This example demonstrates how to run PPO training on GSM8K. See :doc:`../getting-started/quickstart` for a similar setup process for GRPO!

Dataset Preparation
-------------------

To download and prepare the GSM8K dataset, run the following script.

.. code-block:: bash

   uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

Training Configuration
----------------------

Next, let's set up the training configuration. You can find a complete example in ``examples/ppo/run_ppo.sh`` - we highlight some key parameters here:

.. code-block:: bash

   uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
      # Data setup
      data.train_data="['$HOME/data/gsm8k/train.parquet']" \
      data.val_data="['$HOME/data/gsm8k/validation.parquet']" \

      # Trainer and training algorithm
      trainer.algorithm.advantage_estimator="gae" \
      trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
      trainer.critic.model.path="Qwen/Qwen2.5-1.5B-Instruct" \

      # Model placement and training strategy (colocate or disaggregate, sharding, etc.)
      trainer.strategy=fsdp2 \
      trainer.placement.colocate_all=true \
      trainer.placement.policy_num_gpus_per_node=4 \
      trainer.placement.critic_num_gpus_per_node=4 \

      # Batch sizes for critic and policy forward and training passes
      trainer.policy_mini_batch_size=256 \
      trainer.critic_mini_batch_size=256 \
      trainer.micro_forward_batch_size_per_gpu=64 \
      trainer.micro_train_batch_size_per_gpu=64 \

      # Evaluation and checkpointing
      trainer.eval_batch_size=1024 \
      trainer.eval_before_train=true \
      trainer.eval_interval=5 \
      trainer.ckpt_interval=10 \

      # Generator setup for spinning up InferenceEngines
      generator.backend=vllm \
      generator.num_inference_engines=4 \
      generator.inference_engine_tensor_parallel_size=1 \
      generator.weight_sync_backend=nccl \

      # Environment class for the dataset
      # Can be specified here to apply to the full dataset, or at the per-prompt level during preprocessing
      environment.env_class=gsm8k \

      ... # Other parameters (see `examples/ppo/run_ppo.sh` for more)

Hardware Configuration
----------------------

Depending on your hardware setup, you may want to adjust a few parameters:

1. **GPU Configuration**: Set ``trainer.placement.policy_num_gpus_per_node`` and ``trainer.placement.critic_num_gpus_per_node`` to 
match your available GPU count. Note that since we are setting ``trainer.placement.colocate_all`` to ``true``, 
the total number of GPUs used for the policy model and critic model should be the same. 
Additionally ``generator.num_inference_engines`` * ``generator.inference_engine_tensor_parallel_size`` should 
also be equal to the total number of GPUs used for the policy and critic models.

Launching Your Training Run
---------------------------

Let's get our PPO training run started! First, configure your WandB API key for logging:

.. code-block:: bash

   export WANDB_API_KEY=your_wandb_api_key

Then launch your training run:

.. code-block:: bash

   bash examples/ppo/run_ppo.sh

Congratulations! You've just launched your first PPO training run!

What's Next?
------------

Now that you've got basic colocated PPO training down, you might want to explore some more advanced features:

- :doc:`../tutorials/async`: Asynchronous off-by-one training in < 100 lines of code!
- :doc:`../examples/remote_server`: Training with a remote inference engine


