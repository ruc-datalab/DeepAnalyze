Quick Start: GRPO on GSM8K
==========================

First, follow the guide at :doc:`installation` to get your environment set up.

In this quickstart, we'll walk you through running GRPO training on the GSM8K dataset. You'll prepare the dataset, configure your training parameters, and launch your first SkyRL training run.

Dataset Preparation
-------------------

To download and prepare the GSM8K dataset, run the following script. We provide convenience scripts for GSM8K and several other popular datasets, but you can also use your own custom dataset by following the instructions in the :ref:`dataset-preparation` section.

.. code-block:: bash

   uv run --isolated examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k

Training Configuration
----------------------

Next, let's set up the training configuration. You can find a complete example in ``examples/gsm8k/run_gsm8k.sh``, and we highlight some key parameters here:

.. code-block:: bash

   uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
      # Data setup
      data.train_data="['$HOME/data/gsm8k/train.parquet']" \
      data.val_data="['$HOME/data/gsm8k/validation.parquet']" \

      # Trainer and training algorithm
      trainer.algorithm.advantage_estimator="grpo" \
      trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
      
      # Model placement and training strategy (colocate or disaggregate, sharding, etc.)
      trainer.strategy=fsdp2 \
      trainer.placement.colocate_all=true \
      trainer.placement.policy_num_gpus_per_node=4 \

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

      # WandB logging
      trainer.logger="wandb" \
      trainer.project_name="gsm8k" \
      trainer.run_name="gsm8k_test" \
       
      ... # Other parameters (see `examples/gsm8k/run_gsm8k.sh` for more)

Hardware Configuration
----------------------

Depending on your hardware setup, you may want to adjust a few parameters:

1. **GPU Configuration**: Set ``trainer.placement.policy_num_gpus_per_node`` to match your available GPU count. If you need to change the model size, you can also set ``trainer.policy.model.path`` to ``Qwen/Qwen2.5-0.5B-Instruct`` or ``Qwen/Qwen2.5-7B-Instruct``.

2. **InferenceEngine Matching**: Ensure that ``num_inference_engines * inference_engine_tensor_parallel_size`` equals the total number of GPUs used for the policy model above.


Launching Your Training Run
---------------------------

Now for the exciting part! First, configure your WandB API key for logging:

.. code-block:: bash

   export WANDB_API_KEY=your_wandb_api_key

Then launch your training run:

.. code-block:: bash

   bash examples/gsm8k/run_gsm8k.sh

Congratulations! You've just launched your first SkyRL training run!

Monitoring Progress
-------------------

The training progress will be logged to your terminal, showing you which part of the training loop is executing and how long each step takes. You can monitor detailed metrics and visualizations on WandB, or configure logging to output to the console or your preferred logging backend.

What's Next?
------------

Now that you've got the basics down, you might want to explore:

- :doc:`../tutorials/new_env`: Creating a new environment without touching the training loop
- :doc:`../tutorials/async`: Asynchronous off-by-one training in < 100 lines of code!


