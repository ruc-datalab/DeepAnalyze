Multi-Turn RL for Search with SkyRL
=====================================================

In this example, we walk through an example for training a multi-turn search agent with Qwen2.5-3B-Instruct and GRPO (with VLLM async rollouts), using the dataset and recipe
from `Search-R1 <https://arxiv.org/pdf/2503.09516>`_.

The full implementation of the search environment can be found in :skyrl_gym_link:`skyrl_gym/envs/search/env.py`.

You can find the exact step by step commands to reproduce our results in the :doc:`../recipes/searchr1` recipe, and you can find a link to our training runs 
with 2, 3, and 4 turns for comparison at our `WandB report <https://api.wandb.ai/links/sky-posttraining-uc-berkeley/5kvkzdzr>`_.


Task Overview
-------------

In this task, the agent is given a natural language question and the ability to query a search engine. The agent can use the search engine to help answer the question.
An example prompt is shown below:

.. code-block:: text

    You are a helpful and harmless assistant.
    
    Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> 
    and it will return the top searched results between <information> and </information>. You can search as many times as you want. 
    If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. 
    For example, <answer> Beijing </answer>. 

    
    Question: In what year was the company that was founded as Sound of Music added to the S&P 500?

The agent is given ``n`` turns to output an answer to the question within the ``<answer>`` and ``</answer>`` tags, meaning the agent has ``n - 1`` turns to query the search engine by outputting a query inside the ``<search>`` and ``</search>`` tags. 
A reward of 0 is given for incorrect responses, and a reward of 1 is given for correct responses (we do not apply format rewards).

Training Configuration
----------------------
Let's walk through configuration for running GRPO to train a 4-turn search agent on the SearchR1 dataset

.. code-block:: bash
    :caption: Training configuration at ``skyrl_train/examples/search/run_search.sh``

    # path for dataset (.parquet files) containing the prompts and metadata for each question
    DATA_DIR="$HOME/data/searchR1"

    uv run --isolated --frozen --extra vllm -m skyrl_train.entrypoints.main_base \
        # - Dataset: train/val data paths
        data.train_data="['${DATA_DIR}/train.parquet']" \
        data.val_data="['${DATA_DIR}/validation.parquet']" \

        # - Algorithm: GRPO settings, learning rate, KL loss
        trainer.algorithm.advantage_estimator="grpo" \
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.policy.optimizer_config.max_grad_norm=0.5 \
        trainer.policy.optimizer_config.num_warmup_steps=94 \
        trainer.algorithm.use_kl_loss=true \
        trainer.algorithm.kl_loss_coef=0.001 \

        # - Model: model path, placement, FSDP settings
        trainer.policy.model.path="Qwen/Qwen2.5-3B-Instruct" \
        trainer.placement.colocate_all=true \
        trainer.strategy=fsdp2 \
        trainer.policy.fsdp_config.cpu_offload=false \
        trainer.ref.fsdp_config.cpu_offload=true \
        trainer.placement.policy_num_gpus_per_node=8 \
        trainer.placement.ref_num_gpus_per_node=8 \

        # - Generator: VLLM backend, GPU settings  
        generator.num_inference_engines=4 \
        generator.inference_engine_tensor_parallel_size=2 \
        generator.backend=vllm \
        generator.run_engines_locally=true \
        generator.weight_sync_backend=nccl \
        generator.gpu_memory_utilization=0.5 \

        # - Training: epochs, batch sizes
        trainer.epochs=1 \
        trainer.update_epochs_per_batch=1 \
        trainer.train_batch_size=512 \
        trainer.policy_mini_batch_size=256 \
        trainer.micro_forward_batch_size_per_gpu=4 \
        trainer.micro_train_batch_size_per_gpu=4 \
        
        # - Length limits: prompt and generation lengths
        # trainer.max_prompt_length is the max length of the initial prompt
        trainer.max_prompt_length=2048 \
        # generator.max_input_length is the max length of the input to the model after any number of turns (including the initial prompt)
        generator.max_input_length=4096 \
        # generator.sampling_params.max_generate_length is the max length of the generated response for EACH turn
        generator.sampling_params.max_generate_length=500 \

        # - Generator multi-turn: async rollouts, batching, sampling settings
        # we need to make sure to set async_engine=true for async rollouts
        generator.async_engine=true \
        # we need to make sure to set batched=false for async rollouts
        generator.batched=false \
        generator.n_samples_per_prompt=5 \
        # this is used to set the max turns for the environment
        generator.max_turns=4 \
        # multi-turn generation format - see `skyrl_train/generators/skyrl_gym_generator.py` for more details
        generator.use_conversation_multi_turn=false \
        generator.sampling_params.temperature=1.0 \
        generator.sampling_params.top_p=1.0 \

        # - Environment: environment class, max env workers, search env settings
        environment.env_class="search" \
        environment.skyrl_gym.max_env_workers=16 \
        environment.skyrl_gym.search.log_requests=false \
        environment.skyrl_gym.search.search_url="http://127.0.0.1:8000/retrieve" \
        environment.skyrl_gym.search.topk=3 \

        # - Evaluation: batch size, intervals, sampling params
        trainer.eval_batch_size=256 \
        trainer.eval_before_train=false \
        generator.eval_sampling_params.temperature=0 \
        trainer.eval_interval=50 \
        ... # logging + checkpointing configuration (see `examples/search/run_search.sh` for the full script)
    
To change the number of turns, you can simply change the ``generator.max_turns`` setting.
For more details on environment implementation, see :skyrl_gym_link:`skyrl_gym/envs/search/env.py`.

Launching Your Training Run
---------------------------

Let's get our training run started! Make sure your WandB API key is set, your dataset paths are correctly set, and that you have launched the local retrieval server, following the :doc:`../recipes/searchr1` recipe instructions.

.. code-block:: bash

    export WANDB_API_KEY=your_wandb_api_key
    bash examples/search/run_search.sh

Now just sit back and watch your model learn to search! You can find a link to our training runs with 2, 3, and 4 turns for comparison at our `WandB report <https://api.wandb.ai/links/sky-posttraining-uc-berkeley/5kvkzdzr>`_.

Attribution
-------------
We thank the authors of Search-R1 for their work: `paper <https://arxiv.org/pdf/2503.09516>`_, `code <https://github.com/PeterGriffinJin/Search-R1>`_.
Additionally we thank the SGLang + Verl team for their work reproducing Search-R1 in Verl, which we use to validate our results: `doc <https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md>`_, 
`wandb <https://wandb.ai/lingchang-ustc/search_async_rl/runs/21rubwvs/workspace?nw=nwuserlingchang>`_, and `PR <https://github.com/volcengine/verl/pull/1682>`_.

What's Next?
------------

Now that you've trained a multi-turn search agent, you might want to build your own multi-turn environments:

- :doc:`../tutorials/new_env`: Learn how to build your own multi-turn environments!
- :doc:`../examples/multi_turn_text2sql`: Learn how to train a multi-turn text2sql agent with SkyRL!