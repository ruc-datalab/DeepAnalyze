Multi-Turn GRPO Text2SQL with Async Rollouts
=====================================================

In this example, we walk through how to train an effective multi-turn Text2SQL model (beating GPT-4o) with SkyRL using async rollouts.

We provide an implementation of a multi-turn Text2SQL environment at :skyrl_gym_link:`skyrl_gym/envs/sql/env.py`.

You can find the exact recipe for reproducing our prior `SkyRL-SQL-7B <https://novasky-ai.notion.site/skyrl-sql>`_ release at :doc:`../recipes/skyrl-sql`.

Task Overview
-------------

In this task, the agent is given a natural language question and a database schema, and is tasked with generating a SQL query to answer the question.
An abbreviated example prompt is shown below:

.. code-block:: text

    You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question within limited turns. You should breakdown the problem, draft your reasoning process, and generate the solution.

    Database Engine:
    SQLite

    Database Schema:
    {db_details}
    This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

    External Knowledge:
    {external_knowledge}

    Question:
    {question}

    Instructions:
    - Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
    - The generated query should return all of the information asked in the question without any missing or extra information.
    - Before generating the final SQL query, please think through the steps of how to write the query. It should include detailed considerations such as analyzing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, thinking of how to call SQL tools, and revisiting previous steps.

    Format:
    - Conduct thinking inside <think>...</think> blocks every time you get new observation or information. 
    - You can use SQL tool written within a single <sql>your sql</sql> block to explore or verify. SQL tool output will be shown as dataframe inside <observation>...</observation>. Based on this observation, you can think again and refine.
    - The returned dataframe will be truncated in 50 rows if observation is too long. 
    - If you find no further exploration is needed or reaches max turns, you MUST directly provide the final SQL query solution inside <solution>...</solution>. 
    ...
    {db_details}: 
    CREATE TABLE country_aliases (
        alias_id INTEGER,
        country_id INTEGER,
        alias_name TEXT,
        alias_type TEXT,
        PRIMARY KEY (alias_id),
        FOREIGN KEY (country_id) REFERENCES countries (country_id)
    );
    ...
    {question}: 
    Could you please provide me with the English name, current population, average population over the last five years, and population from last year for each country, based on the country's population metrics? I need this information to analyze population trends.

The agent is given a maximum of 6 turns to generate a valid SQL query with the correct formatting, with negative rewards being assigned for incorrect formatting, 0 rewards for correct formatting but incorrect SQL, and a positive reward for correct SQL.

Data Preparation
----------------

First, follow the :ref:`data preparation instructions <skyrl-sql-data>` to download the dataset, as well as the database files (note that
the database files are quite large - around 50GB in total). We reproduce the commands here:

.. code-block:: bash

    huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset
    huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir <path_to_file.zip>
    unzip <path_to_file.zip>

Training Configuration
----------------------
Now that we have our dataset and database files, let's walk through the some of the key training configurations.


.. code-block:: bash
    :caption: Training configuration at ``skyrl_train/examples/text_to_sql/run_skyrl_sql.sh``

    # path for dataset (.parquet files) containing the prompts and metadata for each question
    DATA_DIR="$HOME/data/sql"
    # path for .db files for environment interaction
    DB_PATH="$HOME/path/to/db_files"

    uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base \
        #### Environment configuration
        environment.env_class=text2sql \
        environment.skyrl_gym.text2sql.db_path=$DB_PATH \
      
        #### Multi-turn Async Rollouts configuration
        # this is used to set the max turns for the environment
        generator.max_turns=6 \
        # we need to make sure to set async_engine=true for async rollouts
        generator.async_engine=true \
        # we need to make sure to set batched=false for async rollouts
        generator.batched=false \

        #### context length related configurations
        # trainer.max_prompt_length is the max length of the initial prompt
        trainer.max_prompt_length=6000 \
        # generator.max_input_length is the max length of the input to the model after any number of turns (including the initial prompt)
        generator.max_input_length=29000 \
        # generator.sampling_params.max_generate_length is the max length of the generated response for EACH turn
        generator.sampling_params.max_generate_length=3000 \

        #### multi-turn generation format - see `skyrl_train/generators/skyrl_gym_generator.py` for more details
        generator.use_conversation_multi_turn=false \
        
        #### data configuration
        data.train_data="['$DATA_DIR/train.parquet']" \
        data.val_data="['$DATA_DIR/validation.parquet']" \

        #### Placement configuration - note since we set use_kl_loss=false below, we don't need to use a ref model
        # policy placement configuration
        trainer.policy.model.path="Qwen/Qwen2.5-Coder-7B-Instruct" \
        trainer.placement.colocate_all=true \
        trainer.placement.policy_num_gpus_per_node=8 \
        # inference engine placement configuration
        generator.num_inference_engines=2 \
        generator.inference_engine_tensor_parallel_size=4 \

        #### algorithm configuration
        trainer.epochs=30 \
        trainer.algorithm.advantage_estimator="grpo" \
        trainer.algorithm.use_kl_loss=false \
        trainer.algorithm.use_kl_loss=false \
        generator.n_samples_per_prompt=5 \

        #### generation sampling params (relevant to algorithm correctness)
        generator.sampling_params.temperature=0.6 \
        generator.sampling_params.top_p=0.95 \

        #### training configuration
        trainer.policy.optimizer_config.lr=1.0e-6 \
        trainer.train_batch_size=256 \
        trainer.policy_mini_batch_size=256 \
        trainer.micro_forward_batch_size_per_gpu=8 \
        trainer.micro_train_batch_size_per_gpu=1 \
        trainer.eval_batch_size=1024 \
        trainer.eval_before_train=true \
        trainer.eval_interval=5 \
        ... # Other parameters (see `examples/text_to_sql/run_skyrl_sql.sh` for the full script)

- All we have to do to enable multi-turn training with async rollouts is to simply set ``generator.max_turns`` to the maximum number of turns we want the agent to take,
  and to make sure ``generator.async_engine=true`` and ``generator.batched=false``. 

- Chat templating and loss masking for multi-turn conversations are handled by the ``SkyRLGymGenerator`` class.

  - In the above example, we set ``use_conversation_multi_turn=false`` to enforce that the multi-turn conversation is formatted as a single assistant response.
  - If you want to use a conversation-based format, you can set ``use_conversation_multi_turn=true`` and the model will generate a separate assistant response for each turn. This is supported only with ``backend="vllm"`` as of now.
  - See :code_link:`skyrl_train/generators/skyrl_gym_generator.py` for more details on both options!

Launching Your Training Run
---------------------------

Let's get our training run started! Make sure to set your WandB API key for logging, and that your database and dataset paths are correctly set.

.. code-block:: bash

    export WANDB_API_KEY=your_wandb_api_key
    bash examples/text_to_sql/run_skyrl_sql.sh

And now watch your model start to learn to generate better SQL queries!

What's Next?
------------

Now that you've seen what's possible with multi-turn training with async rollouts, you might want to start building your own multi-turn environments!

- :doc:`../tutorials/new_env`: Learn how to build your own multi-turn environments!
- :doc:`search`: Learn how to train a multi-turn search agent with SkyRL!