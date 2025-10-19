SkyRL-SQL
=========

We provide scripts to reproduce the results for `SkyRL-SQL-7B <https://novasky-ai.notion.site/skyrl-sql>`_ using SkyRL-train and SkyRL-Gym.

You can find a WandB run for both single-turn and multi-turn Text2SQL training at `this link <https://api.wandb.ai/links/sky-posttraining-uc-berkeley/5df7pt6p>`_.

Pre-requisites 
---------------

Make sure to have followed the installation commands in :doc:`../getting-started/installation`. 


Start Ray
---------

Start ray in your cluster following the guide: https://docs.ray.io/en/latest/ray-core/starting-ray.html. 


.. _skyrl-sql-data:

Data Preparation
----------------


We provide the dataset we used on HuggingFace: https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data-newfmt 
You can download the dataset by running the following command

.. code-block:: bash

    huggingface-cli download NovaSky-AI/SkyRL-SQL-653-data-newfmt --local-dir $HOME/data/sql --repo-type dataset


DB environment 
---------------

Make sure to setup the database files needed for training.  We use the database files from `OmniSQL <https://github.com/RUCKBReasoning/OmniSQL/blob/main/train_and_evaluate/README.md>`_. 

You can download the datasets from:
- `ModelScope-OmniSQL-datasets <https://modelscope.cn/datasets/seeklhy/OmniSQL-datasets/summary>`_
- `HuggingFace-OmniSQL-datasets <https://huggingface.co/datasets/seeklhy/OmniSQL-datasets>`_



The datasets include BIRD, Spider, ScienceBenchmark, EHRSQL, Spider2-SQLite, Spider-DK, Spider-Realistic, Spider-Syn, and SynSQL-2.5M. In our training pipeline, we only need to access databases from SynSQL-2.5M and Spider. 

Unzip `data.zip` in this folder, and set the corresponding `DB_PATH` in the training script below. You can download and unzip the data by running

.. code-block:: bash

    huggingface-cli download seeklhy/OmniSQL-datasets data.zip --repo-type dataset --local-dir <path_to_file.zip>
    unzip <path_to_file.zip>

Running the scripts 
-------------------

We provide a script :code_link:`examples/text_to_sql/run_skyrl_sql.sh` for reproducing the results for SkyRL-SQL-7B. Make sure to substitute the `DB_PATH`  and `DATA_PATH` variables with your own.

.. code-block:: bash

    export WANDB_API_KEY=<wandb-api-key>
    bash examples/text_to_sql/run_skyrl_sql.sh



