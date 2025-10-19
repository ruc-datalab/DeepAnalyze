SearchR1
=========

We provide scripts to reproduce our results for training a multi-turn search agent using the dataset and recipe from `Search-R1 <https://arxiv.org/pdf/2503.09516>`_.
For a more detailed explanation of the task and training configuration, see :doc:`../examples/search`.

Pre-requisites
--------------

Make sure to have followed the installation commands in :doc:`../getting-started/installation`.

Data Preparation
----------------

We provide a script to download the dataset from huggingface, and preprocess it to run in SkyRL-Gym.

.. code-block:: bash

    uv run --isolated examples/search/searchr1_dataset.py --local_dir ~/data/searchR1

Start the Search Engine
------------------------
Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will
use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration.

Retriever environments 
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create and activate the retriever environment with Python 3.10
    conda create -n retriever python=3.10 -y
    conda activate retriever

    # Install PyTorch (with GPU support) and related libraries
    conda install numpy==1.26.4 # needed to stop incompatible version of numpy from being installed via pip
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

    # Install other Python packages
    pip install transformers datasets pyserini huggingface_hub

    # Install the GPU version of faiss
    conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

    # Install the API service framework
    pip install uvicorn fastapi

Download the Index
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    conda activate retriever

    local_dir=~/data/searchR1
    python examples/search/searchr1_download.py --local_dir $local_dir
    cat $local_dir/part_* > $local_dir/e5_Flat.index
    gzip -d $local_dir/wiki-18.jsonl.gz


Prepare Datasets 
~~~~~~~~~~~~~~~~

.. code-block:: bash

    python examples/search/searchr1_dataset.py --local_dir $local_dir


Start the Local Flat e5 Retrieval Server 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: bash

    conda activate retriever

    # redirect the output to a file to avoid cluttering the terminal
    # we have observed outputting to the terminal causing spikes in server response times
    bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 

Start your Training Run
~~~~~~~~~~~~~~~~~~~~~~~
Now from your base environment, you can launch your training run (which will use uv to package dependencies, separately from the retriever environment).

.. code-block:: bash

    export WANDB_API_KEY=your_wandb_api_key
    bash examples/search/run_search.sh

You can find a link to our training runs with 2, 3, and 4 turns for comparison at our `WandB report <https://api.wandb.ai/links/sky-posttraining-uc-berkeley/5kvkzdzr>`_.