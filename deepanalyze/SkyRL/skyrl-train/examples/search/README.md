# SearchR1 Replication Setup Instructions 

We provide scripts to reproduce our results for training a multi-turn search agent using the dataset and recipe from [SearchR1](https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/refs/heads/main/docs/retriever.md).

Additional Reference: [Verl+Sglang Instructions](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md). 

## Prepare Datasets 
```bash
local_dir=~/data/searchR1
uv run --isolated examples/search/searchr1_dataset.py --local_dir $local_dir
```

# Start the Search Engine
Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration.

## Retriever environments 
```bash
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
```

## Download the Index
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

## Start the Local Flat e5 Retrieval Server 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```

## Launch your Training Job
Now from your base environment, you can launch your training run (which will use uv to package dependencies, separately from the retriever environment).

```bash
    export WANDB_API_KEY=your_wandb_api_key
    bash examples/search/run_search.sh
```
