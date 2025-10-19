# SkyAgent + VERL


We've provided example scripts for training with VERL for the SweBench task. The agent yaml is in [verl_oh.yaml](./verl_oh.yaml) and the script to run the training is in [verl_oh.sh](./verl_oh.sh). 


### Configure environment variables and setup remote server

We make use of our [remote server enabled OpenHands fork](https://github.com/NovaSky-AI/SKyRL-OpenHands) for training. For setting up the remote server, please refer to the above link. 

We are actively working on supporting more runtime implementations for training.

Make sure to set the following environment variables: `WANDB_API_KEY`, `ALLHANDS_API_KEY` and `SANDBOX_REMOTE_RUNTIME_API_URL`. 

### Download the dataset

You can download a dataset for training on the SweBench task, such as [NovaSky-AI/SkyRL-v0-293-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-v0-293-data). 
Make sure to download the dataset e.g. via

```bash
uv run huggingface-cli download NovaSky-AI/SkyRL-v0-293-data --repo-type dataset --local-dir <path_to_swegym_dataset>
```

and update the path in `DATA_DIR` in the script.


### Run training


You can run training with:

```bash
bash examples/run_verl/verl_oh.sh
```