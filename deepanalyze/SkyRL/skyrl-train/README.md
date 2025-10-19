# SkyRL-Train: A modular, performant RL framework for post-training LLMs
<div align="center">

[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/) [![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL) [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI) [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI) [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/RBAjeWSA) [![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)

</div>

# Overview

 With a focus on modularity, `skyrl-train` makes it easy to prototype new training algorithms, environments, and execution plans—without compromising usability or speed. 

`skyrl-train` is **for users who want to modify anything:**

- **Quickly develop new environments** without modifying or understanding the training code .
- **Modify the training execution plan** such as model placement, colocation or disaggregation of training and generation, and async RL.
- **Implement custom trajectory generation** specific to your use-case, such as custom sampling methods, tree search, etc.
- … make any other flexible modifications to the RL workflow!


## Key Features
The `skyrl-train` package supports:
- PPO and GRPO
- Training Backends: FSDP, FSDP2, and DeepSpeed
- Inference backends: vLLM, SGLang, and any custom OpenAI API compatible endpoint that exposes a method to perform weight sync
- Ulysses sequence parallelism for long-context training
- Colocated or disaggregated training and generation (including on heterogeneous hardware)
- Synchronous RL or async one-off pipelining
- Simple batched rollouts or Asynchronous rollouts for multi-turn conversations
- Weight sync via NCCL, gloo, or checkpoint-and-load
- Integration with `skyrl-gym` to run any environment in the gynasium
- Sequence packing and Flash Attention 2

## Documentation

Find `skyrl-train` documentation at: [skyrl.readthedocs.io/en/latest/](https://skyrl.readthedocs.io/en/latest/)

## Quick Start

A quick start guide for installation and your first training run is provided below.

### Requirements

The only requirements are:

- CUDA version 12.8
- [uv](https://docs.astral.sh/uv/)

If you're running on an existing Ray cluster, make sure to use Ray 2.48.0 and Python 3.12. If not, proceed with the installation instructions below.


First, clone the repository:

```bash
git clone --recurse-submodules https://github.com/NovaSky-AI/SkyRL
cd SkyRL/skyrl-train
```

Then, create a new virtual environment and install the dependencies:

```bash
# creates a venv at .venv/
uv sync --extra vllm 
source .venv/bin/activate
```

Then, prepare the dataset:

```bash
uv run -- python examples/gsm8k/gsm8k_dataset.py
```

Finally, before training, make sure to configure Ray to use `uv`:

```bash
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
# or add to your .bashrc
# echo 'export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook' >> ~/.bashrc
```

You should now be able to run our example script (assumes at least 4 GPUs):

```bash
export WANDB_API_KEY=<your wandb api key>
bash examples/gsm8k/run_gsm8k.sh
```

For detailed installation instructions, as well as more examples, please refer to our [documentation](https://skyrl.readthedocs.io/en/latest/).

## Training on a new task or environment

To implement a new task or environment using the SkyRL-Gym interface, please see our see our [Walkthrough Docs](https://skyrl.readthedocs.io/en/latest/tutorials/new_env.html).

If you don't want to use the SkyRL-Gym interface, or you have an existing task or agentic pipeline implementation and just want to train with it on top of SkyRL, we recommend you create a simple custom [`Generator`](skyrl_train/generators/base.py), which requires implementing a single method, `generate()`. We have one example of a custom Generator at [`SkyRLGymGenerator`](skyrl_train/generators/skyrl_gym_generator.py) which executes environments written in the SkyRL-Gym interface. We are working to provide more example integrations of agent harnesses -- please reach out if you'd like yours to be one of them!

## Reproducing SkyRL-SQL
We also test SkyRL by reproducing our prior release [SkyRL-SQL](https://novasky-ai.notion.site/skyrl-sql), which enabled efficient Multi-Turn RL for Text2SQL. 
You can find a link to the wandb report [here](https://wandb.ai/sky-posttraining-uc-berkeley/skyrl-sql/reports/SkyRL-SQL---VmlldzoxMzM0MTAyMw), and a detailed walk through of the reproduction in our [documentation](https://skyrl.readthedocs.io/en/latest/examples/multi_turn_text2sql.html).

# Acknowledgement

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambda.ai/), and [**AMD**](https://www.amd.com/en.html).

We adopt many lessons and code from several great projects such as [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). We appreciate each of these teams and their contributions to open-source research!



# Citation

If you find the work in `skyrl-train` helpful, please consider citing:
```bibtex
@misc{griggs2025skrylv01,
      title={Evolving SkyRL into a Highly-Modular RL Framework},
      author={Tyler Griggs and Sumanth Hegde and Eric Tang and Shu Liu and Shiyi Cao and Dacheng Li and Charlie Ruan and Philipp Moritz and Kourosh Hakhamaneshi and Richard Liaw and Akshay Malik and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
      note={Notion Blog}
}
```
