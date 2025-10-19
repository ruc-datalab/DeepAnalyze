<div align="center">

# SkyRL: A Modular Full-stack RL Library for LLMs


[![🌐 NovaSky](https://img.shields.io/badge/-Visit%20Website-5865F2?style=for-the-badge)](https://novasky-ai.github.io/) [![Github](https://img.shields.io/badge/SkyRL-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/NovaSky-AI/SkyRL) [![Twitter](https://img.shields.io/badge/NovaSky-white?style=for-the-badge&logo=X&logoColor=000&color=000&labelColor=white)](https://x.com/NovaSkyAI) [![Hugging Face Collection](https://img.shields.io/badge/NovaSky-fcd022?style=for-the-badge&logo=huggingface&logoColor=000&labelColor)](https://huggingface.co/NovaSky-AI) [![Discord](https://img.shields.io/badge/NovaSky-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/cJF2JUaaAN) [![Documentation](https://img.shields.io/badge/Documentation-blue?style=for-the-badge&logo=readthedocs&logoColor=white)](https://skyrl.readthedocs.io/en/latest/)



<div align="center" style="font-family: Arial, sans-serif;">
  <p>
    <a href="#news" style="text-decoration: none; font-weight: bold;">News</a> •
    <a href="#links" style="text-decoration: none; font-weight: bold;">Links</a> •
    <a href="#getting-started" style="text-decoration: none; font-weight: bold;">Getting Started</a> •
    <a href="#citation" style="text-decoration: none; font-weight: bold;">Citation</a> •
    <a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">Acknowledgement</a> 
  </p>
</div>

</div>

# Overview

SkyRL is a full-stack RL library that provides the following components:

- [`skyagent`](./skyagent): Our agent layer for training long-horizon, real-world agents. Contains code for [SkyRL-v0](https://novasky-ai.notion.site/skyrl-v0).
- (**NEW**) [`skyrl-train`](./skyrl-train): Our modular, performant training framework for RL.
- (**NEW**) [`skyrl-gym`](./skyrl-gym): Our gynasium of tool-use tasks, including a library of math, coding, search and SQL environments implemented in the Gymnasium API.

# Getting Started

For a guide on developing with SkyRL, take at look at our [Development Guide](https://skyrl.readthedocs.io/en/latest/getting-started/development.html) docs.

For model training, checkout [`skyrl-train`](./skyrl-train) to start using, modifying, or building on top of the SkyRL training stack. See our [quickstart docs](https://skyrl.readthedocs.io/en/latest/index.html) to ramp up!

For building environments, checkout [`skyrl-gym`](./skyrl-gym) to integrate your task in the simple gymnasium interface.

For agentic pipelines, checkout [`skyagent`](./skyagent) for our work on optimizing and scaling pipelines for multi-turn tool use LLMs on long-horizon, real-environment tasks.


# News
- **[2025/06/26]** 🎉 We released SkyRL-v0.1: A highly-modular, performant RL training framework. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
- **[2025/06/26]** 🎉 We released SkyRL-Gym: A library of RL environments for LLMs implemented with the Gymnasium API. [[Blog](https://novasky-ai.notion.site/skyrl-v01)]
- **[2025/05/20]** 🎉 We released SkyRL-SQL: a multi-turn RL training pipeline for Text-to-SQL, along with SkyRL-SQL-7B — a model trained on just 653 samples that outperforms both GPT-4o and o4-mini!
- **[2025/05/06]** 🎉 We released SkyRL-v0: our open RL training pipeline for multi-turn tool use LLMs, optimized for long-horizon, real-environment tasks like SWE-Bench!

# Links
- 📜 [SkyRL-v0.1 Blog Post](https://novasky-ai.notion.site/skyrl-v01)
- 📜 [SkyRL-SQL Blog Post](https://novasky-ai.notion.site/skyrl-sql)
- 📜 [SkyRL-v0 Blog Post](https://novasky-ai.notion.site/skyrl-v0)

# Acknowledgement

This work is done at [**Berkeley Sky Computing Lab**](https://sky.cs.berkeley.edu/) in collaboration with [**Anyscale**](https://www.anyscale.com/), with generous compute support from [**Anyscale**](https://www.anyscale.com/), [**Databricks**](https://www.databricks.com/), [**NVIDIA**](https://developer.nvidia.com/brev), [**Lambda Labs**](https://lambdalabs.com/service/gpu-cloud?srsltid=AfmBOop5FnmEFTkavVtdZDsLWvHWNg6peXtat-OXJ9MW5GMNsk756PE5), and [**AMD**](https://www.amd.com/en.html).

We adopt many lessons and code from several great projects such as [veRL](https://github.com/volcengine/verl), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [OpenReasonerZero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero), and [NeMo-RL](https://github.com/NVIDIA-NeMo/RL). We appreciate each of these teams and their contributions to open-source research!


# Citation

If you find the work in this repository helpful, please consider citing:

```bibtex
@misc{cao2025skyrl,
  title     = {SkyRL-v0: Train Real-World Long-Horizon Agents via Reinforcement Learning},
  author    = {Shiyi Cao and Sumanth Hegde and Dacheng Li and Tyler Griggs and Shu Liu and Eric Tang and Jiayi Pan and Xingyao Wang and Akshay Malik and Graham Neubig and Kourosh Hakhamaneshi and Richard Liaw and Philipp Moritz and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
  year      = {2025},
}
```

```bibtex
@misc{liu2025skyrlsql,
      title={SkyRL-SQL: Matching GPT-4o and o4-mini on Text2SQL with Multi-Turn RL},
      author={Shu Liu and Sumanth Hegde and Shiyi Cao and Alan Zhu and Dacheng Li and Tyler Griggs and Eric Tang and Akshay Malik and Kourosh Hakhamaneshi and Richard Liaw and Philipp Moritz and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
}
```

```bibtex
@misc{griggs2025skrylv01,
      title={Evolving SkyRL into a Highly-Modular RL Framework},
      author={Tyler Griggs and Sumanth Hegde and Eric Tang and Shu Liu and Shiyi Cao and Dacheng Li and Charlie Ruan and Philipp Moritz and Kourosh Hakhamaneshi and Richard Liaw and Akshay Malik and Matei Zaharia and Joseph E. Gonzalez and Ion Stoica},
      year={2025},
      note={Notion Blog}
}
```
