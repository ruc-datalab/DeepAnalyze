<p align="center" width="100%">
<img src="assets/logo.png" alt="DeepAnalyze" style="width: 60%; min-width: 300px; display: block; margin: auto;">
</p>

# DeepAnalyze: Agentic Large Language Models for Autonomous Data Science
[![arXiv](https://img.shields.io/badge/arXiv-2510.16872-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.16872)
[![homepage](https://img.shields.io/badge/%F0%9F%8C%90%20Homepage%20-DeepAnalyze%20Cases-blue.svg)](https://ruc-deepanalyze.github.io/)
[![model](https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface%20-DeepAnalyze--8B-orange.svg)](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B)
[![data](https://img.shields.io/badge/%F0%9F%93%9A%20Datasets%20-DataScience--Instruct--500K-darkgreen.svg)](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K)
[![star](https://img.shields.io/github/stars/ruc-datalab/DeepAnalyze?style=social&label=Code+Stars)](https://github.com/ruc-datalab/DeepAnalyze)
![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Fgithub.com%2Fruc-datalab%2FDeepAnalyze&label=Visitors&icon=graph-up&color=%23dc3545&message=&style=flat&tz=UTC)  [![wechat](https://img.shields.io/badge/WeChat-%E5%8A%A0%E5%85%A5DeepAnalyze%E4%BA%A4%E6%B5%81%E8%AE%A8%E8%AE%BA%E7%BE%A4-black?logo=wechat&logoColor=07C160)](./assets/wechat.jpg) 

[![twitter](https://img.shields.io/badge/@Brian%20Roemmele-gray?logo=x&logoColor=white&labelColor=black)](https://x.com/BrianRoemmele/status/1981015483823571352) [![twitter](https://img.shields.io/badge/@Dr%20Singularity-gray?logo=x&logoColor=white&labelColor=black)](https://x.com/Dr_Singularity/status/1981010771338498241) [![twitter](https://img.shields.io/badge/@Gorden%20Sun-gray?logo=x&logoColor=white&labelColor=black)](https://x.com/Gorden_Sun/status/1980573407386423408) [![twitter](https://img.shields.io/badge/@AIGCLINK-gray?logo=x&logoColor=white&labelColor=black)](https://x.com/aigclink/status/1980554517126246642) [![twitter](https://img.shields.io/badge/@Python%20Developer-gray?logo=x&logoColor=white&labelColor=black)](https://x.com/Python_Dv/status/1980667557318377871) [![twitter](https://img.shields.io/badge/@meng%20shao-gray?logo=x&logoColor=white&labelColor=black)](https://x.com/shao__meng/status/1980623242114314531) 


> **Authors**: **[Shaolei Zhang](https://zhangshaolei1998.github.io/), [Ju Fan*](http://iir.ruc.edu.cn/~fanj/), [Meihao Fan](https://scholar.google.com/citations?user=9RTm2qoAAAAJ), [Guoliang Li](https://dbgroup.cs.tsinghua.edu.cn/ligl/), [Xiaoyong Du](http://info.ruc.edu.cn/jsky/szdw/ajxjgcx/jsjkxyjsx1/js2/7374b0a3f58045fc9543703ccea2eb9c.htm)**
>
> Renmin University of China, Tsinghua University


**DeepAnalyze** is the first agentic LLM for autonomous data science. It can autonomously complete a wide range of data-centric tasks without human intervention, supporting:
- 🛠 **Entire data science pipeline**: Automatically perform any data science tasks such as data preparation, analysis, modeling, visualization, and report generation.
- 🔍 **Open-ended data research**: Conduct deep research on diverse data sources, including structured data (Databases, CSV, Excel), semi-structured data (JSON, XML, YAML), and unstructured data (TXT, Markdown), and finally produce analyst-grade research reports.
- 📊 **Fully open-source**: The [model](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B), [code](https://github.com/ruc-datalab/DeepAnalyze), [training data](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K), and [demo](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B) of DeepAnalyze are all open-sourced, allowing you to deploy or extend your own data analysis assistant.

<p align="center" width="100%">
<img src="./assets/deepanalyze.jpg" alt="deepanalyze" style="width: 70%; min-width: 300px; display: block; margin: auto;">
</p>


## 🔥 News
- **[2025.10.28]**: We welcome all contributions, including improving the DeepAnalyze and sharing use cases (see [`CONTRIBUTION.md`](CONTRIBUTION.md)). All merged PRs will be listed as contributors.
- **[2025.10.27]**: DeepAnalyze has attracted widespread attention, gaining **1K+** GitHub stars and **200K+** Twitter views within a week.
- **[2025.10.21]**: DeepAnalyze's [paper](https://arxiv.org/abs/2510.16872), [code](https://github.com/ruc-datalab/DeepAnalyze), [model](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B), [training data](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K) are released!

## 🖥 Demo


<p align="center" width="100%">
Upload the data, DeepAnalyze can perform data-oriented deep research 🔍 and any data-centric tasks 🛠
</p>

- Clone this repo and download [DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B).
- Deploy DeepAnalyze-8B via vllm: `vllm serve DeepAnalyze-8B`

<details>
<summary>WebUI Demo</summary>

https://github.com/user-attachments/assets/04184975-7ee7-4ae0-8761-7a7550c5c8fe

> [!TIP]
>
> Clone this repository to deploy DeepAnalyze locally as your data analyst, completing any data science tasks without any workflow or closed-source APIs.
>
> 🔥 The UI of the demo is an initial version. Welcome to further develop it, and we will include you as a contributor.

- Run these scripts to launch the API and interface, and then interact through the browser (http://localhost:4000):
    ```bash
    cd demo/chat
    npm install
    cd ..
    bash start.sh

    # stop the api and interface
    bash stop.sh
    ```
- If you want to deploy under a specific IP, please replace localhost with your IP address in [./demo/backend.py](./demo/backend.py) and [./demo/chat/lib/config.ts](./demo/chat/lib/config.ts)
</details>

<details>
<summary>JupyterUI Demo</summary>
Familiar with Jupyter Notebook? Try the JupyterUI demo!
This Demo runs Jupyter Lab as frontend, creating a new notebook, converting `<Analyze|Understan|Answer>` to Markdown cells, converting `<Code>` to Code cells and executing them as `<Execute>`.
Go to [`demo/jupyter`](./demo/jupyter) to see more and try!
</details>

## 🚀 Quick Start

### Requirements

- Install packages: `torch`, `transformers`, `vllm>=0.8.5`
    ```bash
    conda create -n deepanalyze python=3.12 -y
    conda activate deepanalyze
    pip install -r requirements.txt
    
    # For training
    (cd ./deepanalyze/ms-swift/ && pip install -e .)
    (cd ./deepanalyze/SkyRL/ && pip install -e .)
    ```
- [`requirements.txt`](requirements.txt) lists the minimal dependencies required for DeepAnalyze inference.
For training, please refer to [`./deepanalyze/ms-swift/requirements.txt`](./deepanalyze/ms-swift/requirements.txt) and [`./deepanalyze/SkyRL/pyproject.toml`](./deepanalyze/SkyRL/pyproject.toml)
- We recommend separating the inference and training environments to avoid dependency conflicts.

### Command Interaction

- Deploy DeepAnalyze-8B via vllm: `vllm serve DeepAnalyze-8B`

- Run these scripts for any data science tasks:
  - You can specify **any data science tasks**, including specific data tasks and open-ended data research.
  - You can specify **any number of data sources**, and DeepAnalyze will automatically explore them.
  - You can specify **any type of data sources**, e.g., structured data (Databases, CSV, Excel), semi-structured data (JSON, XML, YAML), and unstructured data (TXT, Markdown)

  ```python
  from deepanalyze import DeepAnalyzeVLLM
  
  prompt = """# Instruction
  Generate a data science report.
  
  # Data
  File 1: {"name": "bool.xlsx", "size": "4.8KB"}
  File 2: {"name": "person.csv", "size": "10.6KB"}
  File 3: {"name": "disabled.xlsx", "size": "5.6KB"}
  File 4: {"name": "enlist.csv", "size": "6.7KB"}
  File 5: {"name": "filed_for_bankrupcy.csv", "size": "1.0KB"}
  File 6: {"name": "longest_absense_from_school.xlsx", "size": "16.0KB"}
  File 7: {"name": "male.xlsx", "size": "8.8KB"}
  File 8: {"name": "no_payment_due.xlsx", "size": "15.6KB"}
  File 9: {"name": "unemployed.xlsx", "size": "5.6KB"}
  File 10: {"name": "enrolled.csv", "size": "20.4KB"}"""

  workspace = "/home/u2023000922/zhangshaolei/deepanalyze_public/DeepAnalyze/example/analysis_on_student_loan/"
  
  deepanalyze = DeepAnalyzeVLLM(
      "/fs/fast/u2023000922/zhangshaolei/checkpoints/deepanalyze-8b/"
  )
  answer = deepanalyze.generate(prompt, workspace=workspace)
  print(answer["reasoning"])
  ```
  You shoud get a deep research report, which can be rendered as a PDF.:
  ```text
  # Comprehensive Analysis of Student Enrollment Patterns and Institutional Transfers
  
  ## Introduction and Research Context
  
  The analysis of student enrollment patterns represents a critical area of educational research with significant implications for institutional planning, resource allocation, and student support services. This comprehensive study examines a comprehensive dataset encompassing 1,194 enrollment records across six educational institutions, merged with supplementary demographic, financial, and employment status data. The research employs advanced analytical techniques including network analysis, predictive modeling, and temporal pattern recognition to uncover both macro-level institutional trends and micro-level student mobility patterns. The dataset's longitudinal nature, spanning fifteen months of enrollment records, provides unique insights into the complex dynamics of student pathways through higher education systems.
  
  Our methodological approach combines quantitative analysis of enrollment durations, transfer probabilities, and financial indicators with qualitative ...
  
  The research contributes to the growing body of literature on student mobility by providing empirical evidence of institutional transfer networks and their relationship to student outcomes...
  .....
  ```
  <p align="center" width="100%">
    <img src="./assets/report.png" alt="deepanalyze" style="width: 100%; min-width: 300px; display: block; margin: auto;">
  </p>

  > For more examples and task completion details, please refer to [DeepAnalyze's homepage](https://ruc-deepanalyze.github.io/).

### API
- You can build an OpenAI-Style API, using this script (note to change `MODEL_PATH = "DeepAnalyze-8B"` in [demo/backend.py](demo/backend.py) to your vllm model name):

  ```
  python demo/backend.py
  ```

- API usage (streaming response):

  ```
  curl -X POST http://localhost:8200/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
             "messages": [
               {
                 "role": "user",
                 "content": "Generate a data science report."
               }
             ],
             "workspace": "example/student_loan/"
           }'
  ```

  
## 🎈 Develop Your Own DeepAnalyze

### 1. Download Model and Training Data
- Download [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B). Or you can directly finetune based on [DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B).

  - If you use DeepSeek-R1-0528-Qwen3-8B as the base model, you should add the special tokens, using:

    ```shell
    MODEL_PATH=path_to_DeepSeek-R1-0528-Qwen3-8B
    SAVE_PATH=path_to_save_DeepSeek-R1-0528-Qwen3-8B-addvocab
    
    python deepanalyze/add_vocab.py \
      --model_path "$MODEL_PATH" \
      --save_path "$SAVE_PATH" \
      --add_tags
    ```

- Download training data [DataScience-Instruct-500K](https://huggingface.co/datasets/RUC-DataLab/DataScience-Instruct-500K).
  - unzip `DataScience-Instruct-500K/RL/data.zip`


### 2. Curriculum-based Agentic Training
- Single-ability Fine-tuning: [./scripts/single.sh](./scripts/single.sh)
- Multi-ability Agentic Training (cold start): [./scripts/multi_coldstart.sh](./scripts/multi_coldstart.sh)
- Multi-ability Agentic Training (RL): [./scripts/multi_rl.sh](./scripts/multi_rl.sh)

### 3. Evaluation
- We have unified the evaluation of most existing data science benchmarks using vLLM (with more being continuously added...). You can directly follow the introduction in [./playground](./playground) to quickly evaluate DeepAnalyze or your own agent.


## 👏 Contribution
> We welcome all forms of contributions, and merged PRs will be listed as contributors.
### Contribution on Code and Model

- We welcome all forms of contributions on DeepAnalyze's code, model and UI, such as Docker packaging, DeepAnalyze model conversion and quantization, and submitting DeepAnalyze workflows based on closed-source LLMs. 
- You can submit a pull request directly.

### Contribution on Case Study

- We also especially encourage you to share your use cases and feedback when using DeepAnalyze; these are extremely valuable for helping us improve DeepAnalyze.
- You can place your use cases in a new folder under [`.example/`](.example/). We recommend following the folder structure of [`.example/analysis_on_student_loan/`](.example/analysis_on_student_loan/), which includes three parts:
    - `data/`: stores the uploaded files
    - `prompt.txt`: input instructions
    - `README.md`: documentation. We suggest including the input, DeepAnalyze’s output, outputs from other closed-source LLMs (optional), and your evaluation/comments of the case.
- DeepAnalyze only has 8B parameters, so we also welcome examples where DeepAnalyze performs slightly worse than the closed-source LLMs — this will help us improve DeepAnalyze.

## 🤝 Acknowledgement
- Training framework: [ms-swift](https://github.com/modelscope/ms-swift), [SkyRL](https://github.com/NovaSky-AI/SkyRL)
- Source of Training Data: [Reasoning-Table](https://github.com/MJinXiang/Reasoning-Table), [Spider](https://yale-lily.github.io/spider), [BIRD](https://bird-bench.github.io/), [DABStep](https://huggingface.co/blog/dabstep)

## 🖋Citation

If this repository is useful for you, please cite as:

```
@misc{deepanalyze,
      title={DeepAnalyze: Agentic Large Language Models for Autonomous Data Science}, 
      author={Shaolei Zhang and Ju Fan and Meihao Fan and Guoliang Li and Xiaoyong Du},
      year={2025},
      eprint={2510.16872},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.16872}, 
}
```

If you have any questions, please feel free to submit an issue or contact `zhangshaolei98@ruc.edu.cn`.

## 🌟 Misc

Welcome to join the [DeepAnalyze WeChat group](./assets/wechat.jpg), chat and share ideas with others!

<p align="left" width="100%">
<img src="./assets/wechat.jpg" alt="DeepAnalyze" style="width: 30%; min-width: 300px; display: block; margin: auto;">
</p>

If you like DeepAnalyze, give it a GitHub Star ⭐. 

[![Star History Chart](https://api.star-history.com/svg?repos=ruc-datalab/DeepAnalyze&type=date&legend=top-left)](https://www.star-history.com/#ruc-datalab/DeepAnalyze&type=date&legend=top-left)