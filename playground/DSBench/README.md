# DSBench
[![project page](https://img.shields.io/badge/Homepage-DSBench-blue)](https://liqiangjing.github.io/dsbench.github.io/)
[![arxiv](https://img.shields.io/badge/arXiv-2409.07703-b31b1b.svg)](https://arxiv.org/abs/2409.07703)


This repo provides the source code of our paper: [DSBench: How Far are Data Science Agents from Becoming Data Science Experts?](https://arxiv.org/abs/2409.07703)
[[PDF]](https://arxiv.org/pdf/2409.07703)[[Twitter]](https://twitter.com/wyu_nd/status/1835469391426973898) If you discuss or use DSBench in your research, please cite us!
```bibtex
@misc{jing2024dsbenchfardatascience,
      title={DSBench: How Far Are Data Science Agents to Becoming Data Science Experts?}, 
      author={Liqiang Jing and Zhehui Huang and Xiaoyang Wang and Wenlin Yao and Wenhao Yu and Kaixin Ma and Hongming Zhang and Xinya Du and Dong Yu},
      year={2024},
      eprint={2409.07703},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2409.07703}, 
}
```

## Overview
DSBench is a benchmark for evaluating data science agents with 
realistic data analysis and data modeling tasks collected from 
modeloff and kaggle. 
Given a *task instruction (may contain image and table)* and
*data files*, a data science agent is tasked with generating 
a *solution* that resolves the described task.

<p align="center">
<img src="figures/overview.svg">
</p>

## Set Up
For evaluation, you should install the Python packages in the requirments.txt file.

## Usage

1. Clone this repo.
2. Install all the requirments in Set Up.
3. For evaluation on data analysis task, refer to [./data_analysis/readme.md](https://github.com/LiqiangJing/DSBench/blob/main/data_analysis/readme.md).
4. For evaluation on data modeling task, refer to [./data_modeling/readme.md](https://github.com/LiqiangJing/DSBench/blob/main/data_modeling/readme.md).

## Results

<p align="center">
<img src="figures/result1.svg">
</p>

<p align="center">
<img src="figures/result2.svg" width="600">
</p>

##  Disclaimer
The dataset provided is intended solely for educational and research purposes, with the goal of fostering research in related areas. Users of this dataset are required to adhere to the following guidelines:

- Data Source and Accuracy: While efforts have been made to curate and organize the data, we make no guarantees regarding the accuracy, completeness, or timeliness of the dataset. Users are encouraged to independently verify the data's accuracy and assume full responsibility for any conclusions drawn from it.

- Usage Restrictions: This dataset is strictly for non-commercial use. Any commercial development or profit-driven activity requires explicit written permission from the dataset providers.

- Privacy and Compliance: Users must ensure that their use of the dataset complies with all applicable laws and regulations, particularly those related to privacy and data security. The dataset providers are not responsible for any legal consequences arising from improper use of the data.

- Non-Infringement: The pre-processed data provided by us is intended solely for educational and research purposes. We do not claim ownership of the original data, and any use of this data should respect the rights of the original creators. Users are responsible for ensuring that their use of the data does not infringe on any copyrights or other intellectual property rights.

- Disclaimer of Liability: The dataset providers shall not be held liable for any direct or indirect consequences resulting from the use of this dataset, including but not limited to losses, damages, or liabilities arising from reliance on the information contained within the dataset.





