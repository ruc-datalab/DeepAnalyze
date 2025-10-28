# Welcome to Contribution
> We welcome all forms of contributions, and merged PRs will be listed as contributors.

## Contribution on Code and Model

- We welcome all forms of contributions on DeepAnalyze's code and model, such as Docker packaging, DeepAnalyze model conversion and quantization, and submitting DeepAnalyze workflows based on closed-source LLMs. 
- You can submit a pull request directly.

## Contribution on Case Study

- We also especially encourage you to share your use cases and feedback when using DeepAnalyze; these are extremely valuable for helping us improve DeepAnalyze.
- You can place your use cases in a new folder under [`.example/`](.example/). We recommend following the folder structure of [`.example/analysis_on_student_loan/`](.example/analysis_on_student_loan/), which includes three parts:
    - `data/`: stores the uploaded files
    - `prompt.txt`: input instructions
    - `README.md`: documentation. We suggest including the input, DeepAnalyze’s output, outputs from other closed-source LLMs (optional), and your evaluation/comments of the case.
- DeepAnalyze only has 8B parameters, so we also welcome examples where DeepAnalyze performs slightly worse than the closed-source LLMs — this will help us improve DeepAnalyze.