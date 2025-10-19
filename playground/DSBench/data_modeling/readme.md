# Data Modeling task


eval_code_interpreter.ipynb: A sample code to evaluate 
Code Interpreter with GPT series models on our data modeling task.

./output_model/gpt-3.5-turbo-0125: This is evaluation results of Code
Interpreter with gpt-3.5-turbo-0125. It recodes the cost, 
predicted answer, consuming time....

If you want to evaluate AutoGen with eva_autogen_gpt.ipynb, please install all required [environments](https://microsoft.github.io/autogen/docs/installation/).

### Download all competations
Please download all competations from the 
[Kaggle](https://www.kaggle.com/competitions) website.
 The details and download url of each competition can be found in *data.json*
file.
To simplify, you can use the [Kaggle API](https://www.kaggle.com/docs/api) for data dowloading.


### Process the downloaded file with our script.


### For simplicity, we also provide the processed files by [google drive](https://drive.google.com/file/d/1aH5HHnbKxO9Qq0A4d9MUOKmBj8KwU9Pe/view?usp=sharing) and [huggingface](https://huggingface.co/datasets/liqiang888/DSBench/blob/main/data_modeling/data.zip).


*Non-Infringement: The pre-processed data provided by us is intended solely for educational and research purposes. We do not claim ownership of the original data, and any use of this data should respect the rights of the original creators. Users are responsible for ensuring that their use of the data does not infringe on any copyrights or other intellectual property rights.*



## How to evaluate CodeInterpreter on data modeling tasks
1. Unzip data.zip into the current path.
3. Set your OpenAI key in eval_code_interpreter.ipynb.
4. Run eval_code_interpreter.ipynb to evaluate the performance.
5. ```bash
   python score4each_com.py
   ```
    Computer the performance of the model on each competition with the corresponding metric, such as accuracy. Save the results in the directory `./save_performance/gpt-3.5-turbo-0125`.
7. ```bash
   python show_result.py
   ```
   Show the performance of the model based on the results generated from the last step.
