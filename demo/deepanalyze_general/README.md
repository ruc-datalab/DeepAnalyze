# DeepAnalyze General
**Designed dedicated prompts to enable general large models to emulate the execution trajectory of DeepAnalyze-8b `<Analyze>`-> `<Code>` -> `<Execute>`, completing data analysis tasks**

## 🚀 Quick Experience Support

Provides a dedicated `system prompt` and is fully compatible with OpenAI API (such as Alibaba Cloud Bailian platform), allowing users to quickly get started with DeepAnalyze's Agentic paradigm based on general large models, experiencing DeepAnalyze without complex configuration.

## 💬 Added `<Ask>` Interactive Mode

Based on the original `<Action>` tag system, an optional `<Ask>` tag has been added.  
DeepAnalyze can actively ask users questions during task execution to obtain key background information that cannot be inferred independently, thereby:

- Improving task completion quality  
- Avoiding directional errors caused by information gaps  
- Preventing error accumulation and compounding  

## ⚙️ `generate` Process Optimization

To improve execution efficiency and user experience, the code generation and execution process has been partially optimized:

- **Automatic warning filtering**: Automatically suppress `warning` logs when executing `<Code>` blocks to avoid occupying precious context windows.
- **Shared code execution environment**: Fixed the variable isolation issue between multiple `<Code>` blocks. Through a unified namespace, variable and state sharing across code blocks is achieved (for example, `df` defined in a previous `<Code>` can be directly used in subsequent `<Code>`), significantly simplifying logic writing.
- **Pre-import common dependencies**: The execution environment has pre-loaded commonly used Python libraries (such as `pandas`, etc.), eliminating the need to repeat `import` in each `<Code>` block, further streamlining user input. (Depending on the capabilities of the model used, prompts may not always be followed)

## 🧪 Simulated Scenario Examples

In the [`example/financial_insights_and_api_usage_analytics`](./example/financial_insights_and_api_usage_analytics) directory, we provide two test files close to real usage scenarios to help developers quickly understand, verify, and integrate DeepAnalyze's various functions.

## Specific Usage

1. Set the environment variable DASHSCOPE_API_KEY

2. Create a DeepAnalyzeVLLM instance:
   ```python
   deepanalyze = DeepAnalyzeVLLM(
       model_name="qwen3-coder-30b-a3b-instruct",  # Specify the model to use
       is_interactive=True  # Whether to enable interactive mode (allows using <Ask> tag)
   )
   ```

3. Prepare data analysis task description and working directory:
   ```python
   task = "Data analysis task description"
   workspace = "example/financial_insights_and_api_usage_analytics"  # Working directory path
   ```

4. Call the generate method to execute the data analysis task:
   ```python
   task_execute_trace = deepanalyze.generate(
       prompt=task,
       workspace=workspace,
       temperature=0.3,  # Control the randomness of generation
       top_p=1.0,        # Control the diversity of generation
   )
   ```

5. View task execution results:
   ```python
   print(task_execute_trace["reasoning"])  # Output the complete reasoning process, and the console will also gradually print action tags during execution
   ```

### Parameter Description

- `model_name`: Specifies the model name to use, such as "qwen3-coder-30b-a3b-instruct"
- `is_interactive`: Whether to enable interactive mode, default is False
  - True: Enable `<Ask>` tag, allowing the model to ask users questions when needed
  - False: Use basic mode, only supports `<Analyze>`->`<Code>`->`<Execute>` process
- `prompt`: Specific description of the data analysis task
- `workspace`: Working directory path, containing data files to be analyzed
- `temperature`: Controls the randomness of generated text, lower values result in more deterministic outcomes
- `top_p`: Controls the diversity of generated text, higher values result in more diverse outcomes

### Supported Action Tags

- `<Analyze>`: Task planning, reasoning, hypothesis, result interpretation, or reflection
- `<Understand>`: Actively express intent to explore data source structure and semantics
- `<Code>`: Generate executable Python code (pandas/numpy/matplotlib, etc.)
- `<Ask>`: (Only available in interactive mode) Ask users questions to obtain key background information
- `<Execute>`: Automatically injected by the system, containing code execution results or user responses
- `<Finish>`: Output final conclusions, reports, or recommendations

### Reference Task Examples

```plaintext
Data Analysis Task Description:

You are a bank data analyst who now needs to analyze data from a batch of personal loan customers. Please use the provided Excel file (bank_data.xlsx) to complete the following tasks:

High-value customer characteristic analysis:
   - Analyze age distribution characteristics of high-value customers
   - Analyze income level characteristics of high-value customers
   - Analyze loan amount characteristics of high-value customers
```

```plaintext
Data Analysis Task - Interface Call Analysis

Based on the data in the interface_calls.xlsx file, please answer the following questions:

1. Application source analysis: What is the distribution of interface calls from different application sources (web applications, mobile applications, etc.)? Which application source has the highest call volume?
```