import re
import io
import contextlib
import os

from openai import OpenAI

import warnings

# Code执行过程中可能会产生大量warning信息占用上下文窗口
warnings.filterwarnings('ignore')

deepanalyze_system_prompt = """\
你是一个自主数据科学智能体 **DeepAnalyze**，模拟人类数据科学家的“思考–行动–观察”工作流。  
**每次响应必须且只能包含一个动作标签**，格式如下：

```xml
<ActionName>
[具体内容]
</ActionName>
```

#### 动作定义

- **`<Analyze>`**：任务规划、推理、假设、结果解读或反思。  
- **`<Understand>`**：主动表达对数据源（如表、文件）结构与语义的探索意图（例如字段、类型、分布）。**不是向用户提问，而是准备通过代码探查。**  
- **`<Code>`**：生成可执行的 Python 代码（pandas/numpy/matplotlib 等）以操作数据。  
- **`<Execute>`**：由系统自动注入（含上轮 `<Code>` 的执行结果）；你不得生成此标签。  
- **`<Finish>`**：输出最终结论、报告或建议。

#### 核心规则

1. **每次仅输出一个动作标签**。
2. **`<Understand>` 后，下一轮应输出 `<Code>`** 以实际探查数据。
3. **`<Code>` 必须是本轮唯一输出**；系统将在下一轮末尾自动追加 `<Execute>`。
4. **收到 `<Execute>` 后，下一轮必须以 `<Analyze>` 开始**，解读结果。
5. **`<Finish>` 必须是整个任务的最后一个动作**。
6. 不得虚构执行结果；所有结论需基于真实反馈。

---

#### 🧪 One-Shot 示例（供参考，非输出内容）

**用户输入**：  
“哪个产品类别的销售额最高？”

**你的输出序列应为**：

```xml
<Analyze>
需要按产品类别聚合销售额并比较总和。首先确认销售数据是否存在及包含必要字段。
</Analyze>
```

→（系统无注入，继续）

```xml
<Understand>
目标：理解 sales.csv 是否包含 'category' 和 'revenue' 字段。
</Understand>
```

→（你下一轮输出）

```xml
<Code>
import pandas as pd
df = pd.read_csv('sales.csv')
print("Columns:", df.columns.tolist())
print("Sample revenue values:", df['revenue'].head() if 'revenue' in df.columns else "No revenue column")
</Code>
```

→（系统执行后，在下一轮输入末尾自动添加）

```xml
<Execute>
Columns: ['order_id', 'category', 'revenue']
Sample revenue values: 0    120.5, 1    89.0, ...
</Execute>
```

→（你下一轮输出）

```xml
<Analyze>
数据包含 'category' 和 'revenue'。接下来将按类别求和并排序。
</Analyze>
```

→（继续... 最终）

```xml
<Finish>
“Electronics” 类别销售额最高（$1.25M），显著领先其他品类。建议优先分配营销资源至该品类。
</Finish>
```

---

现在请等待用户输入任务，并以 `<Analyze>` 开始你的第一个动作。
"""

deepanalyze_system_prompt_with_ask = """\
你是一个自主数据科学智能体 **DeepAnalyze**，模拟人类数据科学家的“思考–行动–观察”工作流。  
**每次响应必须且只能包含一个动作标签**，格式如下：

```xml
<ActionName>
[具体内容]
</ActionName>
```

#### 动作定义

- **`<Analyze>`**：仅用于任务规划、逻辑推理、**基于已知事实的解释**、结果反思或障碍诊断。**严禁在此阶段引入未经验证的业务假设（如字段含义、用户意图、指标定义等）**。若存在不确定性，应导向 `<Ask>` 或通过 `<Code>` 获取证据。
- **`<Understand>`**：主动表达对数据源（如表、文件）结构与语义的探索意图（例如字段、类型、分布）。**这不是提问，也不是猜测——而是声明即将用代码探查。**
- **`<Code>`**：生成可执行的 Python 代码（pandas/numpy/matplotlib 等）以操作数据。
- **`<Ask>`**：**仅在以下情况才使用**：
  - 数据中缺失定义关键概念所需的信息（如“高价值客户”无对应字段）；
  - 字段名称或值存在歧义，无法从数据本身推断其业务含义；
  - 用户目标依赖外部上下文（如时间范围、成功标准、数据位置）未提供。
  **问题必须具体、单一、可回答，且不能预设答案。**
- **`<Execute>`**：由系统自动注入，包含上一轮 `<Code>` 的执行结果 **或** 用户对 `<Ask>` 的回答；你不得生成此标签。
- **`<Finish>`**：输出最终结论、报告或建议，**所有陈述必须基于已验证的数据或用户确认的信息**。

#### 核心规则（强化版）

1. **每次仅输出一个动作标签**。
2. **`<Understand>` 后，通常应输出 `<Code>`** 以实际探查数据。
3. **`<Code>` 或 `<Ask>` 必须是本轮唯一输出**；系统将在下一轮末尾自动追加 `<Execute>`。
4. **收到 `<Execute>` 后，下一轮必须以 `<Analyze>` 开始**，但仅限解读**已返回的内容**，不得延伸假设。
5. **`<Finish>` 必须是整个任务的最后一个动作**。
6. **绝对禁止在 `<Analyze>` 或 `<Understand>` 中做出未经证实的业务假设**。例如：
   - ❌ 错误：“‘segment’ 很可能代表客户价值等级。”
   - ✅ 正确：“‘segment’ 字段存在，但其业务含义未知，需确认是否与客户价值相关。”
7. 所有代码必须安全、可执行；所有结论必须基于真实反馈或用户明确确认。
8. 若对用户目标中的术语（如“留存”“高价值”“活跃”）缺乏明确定义，**必须先 `<Ask>`，不得自行定义**。

---

#### 🧪 One-Shot 示例（含 `<Ask>` 场景）

**用户输入**：  
“分析高价值客户的留存情况。”

**你的输出序列**：

```xml
<Analyze>
用户提到“高价值客户”和“留存”，但未提供定义。需确认：(1) 如何识别高价值客户？(2) “留存”的时间窗口和判定标准是什么？首先检查数据是否存在相关字段。
</Analyze>
```

```xml
<Understand>
目标：检查 customers.csv 是否包含可用于定义客户价值或活跃状态的字段（如消费金额、套餐类型、最后活跃日期等）。
</Understand>
```

```xml
<Code>
import pandas as pd
df = pd.read_csv('customers.csv')
print("Columns:", df.columns.tolist())
print("Sample rows:")
print(df.head(2))
</Code>
```

→ 系统注入：
```xml
<Execute>
Columns: ['user_id', 'signup_date', 'last_active', 'segment']
Sample rows:
   user_id signup_date last_active segment
0      101  2023-05-01  2024-01-10    A
1      102  2023-06-12  2023-11-05    B
</Execute>
```

```xml
<Analyze>
数据包含 'segment' 和 'last_active' 字段，但无法确定 'segment' 是否表示客户价值等级（A/B/C 的含义未知），也无法确认“留存”的具体定义（如是否指最近90天内活跃）。需向用户澄清这两个关键点。
</Analyze>
```

```xml
<Ask>
有两个问题需要确认：
1. 'segment' 字段的取值 A、B、C 是否代表客户价值等级？如果是，哪个值对应“高价值客户”？
2. “留存”是指用户在最近多少天内有活动？例如，是否以今天为基准，过去90天内有登录/交易即视为留存？
</Ask>
```

→ 用户回答后，系统注入：
```xml
<Execute>
1. 是的，segment='A' 表示高价值客户。
2. 留存定义为：在过去90天内 last_active 有记录。
</Execute>
```

```xml
<Analyze>
已确认：高价值客户为 segment='A'，留存定义为 last_active 在最近90天内。下一步将计算该群体的留存率。
</Analyze>
```

```xml
<Code>
import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('customers.csv')
cutoff = datetime.today() - timedelta(days=90)
df['last_active'] = pd.to_datetime(df['last_active'])
high_value = df[df['segment'] == 'A']
retention_rate = (high_value['last_active'] >= cutoff).mean()
print(f"High-value customer retention rate: {retention_rate:.2%}")
</Code>
```

→ ……最终

```xml
<Finish>
在用户确认的定义下，高价值客户（segment='A'）的90天留存率为78.4%。建议结合流失预警模型进一步优化干预策略。
</Finish>
```

---

现在请等待用户输入任务，并以 `<Analyze>` 开始你的第一个动作。
"""


class DeepAnalyzeVLLM:
    """
    DeepAnalyzeVLLM provides functionality to generate and execute code
    using a vLLM API with multi-round reasoning.
    """

    def __init__(
            self,
            model_name: str,
            api_url: str = "",
            max_rounds: int = 20,
            # 是否可交互（True时，允许模型输出<Ask>标签）
            is_interactive: bool = False,
    ):
        self.model_name = model_name
        self.api_url = api_url
        self.max_rounds = max_rounds
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.is_interactive = is_interactive
        self.env = None

    @classmethod
    def init_code_execute_env(cls):
        namespace = {}
        # 手动导入并注入需要的模块
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        namespace['pd'] = pd
        namespace['plt'] = plt
        namespace['sns'] = sns
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        namespace['__builtins__'] = __builtins__
        return namespace

    def cleanup_namespace(self):
        """安全清理namespace环境，防止敏感数据泄露"""
        if self.env is not None:
            # 清理可能存在的敏感数据
            keys_to_delete = [key for key in self.env.keys() if
                              not key.startswith('__') and key not in ['pd', 'plt', 'sns']]
            for key in keys_to_delete:
                del self.env[key]
            # 重置环境
            self.env = None

    @classmethod
    def extract_xml_content(cls, markdown_str):
        # 使用非贪婪匹配，提取 ```xml 和 ``` 之间的全部内容
        match = re.search(r'```xml\s*(.*?)\s*```', markdown_str, re.DOTALL)
        if match:
            return match.group(1).strip()  # 去除首尾空白
        else:
            # 如果没有匹配到，可选择返回原字符串或报错
            return markdown_str

    def execute_code(self, code_str: str) -> str:
        """
        Executes Python code and captures stdout and stderr outputs.
        Returns the output or formatted error message.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
                    stderr_capture
            ):
                exec(code_str, self.env)
            output = stdout_capture.getvalue()
            if stderr_capture.getvalue():
                output += stderr_capture.getvalue()
            return output
        except Exception as exec_error:
            # 简化异常处理逻辑
            error_message = f"{type(exec_error).__name__}: {str(exec_error)}"
            if stderr_capture.getvalue():
                error_message += f"\n{stderr_capture.getvalue()}"
            return f"[Error]:\n{error_message.strip()}"

    @classmethod
    def get_user_input(cls, ask: str):
        user_input = input(ask)
        return user_input

    def generate(
            self,
            prompt: str,
            workspace: str,
            temperature: float = 0.5,
            max_tokens: int = 8192,
            top_p: float = None,
            enable_thinking: bool = False,
    ) -> dict:
        """
        Generates content using vLLM API and executes any <Code> blocks found.
        Returns a dictionary containing the full reasoning process.
        """
        original_cwd = os.getcwd()
        os.chdir(workspace)
        self.env = self.init_code_execute_env()
        try:
            system_prompt = deepanalyze_system_prompt_with_ask if self.is_interactive else deepanalyze_system_prompt
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            response_message = []
            for round_idx in range(self.max_rounds):
                response_data = self.client.chat.completions.create(
                    model=self.model_name,  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    extra_body={
                        "enable_thinking": enable_thinking
                    }
                )

                ans = self.extract_xml_content(response_data.choices[0].message.content)

                if code_match := re.search(r"<Code>(.*?)</Code>", ans, re.DOTALL):
                    code_content = code_match.group(1).strip()
                    md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
                    code_str = md_match.group(1).strip() if md_match else code_content
                    exe_output = self.execute_code(code_str)
                    exe_output_str = f"<Execute>\n{exe_output}\n</Execute>"
                    ans += f"\n{exe_output_str}"
                elif self.is_interactive and (ask_match := re.search(r"<Ask>(.*?)</Ask>", ans, re.DOTALL)):
                    ask_str = ask_match.group(1).strip()
                    exe_output = self.get_user_input(ask_str)
                    exe_output_str = f"<Execute>\n{exe_output}\n</Execute>"
                    ans += f"\n{exe_output_str}"
                response_message.append(ans.strip())
                print(f"{ans}")
                if "<Finish>" in ans:
                    break
                # Append messages for next round
                messages.append({"role": "assistant", "content": ans})

            reasoning = "\n".join(response_message)
            return {"reasoning": reasoning}
        finally:
            # 安全清理：任务执行完成后清理namespace环境
            self.cleanup_namespace()
            os.chdir(original_cwd)


def execute_data_analyze_task():
    # 目前尝试了qwen3-8b qwen3-coder-30b-a3b-instruct qwen3-32b
    deepanalyze = DeepAnalyzeVLLM(model_name="qwen3-coder-30b-a3b-instruct", is_interactive=True)

    task1 = """\
数据分析任务 - 接口调用情况分析

请基于interface_calls.xlsx文件中的数据，回答以下问题：

1. 应用来源分析：不同应用来源（网页应用, 移动应用等）对接口的调用分布如何？哪种应用来源的调用量最大？    
    """

    # 缺乏背景知识的任务，高价值户是什么含义？文件中没有体现说明。因此需要agent向用户主动询问
    # 高价值户含义：（- 存款余额分类标准如下：* 低价值客户：存款余额 < 10万元 * 中价值客户：10万元 ≤ 存款余额 < 20万元 * 高价值客户：存款余额 ≥ 20万元）
    task2 = """\
数据分析任务说明：

你是一名银行数据分析师，现在需要对一批个人贷款客户的数据进行分析。请使用提供的Excel文件（bank_data.xlsx）完成以下任务：

高价值客户特征分析：
   - 分析高价值客户的年龄分布特点
   - 分析高价值客户的收入水平特点
   - 分析高价值客户的贷款金额特点

    """

    extra_content = """\
```python
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 如果要绘制图表，不要使用plt.show()，仅保存到本地即可
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
```

**注意**:上述代码已实现，在编写<Code>代码</Code>时，不要重复编写，可直接调用    
    """

    task_execute_trace = deepanalyze.generate(
        prompt=f"{task2}\n{extra_content}",
        workspace="example",
        temperature=0.3,
        top_p=1.0,
    )


if __name__ == '__main__':
    execute_data_analyze_task()
