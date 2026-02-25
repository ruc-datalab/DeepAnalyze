### DeepAnalyze 医疗领域应用示例：心衰结局预测

临床结局预测是临床研究中常见且具有实践意义的一类建模任务。围绕明确的结局事件，研究者通常需要在可解释、可复核的前提下完成变量筛选、模型构建与性能评估，并将结果整理成可直接用于论文或课题报告的材料。以心衰队列为例，研究问题往往较为具体：在既定的基线特征条件下，哪些变量更适合纳入模型？不同建模方法在区分能力上表现如何？最终形成的交付内容通常包括建模特征集合、模型性能指标（如 AUC、Accuracy）、ROC/PR 与混淆矩阵等关键图表，以及一段能够直接进入结果解读部分的文字摘要。

本示例选择心衰结局预测作为展示场景，是因为该任务流程完整，涵盖变量筛选、模型训练、指标计算与结果解释等多个环节，同时在临床研究与教学中具有较高的通用性。传统流程下，这类任务通常需要在数据处理脚本、模型构建代码、性能评估与图表绘制之间反复切换；在具备医学背景与一定编程能力的前提下，完成一次可写入初稿的首版结果，往往需要约 2–4 小时，其中相当一部分时间消耗在脚本调试、结果整理与图表规范化处理上。

在本示例中，DeepAnalyze 将“变量筛选—模型训练—性能评估—结果汇总”串联为一条连续流程：首先基于 L1/Lasso 完成特征筛选（本例保留 9 个关键变量），随后在同一特征集合上构建逻辑回归与随机森林模型，并给出对比结果（LR：Accuracy = 0.8000，AUC = 0.8332；RF：Accuracy = 0.7556，AUC = 0.8610）。系统同时生成 ROC、PR、混淆矩阵及特征重要性等图表，并输出结构化的中文结果摘要。实际运行中，从数据读取到形成可用于初稿的完整结果包约需 15–30 分钟；过程中出现一次字段假设错误导致的报错，系统完成自动修正后继续执行，未造成流程中断。本示例基于公开的心衰患者临床记录数据，围绕结局变量 DEATH_EVENT（死亡事件）完成一条完整的建模链路：数据理解 → 变量筛选 → 建模 → 性能评估与对比 → 结果解读。

#### 1. 资源链接

- 直播页面（含回放入口与简介）：https://www.heywhale.com/home/competition/692e87c437fbb22875f6a0ce 
- API Key 申请：https://heywhale.feishu.cn/share/base/form/shrcnnBRgO0x2qhx40yq4m1HxUg  
- 在线 Demo（Gradio）：https://www.heywhale.com/api/model/services/693c1127d702f81868020fcd/app/  

#### 2. 数据说明

##### 2.1 主数据集

- 文件：`heart_failure_clinical_records_dataset.csv`
- 规模：299 行 × 13 列
- 目标变量：`DEATH_EVENT`（0/1）

特征字段（12 项）：

- 连续/数值：
    - `age`（年龄）
    - `creatinine_phosphokinase`（肌酸磷酸激酶）
    - `ejection_fraction`（射血分数）
    - `platelets`（血小板）
    - `serum_creatinine`（血清肌酐）
    - `serum_sodium`（血清钠）
    - `time`（随访时间）
- 二值：
    - `anaemia`（贫血）
    - `diabetes`（糖尿病）
    - `high_blood_pressure`（高血压）
    - `sex`（性别）
    - `smoking`（吸烟）

数据完整性（本次运行观察到）：

- 无缺失值（299 行字段均完整）
- `DEATH_EVENT` 阳性率约 32.1%（96/299）

##### 2.2 数据概览

- 图 1：`DEATH_EVENT` 类别分布

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=NmQwZmE5MDM1YmFiNDBmMDQyMDFmYWQxZjA0NDBhYzJfN2xGZzhSN05neDRUMk1JZ1ZzRk9BbXRXNE0xQ2ZYaFBfVG9rZW46WEZqNGJyOXJzb2xDeVV4Z0wwTGNQdWo3blFmXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

- 图 2：关键连续变量分布（`age`, `ejection_fraction`, `serum_creatinine` 等）

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=OTcxM2IxOWRhOTFlNzllZDJjOWVhMjE5NzA4ZmQ1NTNfRXI1b0REbzRGbmU3UmtzZmNnNjFmc2Fsbm5PV0pLeXVfVG9rZW46UExRcmIzaTRSbzQ3a054a0dwV2N3RXVSbjZmXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

- 图 3：数值特征相关性热力图

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=YWQ1YzkyZjYyMzI4MGY0MmNmZjlmMjUzYTUyOWMwNDRfTVlQVkVmanRWbm9FaFhXMWpmaGhlemxLbFNXTmpiVG9fVG9rZW46QlJXUmJ5bllRb2VkSXN4d3FkRmNhZVQxbjdiXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

#### 3. 任务定义

**任务 A：变量筛选**

问题：哪些变量可以用于构建预测模型？  

目标：识别目标变量与可用特征字段，给出可建模字段清单，并指出建模前的必要处理。

**任务 B：特征筛选后建模对比**

问题：用 L1/Lasso 风格筛选出的变量构建逻辑回归与随机森林模型，哪个更优？  

目标：在同一特征子集上训练两类模型并对比指标。

**评估指标（本示例使用）：**

- Accuracy：分类正确率
- AUC：ROC 曲线下面积，反映排序/区分能力（对类别不均衡更稳定）

#### 4. 提示词（Prompts）

##### Prompt 1：变量可用性

```Plain
# Instruction
用中文回答，哪些变量可以用于构建预测模型？

# Data
File 1:
{"name": "heart_failure_clinical_records_dataset.csv", "size": "12.2KB"}
```

##### Prompt 2：L1/Lasso + 模型对比

```Plain
# Instruction
用中文回答，用lasso回归筛选出的变量构建逻辑回归和随机森林模型，哪个更优？

# Data
File 1:
{"name": "heart_failure_clinical_records_dataset.csv", "size": "12.2KB"}
```

#### 5. 实验设置（本次运行参数）

- 训练/测试划分：70% / 30%
- 随机种子：`random_state = 42`
- 特征标准化：对线性模型使用标准化（StandardScaler）
- 特征筛选：L1/Lasso 风格（本次运行中使用 `alpha = 0.01`）
- 模型：
    - Logistic Regression（`max_iter = 1000`）
    - Random Forest（`n_estimators = 100`, `random_state = 42`）

注意：

- 本次筛选采用了 Lasso 回归的非零系数作为特征子集（L1 风格）。在更严格的二分类建模规范中，可使用 L1 正则的逻辑回归或交叉验证策略进一步稳定筛选结果。

#### 6. 结果汇总

##### 6.1 任务 A：可用变量

- 目标变量：`DEATH_EVENT`
- 可用特征：除 `DEATH_EVENT` 外的全部字段均可作为特征输入

建模前建议：

- 训练/测试划分
- 特征标准化（对逻辑回归等线性模型更重要）
- 类别特征编码（本数据二值 0/1，可直接使用）

##### 6.2 任务 B：L1/Lasso 风格筛选结果

筛选得到的特征（9 个）：

| 特征                |
| ------------------- |
| age                 |
| anaemia             |
| ejection_fraction   |
| high_blood_pressure |
| serum_creatinine    |
| serum_sodium        |
| sex                 |
| smoking             |
| time                |

- 图 4：L1 系数条形图

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=NWI1ZDQzMGU0Zjk2MzZjNGYzNGM5NjQ1ZmNiODA3M2VfbGVoUmc0RUNDbnh3MDhMc3I3ektnejhyZEl2ZlJLSUNfVG9rZW46Rnk2UmI0bjRSb2VlZWR4UGtXR2NYcnlobklSXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

- 图 5：随机森林特征重要性

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=N2VkNGM3ZDgxOTBiZTkzMDk2NzUzNDEwZGQ4YmEzYjFfd3gySk40QVVDRjlzWjZ5ZFF6YUdzQUxvMlo5d2E2TGZfVG9rZW46V3hjNGJnZERab0lwYkd4UjdGNGNSeHRTblBiXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

##### 6.3 模型对比（基于筛选后的特征子集）

| 模型                | Accuracy | AUC    |
| ------------------- | -------- | ------ |
| Logistic Regression | 0.8000   | 0.8332 |
| Random Forest       | 0.7556   | 0.8610 |

结论（本次运行）：

- AUC 更高：Random Forest（0.8610 > 0.8332）
- Accuracy 更高：Logistic Regression（0.8000 > 0.7556）
- 若以区分能力为优先（常见于医疗结局预测）：Random Forest 更优
- 图 6：ROC 曲线（LR vs RF）

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=NDNiNDFmY2VjOTQyNmY3NGNjYTZmYTQzOGZlZGZiMTlfMEdLUkd1Nk94UHA0TklVZ09pUEpOcE50RXhBWnZHbDJfVG9rZW46Qmh4eWJNREVTb3NqOVp4OG15SWNKOFhrbjJjXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

- 图 7：混淆矩阵（LR vs RF）

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=NjkxYTFiNzYxOTQwMGIxZWNmY2NkMzVjNzVhOTZiN2FfSlJQdzQ5cXhmUzAzcDB5d1ZvM3JtaThWcDVocnVkOE9fVG9rZW46S1EydGJ4M3hBb0F4eTN4bVdoVmM4Q3pIbjNiXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

- 图 8：PR 曲线

![img](https://uw5gkcg3u6.feishu.cn/space/api/box/stream/download/asynccode/?code=N2NlYTM4ZmFkMDYxYzMzMWNjODRlOGU2YTdkYzljZmVfRjZNVlNNUWFrdE1vMmwzbDNQejdVc2tpcUl1YUVadklfVG9rZW46WkEwT2JzdkpOb3JrR1N4MUxKbmNJcnE4bmRiXzE3NzIwMDIzNzY6MTc3MjAwNTk3Nl9WNA)

##### **6.4 端到端产出与效率**

本次流程从变量确认、特征筛选到模型训练与评估，最终输出了一组可直接进入初稿的结果包：特征子集（9 个）、LR/RF 的 AUC 与 Accuracy 对比、ROC/PR/混淆矩阵等图表，以及结构化的中文结果摘要。对比传统做法在多个脚本/工具间拆分完成的方式，这类端到端串联输出减少了反复切换与手工整理的工作量；首版结果通常可在 15–30 分钟内产出，而同等完整度的传统流程往往需要 2–4 小时。此外，本次运行中出现一次字段假设错误导致的报错，系统完成自动修正后继续执行，避免了中断式返工。

从工作方式上看，本示例把原本需要分别完成的几件事合并到一次连续流程中：先基于 L1/Lasso 给出更聚焦的特征集合，再在同一特征集合上完成两类模型的训练与对比，并自动产出对应用于报告的评估图表与解释性信息（如特征重要性）。这使得研究者更多把精力放在研究问题、临床合理性与后续验证上，而不是反复写脚本、修图与整理表述。

#### 7. 结果解读

##### 7.1 选中特征的直观含义

本次筛选得到的特征覆盖了：

- 基础信息：`age`, `sex`
- 合并症与危险因素：`anaemia`, `high_blood_pressure`, `smoking`
- 心功能指标：`ejection_fraction`
- 生化指标：`serum_creatinine`, `serum_sodium`
- 随访信息：`time`

这些特征能够反映患者的基础状况、心功能与代谢/肾功能相关指标，对结局预测具有合理性。

##### 7.2 AUC 与 Accuracy 的差异

- Accuracy 受阈值影响较大，在类别分布不均衡时可能不稳定
- AUC 反映模型对正负样本的整体区分能力，通常更适合医疗结局预测的对比

#### 8. 常见问题与注意事项

- 字段含义与单位：若需要更严格医学解释，建议补充数据字典或字段来源说明
- 随访时间 `time`：可能与结局发生相关，建模时需明确其临床含义与使用边界
- 特征筛选稳定性：不同随机划分、不同 `alpha` 可能导致选中特征变化，建议后续加入交叉验证与稳定性分析
- 模型可解释性：
    - 逻辑回归便于解释系数方向与大小
    - 随机森林可通过特征重要性与 SHAP 等方法解释（需环境支持）

#### 9. 复现方式

##### 9.1 在线 Demo

1. 打开在线 Demo：https://www.heywhale.com/api/model/services/693c1127d702f81868020fcd/app/  
2. 上传 `heart_failure_clinical_records_dataset.csv`
3. 依次运行 Prompt 1 与 Prompt 2

##### 9.2 本地运行

- 模型：`DeepAnalyze-8B`
- API：`http://localhost:8000/v1/chat/completions`

建议目录结构：

```Plain
example/medical_heart_failure_prediction/
  README.md
  data/
    heart_failure_clinical_records_dataset.csv
  prompts/
    prompt1.txt
    prompt2.txt
  outputs/
    results_summary.json
    figures/
      figure_01_class_balance.png
      figure_02_distributions.png
      figure_03_corr.png
      figure_04_l1_coeff.png
      figure_05_rf_importance.png
      figure_06_roc.png
      figure_07_confusion.png
      figure_08_pr.png
      figure_09_demo.png
```

#### 10.实际价值

从实际使用体验来看，DeepAnalyze 的价值并不只是“流程顺畅”，而是体现在端到端效率与调试负担上的实质改善。在传统流程中，完成一次可写入初稿的预测建模，通常需要约 3–5 小时：其中变量筛选与数据处理约 1–2 小时，模型构建与指标对比约 1 小时，图表整理与结果撰写约 1–2 小时；若中间出现字段或类型错误，还需额外调试时间。本示例中，从数据读取到得到包含变量筛选结果、模型对比指标与基础图表的“首版结果包”，实际耗时约 20–40 分钟；后续人工复核与微调约 20–30 分钟即可完成初稿整理。整体端到端时间缩短约 60%–80%。

在编码工作量方面，传统流程往往需要分别编写特征筛选、模型训练、指标计算与绘图脚本，并在不同代码块之间反复切换；本例中，变量筛选（Lasso）、模型对比（Logistic / Random Forest）以及 AUC、ROC 等指标输出在一次流程中完成，显著减少了重复编码与脚本维护工作。

此外，在执行过程中曾出现一次字段假设错误（列名问题），系统能够自动定位并修正后继续运行。若采用手工脚本，通常需要中断流程、手动排查并重新运行相关代码。此类自动纠错能力对于减少调试时间具有直接意义。

总体而言，在保持模型性能处于可用水平（AUC 0.83–0.86）的前提下，本示例实现了显著的时间压缩与调试负担下降，使研究者能够将更多精力集中于结果的临床解释与研究设计本身。

#### 11.边界/注意

尽管在常规预测建模流程中表现稳定，但在部分医学统计扩展分析方面仍存在边界。例如，临床预测模型常配套制作列线图（Nomogram），这一任务目前主要依赖 R 语言生态中的成熟包（如 rms）；Python 生态中对应工具相对有限，因此在当前环境下仍需借助其他软件完成。总体而言，在标准预测建模流程内，其输出结果具有良好的可用性；在超出常规机器学习建模范畴的专门医学统计分析上，仍需结合专业统计环境完成。

#### 12.一句话评价

DeepAnalyze为医学数据分析提供了一条较为完整的自动化实现路径，在临床预测模型构建这类常见研究任务中，能够有效整合变量筛选、模型训练、性能评估与结果整理流程，使首版研究结果更快成型，对医学科研与教学场景具有现实意义。
