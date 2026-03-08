export interface PromptPreset {
  id: string;
  label: string;
  description: string;
  prompt: string;
}

export const DEFAULT_SYSTEM_PROMPT = `你是 DeepAnalyze 的数据分析助手。

请默认使用中文回答，并遵循以下原则：
1. 优先理解用户上传的数据、字段含义和业务问题。
2. 输出尽量结构化，结论先行，再给关键依据。
3. 生成代码时优先使用 Python 数据分析栈（pandas、numpy、matplotlib、seaborn、scipy）。
4. 如果会生成文件，请给出文件用途，并保持命名清晰。
5. 如果信息不足，先提出最小必要假设，并说明假设内容。`;

export const DATA_ANALYSIS_PROMPT_PRESETS: PromptPreset[] = [
  {
    id: "eda",
    label: "探索性分析",
    description: "快速理解数据概况、分布、缺失值和异常点。",
    prompt:
      "请对当前数据做探索性分析，先概览字段和数据质量，再给出关键分布、异常点、相关性和后续建议。",
  },
  {
    id: "cleaning",
    label: "数据清洗",
    description: "定位缺失、重复、异常值，并给出清洗方案。",
    prompt:
      "请检查当前数据中的缺失值、重复值、类型问题和异常值，给出清洗策略，并在必要时生成可直接运行的清洗代码。",
  },
  {
    id: "viz",
    label: "可视化报告",
    description: "输出适合汇报的图表和文字解读。",
    prompt:
      "请生成一组适合汇报的数据可视化，突出最重要的趋势、对比和异常，并给出每张图的业务解读。",
  },
  {
    id: "stats",
    label: "统计检验",
    description: "比较组间差异并解释显著性。",
    prompt:
      "请基于当前数据设计合适的统计检验，说明假设、方法选择理由、显著性结果和业务含义。",
  },
  {
    id: "sql",
    label: "SQL 分析",
    description: "面向 SQLite / 表结构的查询分析。",
    prompt:
      "请基于当前数据库或表结构，设计 SQL 分析方案，逐步给出查询语句、结果解释和可视化建议。",
  },
  {
    id: "feature",
    label: "建模前特征分析",
    description: "识别重要特征、目标变量关系和可建模性。",
    prompt:
      "请从建模准备的角度分析当前数据，识别候选目标变量、重要特征、特征质量问题以及下一步建模建议。",
  },
  {
    id: "report",
    label: "结论总结",
    description: "适合直接复制到汇报或文档中的总结。",
    prompt:
      "请把当前分析过程和结果整理成简洁的结论总结，包含关键发现、证据、风险点和下一步行动建议。",
  },
];
