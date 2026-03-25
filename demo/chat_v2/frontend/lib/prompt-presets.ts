export type UILanguage = "en" | "zh";

export interface LocalizedText {
  en: string;
  zh: string;
}

export interface PromptPreset {
  id: string;
  label: LocalizedText;
  description: LocalizedText;
  prompt: LocalizedText;
}

export const DEFAULT_SYSTEM_PROMPT = "";

export const DATA_ANALYSIS_PROMPT_PRESETS: PromptPreset[] = [
  {
    id: "eda",
    label: {
      en: "Exploratory Analysis",
      zh: "探索性分析",
    },
    description: {
      en: "Quickly inspect data quality, distributions, missing values, and anomalies.",
      zh: "快速理解数据质量、分布、缺失值与异常点。",
    },
    prompt: {
      en: "Please perform exploratory data analysis on the current dataset. Start with schema and data quality, then summarize distributions, anomalies, correlations, and recommended next steps.",
      zh: "请对当前数据做探索性分析，先概览字段和数据质量，再总结分布、异常点、相关性以及建议的下一步。",
    },
  },
  {
    id: "cleaning",
    label: {
      en: "Data Cleaning",
      zh: "数据清洗",
    },
    description: {
      en: "Detect missing values, duplicates, type issues, and outliers, then propose a cleaning plan.",
      zh: "定位缺失值、重复值、类型问题和异常值，并给出清洗方案。",
    },
    prompt: {
      en: "Please inspect the current data for missing values, duplicates, data type issues, and outliers. Provide a cleaning strategy and generate runnable cleaning code when helpful.",
      zh: "请检查当前数据中的缺失值、重复值、类型问题和异常值，给出清洗策略，并在必要时生成可直接运行的清洗代码。",
    },
  },
  {
    id: "viz",
    label: {
      en: "Visualization Report",
      zh: "可视化报告",
    },
    description: {
      en: "Create presentation-ready charts with concise interpretation.",
      zh: "输出适合汇报的图表，并配套简洁解读。",
    },
    prompt: {
      en: "Please generate a set of presentation-ready visualizations for the current data, highlight key trends, comparisons, and anomalies, and explain the business meaning of each chart.",
      // en: "Please generate a set of presentation-ready visualizations for the current data, highlight key trends, comparisons, and anomalies, and explain the business meaning of each chart. In the final answer, you need to reference the relevant images in Markdown format like ![xxx](xxx.png).",
      zh: "请为当前数据生成一组适合汇报的可视化，突出关键趋势、对比和异常，并说明每张图的业务含义。在最终的答案中你需要对相关图片进行markdown格式的引用,如![xxx](xxx.png)。",
    },
  },
  {
    id: "stats",
    label: {
      en: "Statistical Testing",
      zh: "统计检验",
    },
    description: {
      en: "Compare groups, explain significance, and interpret practical impact.",
      zh: "比较组间差异，解释显著性与实际意义。",
    },
    prompt: {
      en: "Please design appropriate statistical tests for the current data, explain the hypotheses and method selection, and interpret significance and business implications.",
      zh: "请基于当前数据设计合适的统计检验，说明假设、方法选择理由，并解释显著性结果和业务意义。",
    },
  },
  {
    id: "sql",
    label: {
      en: "SQL Analysis",
      zh: "SQL 分析",
    },
    description: {
      en: "Analyze SQLite tables and generate query-driven insights.",
      zh: "面向 SQLite 或表结构做查询分析。",
    },
    prompt: {
      en: "Please analyze the current database or table structure with SQL. Propose a query plan, provide the SQL statements step by step, and explain the results and visual follow-up ideas.",
      zh: "请基于当前数据库或表结构设计 SQL 分析方案，逐步给出查询语句，并解释结果和后续可视化思路。",
    },
  },
  {
    id: "feature",
    label: {
      en: "Feature Review",
      zh: "特征分析",
    },
    description: {
      en: "Assess feature quality, target candidates, and modeling readiness.",
      zh: "评估特征质量、候选目标变量和建模准备度。",
    },
    prompt: {
      en: "Please review the current data from a modeling-preparation perspective. Identify candidate targets, important features, feature quality issues, and recommended next modeling steps.",
      zh: "请从建模准备的角度分析当前数据，识别候选目标变量、重要特征、特征质量问题以及建议的下一步建模动作。",
    },
  },
  {
    id: "report",
    label: {
      en: "Executive Summary",
      zh: "结论总结",
    },
    description: {
      en: "Turn the analysis into a concise report summary for stakeholders.",
      zh: "整理为适合汇报或文档的简洁总结。",
    },
    prompt: {
      en: "Please summarize the current analysis into an executive-ready report that includes key findings, supporting evidence, risks, and recommended next actions.",
      zh: "请把当前分析整理成适合汇报的结论总结，包含关键发现、证据、风险点和建议的下一步行动。",
    },
  },
];
