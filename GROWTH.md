# GROWTH.md — DeepAnalyze Community Growth Playbook

> A practical guide to increase visibility, build community, and drive adoption for DeepAnalyze.

## 📍 Positioning

**DeepAnalyze** is the first fully open-source agentic LLM built specifically for autonomous data science. Unlike general-purpose agents or closed-source tools, DeepAnalyze offers:

| Differentiator | Why It Matters |
|----------------|----------------|
| **Fully Open** | Model, training data, and code all public — no vendor lock-in |
| **Local-First** | 8B parameters runs on consumer GPUs (16GB+) |
| **Domain-Specific** | Trained on 500K data science instructions, not general chat |
| **End-to-End** | Data prep → Analysis → Modeling → Visualization → Reports |
| **Multi-Format** | CSV, Excel, JSON, XML, Markdown — structured & unstructured |

**Competitors:**
- Julius AI, OpenAI Code Interpreter — proprietary, pay-per-use
- PandasAI — wrapper around existing LLMs, not specialized
- Data-Copilot — limited scope, less mature

**Target Users:**
- Data scientists wanting local, private analysis
- Researchers needing reproducible workflows
- Enterprises with data sovereignty requirements
- Developers building data-centric AI products

---

## ✅ Visibility Checklist

### 1. Awesome Lists (Highest ROI)

| List | Status | Submission Template |
|------|--------|---------------------|
| [awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents) | ❌ Not listed | PR: Add to "Open Source Agents" — `- [DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) - First open-source agentic LLM for autonomous data science. 8B model with WebUI, CLI, and Jupyter interfaces.` |
| [Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) | ❌ Not listed | PR: Add to "Open LLM" section — `- [DeepAnalyze-8B](https://huggingface.co/RUC-DataLab/DeepAnalyze-8B) - Specialized LLM for data science tasks, from Renmin University. [[paper]](https://arxiv.org/abs/2510.16872) [[code]](https://github.com/ruc-datalab/DeepAnalyze)` |
| [awesome-generative-ai](https://github.com/steven2358/awesome-generative-ai) | ❌ Not listed | PR: Add to "Autonomous Agents" — `- [DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) - Open-source agent for autonomous data analysis and research report generation.` |
| [awesome-data-science](https://github.com/academic/awesome-datascience) | ❌ Not listed | PR: Add to "Deep Learning" or create "AI Agents" section |
| [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning) | ❌ Not listed | PR: Python section, "Natural Language Processing" |

**Submission Tips:**
- One PR per list, follow their contribution guide
- Keep descriptions under 200 characters
- Include arXiv link for academic credibility

### 2. Hacker News / Reddit

**HN Title Options:**
```
Show HN: DeepAnalyze – Open-source AI that autonomously analyzes your data and writes reports
DeepAnalyze: We trained an 8B LLM specifically for data science (all open-source)
```

**Best Subreddits:**
- r/MachineLearning — "Project" flair, emphasize research contribution
- r/LocalLLaMA — Focus on 8B model, GPU requirements, quantization
- r/datascience — Practical demo with real dataset
- r/Python — WebUI/CLI code walkthrough

**Reddit Post Template (r/LocalLLaMA):**
```markdown
# DeepAnalyze-8B: First LLM trained specifically for autonomous data science

We released an 8B model that can:
- Analyze CSV/Excel/JSON files autonomously
- Generate professional research reports
- Run on 16GB GPU (4-bit quantized)

All open-source: model, training data (500K examples), and code.

🔗 GitHub: https://github.com/ruc-datalab/DeepAnalyze
🤗 Model: https://huggingface.co/RUC-DataLab/DeepAnalyze-8B
📄 Paper: https://arxiv.org/abs/2510.16872

Demo video in comments 👇
```

### 3. Academic & Research Visibility

- [ ] Submit to Papers With Code (link arXiv to GitHub)
- [ ] Add to Hugging Face paper listing
- [ ] Cross-post to r/LanguageTechnology

---

## 🎯 Community Building

### Content Cadence

| Frequency | Activity |
|-----------|----------|
| Weekly | Share user success stories, interesting analyses |
| Bi-weekly | Tutorial videos (YouTube/Bilibili) |
| Monthly | Blog post on arxiv findings or new use cases |

### Content Ideas

1. **"DeepAnalyze vs GPT-4 Code Interpreter"** — Head-to-head comparison on real datasets
2. **"Analyze Kaggle Competition Data in 5 Minutes"** — Practical demo
3. **"How We Built a Data Science Agent"** — Technical deep-dive for r/MachineLearning
4. **"Private Data Analysis on Your Laptop"** — Privacy-focused angle

### Community Platforms

| Platform | Current Status | Recommendation |
|----------|----------------|----------------|
| WeChat | ✅ Active | Continue — great for China community |
| Discord | ❌ Missing | Create for international users |
| GitHub Discussions | ✅ Enabled | Pin FAQ, showcase user projects |

---

## 📊 Growth Metrics

Track these monthly:

| Metric | Current | 3-Month Target |
|--------|---------|----------------|
| GitHub Stars | 3.8K | 6K |
| Hugging Face Downloads | ? | 10K/month |
| Discord Members | 0 | 500 |
| Awesome-list Inclusions | 0 | 4+ |

---

## 🤝 Contributor Growth

### Good First Issues

- [ ] Add support for Parquet files
- [ ] Streamlit alternative UI
- [ ] Ollama integration guide
- [ ] Multi-language report generation

### Partnership Opportunities

- **Kaggle** — Official notebook showcasing DeepAnalyze
- **Weights & Biases** — Integration for experiment tracking
- **LangChain/LlamaIndex** — Document loader integration

---

## 📚 Resources

- [Gingiris Open Source Marketing Playbook](https://github.com/Gingiris/opensource) — Proven tactics for 10K+ stars
- [Gingiris Launch Playbook](https://github.com/Gingiris/launch) — Product Hunt & HN strategies
- [Awesome List Guidelines](https://github.com/sindresorhus/awesome/blob/main/pull_request_template.md)

---

*This is a community contribution. Feel free to adapt, modify, or extend!*

*Last updated: March 2026*
