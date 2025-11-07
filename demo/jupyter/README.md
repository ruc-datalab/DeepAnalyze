# DeepAnalyze Jupyter Frontend

[中文](./README_ZH.md) | English

## Project Introduction

The DeepAnalyze Jupyter Frontend is a Jupyter Notebook interface implementation for the DeepAnalyze data science agent. This project provides an interactive environment that allows users to interact with the DeepAnalyze model through the familiar Jupyter Notebook interface, enabling complete data science workflows including data preparation, analysis, modeling, visualization, and report generation.

## Quick Start

### Requirements
- uv (Unified management of Python runtime and dependencies)
- Node.js (For running Jupyter MCP server)
- OpenAI-compatible API server (e.g., vLLM deployed DeepAnalyze-8B)

### Installation Steps

1. Install necessary environments
```bash
# Install node.js
curl -qL https://www.npmjs.com/install.sh | sh
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more details, see [uv installation documentation](https://docs.astral.sh/uv/getting-started/installation/) and [Node.js installation documentation](https://nodejs.org/en/download/package-manager).

2. Clone the project and enter the directory
```bash
git clone https://github.com/ruc-datalab/DeepAnalyze.git
cd demo/jupyter
```

3. Install dependencies
```bash
# Use uv sync to complete all environment dependency installation with one command
uv sync
```

You can use the `uv add` command to add new dependencies, for example:
```bash
uv add xgboost
```

4. Configure environment variables and configuration files
```bash
cp .env.example .env
# Edit the .env file and fill in your API configuration
# Edit the config.toml file and fill in relevant configurations
```

5. Start terminal interactive chat
```bash
uv run CLI.py
```

### Usage Example

After running `uv run CLI.py`, a `workspace` directory will be created under the `jupyter` directory as the Jupyter working directory, and `deep_analyze.ipynb` will be created as the analysis file. You can upload the required datasets in Jupyter Lab (or directly copy them to the `workspace` directory).

Enter the following commands in the terminal to start interacting with the DeepAnalyze model:
```
>>> Analyze this dataset and generate visualization charts
>>> Perform regression analysis on this data and explain the results
>>> Create a predictive model and evaluate its performance
```

All analysis tasks will be executed in `deep_analyze.ipynb`, and you can view the results and export reports in Jupyter Lab.

## Project Structure

```
jupyter/
├── CLI.py          # Command-line interactive interface
├── server.py       # Core server logic
├── mcp_tools.py    # MCP toolset for interacting with Jupyter
├── utils.py        # Utility functions
├── config.toml     # Configuration file
├── .env.example    # Environment variable example
└── test/           # Test files
```

This project mainly relies on [datalayer/jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) to implement interaction with Jupyter Notebook.

Jupyter MCP is currently the most mainstream implementation of Jupyter-related MCP projects on GitHub. Welcome to use and Star support.