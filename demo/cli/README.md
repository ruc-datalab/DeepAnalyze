# DeepAnalyze CLI

A command-line interface for DeepAnalyze, powered by the Rich library. Provides an intuitive way to interact with DeepAnalyze for data analysis tasks with file upload support and real-time streaming responses.

## 🚀 Quick Start

### Prerequisites

Please ensure that the vLLM service is running on port 8000 with the DeepAnalyze-8B model loaded.

1. **Start the DeepAnalyze API Server**:
   ```bash
   cd ../../API
   python start_server.py
   ```

2. **Launch the CLI**: 
   
   ```bash
   
   ```
# In another terminal

   # English version
python api_cli.py

   # Chinese version
   python api_cli_ZH.py
   ```

## 📋 Commands

### Basic Commands
- `help` - Display help information
- `quit` / `exit` - Exit the program
- `clear` - Clear conversation history 
- `clear-all` - Clear all content including uploaded files

### File Management
- `files` - View uploaded files
- `upload <file_path>` - Upload new file
- `delete <file_id>` - Delete specified file
- `download <file_id> [save_path]` - Download file

### System & History
- `status` - Display system status
- `history` - Display conversation history
- `fid` - Display all file names and complete IDs

## 💬 Usage Examples

### Basic Chat
   ```
> Analyze this dataset and generate insights
```

### File Upload and Analysis
```
> upload data.csv
✅ File uploaded: file-abc123...

> Analyze the uploaded data and create visualizations
📊 Generating analysis...
📈 Created charts: analysis.png, trends.png
📝 Generated report: report.md
```

### Streaming Responses
The CLI automatically streams responses, showing real-time progress as DeepAnalyze analyzes your data and generates insights.

## 🔧 Configuration

The CLI connects to the DeepAnalyze API server at `http://localhost:8200/v1` by default. Ensure the server is running before launching the CLI.

Command history is saved to `~/.deeppanalyze_history_en` (English) or `~/.deeppanalyze_history_zh` (Chinese).



```