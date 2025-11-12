# DeepAnalyze 命令行界面（CLI）

一个基于 Rich 库构建的 DeepAnalyze 命令行界面（CLI），提供直观的方式与 DeepAnalyze 交互，支持文件上传和实时流式响应，适用于数据分析任务。

## 🚀 快速开始

### 先决条件

请保证vLLM在8000端口正在运行DeepAnalyze-8B

1. **启动 DeepAnalyze API 服务器**：

   ```bash
   cd ../../API
   python start_server.py
   ```

2. **启动 CLI**：

   ```bash
   # 在另一个终端中运行
   
   # 英文版
   python api_cli.py
   
   # 中文版
   python api_cli_ZH.py
   ```

## 📋 命令列表

### 基础命令

- `help` - 显示帮助信息  
- `quit` / `exit` - 退出程序  
- `clear` - 清除对话历史  
- `clear-all` - 清除所有内容，包括已上传的文件

### 文件管理

- `files` - 查看已上传的文件  
- `upload <文件路径>` - 上传新文件  
- `delete <文件ID>` - 删除指定文件  
- `download <文件ID> [保存路径]` - 下载文件

### 系统与历史记录

- `status` - 显示系统状态  
- `history` - 显示对话历史  
- `fid` - 显示所有文件名及其完整 ID

## 💬 使用示例

### 基础聊天

```
> 分析此数据集并生成洞察
```

### 文件上传与分析

```
> upload data.csv
✅ 文件已上传：file-abc123...

> 分析已上传的数据并创建可视化图表
📊 正在生成分析...
📈 已创建图表：analysis.png, trends.png
📝 已生成报告：report.md
```

### 流式响应

CLI 会自动以流式方式显示响应，在 DeepAnalyze 分析数据并生成洞察时实时展示进度。

## 🔧 配置说明

CLI 默认连接到 `http://localhost:8200/v1` 的 DeepAnalyze API 服务器。启动 CLI 前请确保服务器正在运行。

命令历史将分别保存至 `~/.deeppanalyze_history_en`（英文版）或 `~/.deeppanalyze_history_zh`（中文版）。