# DeepAnalyze API Key Usage Guide (English Version)

This English document demonstrates how to use the DeepAnalyze API for online debugging and local deployment of a Gradio application. If you're in mainland China, you could visit this [Chinese doc](https://heywhale.feishu.cn/wiki/TcVmw314liwCiKkxnttc2CnInfg) instead.

You can apply for a 7-day valid DeepAnalyze API key by filling out this Google Form: https://forms.gle/YxVkCzczqq8jeciw9

**🤹♀️ Document Author & Demo Developer**: Li Haoming, currently an AI Data Engineer at Shanghai Hejin Information Technology Co., Ltd. (HeyWhale Technology). Holds a Master's degree in Data Science from City University of Hong Kong. Specializes in large language model (LLM) and agent system development, with extensive experience in end-to-end LLM application deployment. Responsible for model deployment on the ModelWhale platform within the DeepAnalyze ecosystem, frontend interaction, and application iteration. Previously involved in building high-concurrency intelligent programming assistant services, with a tech stack including FastAPI, LangChain, LangGraph, etc., achieving LLM integration, streaming output, and state management. Proficient in the Hadoop/Spark data ecosystem, with experience in large-scale data ETL development and data warehouse operations. Skilled in full-process engineering from algorithm R&D to business delivery, committed to promoting robust deployment and performance optimization of data intelligence systems. GitHub: https://github.com/LHMQ878

# 1. API Interface Requests

Two calling methods are provided:

## 1.1 Pure Prompt Request Example

```
curl -X POST https://www.heywhale.com/api/model/services/691d42c36c6dda33df0bf645/app/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{"messages": [{"role": "user", "content": "Who are you?"}]}'
```

## 1.2 Prompt + File Upload Request Example

First, download the zip package: [https://open-cdn.kesci.com/admin/t7v5mzpmw/da_quick_start.zip](https://open-cdn.kesci.com/admin/t7v5mzpmw/da_quick_start.zip?referrer=grok.com)

Then run the following in your terminal:


```
python test_streaming.py
```
### 1.2.1 Interactive Usage

After starting the script, follow the prompts to operate:

**Step 1**: Enter the API key
```
Enter API Key: your_api_key_here
```
**Step 2**: Select dialog type
```
Select dialog type:
  1. No-file dialog
  2. Dialog with files

Enter choice (1 or 2):
```

**Step 3**: Enter file paths (if option 2 is selected)

```
Enter file paths (comma separated):
```

**Step 4**: Enter analysis instruction (optional)

```
Enter analysis instruction (blank for default):
```

### 1.2.2 Usage Examples

**Example 1**: Analyzing a CSV file

**Scenario**: Analyze the Simpson.csv data file

**Steps**:

1. Run the script:
   - python test_streaming.py
2. Enter API key:
   - Enter API Key: your_api_key
3. Select dialog type:
   - Enter choice (1 or 2): 2
4. Enter file path:
   - Enter file paths (comma separated): Simpson.csv
   - Or use an absolute path:
   - Enter file paths (comma separated): D:\da_gradio\test\Simpson.csv
5. Enter analysis instruction (optional):
   - Enter analysis instruction (blank for default): 
   - If left blank, the default instruction will be used: Analyze the data file, perform EDA, and generate visualizations.

**Expected Output**:

- The script will automatically upload the file to the API server
- Real-time streaming display of analysis results
- If visualization files are generated, the number of generated files will be displayed

**Example 2**: Using a ZIP archive

**Scenario**: Analyze files inside the example.zip archive

**Steps**:

1. Run the script:
   - python test_streaming.py
2. Enter API key:
   - Enter API Key: your_api_key
3. Select dialog type:
   - Enter choice (1 or 2): 2
4. Enter ZIP file path:
   - Enter file paths (comma separated): example.zip
5. Enter custom analysis instruction (optional):
   - Enter analysis instruction (blank for default): Please analyze all files in the archive, identify key patterns and outliers

**Notes**:

- ZIP files will be automatically extracted
- Only supported file formats are processed (see supported formats list below)
- If the ZIP contains multiple files, all supported files will be uploaded and analyzed

**Example 3**: No-file conversation

**Scenario**: Normal conversation without file analysis

**Steps**:

1. Run the script:
   - python test_streaming.py
2. Enter API key:
   - Enter API Key: your_api_key
3. Select dialog type:
   - Enter choice (1 or 2): 1
4. Enter conversation prompt:
   - Enter analysis instruction (blank for default): Please explain what machine learning is

# 2. Local Deployment of Gradio Application

**Purpose**: Help you quickly deploy, launch, and experience the DeepAnalyze Gradio frontend application locally.

## 2.1 Environment and Preparation

**Environment Requirements**:

- Operating System: Windows / macOS / Linux
- Python: 3.8 or higher
- Dependencies: gradio, openai, fastapi, uvicorn, pandas, requests, python-multipart

**Required Files**:

- Code files: [https://open-cdn.kesci.com/admin/t7v5tp139f/DeepAnalyze_Gradio.zip](https://open-cdn.kesci.com/admin/t7v5tp139f/DeepAnalyze_Gradio.zip?referrer=grok.com)
- Test sample files:
  - [Simpson.csv](https://open-cdn.kesci.com/admin/t7r80krss/Simpson.csv?referrer=grok.com) (example dataset)
  - [example.zip](https://open-cdn.kesci.com/admin/t62le3tdx/example.zip?referrer=grok.com) (ZIP example with multiple files)

## 2.2 Installation Steps

1. Navigate to th/e project directory
```
cd project
```

2. Install dependencies (choose one)
```
pip install gradio openai fastapi uvicorn pandas requests python-multipart
```

or
    ```
    pip install -r requirements.txt
    ```

## 2.3 Launching the Gradio Frontend

Two interface options are available—choose one:

- English interface: python app.py
- Chinese interface: python app_ZH.py

After startup, the program will:

1. Start the API server in the background
   - Default: http://localhost:8200
   - File service: http://localhost:8100
   - Health check: http://localhost:8200/health
2. Launch the Gradio frontend (default: http://localhost:8080)

If ports are occupied, you can adjust them in config.py or app.py.

## 2.4 Usage Workflow (Frontend)

1. Open your browser and visit [http://localhost:8080](http://localhost:8080/?referrer=grok.com)
2. Enter the API Key (required)
3. Optionally upload files: Supports single/multiple files; ZIP files are automatically extracted. Common formats include CSV/Excel/PDF/images/code/logs, etc.
4. Enter an analysis instruction, or use preset buttons (e.g., Data Overview, Trend Analysis)
5. Click "Start Analysis" and observe the streaming output on the right, along with file previews and downloads

## 2.5 Quick Test Examples

- Directly upload [Simpson.csv](https://open-cdn.kesci.com/admin/t7r80krss/Simpson.csv?referrer=grok.com) and enter an instruction such as "Please analyze and visualize the relationships between main variables"
- Upload [example.zip](https://open-cdn.kesci.com/admin/t62le3tdx/example.zip?referrer=grok.com) to verify automatic extraction and multi-file analysis