# DeepAnalyze API Server

## 🚀 Quick Start

### Prerequisites

1. **Start vLLM Model Server** (required):
```bash
vllm serve DeepAnalyze-8B --host 0.0.0.0 --port 8000
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

### Starting the API Server

```bash
cd API
python main.py
```

The server will start on multiple ports:
- **API Server**: `http://localhost:8200` (API)
- **File Server**: `http://localhost:8100` (File downloads)
- **Health Check**: `http://localhost:8200/health`

### Quick Test

```bash
cd example
python example.py
```

### Simple File Upload Example

```python
import requests

API_BASE = "http://localhost:8200"

# Upload a file
with open('data.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    data = {'purpose': 'assistants'}
    response = requests.post(f'{API_BASE}/v1/files', files=files, data=data)

if response.status_code == 200:
    file_obj = response.json()
    file_id = file_obj['id']
    print(f"✅ File uploaded successfully: {file_id}")
    print(f"   Filename: {file_obj['filename']}")
    print(f"   Size: {file_obj['bytes']} bytes")
else:
    print(f"❌ Upload failed: {response.text}")
```

Run the interactive example program to test all API features.

## 📚 API Endpoints

### Overview

The API provides these main endpoints:
- **Health Check**: `/health`
- **Files API**: `/v1/files` (File upload/download)
- **Chat API**: `/v1/chat/completions` (Extended with file support)
- **Assistants API**: `/v1/assistants` 
- **Threads API**: `/v1/threads` (Conversation management)
- **Messages API**: `/v1/threads/{thread_id}/messages`
- **Runs API**: `/v1/threads/{thread_id}/runs` (Task execution)
- **Admin API**: `/v1/admin` (System management)

## 🔧 Core Features

### 1. Extended Chat Completions with File Support

The chat completions API extends OpenAI's standard with file attachment capabilities:

```python
import requests
import json

API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"

# Upload file first
with open('data.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    data = {'purpose': 'assistants'}
    response = requests.post(f'{API_BASE}/v1/files', files=files, data=data)

file_obj = response.json()
file_id = file_obj['id']
print(f"✅ File uploaded: {file_id}")

# Chat with file analysis
response = requests.post(f'{API_BASE}/v1/chat/completions', json={
    "model": MODEL,
    "messages": [
        {"role": "user", "content": "Analyze this dataset and provide insights"}
    ],
    "file_ids": [file_id],
    "temperature": 0.3,
    "stream": False,
    "execute_code": True  # Enable automatic code execution
})

result = response.json()
content = result['choices'][0]['message']['content']
generated_files = result.get('generated_files', [])

print("🤖 Analysis Result:")
print(content)

if generated_files:
    print(f"\n📁 Generated {len(generated_files)} files:")
    for f in generated_files:
        print(f"  📄 {f['name']}: {f['url']}")
```



### 2. Multi-Level File Association

Advanced file management across three levels:

```python
API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"

# Assistant-level files (knowledge base, templates)
assistant_response = requests.post(f'{API_BASE}/v1/assistants', json={
    "model": MODEL,
    "name": "Business Analyst",
    "description": "Analyzes business and financial data",
    "instructions": "You are a business analyst. Analyze the provided financial and market data.",
    "file_ids": ["file-template-123"]  # Assistant files
})

assistant = assistant_response.json()
assistant_id = assistant['id']

# Thread-level files (datasets, background materials)
thread_response = requests.post(f'{API_BASE}/v1/threads', json={
    "metadata": {"project": "sales_analysis"},
    "file_ids": ["file-dataset-456"]  # Thread files
})

thread = thread_response.json()
thread_id = thread['id']

# Message-level files (specific query attachments)
message_response = requests.post(f'{API_BASE}/v1/threads/{thread_id}/messages', json={
    "role": "user",
    "content": "Analyze this with the provided templates and datasets",
    "file_ids": ["file-query-789"]  # Message files
})

message = message_response.json()
```

**File Collection**: During run execution, all files from Assistant, Thread, and Message levels are automatically collected and made available to the model.

### 3. Complete Assistants API Workflow

```python
API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"

# 1. Create assistant
assistant_response = requests.post(f'{API_BASE}/v1/assistants', json={
    "model": MODEL,
    "name": "Data Analyst",
    "description": "Professional data analysis assistant",
    "instructions": "You are a professional data analyst. Analyze data and provide comprehensive insights."
})

assistant = assistant_response.json()
assistant_id = assistant['id']

# 2. Create thread
thread_response = requests.post(f'{API_BASE}/v1/threads', json={
    "metadata": {"project": "analysis"}
})

thread = thread_response.json()
thread_id = thread['id']

# 3. Add message with file
message_response = requests.post(
    f'{API_BASE}/v1/threads/{thread_id}/messages',
    json={
        "role": "user",
        "content": "Analyze the uploaded data and create visualizations",
        "file_ids": [file_id]  # file_id should be obtained from file upload
    }
)

message = message_response.json()

# 4. Create and monitor run
run_response = requests.post(
    f'{API_BASE}/v1/threads/{thread_id}/runs',
    json={"assistant_id": assistant_id}
)

run = run_response.json()
run_id = run['id']

# Monitor until completion
import time
while True:
    run_status_response = requests.get(f'{API_BASE}/v1/threads/{thread_id}/runs/{run_id}')
    run_status = run_status_response.json()
    status = run_status['status']

    if status == 'completed':
        print("✅ Analysis completed!")
        break
    elif status in ['failed', 'cancelled', 'expired']:
        print(f"❌ Run failed: {status}")
        break

    time.sleep(2)

# 5. Get results and generated files
messages_response = requests.get(f'{API_BASE}/v1/threads/{thread_id}/messages')
messages = messages_response.json()['data']

files_response = requests.get(f'{API_BASE}/v1/threads/{thread_id}/files')
files = files_response.json()['data']

print(f"Generated {len(files)} files during analysis")
```

## 🏗️ Architecture

### Multi-Port Design

- **Port 8000**: vLLM model server (external)
- **Port 8200**: Main API server (FastAPI)
- **Port 8100**: File HTTP server for downloads

### Core Components

1. **API Gateway** (`main.py`): FastAPI application setup
2. **Chat Engine** (`chat_api.py`): Extended chat completions with file support
3. **Assistant Manager** (`assistants_api.py`): Assistant lifecycle management
4. **File Handler** (`file_api.py`): File upload/download management
5. **Thread Manager** (`threads_api.py`): Conversation thread management
6. **Execution Engine** (`utils.py`): Safe code execution in sandboxed environments
7. **Storage Layer** (`storage.py`): In-memory data management with workspace isolation

## 📊 Configuration

### Environment Variables
```python
# API Configuration
API_HOST = "0.0.0.0"           # API server host
API_PORT = 8200               # API server port
MODEL_PATH = "DeepAnalyze-8B"  # Model name

# vLLM Configuration
API_BASE = "http://localhost:8000/v1"  # vLLM endpoint

# File Management
WORKSPACE_BASE_DIR = "workspace"       # Base workspace directory
HTTP_SERVER_PORT = 8100               # File server port

# Execution Settings
CODE_EXECUTION_TIMEOUT = 120          # Code execution timeout (seconds)
MAX_NEW_TOKENS = 32768               # Maximum response tokens
DEFAULT_TEMPERATURE = 0.4            # Default sampling temperature
```


## 🔧 Management API

### Thread Cleanup
```http
POST /v1/admin/cleanup-threads?timeout_hours=12
```

### Thread Statistics
```http
GET /v1/admin/threads-stats
```

**Response:**
```json
{
  "total_threads": 150,
  "recent_threads": 25,
  "old_threads": 100,
  "expired_threads": 25,
  "timeout_hours": 12,
  "timestamp": 1704067200
}
```

## 📝 Usage Examples

### Example 1: Simple Data Analysis
```python
API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"

# Upload data
with open('sales_data.csv', 'rb') as f:
    files = {'file': ('sales.csv', f, 'text/csv')}
    data = {'purpose': 'assistants'}
    response = requests.post(f'{API_BASE}/v1/files', files=files, data=data)
    file_id = response.json()['id']

# Analyze with streaming
response = requests.post(f'{API_BASE}/v1/chat/completions', json={
    "model": MODEL,
    "messages": [{"role": "user", "content": "Analyze sales trends and forecast next quarter"}],
    "file_ids": [file_id],
    "stream": True,
    "execute_code": True
}, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and chunk['choices']:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        print(delta['content'], end='', flush=True)
            except json.JSONDecodeError:
                pass
```

### Example 2: Multi-File Analysis
```python
API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"

# Upload multiple data sources
file_ids = []
for filename in ['sales.csv', 'marketing.csv', 'customers.csv']:
    with open(filename, 'rb') as f:
        files = {'file': (filename, f, 'text/csv')}
        data = {'purpose': 'assistants'}
        response = requests.post(f'{API_BASE}/v1/files', files=files, data=data)
        file_ids.append(response.json()['id'])

# Comprehensive analysis
response = requests.post(f'{API_BASE}/v1/chat/completions', json={
    "model": MODEL,
    "messages": [{"role": "user", "content": "Create a comprehensive business analysis integrating all data sources"}],
    "file_ids": file_ids,
    "temperature": 0.3,
    "stream": False,
    "execute_code": True
})

result = response.json()
content = result['choices'][0]['message']['content']
generated_files = result.get('generated_files', [])
print(f"Analysis complete. Generated {len(generated_files)} files.")
```

### Example 3: Complete Assistants API Workflow
```python
API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"

# 1. Upload data file
with open('financial_data.csv', 'rb') as f:
    files = {'file': ('financial_data.csv', f, 'text/csv')}
    data = {'purpose': 'assistants'}
    response = requests.post(f'{API_BASE}/v1/files', files=files, data=data)
    file_id = response.json()['id']

# 2. Create assistant
response = requests.post(f'{API_BASE}/v1/assistants', json={
    "model": MODEL,
    "name": "Financial Analyst",
    "description": "Analyzes financial data",
    "instructions": "You are a financial analyst. Analyze data and provide insights.",
    "file_ids": [file_id]  # Optional: associate files at assistant level
})

assistant = response.json()
assistant_id = assistant['id']

# 3. Create thread
response = requests.post(f'{API_BASE}/v1/threads', json={
    "metadata": {"project": "financial_analysis"}
})

thread = response.json()
thread_id = thread['id']

# 4. Add message
response = requests.post(f'{API_BASE}/v1/threads/{thread_id}/messages', json={
    "role": "user",
    "content": "请分析这个财务数据，计算每月的利润，并创建一个可视化图表。"
})

message = response.json()

# 5. Create run
response = requests.post(f'{API_BASE}/v1/threads/{thread_id}/runs', json={
    "assistant_id": assistant_id
})

run = response.json()
run_id = run['id']

# 6. Monitor run status
import time
while True:
    response = requests.get(f'{API_BASE}/v1/threads/{thread_id}/runs/{run_id}')
    run = response.json()
    status = run['status']

    if status == 'completed':
        print("✅ Analysis completed!")
        break
    elif status in ['failed', 'cancelled', 'expired']:
        print(f"❌ Run failed: {status}")
        break

    time.sleep(2)

# 7. Get results
response = requests.get(f'{API_BASE}/v1/threads/{thread_id}/messages')
messages = response.json()['data']

for msg in messages:
    if msg['role'] == 'assistant':
        content = msg['content'][0]['text']['value']
        print(f"Analysis Result:\n{content}")
        break

# 8. Get generated files
response = requests.get(f'{API_BASE}/v1/threads/{thread_id}/files')
files = response.json()['data']

if files:
    print(f"Generated {len(files)} files:")
    for f in files:
        print(f"  📄 {f['name']}: {f['url']}")
```

## 📋 Complete API Reference

### Health Check API

#### GET /health
Health check endpoint to verify API server status.

**Request:**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1704067200
}
```

### Files API

#### POST /v1/files
Upload a file to be used in analysis.

**Request:**
```http
POST /v1/files
Content-Type: multipart/form-data

file: [binary file data]
purpose: assistants
```

**Response:**
```json
{
  "id": "file-abc123...",
  "object": "file",
  "bytes": 1024,
  "created_at": 1704067200,
  "filename": "data.csv",
  "purpose": "assistants"
}
```

#### GET /v1/files
List all uploaded files.

**Request:**
```http
GET /v1/files?purpose=assistants
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "file-abc123...",
      "object": "file",
      "bytes": 1024,
      "created_at": 1704067200,
      "filename": "data.csv",
      "purpose": "assistants"
    }
  ]
}
```

#### GET /v1/files/{file_id}
Retrieve file information.

**Request:**
```http
GET /v1/files/{file_id}
```

**Response:** Same as individual file object above

#### DELETE /v1/files/{file_id}
Delete a file.

**Request:**
```http
DELETE /v1/files/{file_id}
```

**Response:**
```json
{
  "id": "file-abc123...",
  "object": "file",
  "deleted": true
}
```

#### GET /v1/files/{file_id}/content
Download file content.

**Request:**
```http
GET /v1/files/{file_id}/content
```

**Response:** Binary file content

### Chat Completions API

#### POST /v1/chat/completions
Extended chat completion with file support.

**Request:**
```json
{
  "model": "DeepAnalyze-8B",
  "messages": [
    {
      "role": "user",
      "content": "Analyze this dataset and provide insights"
    }
  ],
  "file_ids": ["file-abc123..."],
  "temperature": 0.4,
  "stream": false,
  "execute_code": true
}
```

**Response (non-streaming):**
```json
{
  "id": "chatcmpl-xyz789...",
  "object": "chat.completion",
  "created": 1704067200,
  "model": "DeepAnalyze-8B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Analysis results with <Analyze>, <Code>, <Execute>, <Answer> tags"
      },
      "finish_reason": "stop"
    }
  ],
  "generated_files": [
    {
      "name": "chart.png",
      "url": "http://localhost:8100/thread-123/generated/chart.png"
    }
  ],
  "attached_files": ["file-abc123..."]
}
```

**Response (streaming):**
```
data: {"id": "chatcmpl-xyz789...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "First part"}}]}
data: {"id": "chatcmpl-xyz789...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": " of response"}}], "generated_files": {"name":"file_name","url":"http://localhost:8100/..."}}
data: [DONE]
```

### Assistants API

#### POST /v1/assistants
Create an assistant.

**Request:**
```json
{
  "model": "DeepAnalyze-8B",
  "name": "Data Analyst",
  "description": "Professional data analysis assistant",
  "instructions": "You are a professional data analyst...",
  "tools": [],
  "file_ids": ["file-abc123..."],
  "metadata": {"version": "1.0"}
}
```

**Response:**
```json
{
  "id": "asst-xyz789...",
  "object": "assistant",
  "created_at": 1704067200,
  "name": "Data Analyst",
  "description": "Professional data analysis assistant",
  "model": "DeepAnalyze-8B",
  "instructions": "You are a professional data analyst...",
  "tools": [],
  "file_ids": ["file-abc123..."],
  "metadata": {"version": "1.0"}
}
```

#### GET /v1/assistants
List all assistants.

**Request:**
```http
GET /v1/assistants
```

**Response:**
```json
{
  "object": "list",
  "data": [assistant objects]
}
```

#### GET /v1/assistants/{assistant_id}
Retrieve an assistant.

**Request:**
```http
GET /v1/assistants/{assistant_id}
```

**Response:** Assistant object

#### DELETE /v1/assistants/{assistant_id}
Delete an assistant.

**Request:**
```http
DELETE /v1/assistants/{assistant_id}
```

**Response:**
```json
{
  "id": "asst-xyz789...",
  "object": "assistant",
  "deleted": true
}
```

### Threads API

#### POST /v1/threads
Create a conversation thread.

**Request:**
```json
{
  "metadata": {"project": "analysis"},
  "file_ids": ["file-abc123..."]
}
```

**Response:**
```json
{
  "id": "thread-123...",
  "object": "thread",
  "created_at": 1704067200,
  "last_accessed_at": 1704067200,
  "metadata": {"project": "analysis"},
  "file_ids": ["file-abc123..."]
}
```

#### GET /v1/threads/{thread_id}
Retrieve a thread.

**Request:**
```http
GET /v1/threads/{thread_id}
```

**Response:** Thread object

#### DELETE /v1/threads/{thread_id}
Delete a thread.

**Request:**
```http
DELETE /v1/threads/{thread_id}
```

**Response:**
```json
{
  "id": "thread-123...",
  "object": "thread",
  "deleted": true
}
```

### Messages API

#### POST /v1/threads/{thread_id}/messages
Add a message to a thread.

**Request:**
```json
{
  "role": "user",
  "content": "Analyze the uploaded data",
  "file_ids": ["file-def456..."],
  "metadata": {"source": "web"}
}
```

**Response:**
```json
{
  "id": "msg-456...",
  "object": "thread.message",
  "created_at": 1704067200,
  "thread_id": "thread-123...",
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": {
        "value": "Analyze the uploaded data",
        "annotations": []
      }
    }
  ],
  "file_ids": ["file-def456..."],
  "assistant_id": null,
  "run_id": null,
  "metadata": {"source": "web"}
}
```

#### GET /v1/threads/{thread_id}/messages
List messages in a thread.

**Request:**
```http
GET /v1/threads/{thread_id}/messages
```

**Response:**
```json
{
  "object": "list",
  "data": [message objects]
}
```

### Runs API

#### POST /v1/threads/{thread_id}/runs
Create and execute a run.

**Request:**
```json
{
  "assistant_id": "asst-xyz789...",
  "model": "DeepAnalyze-8B",
  "instructions": "Override instructions for this run"
}
```

**Response:**
```json
{
  "id": "run-789...",
  "object": "thread.run",
  "created_at": 1704067200,
  "thread_id": "thread-123...",
  "assistant_id": "asst-xyz789...",
  "status": "queued",
  "required_action": null,
  "last_error": null,
  "expires_at": 1704070800,
  "started_at": null,
  "cancelled_at": null,
  "failed_at": null,
  "completed_at": null,
  "model": "DeepAnalyze-8B",
  "instructions": "You are a professional data analyst...",
  "tools": [],
  "file_ids": [],
  "metadata": {}
}
```

#### GET /v1/threads/{thread_id}/runs/{run_id}
Retrieve a run status.

**Request:**
```http
GET /v1/threads/{thread_id}/runs/{run_id}
```

**Response:** Run object with updated status

#### GET /v1/threads/{thread_id}/runs
List runs for a thread.

**Request:**
```http
GET /v1/threads/{thread_id}/runs
```

**Response:**
```json
{
  "object": "list",
  "data": [run objects]
}
```

### Thread Files API (Extended)

#### GET /v1/threads/{thread_id}/files
List files associated with a thread (including generated files).

**Request:**
```http
GET /v1/threads/{thread_id}/files
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "file-abc123...",
      "object": "file",
      "filename": "uploaded_data.csv",
      "url": "http://localhost:8100/thread-123/uploads/uploaded_data.csv",
      "created_at": 1704067200
    },
    {
      "id": "generated-chart-1",
      "object": "file",
      "filename": "analysis_chart.png",
      "url": "http://localhost:8100/thread-123/generated/analysis_chart.png",
      "created_at": 1704067260
    }
  ]
}
```

### Admin API

#### POST /v1/admin/cleanup-threads
Manually trigger cleanup of expired threads.

**Request:**
```http
POST /v1/admin/cleanup-threads?timeout_hours=12
```

**Response:**
```json
{
  "status": "success",
  "cleaned_threads": 5,
  "timeout_hours": 12,
  "timestamp": 1704067200
}
```

#### GET /v1/admin/threads-stats
Get thread statistics.

**Request:**
```http
GET /v1/admin/threads-stats
```

**Response:**
```json
{
  "total_threads": 150,
  "recent_threads": 25,
  "old_threads": 100,
  "expired_threads": 25,
  "timeout_hours": 12,
  "timestamp": 1704067200
}
```

## 🔍 Response Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters, file not found
- `404 Not Found`: Resource not found (thread, message, assistant, file)
- `422 Unprocessable Entity`: Validation error (invalid file type, size exceeded)
- `500 Internal Server Error`: Model error, code execution failure

## 🏷️ Object Types

### Status Values (Runs)
- `queued`: Run is queued for execution
- `in_progress`: Run is currently executing
- `requires_action`: Run needs user input
- `cancelling`: Run is being cancelled
- `cancelled`: Run was cancelled
- `failed`: Run failed with error
- `completed`: Run completed successfully
- `expired`: Run expired before completion

### Tool Types
- `NotImplemented`: Placeholder for future tools

