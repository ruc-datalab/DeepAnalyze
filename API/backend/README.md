# DeepAnalyze API Server

## 🚀 Quick Start

### Prerequisites

1. **Start vLLM Model Server**:
```bash
vllm serve DeepAnalyze-8B --host 0.0.0.0 --port 8000
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

### Starting the Server

```bash
cd API
python start_server.py
```

- **API Server**: `http://localhost:8200` (Main API)
- **File Server**: `http://localhost:8100` (File downloads)
- **Health Check**: `http://localhost:8200/health`

The API server will create a new `workspace` folder in the current directory as the working directory. For each conversation, it will generate a `thread` subdirectory under this workspace to perform data analysis and generate files.

### Quick Test

```bash
cd example
python example.py          # Simple requests example
python exampleOpenAI.py    # OpenAI library example
```

## 📚 API Usage

### 1. File Upload

**Requests Example:**
```python
import requests

with open('data.csv', 'rb') as f:
    files = {'file': ('data.csv', f, 'text/csv')}
    response = requests.post('http://localhost:8200/v1/files', files=files)

file_id = response.json()['id']
print(f"File uploaded: {file_id}")
```

**OpenAI Library Example:**
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8200/v1",
    api_key="dummy"
)

with open('data.csv', 'rb') as f:
    file_obj = client.files.create(file=f)

print(f"File uploaded: {file_obj.id}")
```

### 2. Simple Chat (No Files)

**Requests Example:**

```python
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": [
        {"role": "user", "content": "用一句话介绍Python编程语言"}
    ],
    "temperature": 0.4
})

content = response.json()['choices'][0]['message']['content']
print(content)
```

**OpenAI Library Example:**
```python
response = client.chat.completions.create(
    model="DeepAnalyze-8B",
    messages=[
        {"role": "user", "content": "用一句话介绍Python编程语言"}
    ],
    temperature=0.4
)

print(response.choices[0].message.content)
```

### 3. Chat with Files

**Requests Example:**
```python
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": [
        {
            "role": "user",
            "content": "分析这个数据文件，计算各部门的平均薪资，并生成可视化图表。",
            "file_ids": [file_id]  
        }
    ],
    "temperature": 0.4
})

result = response.json()
content = result['choices'][0]['message']['content']
files = result['choices'][0]['message'].get('files', [])

print(f"Response: {content}")
for file_info in files:
    print(f"Generated file: {file_info['name']} - {file_info['url']}")
```

**OpenAI Library Example:**
```python
response = client.chat.completions.create(
    model="DeepAnalyze-8B",
    messages=[
        {
            "role": "user",
            "content": "分析这个数据文件，计算各部门的平均薪资，并生成可视化图表。",
            "file_ids": [file_id]  
        }
    ],
    temperature=0.4
)

message = response.choices[0].message
print(f"Response: {message.content}")

# Access generated files (new format)
if hasattr(message, 'files') and message.files:
    for file_info in message.files:
        print(f"Generated file: {file_info['name']} - {file_info['url']}")
```

### 4. Streaming Chat with Files

**Requests Example:**

```python
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": [
        {
            "role": "user",
            "content": "流式分析这个数据并生成趋势图。",
            "file_ids": [file_id]
        }
    ],
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]
            if data_str == '[DONE]':
                break
            chunk = json.loads(data_str)
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    print(delta['content'], end='', flush=True)
```

**OpenAI Library Example:**
```python
stream = client.chat.completions.create(
    model="DeepAnalyze-8B",
    messages=[
        {
            "role": "user",
            "content": "流式分析这个数据并生成趋势图。",
            "file_ids": [file_id]
        }
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)
    if hasattr(chunk, 'generated_files') and chunk.generated_files:
        collected_files.extend(chunk.generated_files)
```


## 📋 API Reference

### Files API

#### POST /v1/files
Upload a file for analysis.

**Request:**
```http
POST /v1/files
Content-Type: multipart/form-data

file: [binary file data]
```

**Response:**
```json
{
  "id": "file-abc123...",
  "object": "file",
  "bytes": 1024,
  "created_at": 1704067200,
  "filename": "data.csv"
}
```

#### GET /v1/files
List all uploaded files.

**Request:**
```http
GET /v1/files
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
      "filename": "data.csv"
    }
  ]
}
```

#### GET /v1/files/{file_id}/content
Download file content.

**Request:**
```http
GET /v1/files/{file_id}/content
```

**Response:** Binary file content

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
      "content": "分析这个数据文件",
      "file_ids": ["file-abc123"]  // OpenAI compatible: file_ids in messages
    }
  ],
  "file_ids": ["file-def456"],     // Optional: file_ids parameter (backward compatibility)
  "temperature": 0.4,
  "stream": false
}
```

**Response (Non-Streaming):**
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
        "content": "分析结果...",
        "files": [                    // New format: files in message
          {
            "name": "chart.png",
            "url": "http://localhost:8100/thread-123/generated/chart.png"
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "generated_files": [              // Backward compatibility: generated_files field
    {
      "name": "chart.png",
      "url": "http://localhost:8100/thread-123/generated/chart.png"
    }
  ],
  "attached_files": ["file-abc123"] // Input files
}
```

**Response (Streaming):**
```
data: {"id": "chatcmpl-xyz789...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "分析"}}]}
data: {"id": "chatcmpl-xyz789...", "object": "chat.completion.chunk", "choices": [{"delta": {"files": [{"name":"chart.png","url":"..."}]}, "finish_reason": "stop"}]}
data: [DONE]
```






### Health Check API

#### GET /health
Check API server status.

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

## 🏗️ Architecture

### Multi-Port Design

- **Port 8000**: vLLM model server (external)
- **Port 8200**: Main API server (FastAPI)
- **Port 8100**: File HTTP server for downloads

## 🔧 Configuration

### Environment Variables

```python
# API Configuration
API_BASE = "http://localhost:8000/v1"  # vLLM endpoint
MODEL_PATH = "DeepAnalyze-8B"          # Model name
WORKSPACE_BASE_DIR = "workspace"       # File storage
HTTP_SERVER_PORT = 8100               # File server port

# Model Settings
DEFAULT_TEMPERATURE = 0.4            # Default sampling temperature
MAX_NEW_TOKENS = 32768               # Maximum response tokens
STOP_TOKEN_IDS = [32000, 32007]      # Special token IDs
```



## 🛠️ Examples

The `example/` directory contains comprehensive examples:

- `example.py` - Simple requests example
- `exampleOpenAI.py` - OpenAI library example
