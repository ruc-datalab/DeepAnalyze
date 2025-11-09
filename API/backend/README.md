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
    data = {'purpose': 'assistants'}
    response = requests.post('http://localhost:8200/v1/files', files=files, data=data)

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
    file_obj = client.files.create(file=f, purpose="assistants")

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

### 4. Chat with Files (file_ids in top level)

**Requests Example:**
```python
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": [
        {"role": "user", "content": "分析这个数据文件"}
    ],
    "file_ids": [file_id],  # file_ids parameter (old format)
    "temperature": 0.4
})

result = response.json()
content = result['choices'][0]['message']['content']
files = result.get('generated_files', [])  # old format

print(f"Response: {content}")
for file_info in files:
    print(f"Generated file: {file_info['name']} - {file_info['url']}")
```

### 5. Streaming Chat with Files

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

### 6. Complete Assistants Workflow

**Requests Example:**
```python
# 1. Create assistant
assistant = requests.post('http://localhost:8200/v1/assistants', json={
    "model": "DeepAnalyze-8B",
    "name": "Data Analyst",
    "instructions": "分析数据并提供洞察。",
    "tools": [{"type": "analyze"}]
}).json()

# 2. Create thread
thread = requests.post('http://localhost:8200/v1/threads', json={
    "metadata": {"project": "analysis"}
}).json()

# 3. Add message
requests.post(f'http://localhost:8200/v1/threads/{thread["id"]}/messages', json={
    "role": "user",
    "content": "分析这个数据集并确定哪种教学方法效果更好。",
    "file_ids": [file_id]
})

# 4. Create run
run = requests.post(f'http://localhost:8200/v1/threads/{thread["id"]}/runs', json={
    "assistant_id": assistant["id"]
}).json()

# 5. Monitor completion
while run["status"] in ["queued", "in_progress"]:
    run = requests.get(f'http://localhost:8200/v1/threads/{thread["id"]}/runs/{run["id"]}').json()
    time.sleep(1)

# 6. Get results
messages = requests.get(f'http://localhost:8200/v1/threads/{thread["id"]}/messages').json()['data']
for msg in messages:
    if msg["role"] == "assistant":
        content = msg["content"][0]["text"]["value"]
        print(f"Analysis: {content}")
        break
```

**OpenAI Library Example:**
```python
# 1. Create assistant
assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="分析数据并提供洞察。",
    model="DeepAnalyze-8B",
    tools=[{"type": "analyze"}]
)

# 2. Create thread
thread = client.beta.threads.create(
    tool_resources={
        "analyze": {
            "file_ids": [file_id]
        }
    }
)

# 3. Add message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="分析这个数据集并确定哪种教学方法效果更好。"
)

# 4. Create run
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# 5. Monitor completion
while run.status in ["queued", "in_progress"]:
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    time.sleep(1)

# 6. Get results
messages = client.beta.threads.messages.list(thread_id=thread.id)
for msg in messages.data:
    if msg.role == "assistant":
        content = msg.content[0].text.value
        print(f"Analysis: {content}")
        break
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
      "filename": "data.csv",
      "purpose": "assistants"
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
  "tools": [{"type": "analyze"}],
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
  "tools": [{"type": "analyze"}],
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
  "tools": [{"type": "analyze"}],
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

### Thread Files API

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
