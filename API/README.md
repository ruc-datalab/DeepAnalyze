# DeepAnalyze API Server

## 🚀 Quick Start

### Prerequisites

**Start vLLM Model Server**:

```bash
vllm serve DeepAnalyze-8B --host 0.0.0.0 --port 8000
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
python exampleRequest.py          #  requests example
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
        {"role": "user", "content": "Introduce Python programming language in one sentence"}
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
        {"role": "user", "content": "Introduce Python programming language in one sentence"}
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
            "content": "Analyze this data file, calculate average salary by department, and generate visualization charts.",
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
            "content": "Analyze this data file, calculate average salary by department, and generate visualization charts.",
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

### 4. Multi-turn Conversations with Thread ID

Use `thread_id` in the latest message to maintain workspace context across multiple requests. This allows files and generated content to be persisted between conversations.

**Important:** You must maintain the complete conversation history in each request - only add `thread_id` to the latest message.

**Requests Example:**
```python
conversation_history = []

# First request - creates new thread and gets thread_id
conversation_history.append({"role": "user", "content": "Create a Python script that calculates Fibonacci numbers"})
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": conversation_history
})

result = response.json()
thread_id = result['choices'][0]['message']['thread_id']
assistant_response = result['choices'][0]['message']['content']
conversation_history.append({"role": "assistant", "content": assistant_response})

print(f"Thread ID: {thread_id}")

# Second request - includes full history + thread_id in latest message
conversation_history.append({"role": "user", "content": "Now run the script to calculate the first 10 numbers", "thread_id": thread_id})
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": conversation_history
})

assistant_response = response.json()['choices'][0]['message']['content']
conversation_history.append({"role": "assistant", "content": assistant_response})

# Third request - continue with same thread, full history maintained
conversation_history.append({"role": "user", "content": "List all files in the current workspace", "thread_id": thread_id})
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": conversation_history
})
```

**OpenAI Library Example:**
```python
messages = []

# First request
messages.append({"role": "user", "content": "Create a data analysis script"})
response = client.chat.completions.create(
    model="DeepAnalyze-8B",
    messages=messages
)

# Check for thread_id using hasattr()
thread_id = None
if hasattr(response.choices[0].message, 'thread_id'):
    thread_id = response.choices[0].message.thread_id

messages.append({"role": "assistant", "content": response.choices[0].message.content})

# Continue conversation - include full history + thread_id in latest message
messages.append({"role": "user", "content": "Run the analysis script", "thread_id": thread_id})
response = client.chat.completions.create(
    model="DeepAnalyze-8B",
    messages=messages
)

if hasattr(response.choices[0].message, 'thread_id'):
    print(f"Thread ID: {response.choices[0].message.thread_id}")

messages.append({"role": "assistant", "content": response.choices[0].message.content})
```

**Key Points:**
- `thread_id` is returned in the response message
- Include `thread_id` in the **latest** user message of your conversation history
- You should send the **complete conversation history** in each request
- Files created in previous requests remain available in the thread workspace
- The workspace persists across requests with the same `thread_id`

### 5. Streaming Chat with Files

**Requests Example:**

```python
response = requests.post('http://localhost:8200/v1/chat/completions', json={
    "model": "DeepAnalyze-8B",
    "messages": [
        {
            "role": "user",
            "content": "Analyze this data and generate trend charts.",
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
                if 'thread_id' in delta:
                    print(f"\nThread ID: {delta['thread_id']}")
```

**OpenAI Library Example:**
```python
stream = client.chat.completions.create(
    model="DeepAnalyze-8B",
    messages=[
        {
            "role": "user",
            "content": "Analyze this data and generate trend charts.",
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
    if hasattr(chunk.choices[0].delta, 'thread_id'):
        print(f"\nThread ID: {chunk.choices[0].delta.thread_id}")
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
      "file_ids": ["file-abc123"],     // OpenAI compatible: file_ids in messages
      "thread_id": "thread-xyz789..."  // Optional: thread_id in latest message for workspace persistence
    }
  ],
  "file_ids": ["file-def456"],         // Optional: file_ids parameter (backward compatibility)
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
        "thread_id": "thread-abc123...", // Thread ID for workspace persistence
        "files": [                      // New format: files in message
          {
            "name": "chart.png",
            "url": "http://localhost:8100/thread-123/generated/chart.png"
          }
        ]
      },
      "finish_reason": "stop"
    }
  ],
  "generated_files": [                // Backward compatibility: generated_files field
    {
      "name": "chart.png",
      "url": "http://localhost:8100/thread-123/generated/chart.png"
    }
  ],
  "attached_files": ["file-abc123"]   // Input files
}
```

**Response (Streaming):**
```
data: {"id": "chatcmpl-xyz789...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "分析"}}]}
data: {"id": "chatcmpl-xyz789...", "object": "chat.completion.chunk", "choices": [{"delta": {"files": [{"name":"chart.png","url":"..."}], "thread_id": "thread-abc123..."}, "finish_reason": "stop"}]}
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

- `exampleRequest.py` - Simple requests example
- `exampleOpenAI.py` - OpenAI library example
