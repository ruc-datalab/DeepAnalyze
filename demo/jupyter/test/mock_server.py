import json
import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# Test data
TEST_RESPONSES = [
    "<Analyze>\nA1\n</Analyze>\n<Code>\n%pwd\n</Code>",
    "<Understand>\nU1\n</Understand>\n<Analyze>\nA2\n</Analyze>\n<Code>\nimport pandas as pd\ndf = pd.read_csv(\"data/Simpson.csv\")\ndf.head()\n</Code>",
    "<Analyze>\nA3\n</Analyze>\n<Code>\nprint(\"Goodbye!\")\n</Code>",
    "<Answer>\nAA\n</Answer>"
]

current_response_index = 0

# 记录请求计数，用于调试
request_count = 0

app = FastAPI(title="Mock OpenAI API Server")

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    stream: bool = False
    extra_body: Dict[str, Any] = {}

class Choice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str | None = None

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]

@app.get("/v1/models")
async def list_models():
    """Mock OpenAI Model List API"""
    return {
        "object": "list",
        "data": [
            {
                "id": "DeepAnalyze-8B",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "deepanalyze"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Mock OpenAI Chat Completion API"""
    global current_response_index, request_count
    
    request_count += 1
    print(f"Request #{request_count}: current_response_index = {current_response_index}")
    
    # Get current response
    if current_response_index >= len(TEST_RESPONSES):
        # If beyond test data range, return a default response
        response_content = "<Answer>\nTest completed\n</Answer>"
        finish_reason = "stop"
    else:
        response_content = TEST_RESPONSES[current_response_index]
        # Only set finish_reason to "stop" for the last response
        finish_reason = "stop" if current_response_index == len(TEST_RESPONSES) - 1 else None
        current_response_index += 1
    
    # Create response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message={
                    "role": "assistant",
                    "content": response_content
                },
                finish_reason=finish_reason
            )
        ]
    )
    
    print(f"Response #{request_count}: content = {response_content[:50]}..., finish_reason = {finish_reason}")
    
    return response

@app.get("/")
async def root():
    """Root path, check if server is running"""
    return {"message": "Mock OpenAI API Server is running"}

if __name__ == "__main__":
    print("Starting Mock OpenAI API Server on http://localhost:8000")
    print("This server provides test responses for DeepAnalyze testing")
    uvicorn.run(app, host="0.0.0.0", port=8000)