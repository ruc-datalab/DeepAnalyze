"""
Data models for DeepAnalyze API Server
Contains all Pydantic models for OpenAI compatibility
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class FileObject(BaseModel):
    """OpenAI File Object"""
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileDeleteResponse(BaseModel):
    """OpenAI File Delete Response"""
    id: str
    object: Literal["file"] = "file"
    deleted: bool


class AssistantObject(BaseModel):
    """OpenAI Assistant Object"""
    id: str
    object: Literal["assistant"] = "assistant"
    created_at: int
    name: Optional[str] = None
    description: Optional[str] = None
    model: str
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    file_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ThreadObject(BaseModel):
    """OpenAI Thread Object"""
    id: str
    object: Literal["thread"] = "thread"
    created_at: int
    last_accessed_at: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    file_ids: List[str] = Field(default_factory=list)
    tool_resources: Optional[Dict[str, Any]] = Field(default=None)


class MessageObject(BaseModel):
    """OpenAI Message Object"""
    id: str
    object: Literal["thread.message"] = "thread.message"
    created_at: int
    thread_id: str
    role: Literal["user", "assistant"]
    content: List[Dict[str, Any]]
    file_ids: List[str] = Field(default_factory=list)
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunObject(BaseModel):
    """OpenAI Run Object"""
    id: str
    object: Literal["thread.run"] = "thread.run"
    created_at: int
    thread_id: str
    assistant_id: str
    status: Literal[
        "queued",
        "in_progress",
        "requires_action",
        "cancelling",
        "cancelled",
        "failed",
        "completed",
        "expired",
    ]
    required_action: Optional[Dict[str, Any]] = None
    last_error: Optional[Dict[str, Any]] = None
    expires_at: int
    started_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    failed_at: Optional[int] = None
    completed_at: Optional[int] = None
    model: str
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = Field(default_factory=list)
    file_ids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    model: str
    messages: List[Dict[str, Any]]
    file_ids: Optional[List[str]] = Field(default=None)
    temperature: Optional[float] = Field(0.4)
    stream: Optional[bool] = Field(False)


class FileInfo(BaseModel):
    """File information model for OpenAI compatibility"""
    filename: str
    url: str


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model"""
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    generated_files: Optional[List[Dict[str, str]]] = Field(default=None)
    attached_files: Optional[List[str]] = Field(default=None)


class ChatCompletionChunk(BaseModel):
    """Chat completion streaming chunk model"""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    generated_files: Optional[List[Dict[str, str]]] = Field(default=None)


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: int


class ThreadCleanupRequest(BaseModel):
    """Thread cleanup request model"""
    timeout_hours: int = Field(12, description="Timeout in hours for thread cleanup")


class ThreadCleanupResponse(BaseModel):
    """Thread cleanup response model"""
    status: str
    cleaned_threads: int
    timeout_hours: int
    timestamp: int


class ThreadStatsResponse(BaseModel):
    """Thread statistics response model"""
    total_threads: int
    recent_threads: int  # < 1 hour
    old_threads: int     # 1-12 hours
    expired_threads: int # > 12 hours
    timeout_hours: int
    timestamp: int


class ModelObject(BaseModel):
    """OpenAI Model Object"""
    id: str
    object: Literal["model"] = "model"
    created: Optional[int] = None
    owned_by: Optional[str] = None


class ModelsListResponse(BaseModel):
    """OpenAI Models List Response"""
    object: Literal["list"] = "list"
    data: List[ModelObject]