"""
Storage layer for DeepAnalyze API Server
Handles in-memory storage for OpenAI objects
"""

import os
import time
import uuid
import shutil
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

from models import (
    FileObject, ThreadObject, MessageObject
)
from utils import get_thread_workspace, uniquify_path


class Storage:
    """Simple in-memory storage for OpenAI objects"""

    def __init__(self):
        self.files: Dict[str, Dict[str, Any]] = {}
        self.threads: Dict[str, Dict[str, Any]] = {}
        self.messages: Dict[str, List[Dict[str, Any]]] = {}  # thread_id -> messages
        self._lock = threading.Lock()

    def create_file(self, filename: str, filepath: str, purpose: str) -> FileObject:
        """Create a file record"""
        with self._lock:
            file_id = f"file-{uuid.uuid4().hex[:24]}"
            file_size = os.path.getsize(filepath)
            file_obj = {
                "id": file_id,
                "object": "file",
                "bytes": file_size,
                "created_at": int(time.time()),
                "filename": filename,
                "purpose": purpose,
                "filepath": filepath,
            }
            self.files[file_id] = file_obj
            return FileObject(**file_obj)

    def get_file(self, file_id: str) -> Optional[FileObject]:
        """Get a file record"""
        with self._lock:
            if file_id in self.files:
                return FileObject(**self.files[file_id])
            return None

    def delete_file(self, file_id: str) -> bool:
        """Delete a file record"""
        with self._lock:
            if file_id in self.files:
                filepath = self.files[file_id].get("filepath")
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                del self.files[file_id]
                return True
            return False

    def list_files(self, purpose: Optional[str] = None) -> List[FileObject]:
        """List files with optional purpose filter"""
        with self._lock:
            files = list(self.files.values())
            if purpose:
                files = [f for f in files if f.get("purpose") == purpose]
            return [FileObject(**f) for f in files]

  
    def create_thread(
        self,
        metadata: Optional[Dict] = None,
        file_ids: Optional[List[str]] = None,
        tool_resources: Optional[Dict] = None
    ) -> ThreadObject:
        """Create a thread record"""
        with self._lock:
            thread_id = f"thread-{uuid.uuid4().hex[:24]}"
            now = int(time.time())
            thread = {
                "id": thread_id,
                "object": "thread",
                "created_at": now,
                "last_accessed_at": now,
                "metadata": metadata or {},
                "file_ids": file_ids or [],
                "tool_resources": tool_resources,
            }
            self.threads[thread_id] = thread
            self.messages[thread_id] = []

            # Create workspace for this thread
            workspace_dir = get_thread_workspace(thread_id)
            os.makedirs(workspace_dir, exist_ok=True)
            os.makedirs(os.path.join(workspace_dir, "generated"), exist_ok=True)

            # Copy files to thread workspace
            for fid in (file_ids or []):
                if fid in self.files:
                    file_data = self.files[fid]
                    src_path = file_data.get("filepath")
                    if src_path and os.path.exists(src_path):
                        dst_path = uniquify_path(Path(workspace_dir) / file_data["filename"])
                        shutil.copy2(src_path, dst_path)

            return ThreadObject(**thread)

    def get_thread(self, thread_id: str) -> Optional[ThreadObject]:
        """Get a thread record"""
        with self._lock:
            if thread_id in self.threads:
                # Update last accessed time
                self.threads[thread_id]["last_accessed_at"] = int(time.time())
                return ThreadObject(**self.threads[thread_id])
            return None

    def delete_thread(self, thread_id: str) -> bool:
        """Delete a thread record"""
        with self._lock:
            if thread_id in self.threads:
                del self.threads[thread_id]
                if thread_id in self.messages:
                    del self.messages[thread_id]
                # Clean up workspace
                workspace_dir = get_thread_workspace(thread_id)
                if os.path.exists(workspace_dir):
                    shutil.rmtree(workspace_dir)
                return True
            return False

    def create_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
    ) -> MessageObject:
        """Create a message record"""
        with self._lock:
            if thread_id not in self.threads:
                raise ValueError(f"Thread {thread_id} not found")

            message_id = f"msg-{uuid.uuid4().hex[:24]}"
            message = {
                "id": message_id,
                "object": "thread.message",
                "created_at": int(time.time()),
                "thread_id": thread_id,
                "role": role,
                "content": [{"type": "text", "text": {"value": content}}],
                "file_ids": file_ids or [],
                "assistant_id": None,
                "run_id": None,
                "metadata": metadata or {},
            }
            self.messages[thread_id].append(message)
            return MessageObject(**message)

    def list_messages(self, thread_id: str) -> List[MessageObject]:
        """List messages in a thread"""
        with self._lock:
            if thread_id not in self.messages:
                return []
            return [MessageObject(**m) for m in self.messages[thread_id]]

    
    def cleanup_expired_threads(self, timeout_hours: float = 12) -> int:
        """Clean up threads that haven't been accessed for more than timeout_hours"""
        with self._lock:
            now = int(time.time())
            timeout_seconds = int(timeout_hours * 3600)
            expired_threads = []

            for thread_id, thread_data in self.threads.items():
                last_accessed = thread_data.get("last_accessed_at", thread_data.get("created_at", 0))
                if now - last_accessed > timeout_seconds:
                    expired_threads.append(thread_id)

        cleaned_count = 0
        for thread_id in expired_threads:
            try:
                # Delete thread and its workspace
                if self.delete_thread(thread_id):
                    cleaned_count += 1
                    print(f"Cleaned up expired thread: {thread_id}")
            except Exception as e:
                print(f"Error cleaning up thread {thread_id}: {e}")

        return cleaned_count


# Global storage instance
storage = Storage()