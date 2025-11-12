"""
Chat Completions API for DeepAnalyze API Server
Handles extended chat completion with file attachment support
"""

import json
import os
import time
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import HTTPException

import openai
from fastapi import APIRouter, Body
from fastapi.responses import StreamingResponse

from config import API_BASE, DEFAULT_TEMPERATURE, STOP_TOKEN_IDS, MAX_NEW_TOKENS
from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice
from storage import storage
from utils import (
    get_thread_workspace, prepare_vllm_messages, execute_code_safe,
    execute_code_safe_async, WorkspaceTracker,
    generate_report_from_messages, extract_code_from_segment
)


# Initialize OpenAI clients for vllm
vllm_client = openai.OpenAI(base_url=API_BASE, api_key="dummy")
vllm_client_async = openai.AsyncOpenAI(base_url=API_BASE, api_key="dummy")

# Create router for chat endpoints
router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/completions")
async def chat_completions(
    model: str = Body(...),
    messages: List[Dict[str, Any]] = Body(...),
    file_ids: Optional[List[str]] = Body(None),
    temperature: Optional[float] = Body(DEFAULT_TEMPERATURE),
    stream: Optional[bool] = Body(False),
):
    """
    Extended chat completion API with file attachment support.
    Creates a temporary conversation with associated files.

    Parameters:
    - model: Model name
    - messages: List of message objects with role and content
    - file_ids: Optional list of file IDs to attach to the conversation
    - temperature: Sampling temperature (default 0.4)
    - stream: Whether to stream the response (default False)

    Returns:
    - Standard OpenAI chat completion response
    - Additional field 'generated_files' with list of generated file URLs
    """
    # Create temporary thread
    temp_thread = storage.create_thread(metadata={"temporary": True})
    workspace_dir = get_thread_workspace(temp_thread.id)
    generated_dir = os.path.join(workspace_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    try:
        # Collect all file IDs from both parameter and messages
        all_file_ids = set()

        # Add file_ids from parameter (backward compatibility)
        if file_ids:
            all_file_ids.update(file_ids)

        # Extract file_ids from messages (new OpenAI compatibility)
        for message in messages:
            if isinstance(message, dict) and "file_ids" in message:
                message_file_ids = message.get("file_ids", [])
                if isinstance(message_file_ids, list):
                    all_file_ids.update(message_file_ids)

        # Copy files to workspace
        for fid in all_file_ids:
            file_obj = storage.get_file(fid)
            if not file_obj:
                raise HTTPException(status_code=400, detail=f"File {fid} not found")
            src_path = storage.files[fid].get("filepath")
            if src_path and os.path.exists(src_path):
                from utils import uniquify_path
                dst_path = uniquify_path(Path(workspace_dir) / file_obj.filename)
                shutil.copy2(src_path, dst_path)

        # Build messages with DeepAnalyze prompt template
        vllm_messages: List[Dict[str, Any]] = prepare_vllm_messages(
            messages, workspace_dir
        )

        # Track generated files
        generated_files = []

        # Stream response with code execution (always enabled)
        if stream:
            def generate_stream_with_execution():
                assistant_reply = ""
                finished = False
                tracker = WorkspaceTracker(workspace_dir, generated_dir)

                while not finished:
                    response = vllm_client.chat.completions.create(
                        model=model,
                        messages=vllm_messages,
                        temperature=temperature,
                        stream=True,
                        extra_body={
                            "add_generation_prompt": False,
                            "stop_token_ids": STOP_TOKEN_IDS,
                            "max_new_tokens": MAX_NEW_TOKENS,
                        },
                    )

                    cur_res = ""
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            delta = chunk.choices[0].delta.content
                            cur_res += delta
                            assistant_reply += delta

                            chunk_data = {
                                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"

                        if "</Answer>" in cur_res:
                            finished = True
                            break

                    if chunk.choices[0].finish_reason == "stop" and not finished:
                        if not cur_res.endswith("</Code>"):
                            cur_res += "</Code>"
                            assistant_reply += "</Code>"

                    # Handle code execution
                    if "</Code>" in cur_res and not finished:
                        vllm_messages.append({"role": "assistant", "content": cur_res})

                        code_str = extract_code_from_segment(cur_res)
                        if code_str:
                            exe_output = execute_code_safe(code_str, workspace_dir)
                            artifacts = tracker.diff_and_collect()
                            exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
                           
                            assistant_reply += exe_str

                            # Stream execution result
                            for char in exe_str:
                                chunk_data = {
                                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": char},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk_data)}\n\n"

                            vllm_messages.append({"role": "execute", "content": exe_output})

                # Generate and stream report
                report_block = generate_report_from_messages(
                    messages, assistant_reply, workspace_dir, temp_thread.id, generated_files
                )
                if report_block:
                    for char in report_block:
                        chunk_data = {
                            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": char},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

                # Send final chunk with generated files
                final_chunk_data = {}
                if generated_files:
                    final_chunk_data["files"] = generated_files

                final_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": final_chunk_data,
                            "finish_reason": "stop"
                        }
                    ],
                }

                # Keep backward compatibility with generated_files field
                if generated_files:
                    final_chunk["generated_files"] = generated_files

                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream_with_execution(), media_type="text/event-stream")

        else:
            # Non-streaming response processed with code execution workflow (always enabled)
            assistant_reply = ""
            finished = False
            generated_files = []
            tracker = WorkspaceTracker(workspace_dir, generated_dir)

            while not finished:
                # Use async client to avoid blocking
                response = await vllm_client_async.chat.completions.create(
                    model=model,
                    messages=vllm_messages,
                    temperature=temperature,
                    stream=True,
                    extra_body={
                        "add_generation_prompt": False,
                        "stop_token_ids": STOP_TOKEN_IDS,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    },
                )

                cur_res = ""
                last_finish_reason: Optional[str] = None
                # For async streaming, we need to iterate through async chunks
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        delta = chunk.choices[0].delta.content
                        cur_res += delta
                        assistant_reply += delta
                    last_finish_reason = chunk.choices[0].finish_reason
                    if "</Answer>" in cur_res:
                        finished = True
                        break

                if last_finish_reason == "stop" and not finished:
                    if not cur_res.endswith("</Code>"):
                        cur_res += "</Code>"
                        assistant_reply += "</Code>"

                if "</Answer>" in cur_res:
                    finished = True

                if "</Code>" in cur_res and not finished:
                    vllm_messages.append({"role": "assistant", "content": cur_res})
                    code_str = extract_code_from_segment(cur_res)
                    if code_str:
                        # Use async version of execute_code_safe to avoid blocking
                        exe_output = await execute_code_safe_async(code_str, workspace_dir)
                        artifacts = tracker.diff_and_collect()
                        exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
                        assistant_reply += exe_str
                        vllm_messages.append({"role": "execute", "content": exe_output})

            # Generate report
            report_block = generate_report_from_messages(
                messages, assistant_reply, workspace_dir, temp_thread.id, generated_files
            )
            assistant_reply += report_block

            result_content = assistant_reply

            # Prepare message with files for OpenAI compatibility
            message_data = {
                "role": "assistant",
                "content": result_content,
            }

            # Add files to message object (new OpenAI compatibility)
            if generated_files:
                message_data["files"] = generated_files

            result = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": message_data,
                        "finish_reason": "stop",
                    }
                ],
            }

            # Keep backward compatibility with generated_files field
            if generated_files:
                result["generated_files"] = generated_files
            if file_ids:
                result["attached_files"] = file_ids

            return result
    finally:
        # Clean up temporary thread after some time (optional)
        pass