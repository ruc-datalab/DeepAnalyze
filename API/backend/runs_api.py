"""
Runs API for DeepAnalyze API Server
Handles run execution and core analysis logic
"""

import os
import json
import time
import traceback
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any

import openai
from fastapi import APIRouter, HTTPException, Body

from config import API_BASE, SUPPORTED_TOOLS
from models import RunObject
from storage import storage
from utils import (
    get_thread_workspace, extract_text_from_content, prepare_vllm_messages,
    execute_code_safe, WorkspaceTracker, render_file_block,
    collect_all_associated_files, fix_tags_and_codeblock,
    generate_markdown_report
)


# Initialize OpenAI client for vllm
vllm_client = openai.OpenAI(base_url=API_BASE, api_key="dummy")

# Create router for runs endpoints
router = APIRouter(prefix="/v1/threads", tags=["runs"])


def bot_stream_for_run(thread_id: str, assistant):
    """
    Execute the DeepAnalyze model in a streaming fashion.
    This is the core logic that runs the analysis workflow.
    """
    workspace_dir = get_thread_workspace(thread_id)
    generated_dir = os.path.join(workspace_dir, "generated")
    os.makedirs(generated_dir, exist_ok=True)

    # Collect all associated files from assistant, thread, and messages
    all_file_ids = collect_all_associated_files(thread_id, assistant, storage)

    # Copy all associated files to workspace if they don't exist already
    for fid in all_file_ids:
        file_obj = storage.get_file(fid)
        if file_obj:
            src_path = storage.files[fid].get("filepath")
            if src_path and os.path.exists(src_path):
                dst_path = Path(workspace_dir) / file_obj.filename
                # Only copy if file doesn't already exist in workspace
                if not dst_path.exists():
                    from utils import uniquify_path
                    dst_path = uniquify_path(dst_path)
                    import shutil
                    shutil.copy2(src_path, dst_path)

    # Get messages from thread
    messages_list = storage.list_messages(thread_id)

    # Convert to generic dict format for preparation
    conversation_messages = list(messages_list)
    if conversation_messages and conversation_messages[0].role == "assistant":
        conversation_messages = conversation_messages[1:]
    payload_messages: List[Dict[str, Any]] = [
        {"role": m.role, "content": extract_text_from_content(m.content)}
        for m in conversation_messages
    ]
    vllm_messages = prepare_vllm_messages(
        payload_messages, workspace_dir, assistant.instructions
    )

    # Workspace tracker for artifacts
    tracker = WorkspaceTracker(workspace_dir, generated_dir)

    assistant_reply = ""
    finished = False

    while not finished:
        # Call vllm
        response = vllm_client.chat.completions.create(
            model=assistant.model,
            messages=vllm_messages,
            temperature=0.4,
            stream=True,
            extra_body={
                "add_generation_prompt": False,
                "stop_token_ids": [151676, 151645],
                "max_new_tokens": 32768,
            },
        )

        cur_res = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                delta = chunk.choices[0].delta.content
                cur_res += delta
                assistant_reply += delta
                yield delta

            if "</Answer>" in cur_res:
                finished = True
                break

        if chunk.choices[0].finish_reason == "stop" and not finished:
            # Model stopped but didn't finish properly
            if not cur_res.endswith("</Code>"):
                cur_res += "</Code>"
                assistant_reply += "</Code>"
                yield "</Code>"

        # Handle code execution
        if "</Code>" in cur_res and not finished:
            vllm_messages.append({"role": "assistant", "content": cur_res})
            from utils import extract_code_from_segment
            code_str = extract_code_from_segment(cur_res)
            if code_str:
                exe_output = execute_code_safe(code_str, workspace_dir)
                artifacts = tracker.diff_and_collect()
                exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
                file_block = render_file_block(artifacts, workspace_dir, thread_id)
                assistant_reply += exe_str + file_block
                yield exe_str + file_block
                vllm_messages.append({"role": "execute", "content": exe_output})

    final_message_text = fix_tags_and_codeblock(assistant_reply)
    report_block = ""
    try:
        report_path = generate_markdown_report(
            thread_id, messages_list, final_message_text, workspace_dir
        )
    except Exception as report_error:
        report_path = None
        print(f"Report generation error: {report_error}")

    if report_path:
        try:
            rel = report_path.resolve().relative_to(Path(workspace_dir).resolve())
            rel_path = rel.as_posix()
        except Exception:
            rel_path = report_path.name
        from utils import build_download_url
        url = build_download_url(thread_id, rel_path)
        lines = ["<File>", f"- [{report_path.name}]({url})", "</File>"]
        report_block = "\n" + "\n".join(lines) + "\n"
        final_message_text += report_block
        yield report_block

    # Store final assistant message
    storage.create_message(
        thread_id=thread_id,
        role="assistant",
        content=final_message_text,
    )


@router.post("/{thread_id}/runs", response_model=RunObject)
async def create_run(
    thread_id: str,
    assistant_id: str = Body(...),
    model: Optional[str] = Body(None),
    instructions: Optional[str] = Body(None),
    tools: Optional[List[Dict]] = Body(None),
    metadata: Optional[Dict] = Body(None),
):
    """Create a run (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    assistant = storage.get_assistant(assistant_id)
    if not assistant:
        raise HTTPException(status_code=404, detail="Assistant not found")

    # Validate tools
    if tools:
        for tool in tools:
            if tool.get("type") not in SUPPORTED_TOOLS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported tool type: {tool.get('type')}",
                )

    # Create run
    run = storage.create_run(
        thread_id=thread_id,
        assistant_id=assistant_id,
        model=model or assistant.model,
        instructions=instructions,
        tools=tools,
        metadata=metadata,
    )

    # Execute run in background
    def execute_run():
        try:
            storage.update_run_status(
                run.id, "in_progress", started_at=int(time.time())
            )

            # Execute the model
            full_response = ""
            for chunk in bot_stream_for_run(thread_id, assistant):
                full_response += chunk

            storage.update_run_status(
                run.id, "completed", completed_at=int(time.time())
            )
        except Exception as e:
            print(f"Run execution error: {e}")
            traceback.print_exc()
            storage.update_run_status(
                run.id,
                "failed",
                failed_at=int(time.time()),
                last_error={"code": "server_error", "message": str(e)},
            )

    threading.Thread(target=execute_run, daemon=True).start()

    return run


@router.get("/{thread_id}/runs/{run_id}", response_model=RunObject)
async def retrieve_run(thread_id: str, run_id: str):
    """Retrieve a run (OpenAI compatible)"""
    # Update thread access time
    thread = storage.get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    run = storage.get_run(thread_id, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{thread_id}/runs", response_model=dict)
async def list_runs(thread_id: str):
    """List runs for a thread (OpenAI compatible)"""
    thread = storage.get_thread(thread_id)  # This will update last_accessed_at
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    runs = [r for r in storage.runs.values() if r["thread_id"] == thread_id]
    return {"object": "list", "data": [RunObject(**r).dict() for r in runs]}