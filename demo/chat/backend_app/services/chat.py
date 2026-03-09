from __future__ import annotations

import re
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import openai

from .execution import (
    build_file_block,
    collect_artifact_paths,
    execute_code_safe,
    snapshot_workspace_files,
)
from .workspace import collect_file_info, get_session_workspace
from ..settings import CHINESE_MATPLOTLIB_BOOTSTRAP, settings


client = openai.OpenAI(base_url=settings.api_base, api_key="dummy")
_STOP_EVENTS: dict[str, threading.Event] = {}
_STOP_EVENTS_LOCK = threading.Lock()


def _get_or_create_stop_event(session_id: str) -> threading.Event:
    sid = session_id or "default"
    with _STOP_EVENTS_LOCK:
        event = _STOP_EVENTS.get(sid)
        if event is None:
            event = threading.Event()
            _STOP_EVENTS[sid] = event
        return event


def request_stop(session_id: str) -> None:
    _get_or_create_stop_event(session_id).set()


def _resolve_workspace_selection(
    workspace: Iterable[str] | None,
    workspace_dir: str,
) -> list[Path]:
    workspace_root = Path(workspace_dir).resolve()
    resolved_paths: list[Path] = []
    for item in workspace or []:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = (workspace_root / candidate).resolve()
        if candidate.exists() and candidate.is_file():
            resolved_paths.append(candidate)
    return resolved_paths


def _build_user_prompt(messages: list[dict[str, Any]], workspace: list[str], workspace_dir: str) -> None:
    if not messages or messages[-1].get("role") != "user":
        return

    user_message = str(messages[-1].get("content") or "")
    selected_paths = _resolve_workspace_selection(workspace, workspace_dir)
    file_info = collect_file_info(selected_paths if selected_paths else workspace_dir)
    if file_info:
        messages[-1]["content"] = f"# Instruction\n{user_message}\n\n# Data\n{file_info}"
    else:
        messages[-1]["content"] = f"# Instruction\n{user_message}"


def _extract_code_to_execute(content: str) -> str | None:
    code_match = re.search(r"<Code>(.*?)</Code>", content, re.DOTALL)
    if not code_match:
        return None

    code_content = code_match.group(1).strip()
    md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
    code_str = md_match.group(1).strip() if md_match else code_content
    return CHINESE_MATPLOTLIB_BOOTSTRAP + "\n" + code_str


def bot_stream(messages: list[dict[str, Any]], workspace: list[str], session_id: str = "default"):
    stop_event = _get_or_create_stop_event(session_id)
    stop_event.clear()
    conversation = deepcopy(messages or [])
    workspace_paths = list(workspace or [])
    workspace_dir = get_session_workspace(session_id)
    generated_dir = str(Path(workspace_dir) / "generated")
    Path(generated_dir).mkdir(parents=True, exist_ok=True)

    if conversation and conversation[0].get("role") == "assistant":
        conversation = conversation[1:]

    _build_user_prompt(conversation, workspace_paths, workspace_dir)

    initial_workspace = {
        path.resolve() for path in _resolve_workspace_selection(workspace_paths, workspace_dir)
    }
    finished = False

    try:
        while not finished:
            if stop_event.is_set():
                break

            response = client.chat.completions.create(
                model=settings.model_path,
                messages=conversation,
                temperature=0.4,
                stream=True,
                extra_body={
                    "add_generation_prompt": False,
                    "stop_token_ids": [151676, 151645],
                    "max_new_tokens": 32768,
                },
            )

            cur_res = ""
            last_chunk = None
            try:
                for chunk in response:
                    if stop_event.is_set():
                        finished = True
                        break
                    last_chunk = chunk
                    if chunk.choices and chunk.choices[0].delta.content is not None:
                        delta = chunk.choices[0].delta.content
                        cur_res += delta
                        yield delta
                    if "</Answer>" in cur_res:
                        finished = True
                        break
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

            if stop_event.is_set():
                break

            if (
                last_chunk
                and last_chunk.choices[0].finish_reason == "stop"
                and not finished
                and "<Code>" in cur_res
                and "</Code>" not in cur_res
            ):
                missing_tag = "</Code>"
                cur_res += missing_tag
                yield missing_tag

            if "</Code>" not in cur_res or finished:
                continue

            conversation.append({"role": "assistant", "content": cur_res})
            code_str = _extract_code_to_execute(cur_res)
            if not code_str:
                continue

            before_state = snapshot_workspace_files(workspace_dir)
            exe_output = execute_code_safe(code_str, workspace_dir)
            if stop_event.is_set():
                break
            after_state = snapshot_workspace_files(workspace_dir)
            artifact_paths = collect_artifact_paths(
                before_state,
                after_state,
                generated_dir,
                session_id,
            )

            exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
            file_block = build_file_block(artifact_paths, workspace_dir, session_id)
            yield exe_str + file_block

            conversation.append({"role": "execute", "content": exe_output})

            current_files = {
                path.resolve() for path in Path(workspace_dir).rglob("*") if path.is_file()
            }
            new_files = [str(path) for path in current_files - initial_workspace]
            if new_files:
                workspace_paths.extend(new_files)
                initial_workspace.update(Path(path).resolve() for path in new_files)
    finally:
        stop_event.clear()
