from __future__ import annotations

import json
import re
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
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
HEYWHALE_API_BASE = (
    "https://www.heywhale.com/api/model/services/691d42c36c6dda33df0bf645/app/v1"
)
REMOTE_STOP_SEQUENCES = ["</Code>", "</Answer>"]


@dataclass(frozen=True)
class ChatRuntimeConfig:
    provider: str = "local"
    temperature: float = 0.4
    model: str = settings.model_path
    api_key: str = ""
    api_base: str = ""


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


def _normalize_temperature(value: Any) -> float:
    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return 0.4
    return max(0.0, min(2.0, temperature))


def build_chat_runtime_config(payload: dict[str, Any] | None) -> ChatRuntimeConfig:
    body = payload or {}
    provider = str(body.get("provider") or "local").strip().lower() or "local"
    if provider not in {"local", "heywhale"}:
        provider = "local"

    api_base = str(body.get("api_base") or "").strip()
    if provider == "heywhale" and not api_base:
        api_base = HEYWHALE_API_BASE

    model = str(body.get("model") or settings.model_path).strip() or settings.model_path
    api_key = str(body.get("api_key") or "").strip()

    return ChatRuntimeConfig(
        provider=provider,
        temperature=_normalize_temperature(body.get("temperature")),
        model=model,
        api_key=api_key,
        api_base=api_base,
    )


def _infer_missing_close_tag(content: str) -> str | None:
    if "<Code>" in content and "</Code>" not in content:
        return "</Code>"
    if "<Answer>" in content and "</Answer>" not in content:
        return "</Answer>"
    return None


def _starts_with_structured_tag(content: str) -> bool:
    return bool(
        re.match(
            r"^\s*<(Analyze|Understand|Code|Execute|Answer|File)>",
            content or "",
        )
    )


def _iter_local_stream(
    conversation: list[dict[str, Any]],
    runtime_config: ChatRuntimeConfig,
):
    response = client.chat.completions.create(
        model=runtime_config.model,
        messages=conversation,
        temperature=runtime_config.temperature,
        stream=True,
        extra_body={
            "add_generation_prompt": False,
            "stop_token_ids": [151676, 151645],
            "max_new_tokens": 32768,
        },
    )
    try:
        for chunk in response:
            yield chunk.choices[0].delta.content if chunk.choices else None, chunk
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _iter_heywhale_stream(
    conversation: list[dict[str, Any]],
    runtime_config: ChatRuntimeConfig,
):
    if not runtime_config.api_key:
        raise ValueError("HeyWhale API key is required")

    request_body = {
        "messages": conversation,
        "temperature": runtime_config.temperature,
        "stream": True,
        "stop": REMOTE_STOP_SEQUENCES,
    }

    with httpx.Client(timeout=None) as http_client:
        with http_client.stream(
            "POST",
            f"{runtime_config.api_base.rstrip('/')}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {runtime_config.api_key}",
            },
            json=request_body,
        ) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                choice = (payload.get("choices") or [{}])[0]
                delta = (choice.get("delta") or {}).get("content")
                finish_reason = choice.get("finish_reason")
                yield delta, {"choices": [{"finish_reason": finish_reason}]}


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
    if re.search(r"(^|\W)(plt\.|matplotlib|sns\.|seaborn)", code_str, re.IGNORECASE):
        return CHINESE_MATPLOTLIB_BOOTSTRAP + "\n" + code_str
    return code_str


def bot_stream(
    messages: list[dict[str, Any]],
    workspace: list[str],
    session_id: str = "default",
    runtime_config: ChatRuntimeConfig | None = None,
):
    runtime_config = runtime_config or ChatRuntimeConfig()
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
    should_patch_first_assistant_message = not any(
        str(message.get("role") or "") == "assistant" for message in conversation
    )

    try:
        while not finished:
            if stop_event.is_set():
                break

            cur_res = ""
            last_chunk = None
            leading_chunks: list[str] = []
            leading_decided = not should_patch_first_assistant_message
            stream_iter = (
                _iter_heywhale_stream(conversation, runtime_config)
                if runtime_config.provider == "heywhale"
                else _iter_local_stream(conversation, runtime_config)
            )
            try:
                for delta, chunk in stream_iter:
                    if stop_event.is_set():
                        finished = True
                        break
                    last_chunk = chunk
                    if delta is not None:
                        if not leading_decided:
                            leading_chunks.append(delta)
                            combined = "".join(leading_chunks)
                            if not combined.strip():
                                continue
                            leading_decided = True
                            should_prefix = not _starts_with_structured_tag(combined)
                            if should_prefix:
                                cur_res += "<Analyze>\n"
                                yield "<Analyze>\n"
                            cur_res += combined
                            yield combined
                            should_patch_first_assistant_message = False
                            continue
                        cur_res += delta
                        yield delta
                    if "</Answer>" in cur_res:
                        finished = True
                        break
            except httpx.HTTPError as exc:
                raise RuntimeError(f"HeyWhale request failed: {exc}") from exc

            if stop_event.is_set():
                break

            finish_reason = None
            if last_chunk:
                try:
                    finish_reason = last_chunk["choices"][0]["finish_reason"]
                except Exception:
                    finish_reason = getattr(last_chunk.choices[0], "finish_reason", None)

            missing_tag = _infer_missing_close_tag(cur_res)
            if finish_reason == "stop" and not finished and missing_tag:
                cur_res += missing_tag
                yield missing_tag
                if missing_tag == "</Answer>":
                    finished = True

            if "</Code>" not in cur_res or finished:
                continue

            conversation.append({"role": "assistant", "content": cur_res})
            code_str = _extract_code_to_execute(cur_res)
            if not code_str:
                continue

            before_state = snapshot_workspace_files(workspace_dir)
            exe_output = execute_code_safe(code_str, workspace_dir, session_id)
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
