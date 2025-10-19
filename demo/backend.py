import openai
import json
import os
import shutil
import re
import io
import contextlib
import traceback
from pathlib import Path
from urllib.parse import quote
import subprocess
import sys
import tempfile
import requests
import threading
import http.server
from functools import partial
import socketserver
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import httpx
import uvicorn
import os
import re
import json
from fastapi.responses import StreamingResponse
import os
import re
from copy import deepcopy
import openai
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse

import re

os.environ.setdefault("MPLBACKEND", "Agg")


def execute_code(code_str):
    import io
    import contextlib
    import traceback

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            exec(code_str, {})
        output = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            output += stderr_capture.getvalue()
        return output
    except Exception as exec_error:
        code_lines = code_str.splitlines()
        tb_lines = traceback.format_exc().splitlines()
        error_line = None
        for line in tb_lines:
            if 'File "<string>", line' in line:
                try:
                    line_num = int(line.split(", line ")[1].split(",")[0])
                    error_line = line_num
                    break
                except (IndexError, ValueError):
                    continue
        error_message = f"Traceback (most recent call last):\n"
        if error_line is not None and 1 <= error_line <= len(code_lines):
            error_message += f'  File "<string>", line {error_line}, in <module>\n'
            error_message += f"    {code_lines[error_line-1].strip()}\n"
        error_message += f"{type(exec_error).__name__}: {str(exec_error)}"
        if stderr_capture.getvalue():
            error_message += f"\n{stderr_capture.getvalue()}"
        return f"[Error]:\n{error_message.strip()}"


def execute_code_safe(
    code_str: str, workspace_dir: str = None, timeout_sec: int = 120
) -> str:
    """在独立进程中执行代码，支持超时，避免阻塞主进程。"""
    if workspace_dir is None:
        workspace_dir = WORKSPACE_BASE_DIR
    exec_cwd = os.path.abspath(workspace_dir)
    os.makedirs(exec_cwd, exist_ok=True)
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", dir=exec_cwd)
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(code_str)
        print(
            f"[exec] Running script: {tmp_path} (timeout={timeout_sec}s) cwd={exec_cwd}"
        )
        # 在子进程中设置无界面环境变量，避免 GUI 后端
        child_env = os.environ.copy()
        child_env.setdefault("MPLBACKEND", "Agg")
        child_env.setdefault("QT_QPA_PLATFORM", "offscreen")
        child_env.pop("DISPLAY", None)

        completed = subprocess.run(
            [sys.executable, tmp_path],
            cwd=exec_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=child_env,
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        return output
    except subprocess.TimeoutExpired:
        return f"[Timeout]: execution exceeded {timeout_sec} seconds"
    except Exception as e:
        return f"[Error]: {str(e)}"
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# API endpoint and model path
API_BASE = "http://localhost:8000/v1"  # this localhost is for vllm api, do not change
MODEL_PATH = "DeepAnalyze-8B"  # replace to your path to DeepAnalyze-8B


# Initialize OpenAI client
client = openai.OpenAI(base_url=API_BASE, api_key="dummy")

# Workspace directory
WORKSPACE_BASE_DIR = "workspace"
HTTP_SERVER_PORT = 8100
HTTP_SERVER_BASE = (
    f"http://localhost:{HTTP_SERVER_PORT}"  # you can replace localhost to your local ip
)


def get_session_workspace(session_id: str) -> str:
    """返回指定 session 的 workspace 路径（workspace/{session_id}/）。"""
    if not session_id:
        session_id = "default"
    session_dir = os.path.join(WORKSPACE_BASE_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir


def build_download_url(rel_path: str) -> str:
    try:
        encoded = quote(rel_path, safe="/")
    except Exception:
        encoded = rel_path
    return f"{HTTP_SERVER_BASE}/{encoded}"


# FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def start_http_server():
    """启动HTTP文件服务器（不修改全局工作目录）。"""
    os.makedirs(WORKSPACE_BASE_DIR, exist_ok=True)
    handler = partial(
        http.server.SimpleHTTPRequestHandler, directory=WORKSPACE_BASE_DIR
    )
    with socketserver.TCPServer(("", HTTP_SERVER_PORT), handler) as httpd:
        print(f"HTTP Server serving {WORKSPACE_BASE_DIR} at port {HTTP_SERVER_PORT}")
        httpd.serve_forever()


# Start HTTP server in a separate thread
threading.Thread(target=start_http_server, daemon=True).start()


def collect_file_info(directory: str) -> str:
    """收集文件信息"""
    all_file_info_str = ""
    dir_path = Path(directory)
    if not dir_path.exists():
        return ""

    files = sorted([f for f in dir_path.iterdir() if f.is_file()])
    for idx, file_path in enumerate(files, start=1):
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        size_str = f"{size_kb:.1f}KB"
        file_info = {"name": file_path.name, "size": size_str}
        file_info_str = json.dumps(file_info, indent=4, ensure_ascii=False)
        all_file_info_str += f"File {idx}:\n{file_info_str}\n\n"
    return all_file_info_str


def get_file_icon(extension):
    """获取文件图标"""
    ext = extension.lower()
    icons = {
        (".jpg", ".jpeg", ".png", ".gif", ".bmp"): "🖼️",
        (".pdf",): "📕",
        (".doc", ".docx"): "📘",
        (".txt",): "📄",
        (".md",): "📝",
        (".csv", ".xlsx"): "📊",
        (".json", ".sqlite"): "🗄️",
        (".mp4", ".avi", ".mov"): "🎥",
        (".mp3", ".wav"): "🎵",
        (".zip", ".rar", ".tar"): "🗜️",
    }

    for extensions, icon in icons.items():
        if ext in extensions:
            return icon
    return "📁"


def uniquify_path(target: Path) -> Path:
    """若目标已存在，生成 'name (1).ext'、'name (2).ext' 形式的新路径。"""
    if not target.exists():
        return target
    parent = target.parent
    stem = target.stem
    suffix = target.suffix
    import re as _re

    m = _re.match(r"^(.*) \((\d+)\)$", stem)
    base = stem
    start = 1
    if m:
        base = m.group(1)
        try:
            start = int(m.group(2)) + 1
        except Exception:
            start = 1
    i = start
    while True:
        candidate = parent / f"{base} ({i}){suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def execute_code(code_str):
    """执行Python代码"""
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            exec(code_str, {})
        output = stdout_capture.getvalue()
        if stderr_capture.getvalue():
            output += stderr_capture.getvalue()
        return output
    except Exception as exec_error:
        return f"[Error]: {str(exec_error)}"


# API Routes
@app.get("/workspace/files")
async def get_workspace_files(session_id: str = Query("default")):
    """获取工作区文件列表（支持 session 隔离）"""
    workspace_dir = get_session_workspace(session_id)
    generated_dir = Path(workspace_dir) / "generated"
    # 获取 generated 目录下的文件名集合
    generated_files = (
        set(f.name for f in generated_dir.iterdir() if f.is_file())
        if generated_dir.exists()
        else set()
    )

    files = []
    for file_path in Path(workspace_dir).iterdir():
        if file_path.is_file():
            if file_path.name in generated_files:
                continue
            stat = file_path.stat()
            rel_path = f"{session_id}/{file_path.name}"
            files.append(
                {
                    "name": file_path.name,
                    "size": stat.st_size,
                    "extension": file_path.suffix.lower(),
                    "icon": get_file_icon(file_path.suffix),
                    "download_url": build_download_url(rel_path),
                    "preview_url": (
                        build_download_url(rel_path)
                        if file_path.suffix.lower()
                        in [
                            ".jpg",
                            ".jpeg",
                            ".png",
                            ".gif",
                            ".bmp",
                            ".pdf",
                            ".txt",
                            ".doc",
                            ".docx",
                            ".csv",
                            ".xlsx",
                        ]
                        else None
                    ),
                }
            )
    return {"files": files}


# ---------- Workspace Tree & Single File Delete ----------
def _rel_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
        return rel.as_posix()
    except Exception:
        return path.name


def build_tree(path: Path, root: Path | None = None) -> dict:
    if root is None:
        root = path
    node: dict = {
        "name": path.name or "workspace",
        "path": _rel_path(path, root),
        "is_dir": path.is_dir(),
    }
    if path.is_dir():
        children = []

        # 自定义排序：generated 文件夹放在最后，其他按目录优先、名称排序
        def sort_key(p):
            is_generated = p.name == "generated"
            is_dir = p.is_dir()
            return (is_generated, not is_dir, p.name.lower())

        for child in sorted(path.iterdir(), key=sort_key):
            if child.name.startswith("."):
                continue
            children.append(build_tree(child, root))
        node["children"] = children
    else:
        node["size"] = path.stat().st_size
        node["extension"] = path.suffix.lower()
        node["icon"] = get_file_icon(path.suffix)
        rel = _rel_path(path, root)
        node["download_url"] = build_download_url(rel)
    return node


@app.get("/workspace/tree")
async def workspace_tree(session_id: str = Query("default")):
    workspace_dir = get_session_workspace(session_id)
    root = Path(workspace_dir)
    tree_data = build_tree(root, root)

    # 在下载链接前加上 session_id 前缀
    def prefix_urls(node, sid):
        if "download_url" in node and node["download_url"]:
            # 重新构建包含 session_id 的路径
            rel = node.get("path", "")
            node["download_url"] = build_download_url(f"{sid}/{rel}")
        if "children" in node:
            for child in node["children"]:
                prefix_urls(child, sid)

    prefix_urls(tree_data, session_id)
    return tree_data


@app.delete("/workspace/file")
async def delete_workspace_file(
    path: str = Query(..., description="relative path under workspace"),
    session_id: str = Query("default"),
):
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()
    target = (abs_workspace / path).resolve()
    if abs_workspace not in target.parents and target != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if target.is_dir():
        raise HTTPException(status_code=400, detail="Folder deletion not allowed")
    try:
        target.unlink()
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/workspace/move")
async def move_path(
    src: str = Query(..., description="relative source path under workspace"),
    dst_dir: str = Query("", description="relative target directory under workspace"),
    session_id: str = Query("default"),
):
    """在同一 workspace 内移动（或重命名）文件/目录。
    - src: 源相对路径（必填）
    - dst_dir: 目标目录（相对路径，空表示移动到根目录）
    """
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()

    abs_src = (abs_workspace / src).resolve()
    if abs_workspace not in abs_src.parents and abs_src != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid src path")
    if not abs_src.exists():
        raise HTTPException(status_code=404, detail="Source not found")

    abs_dst_dir = (abs_workspace / (dst_dir or "")).resolve()
    if abs_workspace not in abs_dst_dir.parents and abs_dst_dir != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid dst_dir path")
    abs_dst_dir.mkdir(parents=True, exist_ok=True)

    target = abs_dst_dir / abs_src.name
    target = uniquify_path(target)
    try:
        shutil.move(str(abs_src), str(target))
        rel_new = str(target.relative_to(abs_workspace))
        return {"message": "moved", "new_path": rel_new}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Move failed: {e}")


@app.delete("/workspace/dir")
async def delete_workspace_dir(
    path: str = Query(..., description="relative directory under workspace"),
    recursive: bool = Query(True, description="delete directory recursively"),
    session_id: str = Query("default"),
):
    """删除 workspace 下的目录。默认递归删除，禁止删除根目录。"""
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()
    target = (abs_workspace / path).resolve()
    if abs_workspace not in target.parents and target != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid path")
    if target == abs_workspace:
        raise HTTPException(status_code=400, detail="Cannot delete workspace root")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="Not a directory")
    try:
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/proxy")
async def proxy(url: str):
    """Simple CORS proxy for previewing external files.
    WARNING: For production, add domain allowlist and authentication.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
            r = await client.get(url)
        return Response(
            content=r.content,
            media_type=r.headers.get("content-type", "application/octet-stream"),
            headers={"Access-Control-Allow-Origin": "*"},
            status_code=r.status_code,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Proxy fetch failed: {e}")


@app.post("/workspace/upload")
async def upload_files(
    files: List[UploadFile] = File(...), session_id: str = Query("default")
):
    """上传文件到工作区（支持 session 隔离）"""
    workspace_dir = get_session_workspace(session_id)
    uploaded_files = []

    for file in files:
        # 唯一化文件名，避免覆盖
        dst = uniquify_path(Path(workspace_dir) / file.filename)
        with open(dst, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        uploaded_files.append(
            {
                "name": dst.name,
                "size": len(content),
                "path": str(dst.relative_to(Path(workspace_dir))),
            }
        )

    return {
        "message": f"Successfully uploaded {len(uploaded_files)} files",
        "files": uploaded_files,
    }


@app.delete("/workspace/clear")
async def clear_workspace(session_id: str = Query("default")):
    """清空工作区（支持 session 隔离）"""
    workspace_dir = get_session_workspace(session_id)
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    return {"message": "Workspace cleared successfully"}


@app.post("/workspace/upload-to")
async def upload_to_dir(
    dir: str = Query("", description="relative directory under workspace"),
    files: List[UploadFile] = File(...),
    session_id: str = Query("default"),
):
    """上传文件到 workspace 下的指定子目录（仅限工作区内）。"""
    workspace_dir = get_session_workspace(session_id)
    abs_workspace = Path(workspace_dir).resolve()
    target_dir = (abs_workspace / dir).resolve()
    if abs_workspace not in target_dir.parents and target_dir != abs_workspace:
        raise HTTPException(status_code=400, detail="Invalid dir path")
    target_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        dst = uniquify_path(target_dir / f.filename)
        try:
            with open(dst, "wb") as buffer:
                content = await f.read()
                buffer.write(content)
            saved.append(
                {
                    "name": dst.name,
                    "size": len(content),
                    "path": str(dst.relative_to(abs_workspace)),
                }
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Save failed: {e}")
    return {"message": f"uploaded {len(saved)}", "files": saved}


@app.post("/execute")
async def execute_code_api(request: dict):
    """执行 Python 代码"""
    print("🔥 Execute API called:", request)  # Debug log

    try:
        code = request.get("code", "")
        session_id = request.get("session_id", "default")
        workspace_dir = get_session_workspace(session_id)

        if not code:
            raise HTTPException(status_code=400, detail="No code provided")

        print(f"Executing code: {code[:100]}...")  # Debug log (first 100 chars)

        # 使用子进程安全执行，避免 GUI/线程问题（在指定 session workspace 中）
        result = execute_code_safe(code, workspace_dir)
        print(f"✅ Execution result: {result[:200]}...")  # Debug log

        return {
            "success": True,
            "result": result,
            "message": "Code executed successfully",
        }

    except Exception as e:
        print(f"❌ Execution error: {traceback.format_exc()}")  # Debug log
        return {
            "success": False,
            "result": f"Error: {str(e)}",
            "message": "Code execution failed",
        }


def fix_code_block(content):
    def fix_text(text):
        stack = []
        lines = text.splitlines(keepends=True)
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```python"):
                if stack and stack[-1] == "```python":
                    result.append("```\n")
                    stack.pop()
                stack.append("```python")
                result.append(line)
            elif stripped == "```":
                if stack and stack[-1] == "```python":
                    stack.pop()
                result.append(line)
            else:
                result.append(line)
        while stack:
            result.append("```\n")
            stack.pop()
        return "".join(result)

    if isinstance(content, str):
        return fix_text(content)
    elif isinstance(content, tuple):
        text_part = content[0] if content[0] else ""
        return (fix_text(text_part), content[1])
    return content


def fix_tags_and_codeblock(s: str) -> str:
    """
    修复未闭合的tags，并确保</Code>后代码块闭合。
    """
    pattern = re.compile(
        r"<(Analyze|Understand|Code|Execute|Answer)>(.*?)(?:</\1>|(?=$))", re.DOTALL
    )

    # 找所有匹配
    matches = list(pattern.finditer(s))
    if not matches:
        return s  # 没有标签，直接返回

    # 检查最后一个匹配是否闭合
    last_match = matches[-1]
    tag_name = last_match.group(1)
    matched_text = last_match.group(0)

    if not matched_text.endswith(f"</{tag_name}>"):
        # 没有闭合，补上
        if tag_name == "Code":
            s = fix_code_block(s) + f"\n```\n</{tag_name}>"
        else:
            s += f"\n</{tag_name}>"

    return s


def bot_stream(messages, workspace, session_id="default"):
    original_cwd = os.getcwd()
    WORKSPACE_DIR = get_session_workspace(session_id)
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    # 创建 generated 子文件夹用于存放代码生成的文件
    GENERATED_DIR = os.path.join(WORKSPACE_DIR, "generated")
    os.makedirs(GENERATED_DIR, exist_ok=True)
    # print(messages)
    if messages and messages[0]["role"] == "assistant":
        messages = messages[1:]
    if messages and messages[-1]["role"] == "user":
        user_message = messages[-1]["content"]
        file_info = (
            collect_file_info(workspace)
            if workspace
            else collect_file_info(WORKSPACE_DIR)
        )
        messages[-1][
            "content"
        ] = f"# Instruction\n{user_message} Include data preparation, data analysis, visualization, data insight in the report.\n\n# Data\n{file_info}\nUse <Analyze>, <Understand>, <Code>, <Execute>, <Answer> tags."
    # print("111",messages)
    initial_workspace = set(workspace)
    assistant_reply = ""
    finished = False
    exe_output = None
    while not finished:
        response = client.chat.completions.create(
            model=MODEL_PATH,
            messages=messages,
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
                yield assistant_reply
            if "</Answer>" in cur_res:
                finished = True
                break
        if chunk.choices[0].finish_reason == "stop" and not finished:
            if not cur_res.endswith("</Code>"):
                cur_res += "</Code>"
                assistant_reply += "</Code>"
            yield assistant_reply
        if "</Code>" in cur_res and not finished:
            messages.append({"role": "assistant", "content": cur_res})
            code_match = re.search(r"<Code>(.*?)</Code>", cur_res, re.DOTALL)
            if code_match:
                code_content = code_match.group(1).strip()
                md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
                code_str = md_match.group(1).strip() if md_match else code_content
                # 执行前快照（路径 -> (size, mtime)）
                try:
                    before_state = {
                        p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                        for p in Path(WORKSPACE_DIR).rglob("*")
                        if p.is_file()
                    }
                except Exception:
                    before_state = {}
                # 在子进程中以固定工作区执行
                exe_output = execute_code_safe(code_str, WORKSPACE_DIR)
                # 执行后快照
                try:
                    after_state = {
                        p.resolve(): (p.stat().st_size, p.stat().st_mtime_ns)
                        for p in Path(WORKSPACE_DIR).rglob("*")
                        if p.is_file()
                    }
                except Exception:
                    after_state = {}
                # 计算新增与修改
                added_paths = [p for p in after_state.keys() if p not in before_state]
                modified_paths = [
                    p
                    for p in after_state.keys()
                    if p in before_state and after_state[p] != before_state[p]
                ]

                # 将新增和修改的文件移动到 generated 文件夹
                artifact_paths = []
                for p in added_paths:
                    try:
                        # 如果文件不在 generated 文件夹中，移动它
                        if not str(p).startswith(GENERATED_DIR):
                            dest_path = Path(GENERATED_DIR) / p.name
                            dest_path = uniquify_path(dest_path)
                            shutil.copy2(str(p), str(dest_path))
                            artifact_paths.append(dest_path.resolve())
                        else:
                            artifact_paths.append(p)
                    except Exception as e:
                        print(f"Error moving file {p}: {e}")
                        artifact_paths.append(p)

                # 为修改的文件生成副本并移动到 generated 文件夹
                for p in modified_paths:
                    try:
                        dest_name = f"{Path(p).stem}_modified{Path(p).suffix}"
                        dest_path = Path(GENERATED_DIR) / dest_name
                        dest_path = uniquify_path(dest_path)
                        shutil.copy2(p, dest_path)
                        artifact_paths.append(dest_path.resolve())
                    except Exception as e:
                        print(f"Error copying modified file {p}: {e}")

                # 旧：Execute 内部放控制台输出；新：追加 <File> 段落给前端渲染卡片
                exe_str = f"\n<Execute>\n```\n{exe_output}\n```\n</Execute>\n"
                file_block = ""
                if artifact_paths:
                    lines = ["<File>"]
                    for p in artifact_paths:
                        try:
                            rel = (
                                Path(p)
                                .relative_to(Path(WORKSPACE_DIR).resolve())
                                .as_posix()
                            )
                        except Exception:
                            rel = Path(p).name
                        # 在相对路径前加上 session_id 前缀
                        url = build_download_url(f"{session_id}/{rel}")
                        name = Path(p).name
                        lines.append(f"- [{name}]({url})")
                        if Path(p).suffix.lower() in [
                            ".png",
                            ".jpg",
                            ".jpeg",
                            ".gif",
                            ".webp",
                            ".svg",
                        ]:
                            lines.append(f"![{name}]({url})")
                    lines.append("</File>")
                    file_block = "\n" + "\n".join(lines) + "\n"
                assistant_reply += exe_str + file_block
                yield assistant_reply
                messages.append({"role": "execute", "content": f"{exe_output}"})
                # 刷新工作区快照（路径集合）
                current_files = set(
                    [
                        os.path.join(WORKSPACE_DIR, f)
                        for f in os.listdir(WORKSPACE_DIR)
                        if os.path.isfile(os.path.join(WORKSPACE_DIR, f))
                    ]
                )
                new_files = list(current_files - initial_workspace)
                if new_files:
                    workspace.extend(new_files)
                    initial_workspace.update(new_files)
    os.chdir(original_cwd)


@app.post("/chat/completions")
async def chat(body: dict = Body(...)):
    messages = body.get("messages", [])
    workspace = body.get("workspace", [])
    session_id = body.get("session_id", "default")

    def generate():
        for reply in bot_stream(messages, workspace, session_id):
            # result=reply + "\n"
            print(reply)
            result = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "deepanalyze-8b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": fix_tags_and_codeblock(reply),
                        },
                        "finish_reason": "stop",
                    }
                ],
            }

            yield json.dumps(result)

    return StreamingResponse(generate(), media_type="text/plain")


# -------- Export Report (PDF + MD) --------
from datetime import datetime


def _extract_sections_from_messages(messages: list[dict]) -> str:
    """从历史消息中抽取 <Answer>..</Answer> 作为报告主体，其余部分按原始顺序作为 Appendix 拼成 Markdown。"""
    if not isinstance(messages, list):
        return ""
    import re as _re

    parts: list[str] = []
    appendix: list[str] = []

    tag_pattern = r"<(Analyze|Understand|Code|Execute|File|Answer)>([\s\S]*?)</\1>"

    for idx, m in enumerate(messages, start=1):
        role = (m or {}).get("role")
        if role != "assistant":
            continue
        content = str((m or {}).get("content") or "")

        step = 1
        # 按照在文本中的出现顺序依次提取
        for match in _re.finditer(tag_pattern, content, _re.DOTALL):
            tag, seg = match.groups()
            seg = seg.strip()
            if tag == "Answer":
                parts.append(f"{seg}\n")

            appendix.append(f"\n### Step {step}: {tag}\n\n{seg}\n")
            step += 1

    final_text = "".join(parts).strip()
    if appendix:
        final_text += (
            "\n\n\\newpage\n\n# Appendix: Detailed Process\n"
            + "".join(appendix).strip()
        )

    # print(final_text)
    return final_text


def _save_md(md_text: str, base_name: str, workspace_dir: str) -> Path:
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    md_path = uniquify_path(Path(workspace_dir) / f"{base_name}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    return md_path


import pypandoc


def _save_pdf(md_text: str, base_name: str, workspace_dir: str) -> Path:
    Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    pdf_path = uniquify_path(Path(workspace_dir) / f"{base_name}.pdf")

    output = pypandoc.convert_text(
        md_text,
        "pdf",
        format="md",
        outputfile=str(pdf_path),
        extra_args=[
            "--standalone",
            "--pdf-engine=xelatex",
        ],
    )
    return pdf_path


from typing import Optional


def _render_md_to_html(md_text: str, title: Optional[str] = None) -> str:
    """简化为占位实现（仅供未来 PDF 渲染使用）。当前仅生成 MD。"""
    doc_title = (title or "Report").strip() or "Report"
    safe = (md_text or "").replace("<", "&lt;").replace(">", "&gt;")
    return f"<html><head><meta charset='utf-8'><title>{doc_title}</title></head><body><pre>{safe}</pre></body></html>"


def _save_pdf_from_md(html_text: str, base_name: str) -> Path:
    """TODO: 服务端 PDF 渲染未实现。"""
    raise NotImplementedError("TODO: implement server-side PDF rendering")


def _save_pdf_with_chromium(html_text: str, base_name: str) -> Path:
    """TODO: 使用 Chromium 渲染 PDF（暂不实现）。"""
    raise NotImplementedError("TODO: chromium-based PDF rendering")


def _save_pdf_from_text(text: str, base_name: str) -> Path:
    """TODO: 纯文本 PDF 渲染（暂不实现）。"""
    raise NotImplementedError("TODO: text-based PDF rendering")


@app.post("/export/report")
async def export_report(body: dict = Body(...)):
    """
    接收全部聊天历史（messages: [{role, content}...]），抽取 <Analyze>..</Analyze> ~ <Answer>..</Answer>
    仅生成 Markdown 文件并保存到 workspace；PDF 渲染留作 TODO。
    """
    try:
        messages = body.get("messages", [])
        title = (body.get("title") or "").strip()
        session_id = body.get("session_id", "default")
        workspace_dir = get_session_workspace(session_id)

        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="messages must be a list")

        md_text = _extract_sections_from_messages(messages)
        if not md_text:
            md_text = (
                "(No <Analyze>/<Understand>/<Code>/<Execute>/<Answer> sections found.)"
            )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r"[^\w\-_.]+", "_", title) if title else "Report"
        base_name = f"{safe_title}_{ts}" if title else f"Report_{ts}"

        # Save MD into generated/ folder under workspace
        export_dir = os.path.join(workspace_dir, "generated")
        os.makedirs(export_dir, exist_ok=True)

        print(md_text)
        md_path = _save_md(md_text, base_name, export_dir)

        # PDF 暂不生成（TODO）。
        pdf_path = _save_pdf(md_text, base_name, export_dir)

        result = {
            "message": "exported",
            "md": md_path.name,
            "pdf": pdf_path.name if pdf_path else None,
            "download_urls": {
                "md": build_download_url(f"{session_id}/generated/{md_path.name}"),
                "pdf": (
                    build_download_url(f"{session_id}/generated/{pdf_path.name}")
                    if pdf_path
                    else None
                ),
            },
        }
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("🚀 启动后端服务...")
    print(f"   - API服务: http://localhost:8200")
    print(f"   - 文件服务: http://localhost:8100")
    uvicorn.run(app, host="0.0.0.0", port=8200)
