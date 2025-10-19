#!/usr/bin/env python3
"""批量更新后端接口以支持 session 隔离"""
import re

def update_backend():
    with open('/Users/a1234/Desktop/jobs/backend.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 更新 execute_code_safe 中的 WORKSPACE_DIR
    content = re.sub(
        r'exec_cwd = os\.path\.abspath\(WORKSPACE_DIR\)',
        'exec_cwd = os.path.abspath(workspace_dir)',
        content
    )
    
    # 2. 更新 /workspace/clear 接口
    content = re.sub(
        r'@app\.delete\("/workspace/clear"\)\nasync def clear_workspace\(\):\s*"""清空工作区"""[^}]+}',
        '''@app.delete("/workspace/clear")
async def clear_workspace(session_id: str = Query("default")):
    """清空工作区（支持 session 隔离）"""
    workspace_dir = get_session_workspace(session_id)
    if os.path.exists(workspace_dir):
        shutil.rmtree(workspace_dir)
    os.makedirs(workspace_dir, exist_ok=True)
    return {"message": "Workspace cleared successfully"}''',
        content,
        flags=re.DOTALL
    )
    
    # 3. 更新 /execute 接口
    content = re.sub(
        r'@app\.post\("/execute"\)\nasync def execute_code_api\(request: dict\):',
        '@app.post("/execute")\nasync def execute_code_api(request: dict):',
        content
    )
    content = re.sub(
        r'code = request\.get\("code", ""\)',
        'code = request.get("code", "")\n        session_id = request.get("session_id", "default")\n        workspace_dir = get_session_workspace(session_id)',
        content
    )
    content = re.sub(
        r'os\.makedirs\(WORKSPACE_DIR, exist_ok=True\)\s+# 使用子进程安全执行',
        '# 使用子进程安全执行（在指定 session workspace 中）',
        content
    )
    content = re.sub(
        r'result = execute_code_safe\(code\)',
        'result = execute_code_safe(code, workspace_dir)',
        content
    )
    
    # 4. 更新 execute_code_safe 函数签名
    content = re.sub(
        r'def execute_code_safe\(code_str: str, timeout_sec: int = 120\) -> str:',
        'def execute_code_safe(code_str: str, workspace_dir: str = None, timeout_sec: int = 120) -> str:',
        content
    )
    
    # 5. 更新 export_report 接口
    content = re.sub(
        r'messages = body\.get\("messages", \[\]\)\s+title = \(body\.get\("title"\)',
        'messages = body.get("messages", [])\n        session_id = body.get("session_id", "default")\n        workspace_dir = get_session_workspace(session_id)\n        title = (body.get("title")',
        content
    )
    content = re.sub(
        r'md_path = _save_md\(md_text, base_name\)',
        'md_path = _save_md(md_text, base_name, workspace_dir)',
        content
    )
    content = re.sub(
        r'def _save_md\(md_text: str, base_name: str\) -> Path:',
        'def _save_md(md_text: str, base_name: str, workspace_dir: str) -> Path:',
        content
    )
    content = re.sub(
        r'Path\(WORKSPACE_DIR\)\.mkdir\(parents=True, exist_ok=True\)\s+md_path = uniquify_path\(Path\(WORKSPACE_DIR\) / f"\{base_name\}\.md"\)',
        'Path(workspace_dir).mkdir(parents=True, exist_ok=True)\n    md_path = uniquify_path(Path(workspace_dir) / f"{base_name}.md")',
        content
    )
    
    # 6. 更新 /chat/completions 中的 collect_file_info 和 bot_stream
    content = re.sub(
        r'def bot_stream\(messages, workspace\):',
        'def bot_stream(messages, workspace, session_id="default"):',
        content
    )
    content = re.sub(
        r'@app\.post\("/chat/completions"\)\nasync def chat\(body: dict = Body\(\.\.\.\)\):\s+messages = body\.get\("messages", \[\]\)\s+workspace = body\.get\("workspace", \[\]\)',
        '@app.post("/chat/completions")\nasync def chat(body: dict = Body(...)):\n    messages = body.get("messages", [])\n    workspace = body.get("workspace", [])\n    session_id = body.get("session_id", "default")',
        content
    )
    content = re.sub(
        r'for reply in bot_stream\(messages, workspace\):',
        'for reply in bot_stream(messages, workspace, session_id):',
        content
    )
    
    with open('/Users/a1234/Desktop/jobs/backend.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Backend session isolation update complete!")

if __name__ == "__main__":
    update_backend()

