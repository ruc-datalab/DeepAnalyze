import os
import re
import time
import tomllib
import asyncio
import openai
import subprocess
import nbformat
import threading
from pathlib import Path
from dotenv import load_dotenv
from utils import is_port_in_use, jupyter_lab_alive
from mcp_tools import (
    list_workspace_files,
    connect_notebook,
    append_execute_cell,
    insert_cell
)

# Load environment variables and config
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    raise FileNotFoundError(f".env MUST exist in directory {env_path.parent}")
config_path = Path(__file__).parent / "config.toml"
if config_path.exists():
    with config_path.open("rb") as f:
        config = tomllib.load(f)
else:
    raise FileNotFoundError(f"config.toml MUST exist in directory {config_path.parent}")
print("Env and config loaded successfully!")


# Initialize OpenAI client
print("Try to connect OpenAI client...")
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "dummy"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
)
try:
    client.models.list()
    print("OpenAI client connected successfully!")
except Exception as e:
    raise ConnectionError(f"OpenAI client connection failed: {e}")


# Initialize Working Space and deep_analyze.ipynb file
workspace_dir = Path(__file__).parent / "workspace"
workspace_dir.mkdir(exist_ok=True)
notebook = nbformat.v4.new_notebook()
notebook.cells.append(nbformat.v4.new_markdown_cell("The Workspace of Deep Analyze"))
with open(workspace_dir / "deep_analyze.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(notebook, f)
print(f"Workspace successfully initialized in {workspace_dir.as_posix()}")


# Initialize Jupyter Process
jupyter_port = config["JUPYTER"].get("JUPYTER_PORT", 8888)
start_jupyter = config["JUPYTER"].get("START_JUPYTER", True)
jupyter_process = None
if start_jupyter:
    if is_port_in_use(jupyter_port):
        raise RuntimeError(f"Port {jupyter_port} is already in use, cannot start new Jupyter Lab server")
    cmd = [
        "uv", "run",
        "--project", Path(__file__).parent.as_posix(),
        "jupyter", "lab",
        "--port", str(jupyter_port),
    ]
    jupyter_process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=workspace_dir.as_posix()
    )
    def start():
        for line in jupyter_process.stdout:
            pass

    output_thread = threading.Thread(target=start)
    output_thread.daemon = True
    output_thread.start()

    print("Waiting for Jupyter Lab server to start...")
    time.sleep(10)
else:
    # Detect if Jupyter Lab is already running
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        alive = loop.run_until_complete(jupyter_lab_alive(jupyter_port))
    finally:
        loop.close()
    if not alive:
        raise RuntimeError(f"Jupyter Lab server is not running on port {jupyter_port}")
print(f"Jupyter Lab server is running on http://localhost:{jupyter_port}")


async def bot_stream(messages):
    """
    Bot function that processes messages and executes code in Jupyter notebook.
    This is adapted from demo/backend.py but modified to work with Jupyter notebook.
    Returns the complete response in OpenAI format as a dictionary array.
    """
    # Connect to notebook
    mcp_client = await connect_notebook(jupyter_port)
    
    # Get file context
    file_info = await list_workspace_files(mcp_client)
    print(f"Workspace file info: \n{file_info}")
    
    # Process messages
    if messages and messages[0]["role"] == "assistant":
        messages = messages[1:]
    
    if messages and messages[-1]["role"] == "user":
        user_message = messages[-1]["content"]
        if file_info:
            messages[-1]["content"] = f"# Instruction\n{user_message}\n\n# Data\n{file_info}"
        else:
            messages[-1]["content"] = f"# Instruction\n{user_message}"
    
    finished = False
    while not finished:
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "DeepAnalyze-8B"),
            messages=messages,
            temperature=0.4,
            stream=False,  # Changed to False for non-streaming
            extra_body={
                "add_generation_prompt": False,
                "stop_token_ids": [151676, 151645],
                "max_new_tokens": 32768,
            },
        )
        
        # Get the complete response
        cur_res = response.choices[0].message.content

        # Check if finished
        if "</Answer>" in cur_res:
            messages.append({"role": "assistant", "content": cur_res})
            finished = True
            
        if response.choices[0].finish_reason == "stop" and not finished:
            if not cur_res.endswith("</Code>"):
                cur_res += "</Code>"
            
        # Process other tags (Analyze, Understand, Answer) first, before Code
        # Use a single regex to match all tags in order
        all_tags_pattern = r"<(Analyze|Understand|Answer)>(.*?)</\1>"
        tag_matches = re.finditer(all_tags_pattern, cur_res, re.DOTALL)
        for match in tag_matches:
            tag_content = match.group(2).strip()
            await insert_cell(mcp_client, -1, cell_source=tag_content, cell_type="markdown")
            
        # Then process Code tag after other tags
        if "</Code>" in cur_res and not finished:
            messages.append({"role": "assistant", "content": cur_res})
            
            # Extract code from <Code> tag
            code_match = re.search(r"<Code>(.*?)</Code>", cur_res, re.DOTALL)
            if code_match:
                code_content = code_match.group(1).strip()
                md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
                code_str = md_match.group(1).strip() if md_match else code_content
                
                # Execute code in Jupyter notebook
                exe_output = await append_execute_cell(mcp_client, code_str)
                messages.append({"role": "execute", "content": exe_output})
    
    return messages





