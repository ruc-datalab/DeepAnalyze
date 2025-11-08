import socket
import aiohttp
import json
from pathlib import Path

# Detect if port is in use
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

# Detect if Jupyter Lab server is alive
async def jupyter_lab_alive(port: int) -> bool:
    url = f"http://127.0.0.1:{port}/lab"
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            async with session.get(url) as resp:
                return resp.status == 200
    except Exception:
        return False

def load_system_prompt(config: dict) -> str | None:
    """Load system prompt from config and index file."""
    prompt_path = Path(__file__).parent / "prompt"
    index_path = prompt_path / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Prompt index file not found: {index_path}")
    
    with index_path.open("r", encoding="utf-8") as f:
        prompt_index = json.load(f)
    
    prompt_name = config.get("PROMPT", {}).get("PROMPT_TEMPLATE", "general")
    prompt_info = None
    
    for prompt in prompt_index.get("prompts", []):
        if prompt.get("name") == prompt_name:
            prompt_info = {
                "name": prompt_name,
                "path": prompt_path / prompt.get("path", ""),
                "description": prompt.get("description", "")
            }
    
    if prompt_info is None or not prompt_info["path"].exists():
        print(f"[WARNING] Prompt not found, using empty prompt.")
        return None
    with prompt_info["path"].open("r", encoding="utf-8") as f:
        return f.read()