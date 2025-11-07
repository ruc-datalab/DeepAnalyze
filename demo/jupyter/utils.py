import socket
import aiohttp

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