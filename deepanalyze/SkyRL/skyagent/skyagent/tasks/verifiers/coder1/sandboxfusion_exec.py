import requests
from itertools import cycle
import threading
import os
from .utils import _ERROR_MSG_PREFIX, _DEFAULT_TIMEOUT_SECONDS, RunCodeResponse, RunStatus

# Default sandbox servers - can be overridden via environment variable or function parameter
DEFAULT_SANDBOX_SERVERS = [
    # "fs-mbz-gpu-044", # Add more servers here
]

# Thread-safe cycle iterator for round-robin load balancing
server_cycle = None
cycle_lock = threading.Lock()

def _parse_sandbox_servers(servers_input):
    """Parse sandbox servers from various input formats"""
    if not servers_input:
        return DEFAULT_SANDBOX_SERVERS
    
    if isinstance(servers_input, str):
        # Single server or comma-separated servers
        if ',' in servers_input:
            return [server.strip() for server in servers_input.split(',')]
        else:
            return [servers_input.strip()]
    elif isinstance(servers_input, list):
        return servers_input
    else:
        raise ValueError(f"Invalid sandbox servers format: {type(servers_input)}. Expected str or list.")

def _get_next_server(server_cycle):
    """Get the next server in round-robin fashion thread-safely."""
    with cycle_lock:
        return next(server_cycle)

def code_exec_sandboxfusion(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, sandbox_servers=None):
    """
    Execute Python code using SandboxFusion remote service.
    
    Args:
        code: Python code to execute
        stdin: Optional input to pass to the code
        timeout: Timeout in seconds (default from utils)
        sandbox_servers: Optional server names for sandbox servers. Can be:
                        - Single server string: "fs-mbz-gpu-044"
                        - Comma-separated servers: "fs-mbz-gpu-044,fs-mbz-gpu-045"
                        - List of servers: ["fs-mbz-gpu-044", "fs-mbz-gpu-045"]
                        - None: Uses SANDBOX_FUSION_SERVERS environment variable or default
        
    Returns:
        tuple: (success: bool, output: str)
    """
    try:
        # Determine sandbox servers to use
        if sandbox_servers is None:
            sandbox_servers = os.getenv('SANDBOX_FUSION_SERVERS', '')
        
        servers = _parse_sandbox_servers(sandbox_servers)
        server_cycle = cycle(servers)
        
        if not servers:
            return False, _ERROR_MSG_PREFIX + "No sandbox servers configured. Set SANDBOX_FUSION_SERVERS environment variable or pass sandbox_servers parameter."
        
        request_data = {
            "language": "python",
            "code": code,
            "stdin": stdin,
            "run_timeout": timeout
        }
        
        # Try each server (for load balancing/failover)
        for _ in range(len(servers)):
            try:
                server = _get_next_server(server_cycle)
                url = f"http://{server}:8080/run_code"
                response = requests.post(url, json=request_data, timeout=timeout + 2)
                
                if response.status_code != 200:
                    continue  # Try next server
                    
                result = RunCodeResponse(**response.json())
                if result.status == RunStatus.Success:
                    return True, result.run_result.stdout
                else:
                    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{result.run_result.stdout}\n\nSTDERR:\n{result.run_result.stderr}"
                    
            except requests.exceptions.RequestException:
                continue  # Try next server
        
        # If we get here, all servers failed
        return False, _ERROR_MSG_PREFIX + f"All sandbox servers failed to process the request. Servers tried: {servers}"
            
    except Exception as e:
        return False, _ERROR_MSG_PREFIX + f"Execution error: {str(e)}"

def code_exec_sandboxfusion_with_pytest(code, pytest_code, timeout=_DEFAULT_TIMEOUT_SECONDS, sandbox_servers=None):
    """
    Execute Python code with pytest using SandboxFusion remote service.
    
    Args:
        code: Python solution code
        pytest_code: Pytest test code
        timeout: Timeout in seconds
        sandbox_servers: Optional server names for sandbox servers (same format as code_exec_sandboxfusion)
        
    Returns:
        tuple: (success: bool, output: str)
    """
    # Combine the solution code and test code
    combined_code = f"""
{code}

{pytest_code}
"""
    return code_exec_sandboxfusion(combined_code, timeout=timeout, sandbox_servers=sandbox_servers)