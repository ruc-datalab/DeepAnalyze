import re
from fastmcp import Client
from fastmcp.client.client import CallToolResult
from mcp.types import TextContent, ImageContent
from fastmcp.client.transports import StdioTransport

def convert_to_backend_format(mcp_result: CallToolResult) -> str:
    """
    Convert a CallToolResult object to the backend.py Execute API format.
    """
    text = []
    for content in mcp_result.content:
        if isinstance(content, TextContent):
            text.append(content.text)
        elif isinstance(content, ImageContent):
            text.append(f"[IMG OUTPUT]")
        else:
            text.append(f"[UNKNOWN CONTENT TYPE]")
    
    return "\n".join(text) if text else ""

async def connect_notebook(jupyter_port: int) -> Client:
    """
    Connect to the deep_analyze.ipynb notebook.
    """
    # Initialize MCP Client
    transport = StdioTransport(
        command="npx",
        args=["mcp-remote", f"http://127.0.0.1:{jupyter_port}/mcp"],
    )
    client = Client(transport)

    # Connect to the notebook
    async with client:
        await client.call_tool("use_notebook", {
            "notebook_name": "deep_analyze",
            "notebook_path": "deep_analyze.ipynb",
            "mode": "connect",
        })
    
    return client

async def list_workspace_files(client: Client) -> str:
    """
    List all files in the workspace directory.
    """
    async with client:
        result = await client.call_tool("list_files", {
            "max_depth": 3,
            "limit": 0
        })

    return convert_to_backend_format(result)
        

async def insert_cell(client: Client, cell_index: int, cell_type: str, cell_source: str) -> int:
    """
    Insert a cell at the specified index in the notebook.
    """
    async with client:
        result = await client.call_tool("insert_cell", {
            "cell_index": cell_index,
            "cell_type": cell_type,
            "cell_source": cell_source,
        })
    content = result.content[0].text
    match = re.search(r"Cell inserted successfully at index (\d+)", content)
    actual_index = int(match.group(1)) if match else None
    return actual_index

async def append_execute_cell(client: Client, cell_source: str, timeout: int = 90) -> str:
    """
    Append a code cell with the given source and execute it.
    """
    index = await insert_cell(client, -1, "code", cell_source)
    async with client:
        result = await client.call_tool("execute_cell", {
            "cell_index": index,
            "timeout": timeout
        })

    return convert_to_backend_format(result)
