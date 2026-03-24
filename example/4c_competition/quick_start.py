#!/usr/bin/env python3
"""
Simple streaming chat test script.
Supports streaming conversations with and without files.
"""

import openai
import os
import sys
import tempfile
import shutil
import requests
import zipfile
import subprocess
import threading
import time
from pathlib import Path

# Default configuration
DEFAULT_API_BASE = "http://localhost:8200/v1"
DEFAULT_MODEL = "deepanalyze-8b"

# Global client variable
client = None


def check_api_server_connection(api_base):
    """Check API server connectivity."""
    try:
        response = requests.get(f"{api_base}/models", timeout=3)
        return response.status_code == 200
    except:
        return False


def start_api_server():
    """Start API server in background."""
    script_dir = Path(__file__).parent
    main_py = script_dir / "main.py"
    
    if not main_py.exists():
        return None
    
    try:
        # Start server in background, silence output
        kwargs = {
            "cwd": str(script_dir),
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        
        process = subprocess.Popen(
            [sys.executable, str(main_py)],
            **kwargs
        )
        return process
    except:
        return None


def wait_for_server(api_base, max_wait=30, server_process=None):
    """Wait for server to start; show progress so the console does not look frozen."""
    for i in range(max_wait):
        if server_process is not None and server_process.poll() is not None:
            code = server_process.returncode
            print(
                f"\n❌ API server process exited early (code {code}). "
                "Run `python main.py` in another terminal to see the error."
            )
            return False
        if check_api_server_connection(api_base):
            if i > 0:
                print()
            return True
        # Same line updates so output stays readable
        print(f"\rWaiting for API at {api_base}... {i + 1}s / {max_wait}s", end="", flush=True)
        time.sleep(1)
    print()
    return False


def get_supported_file_extensions():
    """Supported file extensions."""
    return [
        '.csv', '.txt', '.json', '.xlsx', '.xls', 
        '.pdf', '.doc', '.docx', '.py', '.js', '.html',
        '.xml', '.yaml', '.yml', '.md', '.log'
    ]


def is_supported_file(file_path):
    """Check if file type is supported."""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in get_supported_file_extensions()


def extract_zip_file(zip_path, extract_to):
    """Extract ZIP file to target directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            zip_ref.extractall(extract_to)
            
            extracted_files = []
            for file_name in file_list:
                if not file_name.endswith('/'):
                    file_path = os.path.join(extract_to, file_name)
                    if os.path.exists(file_path):
                        extracted_files.append(file_path)
            
            return extracted_files
    except:
        return []


def download_file_from_url(url, filename, temp_dir):
    """Download file from URL to temp dir."""
    try:
        file_path = os.path.join(temp_dir, filename)
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return file_path
        return None
    except:
        return None


def process_streaming_chat(uploaded_files, user_instruction, api_key):
    """Run streaming chat analysis."""
    global client
    
    # Initialize client
    client = openai.OpenAI(
        base_url=DEFAULT_API_BASE,
        api_key=api_key,
    )
    
    print("🔄 Starting analysis...")
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    files_to_upload = []
    file_objects = []
    supported_extensions = get_supported_file_extensions()
    
    try:
        # Handle uploaded files
        if uploaded_files:
            for file_path in uploaded_files:
                if not os.path.exists(file_path):
                    continue
                
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_name)[1].lower()
                
                # Check for ZIP
                if file_ext == '.zip':
                    extract_dir = os.path.join(temp_dir, f"extracted_{os.path.splitext(file_name)[0]}")
                    os.makedirs(extract_dir, exist_ok=True)
                    extracted_files = extract_zip_file(file_path, extract_dir)
                    
                    if extracted_files:
                        for extracted_file in extracted_files:
                            extracted_name = os.path.basename(extracted_file)
                            extracted_ext = os.path.splitext(extracted_name)[1].lower()
                            
                            if extracted_ext in supported_extensions:
                                dest_path = os.path.join(temp_dir, extracted_name)
                                counter = 1
                                while os.path.exists(dest_path):
                                    name, ext = os.path.splitext(extracted_name)
                                    dest_path = os.path.join(temp_dir, f"{name}_{counter}{ext}")
                                    counter += 1
                                
                                shutil.copy2(extracted_file, dest_path)
                                files_to_upload.append(dest_path)
                else:
                    if file_ext in supported_extensions:
                        dest_path = os.path.join(temp_dir, file_name)
                        shutil.copy2(file_path, dest_path)
                        files_to_upload.append(dest_path)
            
            # Upload files to API
            for file_path in files_to_upload:
                try:
                    with open(file_path, "rb") as f:
                        file_obj = client.files.create(file=f, purpose="file-extract")
                        file_objects.append(file_obj)
                except:
                    pass
        
        file_names = [os.path.basename(path) for path in files_to_upload]
        
        # Use provided or default instruction
        if not user_instruction.strip():
            if files_to_upload:
                user_instruction = (
                    f"Please analyze the following data files {', '.join(file_names)}, "
                    "perform EDA, and generate visualizations. Focus on relationships, trends, and key insights."
                )
            else:
                user_instruction = "Please conduct a conversational analysis and provide detailed insights."
        
        print("\n" + "=" * 60)
        
        # Build messages
        if files_to_upload:
            messages = [
                {
                    "role": "user",
                    "content": user_instruction,
                    "file_ids": [file_obj.id for file_obj in file_objects],
                }
            ]
        else:
            messages = [{"role": "user", "content": user_instruction}]
        
        # Pass api_key via extra_body
        extra_body = {"api_key": api_key} if api_key else {}
        
        # Create streaming request
        try:
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                stream=True,
                extra_body=extra_body,
            )
        except openai.InternalServerError as e:
            raise Exception(f"❌ API server error: {e}")
        except openai.APIError as e:
            raise Exception(f"❌ API error: {e}")
        except Exception as e:
            raise Exception(f"❌ Connection error: {e}")
        
        full_response = ""
        collected_files = []
        downloadable_files = []
        
        # Stream output
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
            
            if hasattr(chunk, "generated_files") and chunk.generated_files:
                collected_files.extend(chunk.generated_files)
        
        print("\n" + "=" * 60)
        
        # Download generated files
        if collected_files:
            for file_info in collected_files:
                filename = file_info.get("name", f"generated_{len(downloadable_files)}.txt")
                url = file_info.get("url", "")
                if url:
                    local_path = download_file_from_url(url, filename, temp_dir)
                    if local_path:
                        downloadable_files.append(local_path)
        
        print(f"\n✅ Analysis complete (generated files: {len(collected_files)})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Cleanup temp files (optional)
        # if temp_dir and os.path.exists(temp_dir):
        #     shutil.rmtree(temp_dir)
        pass


def main():
    """Entry point."""
    # Check and start API server if needed
    if not check_api_server_connection(DEFAULT_API_BASE):
        print("Starting API server...")
        server_process = start_api_server()
        if server_process:
            if wait_for_server(DEFAULT_API_BASE, server_process=server_process):
                print("✅ API server started")
            else:
                print("❌ API server failed to start")
                return
        else:
            print("❌ Unable to start API server")
            return
    else:
        print("✅ API server already running")
    
    # Input API Key
    api_key = input("\nEnter API Key: ").strip()
    if not api_key:
        print("❌ API Key is required")
        return
    
    # Choose mode
    print("\nSelect dialog type:")
    print("  1. No-file dialog")
    print("  2. Dialog with files")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    uploaded_files = []
    if choice == "2":
        file_input = input("\nEnter file paths (comma separated): ").strip()
        if file_input:
            uploaded_files = [f.strip() for f in file_input.split(',') if f.strip()]
    
    # Input instruction
    user_instruction = input("\nEnter analysis instruction (blank for default): ").strip()
    
    # Start streaming dialog
    try:
        process_streaming_chat(uploaded_files, user_instruction, api_key)
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()

