#!/usr/bin/env python3
"""
DeepAnalyze API CLI - Lightweight and Beautiful Command Line Interface
API client implemented with rich package, supporting file upload and data analysis tasks
"""

import os
import sys
import json
import time
import readline
import atexit
from pathlib import Path
from typing import Optional, List, Dict, Any
import openai
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn
from rich.table import Table
from rich.tree import Tree
from rich.markdown import Markdown
from rich.rule import Rule
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.syntax import Syntax
from rich.filesize import decimal

console = Console()

class DeepAnalyzeCLI:
    def __init__(self):
        """Initialize CLI client"""
        self.api_base = "http://localhost:8200/v1"
        self.model = "DeepAnalyze-8B"
        self.client = None
        self.uploaded_files = []
        self.current_thread_id = None
        self.chat_history = []  # Chat history
        self.generated_files = []  # Generated files (including reports, images, etc.)
        self.intermediate_files = []  # Intermediate files (uploaded generated files for conversation)
        self.setup_command_history()

    def setup_command_history(self):
        """Setup command history functionality"""
        history_file = os.path.expanduser("~/.deeppanalyze_history_en")

        # Try to read history file
        try:
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
            # Set history file length limit
            readline.set_history_length(1000)
            # Save history after each command
            readline.set_auto_history(True)
        except Exception as e:
            # Silently handle failures
            pass

        self.history_file = history_file

    def save_history(self):
        """Save history to file"""
        try:
            if hasattr(self, 'history_file'):
                readline.write_history_file(self.history_file)
        except Exception as e:
            # Silently handle failures
            pass

    def initialize_client(self):
        """Initialize OpenAI client"""
        try:
            self.client = openai.OpenAI(
                api_key="dummy",  # DeepAnalyze API uses dummy key
                base_url=self.api_base
            )
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to initialize client: {e}[/red]")
            return False

    def check_server(self) -> bool:
        """Check if API server is running"""
        try:
            import requests
            # First try to check health endpoint
            response = requests.get(f"http://localhost:8200/health", timeout=5)
            if response.status_code == 200:
                return True

            # If health endpoint is unavailable, try to check model list
            temp_client = openai.OpenAI(api_key="dummy", base_url=self.api_base)
            models = temp_client.models.list()
            return True
        except:
            return False

    def display_header(self):
        """Display program header information"""
        header_content = """[bold cyan]🚀 DeepAnalyze API Client[/bold cyan]
[dim]API Server: http://localhost:8200 | Model: DeepAnalyze-8B[/dim]"""

        console.print(Panel(header_content, title="DeepAnalyze CLI", border_style="cyan"))

    def upload_file(self, file_path: str) -> Optional[str]:
        """Upload file to API server"""
        try:
            full_path = Path(file_path).expanduser().resolve()
            if not full_path.exists():
                console.print(f"[red]❌ File does not exist: {file_path}[/red]")
                return None

            if not self.client:
                if not self.initialize_client():
                    return None

            file_size = full_path.stat().st_size

            # Safely handle filename to avoid encoding errors
            safe_filename = full_path.name
            try:
                # Ensure filename can be safely encoded
                safe_filename.encode('utf-8')
            except UnicodeEncodeError:
                # If filename contains invalid characters, use safe filename
                safe_filename = f"file_{int(time.time())}{full_path.suffix}"

            # Display upload start message
            console.print(f"[cyan]📤 Uploading {safe_filename}...[/cyan]")

            # Upload file using OpenAI library
            with open(full_path, 'rb') as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )

            # Safely handle returned filename
            safe_response_filename = file_obj.filename
            try:
                safe_response_filename.encode('utf-8')
            except UnicodeEncodeError:
                safe_response_filename = safe_filename  # Use our safe filename

            self.uploaded_files.append({
                'id': file_obj.id,
                'name': safe_response_filename,
                'path': str(full_path),
                'size': file_size,
                'purpose': file_obj.purpose
            })

            console.print("[green]✅ File uploaded successfully![/green]")
            console.print(f"[dim]File ID: {file_obj.id}[/dim]")
            console.print(f"[dim]Filename: {safe_response_filename}[/dim]")
            console.print(f"[dim]File size: {decimal(file_size)}[/dim]")
            console.print(f"[dim]Purpose: {file_obj.purpose}[/dim]")
            return file_obj.id

        except Exception as e:
            console.print(f"[red]❌ Upload error: {e}[/red]")
            return None

    def list_uploaded_files(self):
        """Display all files list (user uploaded files, intermediate files and output files)"""
        # Get output files (images and MD reports)
        output_files = [f for f in self.generated_files if f.get('type') == 'output']

        # Check if there are any files
        if not self.uploaded_files and not self.intermediate_files and not output_files:
            console.print("[yellow]📝 No files[/yellow]")
            return

        # Display user uploaded files
        if self.uploaded_files:
            table = Table(title="User Uploaded Files", show_header=True, header_style="bold magenta")
            table.add_column("Filename", style="cyan", no_wrap=True)
            table.add_column("File ID", style="green")
            table.add_column("File Size", style="yellow")
            table.add_column("Purpose", style="blue")
            table.add_column("Status", style="green")

            for file_info in self.uploaded_files:
                table.add_row(
                    file_info['name'],
                    file_info['id'][:8] + "...",
                    decimal(file_info['size']),
                    file_info.get('purpose', 'assistants'),
                    "✅ Uploaded"
                )

            console.print(table)

        # Display intermediate files
        if self.intermediate_files:
            if self.uploaded_files:
                console.print()  # Add empty line separator

            intermediate_table = Table(title="Generated Intermediate Files", show_header=True, header_style="bold cyan")
            intermediate_table.add_column("Filename", style="cyan", no_wrap=True)
            intermediate_table.add_column("File ID", style="green")
            intermediate_table.add_column("URL", style="blue")
            intermediate_table.add_column("Source", style="yellow")
            intermediate_table.add_column("Purpose", style="blue")
            intermediate_table.add_column("Status", style="orange3")

            for file_info in self.intermediate_files:
                file_name = file_info['name']
                original_url = file_info.get('original_url', '')

                # Create hyperlink URL display for intermediate files
                if original_url:
                    display_url = original_url[:60] + "..." if len(original_url) > 60 else original_url
                    url_text = Text(display_url, style="blue")
                    url_text.stylize(f"link {original_url}")
                else:
                    url_text = Text("No URL", style="blue")

                intermediate_table.add_row(
                    file_name,
                    file_info['id'][:8] + "...",
                    url_text,
                    "AI Generated",
                    file_info.get('purpose', 'assistants'),
                    "🔄 Intermediate File"
                )

            console.print(intermediate_table)

        # Display output files (images and MD reports)
        if output_files:
            if self.uploaded_files or self.intermediate_files:
                console.print()  # Add empty line separator

            output_table = Table(title="Generated Output Files", show_header=True, header_style="bold green")
            output_table.add_column("Filename", style="cyan", no_wrap=True)
            output_table.add_column("URL", style="blue")
            output_table.add_column("Source", style="yellow")
            output_table.add_column("Size", style="magenta")
            output_table.add_column("Status", style="bright_blue")

            for file_info in output_files:
                file_name = file_info.get('name', 'Unknown file')
                file_url = file_info.get('url', 'No URL')
                file_size = file_info.get('size', 'Unknown')

                # Determine file type based on extension
                if file_name.lower().endswith(('.md', '.markdown')):
                    file_type = "Report"
                elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp')):
                    file_type = "Image"
                else:
                    file_type = "Output"

                # Create hyperlink URL display (show truncated text, but link to full URL)
                if file_url != 'No URL':
                    display_url = file_url[:60] + "..." if len(file_url) > 60 else file_url
                    url_text = Text(display_url, style="blue")
                    url_text.stylize(f"link {file_url}")
                else:
                    url_text = Text("No URL", style="blue")

                # Determine file size display
                size_display = str(file_size) if file_size != 'Unknown' else "Unknown"

                output_table.add_row(
                    file_name,
                    url_text,
                    file_type,
                    size_display,
                    "📋 Generated"
                )

            console.print(output_table)

        # Display explanation information
        if self.intermediate_files or output_files:
            console.print()
            explanations = []
            if self.intermediate_files:
                explanations.append("🔄 Intermediate files: AI-generated data files, automatically uploaded for subsequent conversation context")
            if output_files:
                explanations.append("📋 Output files: AI-generated reports and images, directly accessible via URL")

            for explanation in explanations:
                console.print(f"[dim]{explanation}[/dim]")

    def is_intermediate_file(self, file_info: Dict[str, Any]) -> bool:
        """Determine if file should be uploaded as intermediate file (exclude reports and images)"""
        file_name = file_info.get('name', '').lower()

        # Exclude report files and image files
        intermediate_extensions = ['.md', '.markdown', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp']

        # Check file extension
        for ext in intermediate_extensions:
            if file_name.endswith(ext):
                return False

        # Other files are treated as intermediate files
        return True

    def upload_intermediate_file(self, file_info: Dict[str, Any]) -> Optional[str]:
        """Upload intermediate file and return file_id"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return None

            file_name = file_info.get('name', 'unknown_file')
            file_url = file_info.get('url', '')

            # Safely handle filename
            safe_file_name = file_name
            try:
                safe_file_name.encode('utf-8')
            except UnicodeEncodeError:
                # If filename contains invalid characters, use safe filename
                import time
                file_ext = os.path.splitext(file_name)[1]
                safe_file_name = f"intermediate_file_{int(time.time())}{file_ext}"

            console.print(f"[dim]📤 Uploading intermediate file: {safe_file_name}[/dim]")

            # Try to download file content from URL and upload
            import requests
            import tempfile
            import os

            # Download file
            response = requests.get(file_url)
            if response.status_code == 200:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(safe_file_name)[1]) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name

                try:
                    # Upload to API
                    with open(temp_file_path, 'rb') as f:
                        file_obj = self.client.files.create(
                            file=f,
                            purpose="assistants"
                        )

                    # Save to intermediate file list
                    self.intermediate_files.append({
                        'id': file_obj.id,
                        'name': safe_file_name,
                        'original_url': file_url,
                        'purpose': file_obj.purpose
                    })

                    console.print(f"[dim]✅ Intermediate file uploaded successfully: {safe_file_name} -> {file_obj.id}[/dim]")
                    return file_obj.id

                finally:
                    # Delete temporary file
                    os.unlink(temp_file_path)
            else:
                console.print(f"[red]❌ Failed to download intermediate file: {safe_file_name}[/red]")
                return None

        except Exception as e:
            console.print(f"[red]❌ Failed to upload intermediate file {safe_file_name}: {e}[/red]")
            return None

    def chat_with_file(self, message: str, file_ids: List[str] = None, stream: bool = True):
        """Chat with AI for analysis"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return

            # Add user message to history
            self.chat_history.append({"role": "user", "content": message})
            if file_ids:
                self.chat_history[-1]["file_ids"] = file_ids

            # Build message list, including historical conversation
            messages = []

            # Add historical conversation (excluding file_ids)
            for msg in self.chat_history[:-1]:  # Exclude the newly added user message
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})

            # Get all file IDs: uploaded files + intermediate files
            all_file_ids = []

            # Add uploaded file IDs
            uploaded_file_ids = [f['id'] for f in self.uploaded_files]
            all_file_ids.extend(uploaded_file_ids)

            # Add intermediate file IDs (uploaded generated files)
            intermediate_file_ids = [f['id'] for f in self.intermediate_files]
            all_file_ids.extend(intermediate_file_ids)

            # Remove duplicates
            all_file_ids = list(set(all_file_ids))

            # Add current user message (only this message contains file_ids)
            current_message = {"role": "user", "content": message}
            if all_file_ids:
                current_message["file_ids"] = all_file_ids
            messages.append(current_message)

            console.print("[cyan]💭 Analyzing...[/cyan]")
            if all_file_ids:
                console.print(f"[dim]Using files: {len(uploaded_file_ids)} uploaded files, {len(intermediate_file_ids)} intermediate files[/dim]")

            # Default to streaming response
            console.print("[dim]📡 Streaming response...[/dim]")
            response_text = ""
            collected_files = []

            console.print("\n[bold yellow]🤖 AI Response:[/bold yellow]")

            stream_response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                stream=True
            )

            for chunk in stream_response:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        response_text += content
                        console.print(content, end='')

                    # Collect generated files (in stream)
                    if hasattr(chunk, 'generated_files') and chunk.generated_files:
                        collected_files.extend(chunk.generated_files)

            console.print()  # Newline

            # Add AI response to history
            self.chat_history.append({"role": "assistant", "content": response_text})

            # Handle generated files
            if collected_files:
                console.print(f"\n[green]📁 Generated {len(collected_files)} files[/green]")

                intermediate_count = 0
                for file_info in collected_files:
                    file_name = file_info.get('name', 'Unknown file')
                    file_url = file_info.get('url', '')
                    file_id = file_info.get('id', '')

                    # Determine if it's an intermediate file
                    if self.is_intermediate_file(file_info):
                        # Upload intermediate file
                        uploaded_id = self.upload_intermediate_file(file_info)
                        if uploaded_id:
                            intermediate_count += 1
                        # Still save to generated_files for statistics
                        self.generated_files.append({
                            **file_info,
                            'uploaded_id': uploaded_id,
                            'type': 'intermediate'
                        })
                    else:
                        # Report and image files, save directly
                        # Try to get file size from URL
                        file_size = file_info.get('size', 'Unknown')
                        if file_size == 'Unknown' and file_url:
                            try:
                                import requests
                                response = requests.head(file_url, timeout=5)
                                if response.status_code == 200 and 'content-length' in response.headers:
                                    size_bytes = int(response.headers['content-length'])
                                    file_size = decimal(size_bytes)
                                else:
                                    # If HEAD request fails, try full download
                                    response = requests.get(file_url, timeout=10)
                                    if response.status_code == 200:
                                        size_bytes = len(response.content)
                                        file_size = decimal(size_bytes)
                            except Exception:
                                # If retrieval fails, keep as 'Unknown'
                                pass

                        self.generated_files.append({
                            **file_info,
                            'type': 'output',
                            'size': file_size
                        })
                        console.print(f"[dim]• {file_name}: {file_url or file_id}[/dim]")

                if intermediate_count > 0:
                    console.print(f"[dim]✅ {intermediate_count} files uploaded as intermediate files, available for subsequent conversations[/dim]")

            return response_text

        except Exception as e:
            console.print(f"[red]❌ Conversation error: {e}[/red]")
            return None


    def delete_file_by_id(self, file_id: str):
        """Delete file by ID"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return False

            console.print(f"[yellow]🗑️  Deleting file: {file_id}[/yellow]")
            self.client.files.delete(file_id)

            # Remove from local list
            self.uploaded_files = [f for f in self.uploaded_files if f['id'] != file_id]
            console.print(f"[green]✅ File deleted successfully[/green]")
            return True

        except Exception as e:
            console.print(f"[red]❌ Failed to delete file: {e}[/red]")
            return False

    def download_file_by_id(self, file_id: str, save_path: str = None):
        """Download file by ID"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return

            console.print(f"[cyan]📥 Downloading file: {file_id}[/cyan]")
            file_content = self.client.files.content(file_id)

            # Determine save path
            file_info = next((f for f in self.uploaded_files if f['id'] == file_id), None)
            if file_info:
                filename = file_info['name']
            else:
                filename = f"downloaded_file_{file_id[:8]}"

            if save_path:
                save_path = Path(save_path)
                if save_path.is_dir():
                    save_path = save_path / filename
            else:
                save_path = Path(filename)

            # Write file
            with open(save_path, 'wb') as f:
                f.write(file_content.content)

            console.print(f"[green]✅ File downloaded successfully: {save_path}[/green]")
            console.print(f"[dim]File size: {decimal(len(file_content.content))}[/dim]")

        except Exception as e:
            console.print(f"[red]❌ Failed to download file: {e}[/red]")

    def show_history(self):
        """Display conversation history"""
        if not self.chat_history:
            console.print("[yellow]📝 No conversation history[/yellow]")
            return

        output_files = [f for f in self.generated_files if f.get('type') != 'intermediate']
        intermediate_files = [f for f in self.generated_files if f.get('type') == 'intermediate']

        console.print(Panel(
            f"[bold]Conversation rounds:[/bold] {len(self.chat_history) // 2}\n"
            f"[bold]User messages:[/bold] {len([m for m in self.chat_history if m['role'] == 'user'])}\n"
            f"[bold]AI responses:[/bold] {len([m for m in self.chat_history if m['role'] == 'assistant'])}\n"
            f"[bold]Output files:[/bold] {len(output_files)}\n"
            f"[bold]Intermediate files:[/bold] {len(intermediate_files)}",
            title="Conversation History Statistics",
            border_style="blue"
        ))

        # Display recent conversations
        console.print("\n[bold]Recent conversation records:[/bold]")
        recent_messages = self.chat_history[-6:]  # Show last 6 messages

        for i, msg in enumerate(recent_messages):
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            role_color = "blue" if msg['role'] == 'user' else "green"

            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            console.print(f"[{role_color}]{role_emoji} {msg['role'].title()}:[/{role_color}] {content}")

            if i < len(recent_messages) - 1:
                console.print()

    def clear_chat_history(self):
        """Clear conversation history and generated intermediate files"""
        # Delete intermediate files
        if self.intermediate_files:
            console.print("[yellow]🗑️  Deleting intermediate files...[/yellow]")
            for file_info in self.intermediate_files:
                try:
                    self.client.files.delete(file_info['id'])
                    console.print(f"[green]✅ Deleted intermediate file: {file_info['name']}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to delete intermediate file {file_info['name']}: {e}[/red]")

        # Clear local lists
        self.chat_history.clear()
        self.generated_files.clear()
        self.intermediate_files.clear()

        console.print("[green]✅ Conversation history cleared[/green]")
        console.print("[green]✅ Generated file records cleared[/green]")
        console.print("[green]✅ Intermediate files deleted[/green]")

    def clear_all(self):
        """Clear all content (including uploaded files)"""
        try:
            # Delete server files - including uploaded files and intermediate files
            if self.uploaded_files:
                for file_info in self.uploaded_files:
                    try:
                        self.client.files.delete(file_info['id'])
                        console.print(f"[green]✅ Deleted uploaded file: {file_info['name']}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ Failed to delete uploaded file {file_info['name']}: {e}[/red]")

            if self.intermediate_files:
                for file_info in self.intermediate_files:
                    try:
                        self.client.files.delete(file_info['id'])
                        console.print(f"[green]✅ Deleted intermediate file: {file_info['name']}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ Failed to delete intermediate file {file_info['name']}: {e}[/red]")

            # Clear local lists
            self.chat_history.clear()
            self.generated_files.clear()
            self.intermediate_files.clear()
            self.uploaded_files.clear()

            console.print("[green]✅ All content cleared[/green]")
            console.print("[green]✅ Conversation history, generated files, uploaded files, intermediate files all cleared[/green]")

        except Exception as e:
            console.print(f"[red]❌ Error clearing all content: {e}[/red]")

    def get_system_status(self):
        """Get system status"""
        try:
            console.print("[cyan]🔍 Getting system status...[/cyan]")

            # Server status
            server_status = "✅ Online" if self.check_server() else "❌ Offline"

            # Statistics
            output_files = [f for f in self.generated_files if f.get('type') == 'output']
            status_panel = Panel(
                f"[bold]API Server:[/bold] {server_status}\n"
                f"[bold]API Endpoint:[/bold] {self.api_base}\n"
                f"[bold]Current Model:[/bold] {self.model}\n"
                f"[bold]Uploaded Files:[/bold] {len(self.uploaded_files)}\n"
                f"[bold]Intermediate Files:[/bold] {len(self.intermediate_files)}\n"
                f"[bold]Output Files:[/bold] {len(output_files)}\n"
                f"[bold]Conversation Rounds:[/bold] {len([m for m in self.chat_history if m['role'] == 'user'])}",
                title="System Status",
                border_style="cyan"
            )
            console.print(status_panel)

        except Exception as e:
            console.print(f"[red]❌ Failed to get system status: {e}[/red]")

    def interactive_mode(self):
        """Interactive conversation mode"""
        console.print("\n[bold green]💬 Entering interactive conversation mode[/bold green]")

        # Display help information
        self.show_help()

        while True:
            try:
                # 使用简单的输入提示，避免使用终端控制序列
                user_input = input("You: ").strip()

                # Save history after each valid input
                if user_input:
                    self.save_history()

                if user_input.lower() in ['quit', 'exit']:
                    console.print("[green]👋 Goodbye![/green]")
                    break

                # Handle various commands
                if self.handle_command(user_input):
                    continue

                if not user_input:
                    continue

                # Get currently uploaded file IDs
                file_ids = [f['id'] for f in self.uploaded_files]

                # Execute conversation (default streaming output)
                self.chat_with_file(user_input, file_ids if file_ids else None, stream=True)

            except KeyboardInterrupt:
                console.print("\n[green]👋 Goodbye![/green]")
                break
            except EOFError:
                console.print("\n[green]👋 Goodbye![/green]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")

    def show_help(self):
        """Display help information"""
        help_text = """
[bold cyan]📋 Available Commands:[/bold cyan]

[basic commands]
• [yellow]help[/yellow] - Display this help information
• [yellow]quit/exit[/yellow] - Exit the program
• [yellow]clear[/yellow] - Clear conversation history and generated intermediate files
• [yellow]clear-all[/yellow] - Clear all content (including uploaded files)

[file management]
• [yellow]files[/yellow] - View uploaded files
• [yellow]upload <file_path>[/yellow] - Upload new file
• [yellow]delete <file_id>[/yellow] - Delete specified file
• [yellow]download <file_id> [save_path][/yellow] - Download file

[system & history]
• [yellow]status[/yellow] - Display system status
• [yellow]history[/yellow] - Display conversation history
• [yellow]fid[/yellow] - Display all file names and complete IDs

[dim]Directly input text to start conversation, system will automatically use uploaded and generated files[/dim]
"""
        console.print(Panel(help_text, title="Command Help", border_style="blue"))

    def handle_command(self, user_input: str) -> bool:
        """Handle command, return True if it's a command"""
        cmd = user_input.lower().strip()

        # Help command
        if cmd in ['help', 'h']:
            self.show_help()
            return True

        # Clear conversation history
        elif cmd in ['clear']:
            if Confirm.ask("Are you sure you want to clear conversation history and generated intermediate files?"):
                self.clear_chat_history()
            return True

        # Clear all content
        elif cmd in ['clear-all']:
            if Confirm.ask("Are you sure you want to clear all content? This will delete all uploaded files"):
                self.clear_all()
            return True

        # File management commands
        elif cmd in ['files', 'ls']:
            self.list_uploaded_files()
            return True

        elif cmd.startswith('upload '):
            file_path = user_input[7:].strip()
            if file_path:
                self.upload_file(file_path)
            return True

        elif cmd.startswith('delete '):
            file_id = user_input[7:].strip()
            if file_id:
                self.delete_file_by_id(file_id)
            return True

        elif cmd.startswith('download '):
            parts = user_input.split()
            if len(parts) >= 2:
                file_id = parts[1]
                save_path = parts[2] if len(parts) > 2 else None
                self.download_file_by_id(file_id, save_path)
            return True

        # System commands
        elif cmd in ['status']:
            self.get_system_status()
            return True

        # History commands
        elif cmd in ['history']:
            self.show_history()
            return True

        # File ID commands
        elif cmd in ['fid']:
            self.show_file_ids()
            return True

        # Not a command
        return False

    def show_file_ids(self):
        """Display all file names and complete IDs (including uploaded files and intermediate files)"""
        # Check if there are any files
        if not self.uploaded_files and not self.intermediate_files:
            console.print("[yellow]📝 No files[/yellow]")
            return

        # Create comprehensive table
        table = Table(title="All Files and IDs", show_header=True, header_style="bold magenta")
        table.add_column("File Name", style="cyan", no_wrap=True)
        table.add_column("Complete File ID", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Status", style="blue")

        # Display uploaded files
        if self.uploaded_files:
            for file_info in self.uploaded_files:
                table.add_row(
                    file_info['name'],
                    file_info['id'],  # Complete ID
                    "📁 User Uploaded",
                    "✅ Uploaded"
                )

        # Display intermediate files
        if self.intermediate_files:
            for file_info in self.intermediate_files:
                table.add_row(
                    file_info['name'],
                    file_info['id'],  # Complete ID
                    "🔄 Intermediate File",
                    "🔄 AI Generated"
                )

        console.print(table)

        # Display summary
        console.print()
        console.print(f"[dim]Total files: {len(self.uploaded_files) + len(self.intermediate_files)}[/dim]")
        console.print(f"[dim]User uploaded: {len(self.uploaded_files)}[/dim]")
        console.print(f"[dim]Intermediate files: {len(self.intermediate_files)}[/dim]")

    def cleanup_files(self):
        """Clean up uploaded files"""
        if not self.uploaded_files:
            return

        if not self.client:
            self.initialize_client()

        console.print("[yellow]🧹 Cleaning up uploaded files...[/yellow]")

        for file_info in self.uploaded_files:
            try:
                # Delete file using OpenAI library
                self.client.files.delete(file_info['id'])
                console.print(f"[green]✅ Deleted: {file_info['name']}[/green]")
            except Exception as e:
                console.print(f"[red]❌ Delete error {file_info['name']}: {e}[/red]")

        # Clear local list
        self.uploaded_files.clear()

    def run(self):
        """Run main program - directly enter interactive mode"""
        try:
            # Check server status
            if not self.check_server():
                console.print("[red]❌ API server is not running![/red]")
                console.print("[yellow]Please start the API server first: python backend/main.py[/yellow]")
                return

            self.display_header()
            console.print("[green]✅ API server connection successful[/green]")
            console.print(f"[dim]Current model: {self.model}[/dim]")
            console.print(f"[dim]API endpoint: {self.api_base}[/dim]\n")

            # Directly enter interactive mode
            self.interactive_mode()

        except KeyboardInterrupt:
            console.print("\n[green]👋 Program terminated[/green]")
            self.cleanup_files()
        except Exception as e:
            console.print(f"[red]❌ Program error: {e}[/red]")
            self.cleanup_files()


def main():
    """Main function"""
    cli = DeepAnalyzeCLI()
    cli.run()


if __name__ == "__main__":
    main()