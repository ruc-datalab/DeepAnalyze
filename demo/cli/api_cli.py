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
        """Display all files list (user uploaded files and AI generated files in workspace)"""
        # Get AI generated files from workspace (all generated files)
        workspace_files = self.generated_files

        # Check if there are any files
        if not self.uploaded_files and not workspace_files:
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

        # Display AI generated files in workspace
        if workspace_files:
            if self.uploaded_files:
                console.print()  # Add empty line separator

            workspace_table = Table(title="AI Generated Files in Workspace", show_header=True, header_style="bold green")
            workspace_table.add_column("Filename", style="cyan", no_wrap=True)
            workspace_table.add_column("URL", style="blue")
            workspace_table.add_column("File Type", style="yellow")
            workspace_table.add_column("Size", style="magenta")
            workspace_table.add_column("Status", style="bright_blue")

            for file_info in workspace_files:
                file_name = file_info.get('name', 'Unknown file')
                file_url = file_info.get('url', 'No URL')
                file_size = file_info.get('size', 'Unknown')

                # Determine file type based on extension
                if file_name.lower().endswith(('.md', '.markdown')):
                    file_type = "Report"
                elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp')):
                    file_type = "Image"
                else:
                    file_type = "Generated File"

                # Create hyperlink URL display (show truncated text, but link to full URL)
                if file_url != 'No URL':
                    display_url = file_url[:60] + "..." if len(file_url) > 60 else file_url
                    url_text = Text(display_url, style="blue")
                    url_text.stylize(f"link {file_url}")
                else:
                    url_text = Text("No URL", style="blue")

                # Determine file size display
                size_display = str(file_size) if file_size != 'Unknown' else "Unknown"

                workspace_table.add_row(
                    file_name,
                    url_text,
                    file_type,
                    size_display,
                    "📋 Generated"
                )

            console.print(workspace_table)

        # Display explanation information
        if workspace_files:
            console.print()
            console.print("[dim]📋 Generated files: AI-created reports, images, and data files, automatically accessible via thread workspace[/dim]")

  
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

            # Only use user uploaded file IDs
            all_file_ids = [f['id'] for f in self.uploaded_files]

            # Add current user message with thread_id and file_ids
            current_message = {"role": "user", "content": message}
            if all_file_ids:
                current_message["file_ids"] = all_file_ids
            if self.current_thread_id:
                current_message["thread_id"] = self.current_thread_id
            messages.append(current_message)

            console.print("[cyan]💭 Analyzing...[/cyan]")
            if all_file_ids:
                console.print(f"[dim]Using {len(all_file_ids)} uploaded files[/dim]")
            if self.current_thread_id:
                console.print(f"[dim]Using thread: {self.current_thread_id}[/dim]")

            # Default to streaming response
            console.print("[dim]📡 Streaming response...[/dim]")
            response_text = ""
            collected_files = []
            response_thread_id = None

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

                    # Extract thread_id and collect generated files
                    if hasattr(chunk.choices[0].delta, 'thread_id') and chunk.choices[0].delta.thread_id:
                        response_thread_id = chunk.choices[0].delta.thread_id

                if hasattr(chunk, 'generated_files') and chunk.generated_files:
                    collected_files.extend(chunk.generated_files)

            console.print()  # Newline

            # Update thread_id from response
            if response_thread_id and response_thread_id != self.current_thread_id:
                self.current_thread_id = response_thread_id
                console.print(f"[dim]🧵 Thread updated: {self.current_thread_id}[/dim]")

            # Add AI response to history
            self.chat_history.append({"role": "assistant", "content": response_text})

            # Handle generated files
            if collected_files:
                console.print(f"\n[green]📁 Generated {len(collected_files)} files[/green]")

                for file_info in collected_files:
                    file_name = file_info.get('name', 'Unknown file')
                    file_url = file_info.get('url', '')

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

                    # Save to generated_files list
                    self.generated_files.append({
                        **file_info,
                        'size': file_size
                    })
                    console.print(f"[dim]• {file_name}: {file_url}[/dim]")

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

        generated_files_count = len(self.generated_files)

        console.print(Panel(
            f"[bold]Conversation rounds:[/bold] {len(self.chat_history) // 2}\n"
            f"[bold]User messages:[/bold] {len([m for m in self.chat_history if m['role'] == 'user'])}\n"
            f"[bold]AI responses:[/bold] {len([m for m in self.chat_history if m['role'] == 'assistant'])}\n"
            f"[bold]Generated files:[/bold] {generated_files_count}\n"
            f"[bold]Thread ID:[/bold] {self.current_thread_id or 'None'}",
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
        """Clear conversation history and reset thread"""
        # Clear local lists
        self.chat_history.clear()
        self.generated_files.clear()
        self.current_thread_id = None

        console.print("[green]✅ Conversation history cleared[/green]")
        console.print("[green]✅ Generated file records cleared[/green]")
        console.print("[green]✅ Thread reset[/green]")

    def clear_all(self):
        """Clear all content (including uploaded files)"""
        try:
            # Delete server files - uploaded files
            if self.uploaded_files:
                for file_info in self.uploaded_files:
                    try:
                        self.client.files.delete(file_info['id'])
                        console.print(f"[green]✅ Deleted uploaded file: {file_info['name']}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ Failed to delete uploaded file {file_info['name']}: {e}[/red]")

            # Clear local lists
            self.chat_history.clear()
            self.generated_files.clear()
            self.uploaded_files.clear()
            self.current_thread_id = None

            console.print("[green]✅ All content cleared[/green]")
            console.print("[green]✅ Conversation history, generated files, uploaded files all cleared[/green]")
            console.print("[green]✅ Thread reset[/green]")

        except Exception as e:
            console.print(f"[red]❌ Error clearing all content: {e}[/red]")

    def get_system_status(self):
        """Get system status"""
        try:
            console.print("[cyan]🔍 Getting system status...[/cyan]")

            # Server status
            server_status = "✅ Online" if self.check_server() else "❌ Offline"

            # Statistics
            generated_files_count = len(self.generated_files)
            status_panel = Panel(
                f"[bold]API Server:[/bold] {server_status}\n"
                f"[bold]API Endpoint:[/bold] {self.api_base}\n"
                f"[bold]Current Model:[/bold] {self.model}\n"
                f"[bold]Thread ID:[/bold] {self.current_thread_id or 'None'}\n"
                f"[bold]Uploaded Files:[/bold] {len(self.uploaded_files)}\n"
                f"[bold]Generated Files:[/bold] {generated_files_count}\n"
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
• [yellow]clear[/yellow] - Clear conversation history and reset thread
• [yellow]clear-all[/yellow] - Clear all content (including uploaded files)

[file management]
• [yellow]files[/yellow] - View uploaded files and generated workspace files
• [yellow]upload <file_path>[/yellow] - Upload new file
• [yellow]delete <file_id>[/yellow] - Delete specified uploaded file
• [yellow]download <file_id> [save_path][/yellow] - Download uploaded file

[system & history]
• [yellow]status[/yellow] - Display system status and thread information
• [yellow]history[/yellow] - Display conversation history
• [yellow]fid[/yellow] - Display uploaded file names and complete IDs

[dim]Files are automatically managed via thread workspace. Generated files persist across conversations.[/dim]
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
        """Display user uploaded file names and complete IDs"""
        # Check if there are any uploaded files
        if not self.uploaded_files:
            console.print("[yellow]📝 No uploaded files[/yellow]")
            return

        # Create table for uploaded files
        table = Table(title="User Uploaded Files and IDs", show_header=True, header_style="bold magenta")
        table.add_column("File Name", style="cyan", no_wrap=True)
        table.add_column("Complete File ID", style="green")
        table.add_column("File Size", style="yellow")
        table.add_column("Status", style="blue")

        # Display uploaded files
        for file_info in self.uploaded_files:
            table.add_row(
                file_info['name'],
                file_info['id'],  # Complete ID
                decimal(file_info['size']),
                "✅ Uploaded"
            )

        console.print(table)

        # Display summary
        console.print()
        console.print(f"[dim]Total uploaded files: {len(self.uploaded_files)}[/dim]")
        console.print(f"[dim]Generated files are accessible via workspace (thread_id: {self.current_thread_id or 'None'})[/dim]")

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