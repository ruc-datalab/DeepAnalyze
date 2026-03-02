#!/usr/bin/env python3
"""
DeepAnalyze API CLI - 轻量级美观的命令行界面 (中文版)
使用 rich 包实现的 API 客户端，支持文件上传和数据分析任务
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
        """初始化CLI客户端"""
        self.api_base = "http://localhost:8200/v1"
        self.model = "DeepAnalyze-8B"
        self.client = None
        self.uploaded_files = []
        self.current_thread_id = None
        self.chat_history = []  # 聊天历史
        self.generated_files = []  # 生成的文件（包括报告、图片等）
        self.setup_command_history()

    def setup_command_history(self):
        """设置命令历史功能"""
        history_file = os.path.expanduser("~/.deeppanalyze_history_zh")

        # 尝试读取历史文件
        try:
            if os.path.exists(history_file):
                readline.read_history_file(history_file)
            # 设置历史文件长度限制
            readline.set_history_length(1000)
            # 每次命令后自动保存历史
            readline.set_auto_history(True)
        except Exception as e:
            # 静默处理失败
            pass

        self.history_file = history_file

    def save_history(self):
        """保存历史到文件"""
        try:
            if hasattr(self, 'history_file'):
                readline.write_history_file(self.history_file)
        except Exception as e:
            # 静默处理失败
            pass

    def initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            self.client = openai.OpenAI(
                api_key="dummy",  # DeepAnalyze API 使用虚拟密钥
                base_url=self.api_base
            )
            return True
        except Exception as e:
            console.print(f"[red]❌ 客户端初始化失败: {e}[/red]")
            return False

    def check_server(self) -> bool:
        """检查API服务器是否运行"""
        try:
            import requests
            # 首先尝试检查健康端点
            response = requests.get(f"http://localhost:8200/health", timeout=5)
            if response.status_code == 200:
                return True

            # 如果健康端点不可用，尝试检查模型列表
            temp_client = openai.OpenAI(api_key="dummy", base_url=self.api_base)
            models = temp_client.models.list()
            return True
        except Exception:
            return False

    def display_header(self):
        """显示程序头部信息"""
        header_content = """[bold cyan]🚀 DeepAnalyze API 客户端[/bold cyan]
[dim]API服务器: http://localhost:8200 | 模型: DeepAnalyze-8B[/dim]"""

        console.print(Panel(header_content, title="DeepAnalyze CLI", border_style="cyan"))

    def upload_file(self, file_path: str) -> Optional[str]:
        """上传文件到API服务器"""
        try:
            full_path = Path(file_path).expanduser().resolve()
            if not full_path.exists():
                console.print(f"[red]❌ 文件不存在: {file_path}[/red]")
                return None

            if not self.client:
                if not self.initialize_client():
                    return None

            file_size = full_path.stat().st_size

            # 安全处理文件名以避免编码错误
            safe_filename = full_path.name
            try:
                # 确保文件名可以安全编码
                safe_filename.encode('utf-8')
            except UnicodeEncodeError:
                # 如果文件名包含无效字符，使用安全文件名
                safe_filename = f"file_{int(time.time())}{full_path.suffix}"

            # 显示上传开始消息
            console.print(f"[cyan]📤 正在上传 {safe_filename}...[/cyan]")

            # 使用OpenAI库上传文件
            with open(full_path, 'rb') as f:
                file_obj = self.client.files.create(
                    file=f,
                    purpose="assistants"
                )

            # 安全处理返回的文件名
            safe_response_filename = file_obj.filename
            try:
                safe_response_filename.encode('utf-8')
            except UnicodeEncodeError:
                safe_response_filename = safe_filename  # 使用我们的安全文件名

            self.uploaded_files.append({
                'id': file_obj.id,
                'name': safe_response_filename,
                'path': str(full_path),
                'size': file_size,
                'purpose': file_obj.purpose
            })

            console.print("[green]✅ 文件上传成功![/green]")
            console.print(f"[dim]文件ID: {file_obj.id}[/dim]")
            console.print(f"[dim]文件名: {safe_response_filename}[/dim]")
            console.print(f"[dim]文件大小: {decimal(file_size)}[/dim]")
            console.print(f"[dim]用途: {file_obj.purpose}[/dim]")
            return file_obj.id

        except Exception as e:
            console.print(f"[red]❌ 上传错误: {e}[/red]")
            return None

    def list_uploaded_files(self):
        """显示所有文件列表（用户上传文件和AI生成的工作区文件）"""
        # 获取AI生成的工作区文件
        workspace_files = self.generated_files

        # 检查是否有任何文件
        if not self.uploaded_files and not workspace_files:
            console.print("[yellow]📝 暂无文件[/yellow]")
            return

        # 显示用户上传文件
        if self.uploaded_files:
            table = Table(title="用户上传文件", show_header=True, header_style="bold magenta")
            table.add_column("文件名", style="cyan", no_wrap=True)
            table.add_column("文件ID", style="green")
            table.add_column("文件大小", style="yellow")
            table.add_column("用途", style="blue")
            table.add_column("状态", style="green")

            for file_info in self.uploaded_files:
                table.add_row(
                    file_info['name'],
                    file_info['id'][:8] + "...",
                    decimal(file_info['size']),
                    file_info.get('purpose', 'assistants'),
                    "✅ 已上传"
                )

            console.print(table)

        # 显示AI生成的工作区文件
        if workspace_files:
            if self.uploaded_files:
                console.print()  # 添加空行分隔符

            workspace_table = Table(title="AI生成的工作区文件", show_header=True, header_style="bold green")
            workspace_table.add_column("文件名", style="cyan", no_wrap=True)
            workspace_table.add_column("URL", style="blue")
            workspace_table.add_column("文件类型", style="yellow")
            workspace_table.add_column("大小", style="magenta")
            workspace_table.add_column("状态", style="bright_blue")

            for file_info in workspace_files:
                file_name = file_info.get('name', '未知文件')
                file_url = file_info.get('url', '无URL')
                file_size = file_info.get('size', '未知')

                # 根据扩展名确定文件类型
                if file_name.lower().endswith(('.md', '.markdown')):
                    file_type = "报告"
                elif file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp')):
                    file_type = "图片"
                else:
                    file_type = "生成文件"

                # 创建超链接URL显示（显示截断文本，但链接到完整URL）
                if file_url != '无URL':
                    display_url = file_url[:60] + "..." if len(file_url) > 60 else file_url
                    url_text = Text(display_url, style="blue")
                    url_text.stylize(f"link {file_url}")
                else:
                    url_text = Text("无URL", style="blue")

                # 确定文件大小显示
                size_display = str(file_size) if file_size != '未知' else "未知"

                workspace_table.add_row(
                    file_name,
                    url_text,
                    file_type,
                    size_display,
                    "📋 已生成"
                )

            console.print(workspace_table)

        # 显示说明信息
        if workspace_files:
            console.print()
            console.print("[dim]📋 生成文件: AI创建的报告、图片和数据文件，通过线程工作区自动访问[/dim]")

    
    def chat_with_file(self, message: str, file_ids: List[str] = None, stream: bool = True):
        """与AI聊天进行分析"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return

            # 添加用户消息到历史
            self.chat_history.append({"role": "user", "content": message})
            if file_ids:
                self.chat_history[-1]["file_ids"] = file_ids

            # 构建消息列表，包括历史对话
            messages = []

            # 添加历史对话（排除file_ids）
            for msg in self.chat_history[:-1]:  # 排除新添加的用户消息
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})

            # 只使用用户上传的文件ID
            all_file_ids = [f['id'] for f in self.uploaded_files]

            # 添加当前用户消息，包含thread_id和file_ids
            current_message = {"role": "user", "content": message}
            if all_file_ids:
                current_message["file_ids"] = all_file_ids
            if self.current_thread_id:
                current_message["thread_id"] = self.current_thread_id
            messages.append(current_message)

            console.print("[cyan]💭 正在分析...[/cyan]")
            if all_file_ids:
                console.print(f"[dim]使用 {len(all_file_ids)} 个上传文件[/dim]")
            if self.current_thread_id:
                console.print(f"[dim]使用线程: {self.current_thread_id}[/dim]")

            # 默认流式响应
            console.print("[dim]📡 流式响应中...[/dim]")
            response_text = ""
            collected_files = []
            response_thread_id = None

            console.print("\n[bold yellow]🤖 AI回复:[/bold yellow]")

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

                    # 提取thread_id并收集生成的文件
                    if hasattr(chunk.choices[0].delta, 'thread_id') and chunk.choices[0].delta.thread_id:
                        response_thread_id = chunk.choices[0].delta.thread_id

                if hasattr(chunk, 'generated_files') and chunk.generated_files:
                    collected_files.extend(chunk.generated_files)

            console.print()  # 换行

            # 从响应更新thread_id
            if response_thread_id and response_thread_id != self.current_thread_id:
                self.current_thread_id = response_thread_id
                console.print(f"[dim]🧵 线程已更新: {self.current_thread_id}[/dim]")

            # 添加AI回复到历史
            self.chat_history.append({"role": "assistant", "content": response_text})

            # 处理生成的文件
            if collected_files:
                console.print(f"\n[green]📁 生成了 {len(collected_files)} 个文件[/green]")

                for file_info in collected_files:
                    file_name = file_info.get('name', '未知文件')
                    file_url = file_info.get('url', '')

                    # 尝试从URL获取文件大小
                    file_size = file_info.get('size', '未知')
                    if file_size == '未知' and file_url:
                        try:
                            import requests
                            response = requests.head(file_url, timeout=5)
                            if response.status_code == 200 and 'content-length' in response.headers:
                                size_bytes = int(response.headers['content-length'])
                                file_size = decimal(size_bytes)
                            else:
                                # 如果HEAD请求失败，尝试完整下载
                                response = requests.get(file_url, timeout=10)
                                if response.status_code == 200:
                                    size_bytes = len(response.content)
                                    file_size = decimal(size_bytes)
                        except Exception:
                            # 如果获取失败，保持为'未知'
                            pass

                    # 保存到generated_files列表
                    self.generated_files.append({
                        **file_info,
                        'size': file_size
                    })
                    console.print(f"[dim]• {file_name}: {file_url}[/dim]")

            return response_text

        except Exception as e:
            console.print(f"[red]❌ 对话错误: {e}[/red]")
            return None


    def delete_file_by_id(self, file_id: str):
        """根据ID删除文件"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return False

            console.print(f"[yellow]🗑️  正在删除文件: {file_id}[/yellow]")
            self.client.files.delete(file_id)

            # 从本地列表中移除
            self.uploaded_files = [f for f in self.uploaded_files if f['id'] != file_id]
            console.print(f"[green]✅ 文件删除成功[/green]")
            return True

        except Exception as e:
            console.print(f"[red]❌ 删除文件失败: {e}[/red]")
            return False

    def download_file_by_id(self, file_id: str, save_path: str = None):
        """根据ID下载文件"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return

            console.print(f"[cyan]📥 正在下载文件: {file_id}[/cyan]")
            file_content = self.client.files.content(file_id)

            # 确定保存路径
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

            # 写入文件
            with open(save_path, 'wb') as f:
                f.write(file_content.content)

            console.print(f"[green]✅ 文件下载成功: {save_path}[/green]")
            console.print(f"[dim]文件大小: {decimal(len(file_content.content))}[/dim]")

        except Exception as e:
            console.print(f"[red]❌ 下载文件失败: {e}[/red]")

    def show_history(self):
        """显示对话历史"""
        if not self.chat_history:
            console.print("[yellow]📝 暂无对话历史[/yellow]")
            return

        generated_files_count = len(self.generated_files)

        console.print(Panel(
            f"[bold]对话轮数:[/bold] {len(self.chat_history) // 2}\n"
            f"[bold]用户消息:[/bold] {len([m for m in self.chat_history if m['role'] == 'user'])}\n"
            f"[bold]AI回复:[/bold] {len([m for m in self.chat_history if m['role'] == 'assistant'])}\n"
            f"[bold]生成文件:[/bold] {generated_files_count}\n"
            f"[bold]线程ID:[/bold] {self.current_thread_id or '无'}",
            title="对话历史统计",
            border_style="blue"
        ))

        # 显示最近对话
        console.print("\n[bold]最近对话记录:[/bold]")
        recent_messages = self.chat_history[-6:]  # 显示最近6条消息

        for i, msg in enumerate(recent_messages):
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            role_color = "blue" if msg['role'] == 'user' else "green"

            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            console.print(f"[{role_color}]{role_emoji} {msg['role'].title()}:[/{role_color}] {content}")

            if i < len(recent_messages) - 1:
                console.print()


    def clear_chat_history(self):
        """清除对话历史并重置线程"""
        # 清空本地列表
        self.chat_history.clear()
        self.generated_files.clear()
        self.current_thread_id = None

        console.print("[green]✅ 对话历史已清除[/green]")
        console.print("[green]✅ 生成文件记录已清除[/green]")
        console.print("[green]✅ 线程已重置[/green]")

    def clear_all(self):
        """清除所有内容（包括上传文件）"""
        try:
            # 删除服务器文件 - 上传文件
            if self.uploaded_files:
                for file_info in self.uploaded_files:
                    try:
                        self.client.files.delete(file_info['id'])
                        console.print(f"[green]✅ 已删除上传文件: {file_info['name']}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ 删除上传文件失败 {file_info['name']}: {e}[/red]")

            # 清空本地列表
            self.chat_history.clear()
            self.generated_files.clear()
            self.uploaded_files.clear()
            self.current_thread_id = None

            console.print("[green]✅ 所有内容已清除[/green]")
            console.print("[green]✅ 对话历史、生成文件、上传文件全部清除[/green]")
            console.print("[green]✅ 线程已重置[/green]")

        except Exception as e:
            console.print(f"[red]❌ 清除所有内容时出错: {e}[/red]")

    def get_system_status(self):
        """获取系统状态"""
        try:
            console.print("[cyan]🔍 正在获取系统状态...[/cyan]")

            # 服务器状态
            server_status = "✅ 在线" if self.check_server() else "❌ 离线"

            # 统计
            generated_files_count = len(self.generated_files)
            status_panel = Panel(
                f"[bold]API服务器:[/bold] {server_status}\n"
                f"[bold]API端点:[/bold] {self.api_base}\n"
                f"[bold]当前模型:[/bold] {self.model}\n"
                f"[bold]线程ID:[/bold] {self.current_thread_id or '无'}\n"
                f"[bold]上传文件:[/bold] {len(self.uploaded_files)}\n"
                f"[bold]生成文件:[/bold] {generated_files_count}\n"
                f"[bold]对话轮数:[/bold] {len([m for m in self.chat_history if m['role'] == 'user'])}",
                title="系统状态",
                border_style="cyan"
            )
            console.print(status_panel)

        except Exception as e:
            console.print(f"[red]❌ 获取系统状态失败: {e}[/red]")

    def interactive_mode(self):
        """交互式对话模式"""
        console.print("\n[bold green]💬 进入交互式对话模式[/bold green]")

        # 显示帮助信息
        self.show_help()

        while True:
            try:
                # 使用简单的输入提示，避免使用终端控制序列
                user_input = input("您: ").strip()

                # 每次有效输入后保存历史
                if user_input:
                    self.save_history()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    console.print("[green]👋 再见![/green]")
                    break

                # 处理各种命令
                if self.handle_command(user_input):
                    continue

                if not user_input:
                    continue

                # 获取当前上传的文件ID
                file_ids = [f['id'] for f in self.uploaded_files]

                # 执行对话（默认流式输出）
                self.chat_with_file(user_input, file_ids if file_ids else None, stream=True)

            except KeyboardInterrupt:
                console.print("\n[green]👋 再见![/green]")
                break
            except EOFError:
                console.print("\n[green]👋 再见![/green]")
                break
            except Exception as e:
                console.print(f"[red]❌ 错误: {e}[/red]")

    def show_help(self):
        """显示帮助信息"""
        help_text = """
[bold cyan]📋 可用命令:[/bold cyan]

[基本命令]
• [yellow]help[/yellow] - 显示此帮助信息
• [yellow]quit/exit/退出[/yellow] - 退出程序
• [yellow]clear[/yellow] - 清除对话历史并重置线程
• [yellow]clear-all[/yellow] - 清除所有内容（包括上传文件）

[文件管理]
• [yellow]files[/yellow] - 查看上传文件和生成的工作区文件
• [yellow]upload <file_path>[/yellow] - 上传新文件
• [yellow]delete <file_id>[/yellow] - 删除指定的上传文件
• [yellow]download <file_id> [save_path][/yellow] - 下载上传文件

[系统 & 历史]
• [yellow]status[/yellow] - 显示系统状态和线程信息
• [yellow]history[/yellow] - 显示对话历史
• [yellow]fid[/yellow] - 显示上传文件名和完整ID

[dim]文件通过线程工作区自动管理。生成文件在对话间持久保存。[/dim]
"""
        console.print(Panel(help_text, title="命令帮助", border_style="blue"))

    def handle_command(self, user_input: str) -> bool:
        """处理命令，如果是命令返回True"""
        cmd = user_input.lower().strip()

        # 帮助命令
        if cmd in ['help', 'h', '帮助']:
            self.show_help()
            return True

        # 清除对话历史
        elif cmd in ['clear', '清除']:
            if Confirm.ask("确定要清除对话历史和生成的中间文件吗？"):
                self.clear_chat_history()
            return True

        # 清除所有内容
        elif cmd in ['clear-all', '全部清除']:
            if Confirm.ask("确定要清除所有内容吗？这将删除所有上传文件"):
                self.clear_all()
            return True

        # 文件管理命令
        elif cmd in ['files', 'ls', '文件']:
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

        # 系统命令
        elif cmd in ['status', '状态']:
            self.get_system_status()
            return True

        # 历史命令
        elif cmd in ['history', '历史']:
            self.show_history()
            return True

        # 文件ID命令
        elif cmd in ['fid']:
            self.show_file_ids()
            return True

        # 不是命令
        return False

    def show_file_ids(self):
        """显示用户上传的文件名和完整ID"""
        # 检查是否有上传文件
        if not self.uploaded_files:
            console.print("[yellow]📝 暂无上传文件[/yellow]")
            return

        # 创建上传文件表格
        table = Table(title="用户上传文件和ID", show_header=True, header_style="bold magenta")
        table.add_column("文件名", style="cyan", no_wrap=True)
        table.add_column("完整文件ID", style="green")
        table.add_column("文件大小", style="yellow")
        table.add_column("状态", style="blue")

        # 显示上传文件
        for file_info in self.uploaded_files:
            table.add_row(
                file_info['name'],
                file_info['id'],  # 完整ID
                decimal(file_info['size']),
                "✅ 已上传"
            )

        console.print(table)

        # 显示摘要
        console.print()
        console.print(f"[dim]上传文件总数: {len(self.uploaded_files)}[/dim]")
        console.print(f"[dim]生成文件通过工作区访问 (线程ID: {self.current_thread_id or '无'})[/dim]")

    def cleanup_files(self):
        """清理上传文件"""
        if not self.uploaded_files:
            return

        if not self.client:
            self.initialize_client()

        console.print("[yellow]🧹 正在清理上传文件...[/yellow]")

        for file_info in self.uploaded_files:
            try:
                # 使用OpenAI库删除文件
                self.client.files.delete(file_info['id'])
                console.print(f"[green]✅ 已删除: {file_info['name']}[/green]")
            except Exception as e:
                console.print(f"[red]❌ 删除错误 {file_info['name']}: {e}[/red]")

        # 清空本地列表
        self.uploaded_files.clear()

    def run(self):
        """运行主程序 - 直接进入交互模式"""
        try:
            # 检查服务器状态
            if not self.check_server():
                console.print("[red]❌ API服务器未运行![/red]")
                console.print("[yellow]请先启动API服务器: python backend/main.py[/yellow]")
                return

            self.display_header()
            console.print("[green]✅ API服务器连接成功[/green]")
            console.print(f"[dim]当前模型: {self.model}[/dim]")
            console.print(f"[dim]API端点: {self.api_base}[/dim]\n")

            # 直接进入交互模式
            self.interactive_mode()

        except KeyboardInterrupt:
            console.print("\n[green]👋 程序终止[/green]")
            self.cleanup_files()
        except Exception as e:
            console.print(f"[red]❌ 程序错误: {e}[/red]")
            self.cleanup_files()


def main():
    """主函数"""
    cli = DeepAnalyzeCLI()
    cli.run()


if __name__ == "__main__":
    main()