#!/usr/bin/env python3
"""
DeepAnalyze API CLI - 轻量美观的命令行交互程序
基于rich包实现的API客户端，支持文件上传和数据分析任务
"""

import os
import sys
import json
import time
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
        self.chat_history = []  # 对话历史
        self.generated_files = []  # 生成的文件（包含报告、图片等）
        self.intermediate_files = []  # 中间文件（已上传的生成文件，用于对话）

    def initialize_client(self):
        """初始化OpenAI客户端"""
        try:
            self.client = openai.OpenAI(
                api_key="dummy",  # DeepAnalyze API使用dummy key
                base_url=self.api_base
            )
            return True
        except Exception as e:
            console.print(f"[red]❌ 初始化客户端失败: {e}[/red]")
            return False

    def check_server(self) -> bool:
        """检查API服务器是否运行"""
        try:
            import requests
            # 首先尝试检查health端点
            response = requests.get(f"http://localhost:8200/health", timeout=5)
            if response.status_code == 200:
                return True

            # 如果health端点不可用，尝试检查模型列表
            temp_client = openai.OpenAI(api_key="dummy", base_url=self.api_base)
            models = temp_client.models.list()
            return True
        except:
            return False

    def display_header(self):
        """显示程序头部信息"""
        header_content = """[bold cyan]🚀 DeepAnalyze API 客户端[/bold cyan]

[green]功能特性:[/green]
• 📁 文件上传与管理
• 💬 智能对话分析
• 📊 数据分析任务
• 🎨 美观的命令行界面
• 📝 实时响应显示

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

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:

                task = progress.add_task(f"[cyan]上传 {full_path.name}...", total=100)

                # 模拟上传进度
                for i in range(0, 101, 10):
                    time.sleep(0.05)
                    progress.update(task, completed=i)

                # 使用OpenAI库上传文件
                with open(full_path, 'rb') as f:
                    file_obj = self.client.files.create(
                        file=f,
                        purpose="assistants"
                    )

                progress.update(task, completed=100)

            self.uploaded_files.append({
                'id': file_obj.id,
                'name': file_obj.filename,
                'path': str(full_path),
                'size': file_size,
                'purpose': file_obj.purpose
            })

            console.print(f"[green]✅ 文件上传成功![/green]")
            console.print(f"[dim]文件ID: {file_obj.id}[/dim]")
            console.print(f"[dim]文件名: {file_obj.filename}[/dim]")
            console.print(f"[dim]文件大小: {decimal(file_size)}[/dim]")
            console.print(f"[dim]用途: {file_obj.purpose}[/dim]")
            return file_obj.id

        except Exception as e:
            console.print(f"[red]❌ 上传错误: {e}[/red]")
            return None

    def list_uploaded_files(self):
        """显示已上传的文件列表"""
        if not self.uploaded_files:
            console.print("[yellow]📝 暂无已上传的文件[/yellow]")
            return

        table = Table(title="已上传文件", show_header=True, header_style="bold magenta")
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

    def is_intermediate_file(self, file_info: Dict[str, Any]) -> bool:
        """判断文件是否应该作为中间文件上传（排除报告和图片）"""
        file_name = file_info.get('name', '').lower()

        # 排除报告文件和图片文件
        intermediate_extensions = ['.md', '.markdown', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp']

        # 检查文件扩展名
        for ext in intermediate_extensions:
            if file_name.endswith(ext):
                return False

        # 其他文件都作为中间文件
        return True

    def upload_intermediate_file(self, file_info: Dict[str, Any]) -> Optional[str]:
        """上传中间文件并返回file_id"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return None

            file_name = file_info.get('name', 'unknown_file')
            file_url = file_info.get('url', '')

            console.print(f"[dim]📤 上传中间文件: {file_name}[/dim]")

            # 尝试从URL下载文件内容并上传
            import requests
            import tempfile
            import os

            # 下载文件
            response = requests.get(file_url)
            if response.status_code == 200:
                # 创建临时文件
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_file:
                    temp_file.write(response.content)
                    temp_file_path = temp_file.name

                try:
                    # 上传到API
                    with open(temp_file_path, 'rb') as f:
                        file_obj = self.client.files.create(
                            file=f,
                            purpose="assistants"
                        )

                    # 保存到中间文件列表
                    self.intermediate_files.append({
                        'id': file_obj.id,
                        'name': file_name,
                        'original_url': file_url,
                        'purpose': file_obj.purpose
                    })

                    console.print(f"[dim]✅ 中间文件上传成功: {file_name} -> {file_obj.id}[/dim]")
                    return file_obj.id

                finally:
                    # 删除临时文件
                    os.unlink(temp_file_path)
            else:
                console.print(f"[red]❌ 下载中间文件失败: {file_name}[/red]")
                return None

        except Exception as e:
            console.print(f"[red]❌ 上传中间文件失败 {file_name}: {e}[/red]")
            return None

    def chat_with_file(self, message: str, file_ids: List[str] = None, stream: bool = True):
        """与AI进行对话分析"""
        try:
            if not self.client:
                if not self.initialize_client():
                    return

            # 添加用户消息到历史
            self.chat_history.append({"role": "user", "content": message})
            if file_ids:
                self.chat_history[-1]["file_ids"] = file_ids

            # 构建消息列表，包含历史对话
            messages = []

            # 添加历史对话（排除file_ids）
            for msg in self.chat_history[:-1]:  # 排除刚添加的用户消息
                if msg["role"] == "user":
                    messages.append({"role": "user", "content": msg["content"]})
                elif msg["role"] == "assistant":
                    messages.append({"role": "assistant", "content": msg["content"]})

            # 获取所有文件ID：上传的文件 + 中间文件
            all_file_ids = []

            # 添加上传的文件ID
            uploaded_file_ids = [f['id'] for f in self.uploaded_files]
            all_file_ids.extend(uploaded_file_ids)

            # 添加中间文件ID（已上传的生成文件）
            intermediate_file_ids = [f['id'] for f in self.intermediate_files]
            all_file_ids.extend(intermediate_file_ids)

            # 去重
            all_file_ids = list(set(all_file_ids))

            # 添加当前用户消息（只有这条消息包含file_ids）
            current_message = {"role": "user", "content": message}
            if all_file_ids:
                current_message["file_ids"] = all_file_ids
            messages.append(current_message)

            console.print("[cyan]💭 正在分析...[/cyan]")
            if all_file_ids:
                console.print(f"[dim]使用文件: {len(uploaded_file_ids)} 个上传文件, {len(intermediate_file_ids)} 个中间文件[/dim]")

            # 默认使用流式响应
            console.print("[dim]📡 流式响应中...[/dim]")
            response_text = ""
            collected_files = []

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

                    # 收集生成的文件（在流中）
                    if hasattr(chunk, 'generated_files') and chunk.generated_files:
                        collected_files.extend(chunk.generated_files)

            console.print()  # 换行

            # 添加AI回复到历史
            self.chat_history.append({"role": "assistant", "content": response_text})

            # 处理生成的文件
            if collected_files:
                console.print(f"\n[green]📁 生成了 {len(collected_files)} 个文件[/green]")

                intermediate_count = 0
                for file_info in collected_files:
                    file_name = file_info.get('name', '未知文件')
                    file_url = file_info.get('url', '')
                    file_id = file_info.get('id', '')

                    # 判断是否为中间文件
                    if self.is_intermediate_file(file_info):
                        # 上传中间文件
                        uploaded_id = self.upload_intermediate_file(file_info)
                        if uploaded_id:
                            intermediate_count += 1
                        # 仍然保存到generated_files用于统计
                        self.generated_files.append({
                            **file_info,
                            'uploaded_id': uploaded_id,
                            'type': 'intermediate'
                        })
                    else:
                        # 报告和图片文件，直接保存
                        self.generated_files.append({
                            **file_info,
                            'type': 'output'
                        })
                        console.print(f"[dim]• {file_name}: {file_url or file_id}[/dim]")

                if intermediate_count > 0:
                    console.print(f"[dim]✅ {intermediate_count} 个文件已作为中间文件上传，可用于后续对话[/dim]")

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

            console.print(f"[yellow]🗑️  删除文件: {file_id}[/yellow]")
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

            console.print(f"[cyan]📥 下载文件: {file_id}[/cyan]")
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

        output_files = [f for f in self.generated_files if f.get('type') != 'intermediate']
        intermediate_files = [f for f in self.generated_files if f.get('type') == 'intermediate']

        console.print(Panel(
            f"[bold]对话轮次:[/bold] {len(self.chat_history) // 2}\n"
            f"[bold]用户消息:[/bold] {len([m for m in self.chat_history if m['role'] == 'user'])}\n"
            f"[bold]AI回复:[/bold] {len([m for m in self.chat_history if m['role'] == 'assistant'])}\n"
            f"[bold]输出文件:[/bold] {len(output_files)}\n"
            f"[bold]中间文件:[/bold] {len(intermediate_files)}",
            title="对话历史统计",
            border_style="blue"
        ))

        # 显示最近几条对话
        console.print("\n[bold]最近对话记录:[/bold]")
        recent_messages = self.chat_history[-6:]  # 显示最近6条消息

        for i, msg in enumerate(recent_messages):
            role_emoji = "👤" if msg['role'] == 'user' else "🤖"
            role_color = "blue" if msg['role'] == 'user' else "green"

            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            console.print(f"[{role_color}]{role_emoji} {msg['role'].title()}:[/{role_color}] {content}")

            if i < len(recent_messages) - 1:
                console.print()

    def show_generated_files(self):
        """显示生成的输出文件列表（不包括中间文件）"""
        output_files = [f for f in self.generated_files if f.get('type') != 'intermediate']

        if not output_files:
            console.print("[yellow]📁 暂无生成的输出文件[/yellow]")
            return

        table = Table(title="生成的输出文件", show_header=True, header_style="bold magenta")
        table.add_column("文件名", style="cyan", no_wrap=True)
        table.add_column("URL/路径", style="green")
        table.add_column("大小", style="yellow")
        table.add_column("类型", style="blue")

        for file_info in output_files:
            file_name = file_info.get('name', '未知文件')
            file_url = file_info.get('url', '无URL')
            file_size = file_info.get('size', '未知')

            table.add_row(
                file_name,
                file_url[:50] + "..." if len(file_url) > 50 else file_url,
                str(file_size),
                "📄 报告" if file_name.lower().endswith(('.md', '.markdown')) else "🖼️ 图片"
            )

        console.print(table)

        # 显示统计信息
        intermediate_count = len([f for f in self.generated_files if f.get('type') == 'intermediate'])
        if intermediate_count > 0:
            console.print(f"[dim]💡 另有 {intermediate_count} 个中间文件用于对话处理，不在此显示[/dim]")

    def clear_chat_history(self):
        """清空对话历史和生成的中间文件"""
        # 删除中间文件
        if self.intermediate_files:
            console.print("[yellow]🗑️  正在删除中间文件...[/yellow]")
            for file_info in self.intermediate_files:
                try:
                    self.client.files.delete(file_info['id'])
                    console.print(f"[green]✅ 已删除中间文件: {file_info['name']}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ 删除中间文件失败 {file_info['name']}: {e}[/red]")

        # 清空本地列表
        self.chat_history.clear()
        self.generated_files.clear()
        self.intermediate_files.clear()

        console.print("[green]✅ 对话历史已清空[/green]")
        console.print("[green]✅ 生成的文件记录已清空[/green]")
        console.print("[green]✅ 中间文件已删除[/green]")

    def clear_all(self):
        """清空所有内容（包括上传的文件）"""
        try:
            # 删除服务器上的文件 - 包括上传文件和中间文件
            if self.uploaded_files:
                for file_info in self.uploaded_files:
                    try:
                        self.client.files.delete(file_info['id'])
                        console.print(f"[green]✅ 已删除上传文件: {file_info['name']}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ 删除上传文件失败 {file_info['name']}: {e}[/red]")

            if self.intermediate_files:
                for file_info in self.intermediate_files:
                    try:
                        self.client.files.delete(file_info['id'])
                        console.print(f"[green]✅ 已删除中间文件: {file_info['name']}[/green]")
                    except Exception as e:
                        console.print(f"[red]❌ 删除中间文件失败 {file_info['name']}: {e}[/red]")

            # 清空本地列表
            self.chat_history.clear()
            self.generated_files.clear()
            self.intermediate_files.clear()
            self.uploaded_files.clear()

            console.print("[green]✅ 所有内容已清空[/green]")
            console.print("[green]✅ 对话历史、生成文件、上传文件、中间文件均已清空[/green]")

        except Exception as e:
            console.print(f"[red]❌ 清空所有内容时出错: {e}[/red]")

    def get_system_status(self):
        """获取系统状态"""
        try:
            console.print("[cyan]🔍 获取系统状态...[/cyan]")

            # 服务器状态
            server_status = "✅ 在线" if self.check_server() else "❌ 离线"

            # 统计信息
            output_files = [f for f in self.generated_files if f.get('type') == 'output']
            status_panel = Panel(
                f"[bold]API服务器:[/bold] {server_status}\n"
                f"[bold]API端点:[/bold] {self.api_base}\n"
                f"[bold]当前模型:[/bold] {self.model}\n"
                f"[bold]上传文件:[/bold] {len(self.uploaded_files)}\n"
                f"[bold]中间文件:[/bold] {len(self.intermediate_files)}\n"
                f"[bold]输出文件:[/bold] {len(output_files)}\n"
                f"[bold]对话轮次:[/bold] {len([m for m in self.chat_history if m['role'] == 'user'])}",
                title="系统状态",
                border_style="cyan"
            )
            console.print(status_panel)

        except Exception as e:
            console.print(f"[red]❌ 获取系统状态失败: {e}[/red]")

    def interactive_mode(self):
        """交互式对话模式"""
        console.print("\n[bold green]💬 进入交互对话模式[/bold green]")

        # 显示帮助信息
        self.show_help()

        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]您[/bold blue]", default="").strip()

                if user_input.lower() in ['quit', 'exit', '退出']:
                    console.print("[green]👋 再见![/green]")
                    break

                # 处理各种命令
                if self.handle_command(user_input):
                    continue

                if not user_input:
                    continue

                # 获取当前已上传文件的ID
                file_ids = [f['id'] for f in self.uploaded_files]

                # 执行对话（默认流式输出）
                self.chat_with_file(user_input, file_ids if file_ids else None, stream=True)

            except KeyboardInterrupt:
                console.print("\n[green]👋 再见![/green]")
                break
            except Exception as e:
                console.print(f"[red]❌ 错误: {e}[/red]")

    def show_help(self):
        """显示帮助信息"""
        help_text = """
[bold cyan]📋 可用命令列表:[/bold cyan]

[basic commands]
• [yellow]help[/yellow] - 显示此帮助信息
• [yellow]quit/exit[/yellow] - 退出程序
• [yellow]clear-history[/yellow] - 清空对话历史和生成的中间文件
• [yellow]clear-all[/yellow] - 清空所有内容（包括上传的文件）

[file management]
• [yellow]files[/yellow] - 查看已上传文件
• [yellow]upload <文件路径>[/yellow] - 上传新文件
• [yellow]delete <文件ID>[/yellow] - 删除指定文件
• [yellow]download <文件ID> [保存路径][/yellow] - 下载文件
• [yellow]generated-files[/yellow] - 查看生成的中间文件

[system & history]
• [yellow]status[/yellow] - 显示系统状态
• [yellow]history[/yellow] - 显示对话历史

[dim]直接输入文本即可开始对话，系统会自动使用已上传文件和生成文件[/dim]
"""
        console.print(Panel(help_text, title="命令帮助", border_style="blue"))

    def handle_command(self, user_input: str) -> bool:
        """处理命令，返回True表示是命令"""
        cmd = user_input.lower().strip()

        # 帮助命令
        if cmd in ['help', '帮助', 'h']:
            self.show_help()
            return True

        # 清空对话历史
        elif cmd in ['clear-history', 'clear', '清空历史']:
            if Confirm.ask("确定要清空对话历史和生成的中间文件吗?"):
                self.clear_chat_history()
            return True

        # 清空所有内容
        elif cmd in ['clear-all', '清空所有']:
            if Confirm.ask("确定要清空所有内容吗? 这将删除所有上传的文件"):
                self.clear_all()
            return True

        # 文件管理命令
        elif cmd in ['files', '文件', 'ls']:
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

        elif cmd in ['generated-files', 'gen-files', '生成文件']:
            self.show_generated_files()
            return True

        # 系统命令
        elif cmd in ['status', '系统状态']:
            self.get_system_status()
            return True

        # 历史命令
        elif cmd in ['history', '历史']:
            self.show_history()
            return True

        # 不是命令
        return False

    
    def cleanup_files(self):
        """清理已上传的文件"""
        if not self.uploaded_files:
            return

        if not self.client:
            self.initialize_client()

        console.print("[yellow]🧹 清理已上传的文件...[/yellow]")

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
            console.print("\n[green]👋 程序已终止[/green]")
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