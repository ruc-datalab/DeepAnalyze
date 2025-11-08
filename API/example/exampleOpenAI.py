"""
Example usage of DeepAnalyze OpenAI-Compatible API with OpenAI library
Demonstrates assistant workflow with analyze tool
"""

import openai
import time
import re
from pathlib import Path

# Configure OpenAI client for DeepAnalyze
API_BASE = "http://localhost:8200/v1"
MODEL = "DeepAnalyze-8B"

client = openai.OpenAI(
    base_url=API_BASE,
    api_key="dummy"  # DeepAnalyze doesn't require a real API key
)


def extract_files_from_content(content):
    """从assistant回复内容中提取文件信息"""
    files_dict = {}
    file_patterns = [
        r'<File>\s*-?\s*\[([^\]]+)\]\(([^)]+)\)\s*</File>',  # 单个文件
        r'<File>(.*?)</File>',  # 整个File标签内容，然后提取其中的链接
    ]

    for pattern in file_patterns:
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            if pattern == file_patterns[1]:  # File标签内容的模式
                # 从File标签内容中提取所有链接
                link_pattern = r'-?\s*\[([^\]]+)\]\(([^)]+)\)'
                link_matches = re.findall(link_pattern, match)
                for filename, url in link_matches:
                    files_dict[filename] = url.strip()
            else:  # 单个文件模式
                filename, url = match
                files_dict[filename] = url.strip()

    return files_dict


def file_api_examples():
    """Demonstrate various file API operations"""
    try:
        # Create test file
        test_file_path = Path("test.txt")
        test_file_path.write_text("Test content")

        # Create files with different purposes
        file1 = client.files.create(file=test_file_path, purpose="file-extract")
        file2 = client.files.create(file=test_file_path, purpose="assistants")

        print(f"Created files: {file1.id} (extract), {file2.id} (assistants)")

        # List files
        files_list = client.files.list()
        print(f"Total files: {len(files_list.data)}")

        # Get content (file-extract purpose)
        if file1.purpose == "file-extract":
            content = client.files.content(file1.id)
            print(f"File content: {content.text}")

        # Cleanup
        client.files.delete(file1.id)
        client.files.delete(file2.id)
        test_file_path.unlink()
        print("✅ File API examples completed")

    except Exception as e:
        print(f"❌ Error: {e}")


def chat_completion_with_message_file_ids():
    """Chat completion with file_ids in messages (OpenAI compatibility)"""
    try:
        # Use existing Simpson.csv file
        with open("./Simpson.csv", "rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")

        # New format: file_ids in messages
        messages = [
            {
                "role": "user",
                "content": "分析数据，总结主要发现。",
                "file_ids": [file_obj.id]
            }
        ]

        response = client.chat.completions.create(model=MODEL, messages=messages)
        message = response.choices[0].message

        print(f"Response: {message.content[:100]}...")

        # Show files from both formats
        if hasattr(message, 'files') and message.files:
            print(f"Files (message): {len(message.files)}")
        if hasattr(response, 'generated_files') and response.generated_files:
            print(f"Files (response): {len(response.generated_files)}")

        # Backward compatibility example
        response2 = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "生成图表"}],
            file_ids=[file_obj.id]
        )
        print(f"Response 2: {response2.choices[0].message.content[:100]}...")

        client.files.delete(file_obj.id)
        print("✅ Chat completion examples completed")

    except Exception as e:
        print(f"❌ Error: {e}")


def streaming_chat_completion_with_files():
    """Streaming chat completion with file handling"""
    try:
        # Use existing Simpson.csv file
        with open("./Simpson.csv", "rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")

        # Streaming with file_ids in messages
        messages = [
            {
                "role": "user",
                "content": "分析数据并生成可视化图表。",
                "file_ids": [file_obj.id]
            }
        ]

        print("Streaming...")
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )

        full_response = ""
        collected_files = []

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content

            # Collect files from chunks
            if hasattr(chunk, 'generated_files') and chunk.generated_files:
                collected_files.extend(chunk.generated_files)

        print(f"\n✅ Streaming complete ({len(full_response)} chars, {len(collected_files)} files)")

        client.files.delete(file_obj.id)

    except Exception as e:
        print(f"❌ Error: {e}")

def assistant_with_analyze_tool():
    """Assistant with analyze tool and file analysis"""
    try:
        # Upload file and create assistant
        with open("./Simpson.csv", "rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")

        assistant = client.beta.assistants.create(
            name="Data Analysis Assistant",
            instructions="Analyze data and provide insights.",
            model=MODEL,
            tools=[{"type": "analyze"}],
        )

        # Create thread with file
        thread = client.beta.threads.create(
            tool_resources={"analyze": {"file_ids": [file_obj.id]}}
        )

        # Create message and run
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="分析数据并确定哪种教学方法效果更好。",
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Wait for completion
        while run.status in ["queued", "in_progress"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for msg in messages.data:
                if msg.role == "assistant":
                    content = msg.content[0].text.value
                    print(f"Assistant: {content[:200]}...")

                    files_from_message = extract_files_from_content(content)
                    if files_from_message:
                        print(f"Generated files: {len(files_from_message)}")
                        return files_from_message

        # Cleanup
        client.files.delete(file_obj.id)
        client.beta.assistants.delete(assistant.id)
        client.beta.threads.delete(thread.id)

        return {}

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}





def main():
    """Interactive example selector"""
    print("🚀 DeepAnalyze API Examples")
    print("API: localhost:8200 | Model: localhost:8000\n")

    examples = {
        "1": ("File API", file_api_examples),
        "2": ("Assistant", assistant_with_analyze_tool),
        "3": ("Chat Completion", chat_completion_with_message_file_ids),
        "4": ("Streaming", streaming_chat_completion_with_files),
        "5": ("All Examples", None)
    }

    while True:
        print("📋 Examples:")
        for num, (name, _) in examples.items():
            print(f"{num}. {name}")
        print("0. Exit")

        choice = input("\nSelect (0-5): ").strip()

        if choice == "0":
            print("👋 Goodbye!")
            break

        if choice not in examples:
            print("❌ Invalid choice")
            continue

        try:
            # Test connection
            models = client.models.list()
            print(f"✅ Connected: {[m.id for m in models.data]}")

            if choice == "5":  # Run all examples
                for num, (name, func) in list(examples.items())[:-1]:
                    print(f"\n{name}:")
                    func()
            else:
                name, func = examples[choice]
                print(f"\n{name}:")
                func()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("Ensure API server (localhost:8200) and model server (localhost:8000) are running")
            if choice in ["2", "5"]:
                print("And Simpson.csv exists in current directory")


if __name__ == "__main__":
    main()