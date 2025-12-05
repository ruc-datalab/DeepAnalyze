"""
Example usage of DeepAnalyze OpenAI-Compatible API with OpenAI library
Demonstrates 2-turn data analysis workflow with file attachments and streaming
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

def file_api_examples():
    """Demonstrate various file API operations"""
    try:
        # Create test file
        test_file_path = Path("test.txt")
        test_file_path.write_text("Test content")

        # Create files with different purposes
        file1 = client.files.create(file=test_file_path, purpose="file-extract")
        file2 = client.files.create(file=test_file_path, purpose="file-extract")

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
            file_obj = client.files.create(file=f, purpose="file-extract")

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

        print(f"Response: {message.content}")

        # Check for thread_id using hasattr()
        if hasattr(message, 'thread_id'):
            print(f"Thread ID: {message.thread_id}")

        # Show files from both formats
        if hasattr(message, 'files') and message.files:
            print(f"Files (message): {len(message.files)}")
        if hasattr(response, 'generated_files') and response.generated_files:
            print(f"Files (response): {len(response.generated_files)}")
        for file in response.generated_files:
            print(f"- {file['name']}: {file['url']}")


    except Exception as e:
        print(f"❌ Error: {e}")


def multi_turn_example():
    """Demonstrate 2-turn thread_id workflow with OpenAI library using streaming and file attachments"""
    try:
        messages = []

        # Upload data file
        print("📁 Uploading Simpson.csv data file...")
        with open("./Simpson.csv", "rb") as f:
            file_obj = client.files.create(file=f, purpose="file-extract")
        print(f"✅ File uploaded successfully: {file_obj.id[:12]}...")

        # First request - creates new thread, examine data structure
        print("1️⃣ First request - examining data structure...")
        messages.append({
            "role": "user",
            "content": "请查看这个数据文件的结构，告诉我有哪些字段、数据类型和基本统计信息。",
            "file_ids": [file_obj.id]
        })

        print("Streaming response...")
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )

        full_response = ""
        received_thread_id = None
        collected_files = []

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content

            # Check for thread_id in streaming chunks
            if hasattr(chunk.choices[0].delta, 'thread_id') and chunk.choices[0].delta.thread_id:
                received_thread_id = chunk.choices[0].delta.thread_id

            # Collect files from chunks
            if hasattr(chunk, 'generated_files') and chunk.generated_files:
                collected_files.extend(chunk.generated_files)

        print()  # New line after streaming
        thread_id = received_thread_id
        print(f"📝 Response received with thread_id: {thread_id}")

        if collected_files:
            print(f"📁 Files generated: {len(collected_files)}")

        messages.append({"role": "assistant", "content": full_response})

        # Second request with thread_id - generate analysis report
        print(f"\n2️⃣ Second request - generating analysis report (with thread_id: {thread_id[:12] if thread_id else 'None'}...)...")
        messages.append({
            "role": "user",
            "content": "基于刚才的数据结构分析，请生成一个详细的数据分析报告，包括：\n1. 数据质量评估\n2. 各字段的数据分布\n3. 相关性分析\n4. 主要发现和洞察",
            "file_ids": [file_obj.id]
        })
        if thread_id:
            messages[-1]["thread_id"] = thread_id

        print("Streaming response...")
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True
        )

        full_response2 = ""
        returned_thread_id = None
        collected_files2 = []

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response2 += content

            # Check for thread_id in streaming chunks
            if hasattr(chunk.choices[0].delta, 'thread_id') and chunk.choices[0].delta.thread_id:
                returned_thread_id = chunk.choices[0].delta.thread_id

            # Collect files from chunks
            if hasattr(chunk, 'generated_files') and chunk.generated_files:
                collected_files2.extend(chunk.generated_files)

        print()  # New line after streaming
        print(f"📝 Response thread_id: {returned_thread_id[:12] if returned_thread_id else 'None'}...")
        print(f"✅ Thread ID match: {thread_id == returned_thread_id}")

        if collected_files2:
            print(f"📁 Files generated: {len(collected_files2)}")

        # Cleanup uploaded file
        client.files.delete(file_obj.id)
        print("🗑️  Uploaded file cleaned up")

        print("\n✅ 2-turn data analysis workflow completed!")

    except Exception as e:
        print(f"❌ Error: {e}")


def streaming_chat_completion_with_files():
    """Streaming chat completion with file handling"""
    try:
        # Use existing Simpson.csv file
        with open("./Simpson.csv", "rb") as f:
            file_obj = client.files.create(file=f, purpose="file-extract")

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
        received_thread_id = None

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content

            # Check for thread_id in streaming chunks
            if hasattr(chunk.choices[0].delta, 'thread_id') and chunk.choices[0].delta.thread_id:
                received_thread_id = chunk.choices[0].delta.thread_id

            # Collect files from chunks
            if hasattr(chunk, 'generated_files') and chunk.generated_files:
                collected_files.extend(chunk.generated_files)

        if received_thread_id:
            print(f"\nThread ID: {received_thread_id}")

        print(f"\n✅ Streaming complete ({len(full_response)} chars, {len(collected_files)} files)")
        for file in collected_files:
            print(f"- {file['name']}: {file['url']}")

    except Exception as e:
        print(f"❌ Error: {e}")






def main():
    """Interactive example selector"""
    print("🚀 DeepAnalyze API Examples")
    print("API: localhost:8200 | Model: localhost:8000\n")

    examples = {
        "1": ("File API", file_api_examples),
        "2": ("Chat Completion", chat_completion_with_message_file_ids),
        "3": ("2-Turn Data Analysis", multi_turn_example),
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
            if choice in ["2", "3", "4", "5"]:
                print("And Simpson.csv exists in current directory")


if __name__ == "__main__":
    main()