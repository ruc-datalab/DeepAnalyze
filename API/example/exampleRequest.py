#!/usr/bin/env python3
"""
Example usage of DeepAnalyze OpenAI-Compatible API
Demonstrates common use cases including 2-turn data analysis with file attachments
"""

import requests
import time
import json

API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"


def simple_chat():
    """Simple chat without files"""
    response = requests.post(f"{API_BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "用一句话介绍Python编程语言"}
        ],
        "temperature": 0.3
    })

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"Assistant: {content[:100]}...")
    else:
        print(f"❌ Error: {response.text}")


def chat_with_file():
    """Chat with file attachment"""
    # Upload file
    with open("./Simpson.csv", 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'file-extract'}
        response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return

    file_id = response.json()['id']

    # Chat with file
    response = requests.post(f"{API_BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "分析哪种教学方法效果更好。"}
        ],
        "file_ids": [file_id],
        "temperature": 0.3
    })

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"Response: {content}...")

        files = result.get('generated_files', [])
        if files:
            print(f"Files: {len(files)} generated")

    # Cleanup
    # requests.delete(f"{API_BASE}/v1/files/{file_id}")




def file_ids_in_messages():
    """Chat completion with file_ids in messages (OpenAI compatibility)"""
    # Upload file
    with open("./Simpson.csv", 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'file-extract'}
        response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

    file_id = response.json()['id']

    # New format: file_ids in messages
    response = requests.post(f"{API_BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "分析数据并生成可视化图表。",
                "file_ids": [file_id]  # can both be inside message and top-level, for compatibility with OpenAI API
            }
        ],
        "temperature": 0.3
    })

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"Response: {content[:100]}...")

        # Check for files in message (new format)
        message = result['choices'][0]['message']
        if 'files' in message:
            print(f"Files in message: {len(message['files'])}")

        # Check for generated_files (backward compatibility)
        if 'generated_files' in result:
            print(f"Generated files: {len(result['generated_files'])}")

    # Cleanup
    # requests.delete(f"{API_BASE}/v1/files/{file_id}")


def streaming_chat():
    """Streaming chat response"""
    # Upload file
    with open("./Simpson.csv", 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'file-extract'}
        response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

    file_id = response.json()['id']

    print("Streaming response...")
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "分析数据并生成趋势图。",
                    "file_ids": [file_id]
                }
            ],
            "temperature": 0.3,
            "stream": True
        },
        stream=True
    )

    if response.status_code == 200:
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data_str)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                print(delta['content'], end='', flush=True)
                        if 'generated_files' in chunk:
                            print(f"\n\n📁 New file generated: {chunk['generated_files']}")
                    except json.JSONDecodeError:
                        pass
        print("\n✅ Streaming complete")

    


def check_server():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def multi_turn_example():
    """Demonstrate thread_id workflow for 2-turn data analysis conversation with streaming and file attachments"""
    print("🧵 Testing Thread ID Workflow with Streaming for Data Analysis...")

    conversation_history = []

    # Upload data file
    with open("./Simpson.csv", 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'file-extract'}
        response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

    if response.status_code != 200:
        print(f"❌ File upload failed: {response.text}")
        return

    file_id = response.json()['id']

    # First request - creates new thread, examine data structure
    print("\n1️⃣ First request - examining data structure...")
    conversation_history.append({
        "role": "user",
        "content": "请查看这个数据文件的结构，告诉我有哪些字段、数据类型和基本统计信息。"
    })

    print("Streaming response...")
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": conversation_history,
            "file_ids": [file_id],
            "temperature": 0.3,
            "stream": True
        },
        stream=True
    )

    if response.status_code != 200:
        print(f"❌ First request failed: {response.text}")
        return

    full_response = ""
    received_thread_id = None
    generated_files = []

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            print(content, end='', flush=True)
                            full_response += content
                    if 'thread_id' in chunk:
                        received_thread_id = chunk['thread_id']
                    if 'generated_files' in chunk:
                        generated_files.extend(chunk['generated_files'])
                except json.JSONDecodeError:
                    pass

    print()  # New line after streaming
    thread_id = received_thread_id
    print(f"📝 Response received with thread_id: {thread_id}")

    # Add assistant response to history
    conversation_history.append({"role": "assistant", "content": full_response})

    # Check for generated files
    if generated_files:
        print(f"📁 Files generated: {len(generated_files)}")

    # Second request with thread_id - generate analysis report
    print(f"\n2️⃣ Second request - generating analysis report (with thread_id: {thread_id[:12] if thread_id else 'None'}...)...")
    conversation_history.append({
        "role": "user",
        "content": "基于刚才的数据结构分析，请生成一个详细的数据分析报告，包括：\n1. 数据质量评估\n2. 各字段的数据分布\n3. 相关性分析\n4. 主要发现和洞察"
    })
    if thread_id:
        conversation_history[-1]["thread_id"] = thread_id

    print("Streaming response...")
    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": conversation_history,
            "file_ids": [file_id],
            "temperature": 0.3,
            "stream": True
        },
        stream=True
    )

    if response.status_code != 200:
        print(f"❌ Second request failed: {response.text}")
        return

    full_response2 = ""
    returned_thread_id = None
    generated_files2 = []

    for line in response.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith('data: '):
                data_str = line_str[6:]
                if data_str == '[DONE]':
                    break
                try:
                    chunk = json.loads(data_str)
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            print(content, end='', flush=True)
                            full_response2 += content
                    if 'thread_id' in chunk:
                        returned_thread_id = chunk['thread_id']
                    if 'generated_files' in chunk:
                        generated_files2.extend(chunk['generated_files'])
                except json.JSONDecodeError:
                    pass

    print()  # New line after streaming
    print(f"📝 Response thread_id: {returned_thread_id[:12] if returned_thread_id else 'None'}...")
    print(f"✅ Thread ID match: {thread_id == returned_thread_id}")

    if generated_files2:
        print(f"📁 Files generated: {len(generated_files2)}")

    # Cleanup uploaded file
    requests.delete(f"{API_BASE}/v1/files/{file_id}")
    print("🗑️  Uploaded file cleaned up")

    print("\n✅ 2-turn data analysis workflow completed successfully!")



def main():
    """Run examples"""
    print("🚀 DeepAnalyze API Examples")
    print("API: localhost:8200 | Model: localhost:8000")

    examples = {
        "1": ("Simple Chat", simple_chat),
        "2": ("Chat with File", chat_with_file),
        "3": ("File IDs in Messages", file_ids_in_messages),
        "4": ("Streaming Chat", streaming_chat),
        "5": ("2-Turn Data Analysis", multi_turn_example),
        "6": ("All Examples", None)
    }

    while True:
        print("\n📋 Examples:")
        for num, (name, _) in examples.items():
            print(f"{num}. {name}")
        print("0. Exit")

        choice = input("\nSelect (0-6): ").strip()

        if choice == "0":
            print("👋 Goodbye!")
            break

        if choice not in examples:
            print("❌ Invalid choice")
            continue

        try:
            if choice == "6":  # Run all examples
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


if __name__ == "__main__":
    main()
