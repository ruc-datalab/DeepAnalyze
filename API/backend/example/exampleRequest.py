#!/usr/bin/env python3
"""
Example usage of DeepAnalyze OpenAI-Compatible API
Demonstrates common use cases
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
        data = {'purpose': 'assistants'}
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
        print(f"Response: {content[:100]}...")

        files = result.get('generated_files', [])
        if files:
            print(f"Files: {len(files)} generated")

    # Cleanup
    requests.delete(f"{API_BASE}/v1/files/{file_id}")


def assistants_workflow():
    """Full Assistants API workflow with data analysis"""
    # Use existing Simpson.csv file
    with open("./Simpson.csv", 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'assistants'}
        response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

    file_id = response.json()['id']

    try:
        # Create assistant
        assistant = requests.post(f"{API_BASE}/v1/assistants", json={
            "model": MODEL,
            "name": "Data Analyst",
            "instructions": "Analyze data and provide insights.",
            "file_ids": [file_id]
        }).json()

        # Create thread
        thread = requests.post(f"{API_BASE}/v1/threads", json={
            "metadata": {"example": "workflow"}
        }).json()

        # Add message
        requests.post(
            f"{API_BASE}/v1/threads/{thread['id']}/messages",
            json={
                "role": "user",
                "content": "分析数据并确定哪种教学方法效果更好。"
            }
        )

        # Create run
        run = requests.post(
            f"{API_BASE}/v1/threads/{thread['id']}/runs",
            json={"assistant_id": assistant['id']}
        ).json()

        # Wait for completion
        for i in range(60):
            status = requests.get(
                f"{API_BASE}/v1/threads/{thread['id']}/runs/{run['id']}"
            ).json()['status']

            if status == 'completed':
                break
            elif status in ['failed', 'cancelled', 'expired']:
                print(f"❌ Run {status}")
                return
            time.sleep(2)

        # Get results
        messages = requests.get(f"{API_BASE}/v1/threads/{thread['id']}/messages").json()['data']
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content'][0]['text']['value']
                print(f"Assistant: {content[:100]}...")
                break

        # Check generated files
        files = requests.get(f"{API_BASE}/v1/threads/{thread['id']}/files").json()['data']
        if files:
            print(f"Files: {len(files)} generated")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        requests.delete(f"{API_BASE}/v1/files/{file_id}")


def file_ids_in_messages():
    """Chat completion with file_ids in messages (OpenAI compatibility)"""
    # Upload file
    with open("./Simpson.csv", 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'assistants'}
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
    requests.delete(f"{API_BASE}/v1/files/{file_id}")


def streaming_chat():
    """Streaming chat response"""
    # Upload file
    with open("./Simpson.csv", 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'assistants'}
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
                    except json.JSONDecodeError:
                        pass
        print("\n✅ Streaming complete")

    # Cleanup
    requests.delete(f"{API_BASE}/v1/files/{file_id}")


def check_server():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Run examples"""
    print("🚀 DeepAnalyze API Examples")
    print("API: localhost:8200 | Model: localhost:8000")

    examples = {
        "1": ("Simple Chat", simple_chat),
        "2": ("Chat with File", chat_with_file),
        "3": ("Assistants Workflow", assistants_workflow),
        "4": ("File IDs in Messages", file_ids_in_messages),
        "5": ("Streaming Chat", streaming_chat),
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
