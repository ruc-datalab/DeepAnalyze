#!/usr/bin/env python3
"""
Example usage of DeepAnalyze OpenAI-Compatible API
Demonstrates common use cases
"""

import requests
import time
import json
from typing import Optional

API_BASE = "http://localhost:8200"
MODEL = "DeepAnalyze-8B"


def example_1_simple_chat():
    """Example 1: Simple chat without files"""
    print("\n" + "="*60)
    print("Example 1: Simple Chat (No Files)")
    print("="*60)
    
    response = requests.post(f"{API_BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "用一句话介绍Python编程语言"}
        ],
        "temperature": 0.3,
        "stream": False,
        "execute_code": False
    })
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"\n🤖 Assistant: {content}\n")
    else:
        print(f"❌ Error: {response.text}")


def example_2_chat_with_file():
    """Example 2: Chat with file attachment"""
    print("\n" + "="*60)
    print("Example 2: Chat with File Attachment")
    print("="*60)

    # Use Simpson.csv file
    csv_file_path = "./Simpson.csv"

    # Upload file
    print("📤 Uploading Simpson.csv file...")
    with open(csv_file_path, 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'assistants'}
        response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return

    file_obj = response.json()
    file_id = file_obj['id']
    print(f"✅ File uploaded: {file_id}")

    # Chat with file
    print("💬 Chatting with file...")
    response = requests.post(f"{API_BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Analyze which teaching method (treatment) performs better."}
        ],
        "file_ids": [file_id],
        "temperature": 0.3,
        "stream": False,
        "execute_code": False
    })

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"\n🤖 Assistant: {content}\n")
        files = result.get('generated_files', [])
        if files:
            print("📁 Generated Files:")
            for f in files:
                print(f"- {f['name']}: {f['url']}")
    else:
        print(f"❌ Error: {response.text}")

    # Cleanup
    requests.delete(f"{API_BASE}/v1/files/{file_id}")
    print("🧹 Cleaned up")


def example_3_assistants_workflow():
    """Example 3: Full Assistants API workflow with data analysis"""
    print("\n" + "="*60)
    print("Example 3: Assistants API Workflow (With Code Execution)")
    print("="*60)

    # Create sample data file
    import tempfile
    csv_content = """Month,Revenue,Expenses
Jan,50000,30000
Feb,55000,32000
Mar,60000,35000
Apr,58000,33000
May,62000,36000"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        temp_file = f.name

    try:
        # Step 1: Upload file
        print("📤 Step 1: Uploading file...")
        with open(temp_file, 'rb') as f:
            files = {'file': ('financial_data.csv', f, 'text/csv')}
            data = {'purpose': 'assistants'}
            response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

        file_obj = response.json()
        file_id = file_obj['id']
        print(f"✅ File uploaded: {file_id}")

        # Step 2: Create assistant
        print("\n👤 Step 2: Creating assistant...")
        response = requests.post(f"{API_BASE}/v1/assistants", json={
            "model": MODEL,
            "name": "Financial Analyst",
            "description": "Analyzes financial data",
            "instructions": "You are a financial analyst. Analyze data and provide insights.",
            "file_ids": [file_id]

        })

        assistant = response.json()
        assistant_id = assistant['id']
        print(f"✅ Assistant created: {assistant_id}")

        # Step 3: Create thread
        print("\n💬 Step 3: Creating thread...")
        response = requests.post(f"{API_BASE}/v1/threads", json={
            "metadata": {"example": "3"}
        })

        thread = response.json()
        thread_id = thread['id']
        print(f"✅ Thread created: {thread_id}")

        # Step 4: Add message
        print("\n📝 Step 4: Adding message with file...")
        response = requests.post(
            f"{API_BASE}/v1/threads/{thread_id}/messages",
            json={
                "role": "user",
                "content": "请分析这个财务数据，计算每月的利润，并创建一个可视化图表。"
            }
        )

        message = response.json()
        print(f"✅ Message added: {message['id']}")

        # Step 5: Create and monitor run
        print("\n🚀 Step 5: Starting analysis run...")
        response = requests.post(
            f"{API_BASE}/v1/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id}
        )

        run = response.json()
        run_id = run['id']
        print(f"✅ Run started: {run_id}")

        # Monitor run status
        print("\n⏳ Monitoring run progress...")
        max_attempts = 60
        for i in range(max_attempts):
            response = requests.get(
                f"{API_BASE}/v1/threads/{thread_id}/runs/{run_id}"
            )
            run = response.json()
            status = run['status']

            print(f"  [{i+1}/{max_attempts}] Status: {status}", end='\r')

            if status == 'completed':
                print(f"\n✅ Run completed!                    ")
                break
            elif status in ['failed', 'cancelled', 'expired']:
                print(f"\n❌ Run {status}: {run.get('last_error', {})}")
                return

            time.sleep(2)
        else:
            print("\n⚠️ Run timeout")
            return

        # Step 6: Get results
        print("\n📊 Step 6: Getting results...")
        response = requests.get(f"{API_BASE}/v1/threads/{thread_id}/messages")
        messages = response.json()['data']

        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content'][0]['text']['value']
                print(f"\n🤖 Assistant Response:\n{content}\n")
                break

        # Step 7: Get generated files
        print("📁 Step 7: Checking generated files...")
        response = requests.get(f"{API_BASE}/v1/threads/{thread_id}/files")
        files = response.json()['data']

        if files:
            print(f"✅ Found {len(files)} generated file(s):")
            for f in files:
                print(f"  📄 {f['name']}")
                print(f"     📥 Download: {f['url']}")
        else:
            print("ℹ️ No files generated")

        # Cleanup
        # print("\n🧹 Cleaning up...")
        # requests.delete(f"{API_BASE}/v1/files/{file_id}")
        # requests.delete(f"{API_BASE}/v1/threads/{thread_id}")
        # requests.delete(f"{API_BASE}/v1/assistants/{assistant_id}")
        # print("✅ Cleanup complete")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        import os
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def example_4_multi_level_file_association():
    """Example 4: Multi-level File Association (Assistant + Thread + Message)"""
    print("\n" + "="*60)
    print("Example 4: Multi-level File Association")
    print("="*60)

    # Create multiple data files
    import tempfile
    financial_data = """Month,Revenue,Expenses
Jan,50000,30000
Feb,55000,32000
Mar,60000,35000
Apr,58000,33000
May,62000,36000"""

    market_data = """Quarter,Market_Share,Growth_Rate
Q1,15.2,5.3
Q2,16.8,8.1
Q3,18.1,7.2
Q4,19.5,7.7"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f1:
        f1.write(financial_data)
        financial_file = f1.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f2:
        f2.write(market_data)
        market_file = f2.name

    try:
        # Upload financial data file (for assistant)
        print("📤 Step 1: Uploading financial data file...")
        with open(financial_file, 'rb') as f:
            files = {'file': ('financial_data.csv', f, 'text/csv')}
            data = {'purpose': 'assistants'}
            response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

        financial_file_obj = response.json()
        financial_file_id = financial_file_obj['id']
        print(f"✅ Financial file uploaded: {financial_file_id}")

        # Upload market data file (for thread)
        print("\n📤 Step 2: Uploading market data file...")
        with open(market_file, 'rb') as f:
            files = {'file': ('market_data.csv', f, 'text/csv')}
            data = {'purpose': 'assistants'}
            response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

        market_file_obj = response.json()
        market_file_id = market_file_obj['id']
        print(f"✅ Market file uploaded: {market_file_id}")

        # Step 3: Create assistant with financial data file
        print("\n👤 Step 3: Creating assistant with financial data...")
        response = requests.post(f"{API_BASE}/v1/assistants", json={
            "model": MODEL,
            "name": "Business Analyst",
            "description": "Analyzes business and financial data",
            "instructions": "You are a business analyst. Analyze the provided financial and market data to provide comprehensive business insights.",
            "file_ids": [financial_file_id]
        })

        assistant = response.json()
        assistant_id = assistant['id']
        print(f"✅ Assistant created: {assistant_id}")

        # Step 4: Create thread with market data file
        print("\n💬 Step 4: Creating thread with market data...")
        response = requests.post(f"{API_BASE}/v1/threads", json={
            "metadata": {"example": "4"},
            "file_ids": [market_file_id]
        })

        thread = response.json()
        thread_id = thread['id']
        print(f"✅ Thread created: {thread_id}")

        # Step 5: Add message (no additional files)
        print("\n📝 Step 5: Adding analysis request...")
        response = requests.post(
            f"{API_BASE}/v1/threads/{thread_id}/messages",
            json={
                "role": "user",
                "content": "请综合分析财务数据和市场数据，创建一个完整的业务分析报告，包括趋势分析和可视化图表。"
            }
        )

        message = response.json()
        print(f"✅ Message added: {message['id']}")

        # Step 6: Create and monitor run
        print("\n🚀 Step 6: Starting comprehensive analysis run...")
        response = requests.post(
            f"{API_BASE}/v1/threads/{thread_id}/runs",
            json={"assistant_id": assistant_id}
        )

        run = response.json()
        run_id = run['id']
        print(f"✅ Run started: {run_id}")

        # Monitor run status
        print("\n⏳ Monitoring run progress...")
        max_attempts = 60
        for i in range(max_attempts):
            response = requests.get(
                f"{API_BASE}/v1/threads/{thread_id}/runs/{run_id}"
            )
            run = response.json()
            status = run['status']

            print(f"  [{i+1}/{max_attempts}] Status: {status}", end='\r')

            if status == 'completed':
                print(f"\n✅ Run completed!                    ")
                break
            elif status in ['failed', 'cancelled', 'expired']:
                print(f"\n❌ Run {status}: {run.get('last_error', {})}")
                return

            time.sleep(2)
        else:
            print("\n⚠️ Run timeout")
            return

        # Step 7: Get results
        print("\n📊 Step 7: Getting analysis results...")
        response = requests.get(f"{API_BASE}/v1/threads/{thread_id}/messages")
        messages = response.json()['data']

        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content'][0]['text']['value']
                print(f"\n🤖 Assistant Analysis:\n{content}\n")
                break

        # Step 8: Get generated files
        print("📁 Step 8: Checking generated files...")
        response = requests.get(f"{API_BASE}/v1/threads/{thread_id}/files")
        files = response.json()['data']

        if files:
            print(f"✅ Found {len(files)} generated file(s):")
            for f in files:
                print(f"  📄 {f['name']}")
                print(f"     📥 Download: {f['url']}")
        else:
            print("ℹ️ No files generated")

        print("\n📝 Summary:")
        print("  📋 Assistant files: financial_data.csv")
        print("  📋 Thread files: market_data.csv")
        print("  📋 Message files: None")
        print("  🎯 All files were automatically collected and made available during run execution")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        import os
        # Cleanup temporary files
        for temp_file in [financial_file, market_file]:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        # Cleanup uploaded files
        try:
            requests.delete(f"{API_BASE}/v1/files/{financial_file_id}")
            requests.delete(f"{API_BASE}/v1/files/{market_file_id}")
        except:
            pass


def example_5_streaming_chat():
    """Example 5: Streaming chat response"""
    print("\n" + "="*60)
    print("Example 5: Streaming Chat Response")
    print("="*60)

    # Use Simpson.csv file
    csv_file_path = "./Simpson.csv"

    # Upload file first
    print("📤 Uploading Simpson.csv file...")
    with open(csv_file_path, 'rb') as f:
        files = {'file': ('Simpson.csv', f, 'text/csv')}
        data = {'purpose': 'assistants'}
        response = requests.post(f"{API_BASE}/v1/files", files=files, data=data)

    if response.status_code != 200:
        print(f"❌ Upload failed: {response.text}")
        return

    file_obj = response.json()
    file_id = file_obj['id']
    print(f"✅ File uploaded: {file_id}")

    print("\n💬 User: Analyze which teaching method (treatment) performs better.\n")
    print("🤖 Assistant: ", end='', flush=True)

    response = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Analyze which teaching method (treatment) performs better."}
            ],
            "file_ids": [file_id],
            "temperature": 0.3,
            "stream": True,
            "execute_code": True
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
        print("\n")
    else:
        print(f"❌ Error: {response.text}")

    # Cleanup
    requests.delete(f"{API_BASE}/v1/files/{file_id}")
    print("🧹 Cleaned up")


def check_server():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("🚀 DeepAnalyze API Usage Examples")
    print("="*60)

    # Check server
    print("\n🔍 Checking API server...")
    if not check_server():
        print("❌ API server is not running!")
        print("ℹ️  Please start it with: python tryAPI.py")
        return
    print("✅ API server is running")
    while True:
        # Show menu
        print("\nSelect example to run:")
        print("  1) Simple chat (no files)")
        print("  2) Chat with file attachment")
        print("  3) Full Assistants workflow (with code execution)")
        print("  4) Multi-level File Association (Assistant + Thread + Message)")
        print("  5) Streaming chat response")
        print("  6) Run all examples")
        print("  0) Exit")

        choice = input("\nEnter choice [0-6]: ").strip()

        if choice == '1':
            example_1_simple_chat()
        elif choice == '2':
            example_2_chat_with_file()
        elif choice == '3':
            example_3_assistants_workflow()
        elif choice == '4':
            example_4_multi_level_file_association()
        elif choice == '5':
            example_5_streaming_chat()
        elif choice == '6':
            example_1_simple_chat()
            time.sleep(1)
            example_2_chat_with_file()
            time.sleep(1)
            example_3_assistants_workflow()
            time.sleep(1)
            example_4_multi_level_file_association()
            time.sleep(1)
            example_5_streaming_chat()
        elif choice == '0':
            print("\n👋 Goodbye!")
            return
        else:
            print("\n❌ Invalid choice")

        print("\n" + "="*60)
        print("✅ Example completed!")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
