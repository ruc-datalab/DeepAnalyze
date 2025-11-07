"""
Example usage of DeepAnalyze OpenAI-Compatible API with OpenAI library
Demonstrates assistant workflow with analyze tool
"""

import openai
import time
import re

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


def assistant_with_analyze_tool():
    """Assistant with analyze tool and file analysis"""
    print("🚀 DeepAnalyze Analyze Tool Example")
    print("="*50)

    try:
        # Upload file
        print("📤 Uploading Simpson.csv file...")
        with open("./Simpson.csv", "rb") as f:
            file_obj = client.files.create(file=f, purpose="assistants")
        print(f"✅ File uploaded: {file_obj.id}")

        # Create assistant with analyze tool
        assistant = client.beta.assistants.create(
            name="Data Analysis Assistant",
            instructions="You are a data analysis expert. Analyze the provided data and generate insights.",
            model=MODEL,
            tools=[{"type": "analyze"}],
        )
        print(f"✅ Created assistant: {assistant.id}")

        # Create thread with tool_resources (analyze tool files)
        thread = client.beta.threads.create(
            tool_resources={
                "analyze": {
                    "file_ids": [file_obj.id]
                }
            }
        )
        print(f"✅ Created thread: {thread.id}")

        # Create message
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Analyze the Simpson dataset and determine which teaching method performs better. Please provide statistical analysis.",
        )
        print(f"✅ Created message: {message.id}")

        # Create run
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )
        print(f"✅ Created run: {run.id}")

        # Wait for completion
        print("⏳ Waiting for completion...")
        all_generated_files = {}

        while run.status in ["queued", "in_progress"]:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            print(f"   Status: {run.status}")

        if run.status == "completed":
            # Get messages
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            for msg in messages.data:
                if msg.role == "assistant":
                    content = msg.content[0].text.value
                    print(f"\n🤖 Assistant: {content}\n")

                    # 提取文件信息
                    files_from_message = extract_files_from_content(content)
                    if files_from_message:
                        print("📁 在此消息中发现文件:")
                        for filename, url in files_from_message.items():
                            print(f"   - {filename}: {url}")
                            all_generated_files[filename] = url

            # 显示所有收集到的文件
            if all_generated_files:
                print(f"\n📋 总共收集到 {len(all_generated_files)} 个文件:")
                for filename, url in all_generated_files.items():
                    print(f"   📄 {filename}")
                    print(f"      🔗 {url}")
                    print(f"      💾 直接下载: http://localhost:8100/{thread.id}/generated/{filename}")
                    print()

                print("💡 提示: 你可以通过以下方式访问这些文件:")
                print("   1. 直接点击上述URL下载")
                print("   2. 使用 requests.get(url) 下载文件内容")
                print("   3. 文件也存储在 workspace/thread-{id}/generated/ 目录中")
            else:
                print("📝 此分析没有生成文件")

        else:
            print(f"❌ Run failed with status: {run.status}")

        # Cleanup
        client.files.delete(file_obj.id)
        client.beta.assistants.delete(assistant.id)
        client.beta.threads.delete(thread.id)
        print("🧹 Cleaned up")

        return all_generated_files

    except Exception as e:
        print(f"❌ Error: {e}")
        return {}


def main():
    """Run the example"""
    print("Make sure the DeepAnalyze API server is running on localhost:8200")
    print("And the vLLM model server is running on localhost:8000\n")

    try:
        # Test connection
        models = client.models.list()
        print(f"✅ Connected to API. Available models: {[m.id for m in models.data]}\n")

        # Run example and get files
        generated_files = assistant_with_analyze_tool()

        # 演示如何使用返回的文件字典
        if generated_files:
            print(f"\n🎯 文件字典使用示例:")
            print(f"返回的文件字典: {generated_files}")
            print(f"文件数量: {len(generated_files)}")

            # 遍历所有文件
            for filename, url in generated_files.items():
                print(f"\n📄 处理文件: {filename}")
                print(f"   URL: {url}")

    except Exception as e:
        print(f"❌ Connection error: {e}")
        print("\nPlease check that:")
        print("1. DeepAnalyze API server is running on localhost:8200")
        print("2. vLLM model server is running on localhost:8000")
        print("3. Simpson.csv file exists in the current directory")


if __name__ == "__main__":
    main()