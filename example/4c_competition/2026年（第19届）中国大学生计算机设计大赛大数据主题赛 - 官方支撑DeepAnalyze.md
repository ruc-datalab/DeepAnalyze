# api 接口请求

提供两种调用方案：

## 1 纯 prompt 版请求示例

```Python
!curl -X POST https://www.heywhale.com/api/model/services/69b7c9d028cbfe8349df5924/app/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer your_api_key" \
-d '{"messages": [{"role": "user", "content": "Who are you？"}]}'
curl -X POST "https://www.heywhale.com/api/model/services/69b7c9d028cbfe8349df5924/app/v1/chat/completions" -H "Content-Type: application/json" -H "Authorization: Bearer your_api_key" -d "{\"messages\": [{\"role\": \"user\", \"content\": \"What is tensorflow？\"}]}"
```

## 2 prompt+上传文件版请求示例：

将zip文件解压到本地目录

1. **运行脚本**：

```Bash
python quick_start.py
```

2. **按提示操作**：

   \- 输入API密钥

   \- 选择对话类型（1=无文件对话，2=文件分析）

   \- 如选择2，输入文件路径（支持逗号分隔多个文件）

   \- 输入分析指令（可选，留空使用默认指令）

📝 使用示例

示例1：分析CSV文件

```YAML
python quick_start.py
# Enter API Key: your_api_key
# Enter choice (1 or 2): 2
# Enter file paths: Simpson.csv
# Enter analysis instruction: [留空或输入自定义指令]
```

示例2：分析ZIP压缩包

```YAML
python quick_start.py
# Enter API Key: your_api_key
# Enter choice (1 or 2): 2
# Enter file paths: example.zip
# Enter analysis instruction: [留空或输入自定义指令]
```

**注意**：ZIP文件会自动解压，只处理支持的文件格式。

示例3：无文件对话

```Bash
python quick_start.py
# Enter API Key: your_api_key
# Enter choice (1 or 2): 1
# Enter analysis instruction: 请解释一下什么是机器学习
```