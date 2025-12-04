# DeepAnalyze Docker Environment with API Support

这个Docker环境包含完整的DeepAnalyze服务栈：
- **vLLM服务器** (端口 8000): 高性能LLM推理服务
- **API服务器** (端口 8200): OpenAI兼容的API接口
- **文件服务器** (端口 8100): 文件上传和管理服务

## 🚀 快速开始

### 前置要求
- Docker 和 Docker Compose
- NVIDIA GPU 和 Docker NVIDIA Container Toolkit
- 足够的存储空间用于模型文件

### 1. 验证和构建Docker镜像

```bash
# 进入docker-with-api目录
cd /data/cy/personal/DeepAnalyze/DeepAnalyze/docker-with-api

# (可选) 验证配置是否正确
./verify.sh

# 运行构建脚本（会自动复制API源码）
./build.sh
```

> **注意**：构建脚本会自动将`../API`目录复制到构建上下文中，确保包含最新的API源码。

### 2. 准备模型和目录

```bash
# 创建必要的目录
mkdir -p workspace models

# 将DeepAnalyze-8B模型文件放置到models目录中
# 确保模型目录结构如下：
# models/
# └── DeepAnalyze-8B/
#     ├── config.json
#     ├── tokenizer.json
#     └── ... (其他模型文件)
```

### 3. 启动服务

```bash
# 使用docker-compose启动所有服务
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 4. 验证服务

一旦服务启动完成，你可以访问以下URL：

- **vLLM API**: http://localhost:8000
  - 健康检查: http://localhost:8000/health
  - 模型列表: http://localhost:8000/v1/models

- **API服务器**: http://localhost:8200
  - 健康检查: http://localhost:8200/health
  - API文档: http://localhost:8200/docs

- **文件服务器**: http://localhost:8100

## 📋 服务详情

### vLLM服务器 (端口 8000)
- 基于vLLM的高性能推理引擎
- 支持OpenAI API格式
- 自动GPU内存管理
- 模型: DeepAnalyze-8B

### API服务器 (端口 8200)
提供以下API端点：
- `/v1/models` - 模型管理
- `/v1/chat/completions` - 聊天完成
- `/v1/files` - 文件管理
- `/v1/admin` - 管理功能

### 文件服务器 (端口 8100)
- 静态文件服务
- 支持文件上传和下载
- 与API服务器集成

## 🛠️ 配置选项

### 环境变量
可以在`docker-compose.yml`中修改以下环境变量：

```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0           # 指定使用的GPU
  - API_HOST=0.0.0.0                # API服务器监听地址
  - API_PORT=8200                    # API服务器端口
  - HTTP_SERVER_PORT=8100            # 文件服务器端口
  - API_BASE=http://localhost:8000/v1 # vLLM API地址
  - MODEL_PATH=DeepAnalyze-8B        # 模型路径
```

### GPU配置
默认使用所有可用的GPU。如需指定特定GPU：
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0  # 使用第一个GPU
```

### 模型配置
vLLM启动参数在`Dockerfile`中的`start_services.sh`脚本中定义：
```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model DeepAnalyze-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9
```

## 🔧 故障排除

### 常见问题

1. **GPU不可用**
   ```bash
   # 检查NVIDIA Docker支持
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. **模型加载失败**
   - 确保模型文件在正确的目录 (`./models/DeepAnalyze-8B/`)
   - 检查模型文件完整性

3. **端口冲突**
   - 确保端口8000、8100、8200没有被其他服务占用
   - 可以在`docker-compose.yml`中修改端口映射

4. **内存不足**
   - 调整`gpu-memory-utilization`参数
   - 考虑使用更小的模型或更多GPU

### 日志查看
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f deepanalyze

# 查看容器内部状态
docker-compose exec deepanalyze bash
```

### 重新构建
```bash
# 完全重新构建（不使用缓存）
docker-compose build --no-cache

# 或者使用构建脚本
./build.sh
```

## 📁 目录结构

```
docker-with-api/
├── Dockerfile              # Docker镜像定义
├── docker-compose.yml      # 服务编排配置
├── build.sh               # 构建脚本
├── README.md              # 本文档
├── workspace/             # 工作目录挂载点
└── models/                # 模型文件挂载点
    └── DeepAnalyze-8B/    # 模型文件
```

## 🎯 使用示例

### 通过API使用DeepAnalyze

```python
import requests

# API服务器地址
api_base = "http://localhost:8200/v1"

# 聊天请求
response = requests.post(
    f"{api_base}/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "DeepAnalyze-8B",
        "messages": [
            {"role": "user", "content": "你好，请介绍一下你自己。"}
        ],
        "temperature": 0.7
    }
)

print(response.json())
```

## 📝 开发说明

### 修改Docker配置
1. 修改`Dockerfile`来添加依赖或更改配置
2. 修改`docker-compose.yml`来调整服务配置
3. 修改`start_services.sh`来更改启动参数
4. 运行`./build.sh`重新构建镜像

### 调试模式
如需进入容器调试：
```bash
docker-compose exec deepanalyze bash
```

## 📦 镜像大小

- **总大小**: ~18GB
- **基础CUDA**: ~5GB
- **vLLM + PyTorch**: ~10GB
- **数据科学工具**: ~2GB
- **API服务依赖**: ~1GB

## 📄 许可证

本项目遵循与主DeepAnalyze项目相同的许可证。
