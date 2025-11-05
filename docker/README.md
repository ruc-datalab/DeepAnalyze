# DeepAnalyze Docker Environment

A production-ready Docker environment for DeepAnalyze, featuring vLLM for high-performance LLM inference with CUDA 12.1 support.

## 🚀 Quick Start

### Prerequisites

- Docker installed
- NVIDIA GPU with CUDA support (for GPU acceleration)
- NVIDIA Container Toolkit (for GPU support)

## 📦 Deployment Options

### Option 1: Pull from Docker Hub (Recommended)

The easiest way to get started - pull the pre-built image:

```bash
# Pull the image
docker pull facdbe/deepanalyze-env:latest

# Run with GPU support
docker run --gpus all -it --rm \
  -p 8000:8000 \
  facdbe/deepanalyze-env:latest

```

### Option 2: Build from Dockerfile

Build the image from source for customization:

```bash
# Clone the repository
git clone https://github.com/ruc-datalab/DeepAnalyze.git
cd DeepAnalyze/docker

# Build the image
docker build -t deepanalyze-env:latest .

# Run the container
docker run --gpus all -it --rm \
  -p 8000:8000 \
  deepanalyze-env:latest
```

## 🔧 vLLM Server Deployment

### Start vLLM OpenAI-Compatible API Server

```bash
docker run --gpus all -d \
  -p 8000:8000 \
  -v /path/to/models:/models \
  --name deepanalyze-vllm \
  deepanalyze-env:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/your-model-name \
    --host 0.0.0.0 \
    --port 8000
```

### API Endpoints

Once the vLLM server is running, you can access:

- **Base URL**: `http://localhost:8000`
- **OpenAI-compatible endpoint**: `http://localhost:8000/v1/completions`
- **Chat endpoint**: `http://localhost:8000/v1/chat/completions`
- **Models endpoint**: `http://localhost:8000/v1/models`


## 📦 Image Size

- **Total Size**: ~17GB
- **Base CUDA**: ~5GB
- **vLLM + PyTorch**: ~10GB
- **Data Science Tools**: ~2GB


## 🤝 Contributing

Issues and pull requests are welcome!

## 📧 Support

For questions and support, please open an issue on GitHub.
