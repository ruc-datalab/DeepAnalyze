#!/bin/bash

# 验证Docker配置的脚本
echo "🔍 验证Docker配置..."

# 检查文件存在性
echo "检查文件..."
files=("Dockerfile" "docker-compose.yml" "build.sh" "README.md")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file 存在"
    else
        echo "❌ $file 缺失"
        exit 1
    fi
done

# 检查API目录
echo "检查API目录..."
if [ -d "API" ]; then
    echo "✅ API目录存在"
    # 检查关键API文件
    api_files=("API/start_server.py" "API/main.py" "API/chat_api.py" "API/config.py")
    for file in "${api_files[@]}"; do
        if [ -f "$file" ]; then
            echo "  ✅ $file"
        else
            echo "  ❌ $file 缺失"
            exit 1
        fi
    done
else
    echo "❌ API目录缺失"
    exit 1
fi

# 检查Dockerfile关键配置
echo ""
echo "检查Dockerfile配置..."

# 检查端口暴露
if grep -q "EXPOSE 8000 8100 8200" Dockerfile; then
    echo "✅ 端口配置正确"
else
    echo "❌ 端口配置有问题"
    exit 1
fi

# 检查API依赖
if grep -q "openai\|requests" Dockerfile; then
    echo "✅ API依赖已添加"
else
    echo "❌ 缺少API依赖"
    exit 1
fi

# 检查API源码复制
if grep -q "COPY.*API" Dockerfile; then
    echo "✅ API源码复制配置正确"
else
    echo "❌ API源码复制配置有问题"
    exit 1
fi

# 检查启动脚本
if grep -q "start_services.sh" Dockerfile; then
    echo "✅ 启动脚本配置正确"
else
    echo "❌ 启动脚本配置有问题"
    exit 1
fi

# 检查docker-compose.yml配置
echo ""
echo "检查docker-compose.yml配置..."

# 检查端口映射
if grep -q "8000:8000\|8100:8100\|8200:8200" docker-compose.yml; then
    echo "✅ 端口映射正确"
else
    echo "❌ 端口映射有问题"
    exit 1
fi

# 检查GPU支持
if grep -q "nvidia" docker-compose.yml; then
    echo "✅ GPU支持配置正确"
else
    echo "❌ GPU支持配置有问题"
    exit 1
fi

# 检查环境变量
if grep -q "API_HOST\|API_PORT\|HTTP_SERVER_PORT" docker-compose.yml; then
    echo "✅ 环境变量配置正确"
else
    echo "❌ 环境变量配置有问题"
    exit 1
fi

echo ""
echo "🎉 所有配置验证通过！"
echo ""
echo "📋 下一步："
echo "1. 运行 ./build.sh 构建镜像"
echo "2. 创建 models 目录并放入 DeepAnalyze-8B 模型"
echo "3. 运行 docker-compose up -d 启动服务"