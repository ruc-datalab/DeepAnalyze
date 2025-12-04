#!/bin/bash

# 修复Docker构建中的GPG签名问题的脚本
echo "🔧 修复Docker GPG签名问题..."

# 清理Docker缓存
echo "清理Docker构建缓存..."
docker system prune -f

# 清理apt缓存（如果在Ubuntu系统中）
echo "清理本地apt缓存..."
sudo apt-get clean 2>/dev/null || echo "跳过sudo apt清理"

echo ""
echo "✅ 缓存清理完成"
echo ""
echo "📋 现在可以选择以下构建方式："
echo ""
echo "1. 使用原始Dockerfile（已修复GPG问题）："
echo "   ./build.sh"
echo ""
echo "2. 使用备用的PyTorch基础镜像（更稳定）："
echo "   ./build.sh --alternative"
echo ""
echo "如果仍然遇到GPG问题，请选择选项2。"