#!/bin/bash

# =============================================================================
# GeoDeepAnalyze 模型下载脚本: 
#   - 使用 Hugging Face 镜像加速下载 (hf-mirror.com)
#   - 自动检测并选择合适的下载工具 (aria2c > wget)
#   - 支持多线程并发下载
# =============================================================================

# 设置脚本执行权限
chmod a+x "$PWD/hfd.sh"

# 检查 hfd.sh 文件是否存在
if [[ ! -f "$PWD/hfd.sh" ]]; then
    echo "错误: hfd.sh 文件不存在于当前目录"
    exit 1
fi

# 设置 Hugging Face 镜像端点 (国内用户推荐使用镜像加速下载)
export HF_ENDPOINT="https://hf-mirror.com"

# 检查可用的下载工具
if command -v aria2c &>/dev/null; then
    echo "检测到 aria2c 工具，使用多线程下载 (推荐)"
    DOWNLOAD_TOOL="aria2c"
    # -x 4: 使用 4 个线程进行单个文件下载 (提高单个文件下载速度)
    # -j 4: 同时下载 4 个文件 (提高并发下载能力)
    DOWNLOAD_ARGS="-x 4 -j 4"
elif command -v wget &>/dev/null; then
    echo "未检测到aria2c, 使用wget单线程下载"
    DOWNLOAD_TOOL="wget"
    DOWNLOAD_ARGS="--tool wget"
else
    echo "错误: 未找到可用的下载工具 (aria2c 或 wget)"
    exit 1
fi

echo "开始下载 RUC-DataLab/DeepAnalyze-8B 模型..."
echo "下载工具: $DOWNLOAD_TOOL"

# 调用 hfd.sh 脚本下载模型
"$PWD/hfd.sh" RUC-DataLab/DeepAnalyze-8B $DOWNLOAD_ARGS

# 检查下载是否成功
if [[ $? -eq 0 ]]; then
    echo "===================✅ 模型下载完成！==================="
else
    echo "===================❌ 模型下载失败！==================="
    exit 1
fi