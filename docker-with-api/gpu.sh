#!/bin/bash

# 移除可能冲突的现有源
sudo rm -f /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo rm -f /etc/apt/sources.list.d/nvidia-docker.list

# 添加 NVIDIA Container Runtime 官方 GPG 密钥
curl -fsSL https://nvidia.github.io/nvidia-container-runtime/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-runtime-keyring.gpg

# 添加官方仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-runtime-keyring.gpg] https://nvidia.github.io/nvidia-container-runtime/$distribution/ /" | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list

sudo apt-get update