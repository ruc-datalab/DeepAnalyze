# 常见问题解答（持续更新，欢迎贡献）

## 使用问题

### ❓Windows下无法配置vLLM

​	目前vLLM暂不支持windows下运行。可以使用wsl或者docker进行部署，详见docker目录下的readme。

​	后续会更新详细的部署文档。

### ❓模型产生的图表中的文字出现乱码，如何解决？

​	当前中文字体默认使用SimHei。出现乱码可能是因为字体未安装的问题。可以运行下面的命令解决

```bash
wget https://raw.githubusercontent.com/StellarCN/scp_zh/master/fonts/SimHei.ttf
mkdir -p ~/.fonts
mv SimHei.ttf ~/.fonts/
rm -rf ~/.cache/matplotlib
```

