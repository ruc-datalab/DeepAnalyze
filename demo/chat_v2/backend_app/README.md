# 后端结构说明

`demo/chat_v2/backend.py` 现在只保留启动入口，核心逻辑拆到了 `backend_app/`：

- `app.py`：创建 FastAPI 应用并注册路由。
- `settings.py`：集中管理模型、端口、workspace 等配置。
- `routers/`：按接口职责拆分路由。
- `services/workspace.py`：workspace、上传、文件树、代理、静态文件服务。
- `services/execution.py`：代码执行与产物收集。
- `services/chat.py`：聊天流式生成与 `<Code>` 执行链路。
- `services/exporter.py`：报告导出与 Markdown/PDF 保存。

这样做的好处：

- 降低 `backend.py` 体积，入口更清晰。
- 文件服务从导入时启动，改为应用生命周期内启动，减少重复启动副作用。
- workspace 路径校验、上传保存、URL 生成都集中到了同一处，后续更容易维护。
