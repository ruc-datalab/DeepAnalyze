import uvicorn

from backend_app.app import app
from backend_app.settings import settings


if __name__ == "__main__":
    print("🚀 启动后端服务...")
    print(f"   - API服务: http://localhost:{settings.backend_port}")
    print(f"   - 文件服务: {settings.file_server_base}")
    uvicorn.run(app, host=settings.backend_host, port=settings.backend_port)
