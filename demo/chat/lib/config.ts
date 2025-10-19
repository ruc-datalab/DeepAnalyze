// API配置
export const API_CONFIG = {
  // 后端API基础地址
  BACKEND_BASE_URL:
    process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8200",

  // 静态文件服务基础地址（前端可配置下载/预览所使用的文件基址）
  // 例如：http://<server-ip>:8100 或 https://cdn.example.com
  FILE_SERVER_BASE:
    process.env.NEXT_PUBLIC_FILE_SERVER_BASE || "http://localhost:8100",

  // 模拟AI API地址
  AI_API_BASE_URL:
    process.env.NEXT_PUBLIC_AI_API_URL || "http://localhost:8000",

  // WebSocket地址
  WEBSOCKET_URL: process.env.NEXT_PUBLIC_WEBSOCKET_URL || "ws://localhost:8001",

  // API端点
  ENDPOINTS: {
    // 聊天
    CHAT_COMPLETIONS: "/chat/completions",

    // 文件管理
    WORKSPACE_FILES: "/workspace/files",
    WORKSPACE_TREE: "/workspace/tree",
    WORKSPACE_UPLOAD: "/workspace/upload",
    WORKSPACE_CLEAR: "/workspace/clear",
    WORKSPACE_DELETE_FILE: "/workspace/file",
    WORKSPACE_UPLOAD_TO: "/workspace/upload-to",
    WORKSPACE_DELETE_DIR: "/workspace/dir",

    // 代码执行
    EXECUTE_CODE: "/execute",

    // 导出报告
    EXPORT_REPORT: "/export/report",
  },
};

// 构建完整的API URL
export const buildApiUrl = (
  endpoint: string,
  baseUrl: string = API_CONFIG.BACKEND_BASE_URL
) => {
  return `${baseUrl}${endpoint}`;
};

// 预定义的API URLs
export const API_URLS = {
  // 后端服务
  WORKSPACE_FILES: buildApiUrl(API_CONFIG.ENDPOINTS.WORKSPACE_FILES),
  WORKSPACE_TREE: buildApiUrl(API_CONFIG.ENDPOINTS.WORKSPACE_TREE),
  WORKSPACE_UPLOAD: buildApiUrl(API_CONFIG.ENDPOINTS.WORKSPACE_UPLOAD),
  WORKSPACE_CLEAR: buildApiUrl(API_CONFIG.ENDPOINTS.WORKSPACE_CLEAR),
  WORKSPACE_DELETE_FILE: buildApiUrl(
    API_CONFIG.ENDPOINTS.WORKSPACE_DELETE_FILE
  ),
  WORKSPACE_UPLOAD_TO: buildApiUrl(API_CONFIG.ENDPOINTS.WORKSPACE_UPLOAD_TO),
  WORKSPACE_DELETE_DIR: buildApiUrl(API_CONFIG.ENDPOINTS.WORKSPACE_DELETE_DIR),
  EXECUTE_CODE: buildApiUrl(API_CONFIG.ENDPOINTS.EXECUTE_CODE),
  EXPORT_REPORT: buildApiUrl(API_CONFIG.ENDPOINTS.EXPORT_REPORT),

  // AI服务
  CHAT_COMPLETIONS: buildApiUrl(API_CONFIG.ENDPOINTS.CHAT_COMPLETIONS),
};
