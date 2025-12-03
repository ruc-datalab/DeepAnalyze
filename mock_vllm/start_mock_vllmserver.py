import http.server
import json
import time
from typing import Dict, Any

# 原始完整回复文本（不再按行拆分，后续逐字处理）
FULL_RESPONSE_TEXT = """我正在分析您提供的数据...

从数据中可以观察到以下趋势：
1. 第一季度销售额呈现稳步增长
2. 第二季度出现小幅回落
3. 第三、四季度恢复增长态势

生成的可视化图表已准备就绪，您可以通过链接下载查看。
这不是真实的vllm服务，仅用于测试连通性
This is not a real vLLM service; it is only used for connectivity testing."""

# 生成的文件信息
GENERATED_FILES = [
    {
        "name": "sales_trend.png",
        "url": "http://localhost:8100/generated/sales_trend.png"
    }
]

class VLLMHandler(http.server.BaseHTTPRequestHandler):
    # 禁用默认日志
    def log_message(self, format, *args):
        return

    def _send_sse_response(self):
        """发送逐字流式的SSE响应（核心修改）"""
        # 设置SSE标准响应头
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # 固定响应参数
        chunk_id = f"chatcmpl-{int(time.time() * 1000)}"
        created_time = int(time.time())
        model = "DeepAnalyze-8B"

        # 核心修改：将完整文本拆分为单个字符（逐字输出）
        char_list = list(FULL_RESPONSE_TEXT)
        for char in char_list:
            chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": char},  # 每次仅返回一个字符
                        "finish_reason": None
                    }
                ]
            }
            # SSE格式：data: {json}\n\n
            sse_line = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            self.wfile.write(sse_line.encode("utf-8"))
            self.wfile.flush()
            time.sleep(0.05)  # 逐字间隔（可调整：0.05秒/字，更流畅）

        # 发送结束块
        final_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ],
            "generated_files": GENERATED_FILES
        }
        final_sse_line = f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        self.wfile.write(final_sse_line.encode("utf-8"))
        self.wfile.flush()

        # 发送SSE结束标志
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_json_response(self, status_code: int, content: Dict[str, Any]):
        """发送非流式JSON响应"""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.end_headers()
        self.wfile.write(json.dumps(content, ensure_ascii=False).encode("utf-8"))

    def do_POST(self) -> None:
        """处理POST请求"""
        if self.path == "/v1/chat/completions":
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                request_body = self.rfile.read(content_length).decode('utf-8')
                request_data = json.loads(request_body)
                stream = request_data.get('stream', False)

                if stream:
                    self._send_sse_response()
                else:
                    # 非流式返回完整文本
                    full_response = {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "DeepAnalyze-8B",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": FULL_RESPONSE_TEXT,
                                    "files": GENERATED_FILES
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "generated_files": GENERATED_FILES
                    }
                    self._send_json_response(200, full_response)

            except Exception as e:
                self._send_json_response(500, {"error": str(e)})

        elif self.path == "/v1/models":
            models_response = {
                "object": "list",
                "data": [
                    {
                        "id": "DeepAnalyze-8B",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "deepanalyze"
                    }
                ]
            }
            self._send_json_response(200, models_response)

        else:
            self._send_json_response(404, {"error": "Endpoint not found"})

    def do_GET(self) -> None:
        """处理GET请求"""
        if self.path == "/health":
            self._send_json_response(200, {"status": "healthy", "timestamp": int(time.time())})
        elif self.path == "/v1/models":
            self.do_POST()
        else:
            self._send_json_response(404, {"error": "Endpoint not found"})

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """启动模拟vLLM服务器"""
    server = http.server.ThreadingHTTPServer((host, port), VLLMHandler)
    print(f"✅ 模拟vLLM服务器启动成功（逐字流式输出）")
    print(f"   - 地址: http://{host}:{port}")
    print(f"   - 逐字间隔: 0.05秒/字符（可修改time.sleep值调整）")
    print(f"   - 按 Ctrl+C 停止服务器\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 服务器正在停止...")
        server.shutdown()
        server.server_close()
        print("✅ 服务器已停止")

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=8000)