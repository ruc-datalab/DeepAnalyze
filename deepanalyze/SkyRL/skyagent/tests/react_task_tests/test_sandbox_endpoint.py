import requests, json
import os
import threading
import time

results = {"success": 0, "timeout": 0, "error": 0}
results_lock = threading.Lock()


def send_request(json_data, request_id):
    response = None
    try:
        url = f"{os.environ.get('SANDBOX_FUSION_URL')}/run_code"
        start_time = time.time()
        response = requests.post(url, json=json_data, timeout=30)  # Increased timeout
        end_time = time.time()
        result = response.json()
        print(
            f"Request {request_id}: Status {response.status_code}, Time: {end_time - start_time:.2f}s"
        )

        with results_lock:
            results["success"] += 1
        return result
    except Exception as e:
        print(f"Request {request_id}: Error - {str(e)}")
        with results_lock:
            results["error"] += 1
        return {"error": str(e)}


json_data = {
    "compile_timeout": 10,
    "run_timeout": 50,
    "code": "from scipy.optimize import fsolve; print(fsolve(lambda x: np.sin(x), 0))",
    "language": "python",
    "files": {},
    "fetch_files": ["string"],
}

# Launch 1000 concurrent requests
threads = []
start_time = time.time()

for i in range(100):
    thread = threading.Thread(target=send_request, args=(json_data, i))
    threads.append(thread)
    thread.start()

# Wait for all threads to complete
for thread in threads:
    thread.join()

end_time = time.time()
print(f"All 10 requests completed in {end_time - start_time:.2f} seconds")
print(
    f"Results - Success: {results['success']}, Timeout: {results['timeout']}, Error: {results['error']}"
)
