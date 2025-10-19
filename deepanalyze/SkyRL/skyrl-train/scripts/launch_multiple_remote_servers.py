"""
Helper script to launch multiple vLLM remote servers in a Ray cluster.

The main purpose is to be able to test out the remote endpoint functionality in SkyRL easily.

Example usage:
uv run --isolated --frozen --extra vllm scripts/launch_multiple_remote_servers.py --model-path Qwen/Qwen2.5-1.5B-Instruct --tp-size 2 --num-replicas 2 --gpu-memory-utilization 0.9 > my_server_logs.log 2>&1
"""

import argparse
import os
import ray
import subprocess
from ray.util.placement_group import placement_group, PlacementGroupSchedulingStrategy
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.utils.utils import initialize_ray
import socket
import signal
import time
import sys
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
import psutil


def get_free_port():
    """Get a free port on the current node."""
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def wait_for_server(host: str, port: int, timeout_seconds: int = 180) -> bool:
    """Wait for server to be ready by checking if we can connect."""
    start_time = time.time()
    print(f"Waiting for server at {host}:{port}...")

    while True:
        try:
            with socket.socket() as sock:
                sock.settimeout(1)
                sock.connect((host, port))
                return True
        except (socket.timeout, ConnectionRefusedError):
            pass

        if time.time() - start_time >= timeout_seconds:
            print(f"Timeout waiting for server at {host}:{port}")
            return False

        time.sleep(0.5)


@ray.remote
class VLLMServer:
    def __init__(
        self,
        model_path,
        tp_size,
        gpu_memory_utilization=0.7,
        distributed_executor_backend="mp",
        quiet=False,
    ):
        self.model_path = model_path
        self.tp_size = tp_size
        self.port = get_free_port()
        self.host = ray._private.services.get_node_ip_address().strip("[]")
        self.gpu_memory_utilization = gpu_memory_utilization
        self.distributed_executor_backend = distributed_executor_backend
        self.pid = None
        self.quiet = quiet
        self.log_dir = Path(tempfile.gettempdir()) / "skyrl_vllm_server_logs"
        self.log_dir.mkdir(exist_ok=True)

    def start(self):
        """Start the VLLM server process."""
        cmd = [
            "python",
            "-m",
            "skyrl_train.inference_engines.vllm.vllm_server",
            "--model",
            self.model_path,
            "--enforce-eager",
            "--tensor-parallel-size",
            str(self.tp_size),
            "--seed",
            "42",
            "--distributed-executor-backend",
            self.distributed_executor_backend,
            "--enable-prefix-caching",
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--enable-sleep-mode",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--worker-extension-cls",
            "skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
        ]

        # Set CUDA_VISIBLE_DEVICES based on Ray's GPU allocation
        gpu_ids = ray.get_gpu_ids()
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Create log files for this server
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_prefix = f"vllm_server_{self.host}_{self.port}_{timestamp}"
        stdout_log = (
            self.log_dir / f"{log_prefix}_stdout.log" if self.quiet else "sys.stdout"
        )
        stderr_log = (
            self.log_dir / f"{log_prefix}_stderr.log" if self.quiet else "sys.stderr"
        )

        # Configure subprocess output
        if self.quiet:
            stdout = open(stdout_log, "w")
            stderr = open(stderr_log, "w")
        else:
            stdout = sys.stdout
            stderr = sys.stderr

        # Start the server process
        process = subprocess.Popen(cmd, stdout=stdout, stderr=stderr)
        self.pid = process.pid
        return process.pid, f"{self.host}:{self.port}", str(stdout_log), str(stderr_log)

    def kill(self):
        if self.pid is not None:
            try:
                print(f"Killing server {self.pid}...")
                # First kill child processes
                print("Killing child processes...")
                try:
                    parent = psutil.Process(self.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        print(f"Killing child process {child.pid}...")
                        try:
                            child.terminate()
                        except psutil.NoSuchProcess:
                            pass
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # Process already terminated or not accessible

                # Then try SIGTERM on main process
                os.kill(self.pid, signal.SIGTERM)
                # Wait a bit for graceful shutdown
                time.sleep(3)

                # If still running, force kill with SIGKILL
                try:
                    os.kill(self.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already terminated
            except ProcessLookupError:
                pass  # Process already terminated


def wait_for_all_servers(
    server_addresses: List[Tuple[str, int]], timeout_seconds: int = 180
) -> bool:
    """Wait for all servers to be ready using thread pool."""
    with ThreadPoolExecutor(max_workers=len(server_addresses)) as executor:
        futures = [
            executor.submit(wait_for_server, host, port, timeout_seconds)
            for host, port in server_addresses
        ]
        results = [f.result() for f in futures]
    return all(results)


def print_server_info(server_infos: List[Tuple[int, str, str, str]], args):
    for i, (pid, addr, stdout_log, stderr_log) in enumerate(server_infos):
        print(f"Server {i+1}:")
        print(f"TP: {args.tp_size}")
        print(f"  PID: {pid}")
        print(f"  Address: {addr}")
        print(f"  Logs: {stdout_log}, {stderr_log}")


def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple VLLM servers in a Ray cluster"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the model"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        required=True,
        help="Tensor parallel size for each replica",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        required=True,
        help="Number of server replicas to launch",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization",
    )
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds for server readiness",
    )
    args = parser.parse_args()

    cfg = OmegaConf.create()
    cfg.generator = OmegaConf.create()
    cfg.generator.backend = "vllm"
    cfg.generator.weight_sync_backend = "nccl"
    # Initialize Ray
    initialize_ray(cfg)

    # Create placement group for the server
    bundles = [[{"GPU": args.tp_size, "CPU": 1}] for _ in range(args.num_replicas)]
    placement_groups = [placement_group(bundle) for bundle in bundles]
    for pg in placement_groups:
        print("Waiting for Ray placement group to be ready...")
        get_ray_pg_ready_with_timeout(pg, timeout=180)

    # Launch servers
    server_actors = []
    for i in range(args.num_replicas):
        pg = placement_groups[i]
        server = VLLMServer.options(
            num_gpus=args.tp_size,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
            ),
        ).remote(
            model_path=args.model_path,
            tp_size=args.tp_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            quiet=args.quiet,
        )
        server_actors.append(server)

    # Start all servers
    rets = ray.get([server.start.remote() for server in server_actors])

    print(f"Launched {args.num_replicas} VLLM servers:")
    server_addresses = []
    server_infos = []
    for i, ret in enumerate(rets):
        pid, addr, stdout_log, stderr_log = ret
        host, port = addr.split(":")
        server_addresses.append((host, int(port)))
        server_infos.append((pid, addr, stdout_log, stderr_log))

    print_server_info(server_infos, args)

    try:
        print("\nWaiting for all servers to be ready...")
        if wait_for_all_servers(server_addresses, args.timeout):
            print("\nAll servers are ready!")
            print_server_info(server_infos, args)
        else:
            print(
                "\nSome servers failed to start within timeout. Check logs for details."
            )
            print("Shutting down...")
            ray.get([server.kill.remote() for server in server_actors])
            ray.shutdown()
            return

        print("\nPress Ctrl+C to shut down all the servers")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers. Please wait for a few seconds...")
        ray.get([server.kill.remote() for server in server_actors])
        ray.shutdown()


if __name__ == "__main__":
    main()
