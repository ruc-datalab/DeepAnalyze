#!/usr/bin/env python
"""Script to test multi-node NCCL communication with Ray

This script is useful to debug if multi-node communication works and if the right network interfaces (eg: RDMA) is being used.
"""
import os
import sys
import ray
import torch
import torch.distributed as dist
from skyrl_train.utils.utils import initialize_ray
from ray.util.placement_group import placement_group
from omegaconf import OmegaConf
from loguru import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-nodes", type=int, default=2)
parser.add_argument("--master-port", type=int, default=12355)
args = parser.parse_args()


def log_versions(rank):
    logger.info(
        f"{rank} Python version: {sys.version} | "
        f"PyTorch version: {torch.__version__} | "
        f"CUDA available: {torch.cuda.is_available()} | "
        f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'} | "
        f"Ray version: {ray.__version__}"
    )


@ray.remote(num_gpus=1)
class PyTorchDistActor:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.node_ip = self.get_node_ip()
        self.world_size = world_size

        logger.info(
            f"Rank {self.rank} initialized with: node_ip={self.node_ip}, world_size={self.world_size}"
        )
        log_versions(rank)

    def get_node_ip(self):
        return ray.util.get_node_ip_address()

    def set_master_node_addr(self, master_addr, master_port):
        os.environ["MASTER_ADDR"] = str(master_addr)
        os.environ["MASTER_PORT"] = str(master_port)

    def run(self):
        # Initialize the process group
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self.world_size,
            rank=self.rank,
        )

        # Create a dictionary to broadcast
        if self.rank == 0:
            data = {"message": "Hello from rank 0", "value": 123}
        else:
            data = None

        objects = [data]

        # Broadcast the object list
        dist.broadcast_object_list(objects, src=0)

        if self.rank != 0:
            logger.info(f"Rank {self.rank} received data: {objects[0]}")
            assert objects[0] == {
                "message": "Hello from rank 0",
                "value": 123,
            }, f"Data received at rank {self.rank} is not correct, got {objects[0]}"
        else:
            logger.info(f"Rank {self.rank} sent: {data}")

        dist.barrier()
        # Clean up
        dist.destroy_process_group()
        return objects[0]


if __name__ == "__main__":
    # Initialize Ray
    cfg = OmegaConf.create()
    cfg.generator = OmegaConf.create()
    cfg.generator.backend = "vllm"
    initialize_ray(cfg)

    total_ranks = args.num_nodes
    actors = []

    # Create placement group for distributed training
    pg = placement_group(
        bundles=[{"GPU": 1, "CPU": 1}] * total_ranks, strategy="STRICT_SPREAD"
    )
    ray.get(pg.ready())

    # Create actors
    for rank in range(total_ranks):
        actor = PyTorchDistActor.options(placement_group=pg).remote(rank, total_ranks)
        actors.append(actor)

    # set master node addr
    master_addr = ray.get(actors[0].get_node_ip.remote())
    master_port = args.master_port
    ray.get(
        [
            actor.set_master_node_addr.remote(master_addr, master_port)
            for actor in actors
        ]
    )

    # Run the distributed operation
    results = ray.get([actor.run.remote() for actor in actors])
    print("All results:", results)
