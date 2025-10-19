import pytest
import ray
from loguru import logger
from functools import lru_cache
from skyrl_train.utils.utils import peer_access_supported


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


@pytest.fixture
def ray_init_fixture():
    if ray.is_initialized():
        ray.shutdown()
    env_vars = {}
    if not peer_access_supported(max_num_gpus_per_node=2):
        log_once("Disabling NCCL P2P for CI environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    ray.init(runtime_env={"env_vars": env_vars})
    yield
    # call ray shutdown after a test regardless
    ray.shutdown()
