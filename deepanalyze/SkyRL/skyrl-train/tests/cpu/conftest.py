import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    """Initialize Ray once for the entire test session."""
    if not ray.is_initialized():
        ray.init()
    yield
    if ray.is_initialized():
        ray.shutdown()
