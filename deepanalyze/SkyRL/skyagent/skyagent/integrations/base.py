from abc import ABC, abstractmethod
from typing import Any, List, Type, Dict
import importlib


class AsyncInferBackend(ABC):
    @abstractmethod
    async def async_generate_prompts(
        self, prompts: Any, sampling_params: Any, **kwargs
    ) -> List[str]:
        """Generate outputs for a list of prompts."""
        pass

    @abstractmethod
    async def async_generate_ids(
        self, input_ids: Any, sampling_params: Any, **kwargs
    ) -> List[str]:
        """Generate outputs for a list of input_ids."""
        pass


class GeneratorOutput(ABC):
    def __init__(self, result: Any):
        pass


class GeneratorInput(ABC):
    def __init__(self, input_batch: Any):
        pass


class BackendSpec:
    def __init__(
        self,
        infer_backend_cls: Type[AsyncInferBackend],
        generator_output_cls: Type[GeneratorOutput],
        generator_input_cls: Type[GeneratorInput],
    ):
        self.infer_backend_cls = infer_backend_cls
        self.generator_output_cls = generator_output_cls
        self.generator_input_cls = generator_input_cls


def build_backend(name: str, **kwargs):
    spec = BACKEND_REGISTRY.get(name)
    if not spec:
        raise ValueError(f"Unknown backend: {name}")
    infer_backend = spec.infer_backend_cls(**kwargs)
    return infer_backend


def build_generator_output(name: str, result: Any, **kwargs):
    spec = BACKEND_REGISTRY.get(name)
    if not spec:
        raise ValueError(f"Unknown backend: {name}")
    return spec.generator_output_cls(result=result, **kwargs)


def build_generator_input(name: str, input_batch: Any, **kwargs):
    spec = BACKEND_REGISTRY.get(name)
    if not spec:
        raise ValueError(f"Unknown backend: {name}")
    return spec.generator_input_cls(input_batch=input_batch, **kwargs)


def _import_object(path: str):
    """Dynamically import a class or function from a module path."""
    module_path, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


BACKEND_REGISTRY: Dict[str, BackendSpec] = {}


def register_backend(name: str, spec: BackendSpec):
    BACKEND_REGISTRY[name] = spec
