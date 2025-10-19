from .verl_backend import VeRLGeneratorInput, VeRLBackend, VeRLGeneratorOutput
from ..base import register_backend, BackendSpec

register_backend(
    "verl",
    BackendSpec(
        infer_backend_cls=VeRLBackend,
        generator_output_cls=VeRLGeneratorOutput,
        generator_input_cls=VeRLGeneratorInput,
    )
)

__all__ = [
    "VeRLGeneratorInput",
    "VeRLBackend",
    "VeRLGeneratorOutput"
]