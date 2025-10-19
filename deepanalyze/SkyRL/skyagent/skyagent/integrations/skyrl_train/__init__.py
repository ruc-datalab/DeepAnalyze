from .skyrl_train_backend import SkyRLBackend, SkyRLGeneratorInput, SkyRLGeneratorOutput
from ..base import register_backend, BackendSpec

register_backend(
    "skyrl-train",
    BackendSpec(
        infer_backend_cls=SkyRLBackend,
        generator_output_cls=SkyRLGeneratorOutput,
        generator_input_cls=SkyRLGeneratorInput,
    )
)