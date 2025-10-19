import hydra
from omegaconf import DictConfig
from skyrl_train.generators.base import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray

from skyagent import AutoAgentRunner


class CodeActGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        llm_endpoint_client: InferenceEngineClient,
        tokenizer,
    ):
        self.agent_generator = AutoAgentRunner.from_task(
            generator_cfg.task, infer_engine=llm_endpoint_client, tokenizer=tokenizer
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        return await self.agent_generator.run(input_batch)


class CodeActPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, llm_endpoint_client):
        generator = CodeActGenerator(
            generator_cfg=cfg.generator,
            llm_endpoint_client=llm_endpoint_client,
            tokenizer=tokenizer,
        )
        return generator


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = CodeActPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    task = skyrl_entrypoint.remote(cfg)
    try:
        ray.get(task)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
        ray.cancel(task)
        raise


if __name__ == "__main__":
    main()
