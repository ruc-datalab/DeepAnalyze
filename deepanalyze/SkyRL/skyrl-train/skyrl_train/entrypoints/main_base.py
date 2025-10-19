"""

uv run --isolated --extra vllm -m skyrl_train.entrypoints.main_base

"""

from ray.util.placement_group import placement_group, PlacementGroup

from transformers import AutoTokenizer
from skyrl_train.dataset import PromptDataset
from skyrl_train.utils import validate_cfg

from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.remote_inference_engine import (
    create_remote_inference_engines,
)
from skyrl_train.utils.utils import initialize_ray, get_ray_pg_ready_with_timeout
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.base import GeneratorInterface
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import ray

import os
import hydra
from loguru import logger
from skyrl_train.utils.tracking import Tracking
import multiprocessing as mp

# NOTE (sumanthrh): We use ray heavily and thus disable `fork` start method.
# forking within ray leads to undefined behaviour and often causes hard to debug
# memory leaks.  See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
# A common culprit is Pytorch dataloaders which use `fork` by default.
mp.set_start_method("spawn", force=True)

config_dir = str(Path(__file__).parent.parent / "config")
__all__ = ["BasePPOExp", "config_dir"]


def create_ray_wrapped_inference_engines_from_config(
    cfg: DictConfig, colocate_pg, tokenizer
):
    from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
        create_ray_wrapped_inference_engines,
    )

    return create_ray_wrapped_inference_engines(
        num_inference_engines=cfg.generator.num_inference_engines,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        model_dtype=cfg.generator.model_dtype,
        pretrain=cfg.trainer.policy.model.path,
        seed=cfg.trainer.seed,
        vllm_v1_disable_multiproc=cfg.generator.vllm_v1_disable_multiproc,
        enable_prefix_caching=cfg.generator.enable_prefix_caching,
        enforce_eager=cfg.generator.enforce_eager,
        max_model_len=cfg.generator.max_input_length
        + cfg.generator.sampling_params.max_generate_length,
        shared_pg=colocate_pg,
        gpu_memory_utilization=cfg.generator.gpu_memory_utilization,
        inference_engine_enable_sleep=cfg.trainer.placement.colocate_all,
        async_engine=cfg.generator.async_engine,
        max_num_batched_tokens=cfg.generator.max_num_batched_tokens,
        max_num_seqs=cfg.generator.max_num_seqs,
        sampling_params=get_sampling_params_for_backend(
            cfg.generator.backend, cfg.generator.sampling_params
        ),
        tokenizer=tokenizer,
        backend=cfg.generator.backend,
    )


def create_remote_inference_engines_from_config(cfg: DictConfig):
    # TODO(tgriggs): We may want a separate config for the model name in case it's different from the name used in the OpenAI API
    return create_remote_inference_engines(
        urls=cfg.generator.remote_inference_engine_urls,
        model_name=cfg.trainer.policy.model.path,
        engine_backend=cfg.generator.backend,
        tensor_parallel_size=cfg.generator.inference_engine_tensor_parallel_size,
        sampling_params=get_sampling_params_for_backend(
            cfg.generator.backend, cfg.generator.sampling_params
        ),
    )


class BasePPOExp:
    def __init__(self, cfg: DictConfig):
        """
        Initializes a PPO experiment.

        The `cfg` passed here will be the final config from Hydra, including CLI overrides.
        """
        self.cfg = cfg
        self.tokenizer = self.get_tokenizer()
        self.train_dataset = self.get_train_dataset()
        self.eval_dataset = self.get_eval_dataset()
        self.colocate_pg = self.get_colocate_pg()

    @staticmethod
    def get_cfg_as_str(dict_cfg: DictConfig) -> str:
        return OmegaConf.to_yaml(dict_cfg)

    def get_tokenizer(self, padding_side="left"):
        """Initializes a tokenizer for the given model."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.trainer.policy.model.path,
            trust_remote_code=True,
            use_fast=not self.cfg.trainer.disable_fast_tokenizer,
        )
        tokenizer.padding_side = padding_side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            PromptDataset: The training dataset.
        """
        prompts_dataset = PromptDataset(
            self.cfg.data.train_data,
            self.tokenizer,
            self.cfg.trainer.max_prompt_length,
            num_processors=8,
        )
        # make sure the dataset is large enough to train on
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            PromptDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = PromptDataset(
                self.cfg.data.val_data,
                self.tokenizer,
                self.cfg.trainer.max_prompt_length,
                num_processors=8,
            )
            return prompts_dataset
        return None

    def get_colocate_pg(self, timeout: int = 180) -> PlacementGroup:
        """Initializes a placement group for colocated training.

        A single placement group that packs all the inference engines together is created.

        Args:
            timeout (int): The timeout for the placement group to be ready.

        Returns:
            PlacementGroup: The placement group for colocated training.
        """
        if self.cfg.trainer.placement.colocate_all:
            pg = placement_group(
                [{"GPU": 1, "CPU": 1}]
                * self.cfg.generator.num_inference_engines
                * self.cfg.generator.inference_engine_tensor_parallel_size,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(pg, timeout=timeout)
            return pg
        else:
            return None

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """
        from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator

        return SkyRLGymGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator: GeneratorInterface,
        colocate_pg,
    ):
        """Initializes the trainer.

        Returns:
            RayPPOTrainer: The trainer.
        """
        return RayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def get_tracker(self):
        """Initializes the tracker for experiment tracking.

        Returns:
            Tracking: The tracker.
        """
        return Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            default_backend=self.cfg.trainer.logger,
            config=self.cfg,
        )

    def _setup_trainer(self):
        """Setup and return the trainer.

        Instantiates the trainer and all the associated models for training.

        Returns:
            RayPPOTrainer: The trainer.
        """
        logger.info(self.get_cfg_as_str(self.cfg))
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        if self.cfg.trainer.strategy == "deepspeed":
            from skyrl_train.workers.deepspeed.deepspeed_worker import (
                PolicyWorker,
                CriticWorker,
                RefWorker,
                RewardWorker,
            )
        elif self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import (
                PolicyWorker,
                CriticWorker,
                RefWorker,
                RewardWorker,
            )
        else:
            raise ValueError(f"Unknown strategy type: {self.cfg.trainer.strategy}")

        # NOTE (sumanthrh): Instantiate tracker before trainer init.
        # We have custom validation before this step to give better error messages.
        tracker = self.get_tracker()

        tokenizer = self.tokenizer
        if self.cfg.generator.run_engines_locally:
            inference_engines = create_ray_wrapped_inference_engines_from_config(
                self.cfg, self.colocate_pg, tokenizer
            )
        else:
            inference_engines = create_remote_inference_engines_from_config(self.cfg)

        inference_engine_client = InferenceEngineClient(inference_engines)

        generator: GeneratorInterface = self.get_generator(
            self.cfg, tokenizer, inference_engine_client
        )

        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )

        # Build the models
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker, RewardWorker)
        return trainer

    def run(self):
        trainer = self._setup_trainer()
        # Start the training loop
        trainer.train()


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
