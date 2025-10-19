# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from enum import StrEnum
from typing import Callable, List, Tuple, Union, Optional, Literal
from functools import wraps
import torch
import numpy as np

from omegaconf import DictConfig
from skyrl_train.training_batch import TrainingInputBatch
from jaxtyping import Float

import ray
from loguru import logger


# Import cloudpickle for function serialization
try:
    import cloudpickle
except ImportError:
    # Fallback to pickle if cloudpickle is not available
    import pickle as cloudpickle


# TODO (erictang000): unused right now, but will be useful as we add more algorithm support
class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def get_kl_controller(algorithm_cfg: DictConfig):
    if algorithm_cfg.kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=algorithm_cfg.kl_loss_coef)
    elif algorithm_cfg.kl_ctrl.type == "adaptive":
        if algorithm_cfg.kl_ctrl.horizon <= 0:
            raise ValueError(
                f"horizon must be larger than 0. Got {algorithm_cfg.kl_ctrl.horizon}"
            )
        return AdaptiveKLController(
            init_kl_coef=algorithm_cfg.kl_loss_coef,
            target=algorithm_cfg.kl_ctrl.kl_target,
            horizon=algorithm_cfg.kl_ctrl.horizon,
        )
    else:
        raise ValueError(f"Invalid KL controller type: {algorithm_cfg.kl_ctrl.type}")


def masked_mean(
    tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None
) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)


@torch.no_grad()
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    kl_estimator_type: str = "k3",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """
    if kl_estimator_type == "k1":
        kld = log_probs - log_probs_base
    elif kl_estimator_type == "abs":
        kld = (log_probs - log_probs_base).abs()
    elif kl_estimator_type == "k2":
        kld = 0.5 * (log_probs - log_probs_base).square()
    # J. Schulman. Approximating kl divergence, 2020.
    # URL http://joschu.net/blog/kl-approx.html.
    elif kl_estimator_type == "k3":
        kl = log_probs_base - log_probs
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        kld = torch.clamp(kld, min=-10, max=10)
    else:
        raise ValueError(f"Invalid KL estimator type: {kl_estimator_type}")

    if loss_mask is not None:
        kld = kld * loss_mask
    return kld


@torch.no_grad()
def normalize_advantages_dict(data: TrainingInputBatch) -> TrainingInputBatch:
    """Normalizes the advantages in the data batch.

    Expects:
        - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
        - `["response_mask"]`: Float[torch.Tensor, "batch_size seqlen"]
    """
    advantages: Float[torch.Tensor, "batch_size seqlen"] = data["advantages"]
    response_masks: Float[torch.Tensor, "batch_size seqlen"] = data["response_mask"]
    num_actions: float = response_masks.sum()
    # mean
    mean: float = advantages.mean()
    # std
    std: float = ((advantages - mean).pow(2) * response_masks).sum()
    rstd: float = (std / num_actions).clamp(min=1e-8).rsqrt()

    data["advantages"] = (advantages - mean) * rstd
    return data


def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError(
                "The sum of the mask is one, which can cause a division by zero."
            )
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def ppo_critic_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[float]]:

    if config.value_clip is not None:
        values_clipped = old_values + (values - old_values).clamp(
            -config.value_clip, config.value_clip
        )
        surr1 = (values_clipped - returns) ** 2
        surr2 = (values - returns) ** 2
        loss = torch.max(surr1, surr2)
        clipfrac = (
            masked_mean((surr1 > surr2).float(), loss_mask).mean().detach().item()
        )
    else:
        clipfrac = None
        loss = (values - returns) ** 2

    loss = masked_mean(loss, loss_mask, dim=-1).mean()
    return 0.5 * loss, clipfrac


# Shared registry actor class for both policy loss and advantage estimator registries
@ray.remote
class RegistryActor:
    """Shared Ray actor for managing function registries across processes."""

    def __init__(self):
        self.registry = {}

    def register(self, name: str, func_serialized: bytes):
        """Register a serialized function."""
        self.registry[name] = func_serialized

    def get(self, name: str):
        """Get a serialized function by name."""
        return self.registry.get(name)

    def list_available(self):
        """List all available function names."""
        return list(self.registry.keys())

    def unregister(self, name: str):
        """Unregister a function by name."""
        return self.registry.pop(name, None)


class BaseFunctionRegistry:
    """Base class for function registries with Ray actor synchronization."""

    # Subclasses should override these class attributes
    _actor_name = None
    _function_type = "Function"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._functions = {}
        cls._ray_actor = None
        cls._synced_to_actor = False

    @classmethod
    def _get_or_create_actor(cls):
        """Get or create the Ray actor for managing the registry using get_if_exists."""
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot create registry actor")

        if cls._ray_actor is None:
            # Use get_if_exists to create actor only if it doesn't exist
            cls._ray_actor = RegistryActor.options(
                name=cls._actor_name, get_if_exists=True
            ).remote()
        return cls._ray_actor

    @classmethod
    def _sync_local_to_actor(cls):
        """Sync all local functions to Ray actor."""
        if cls._synced_to_actor:
            return
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot sync with actor")

        try:
            actor = cls._get_or_create_actor()
            if actor is not None:
                for name, func in cls._functions.items():
                    func_serialized = cloudpickle.dumps(func)
                    ray.get(actor.register.remote(name, func_serialized))
                cls._synced_to_actor = True
        except Exception as e:
            logger.error(f"Error syncing {cls._function_type} to actor: {e}")
            raise e

    @classmethod
    def sync_with_actor(cls):
        """Sync local registry with Ray actor if Ray is available."""
        # Only try if Ray is initialized
        if not ray.is_initialized():
            raise Exception("Ray is not initialized, cannot sync with actor")

        # First check if the actor is still alive
        # NOTE(Charlie): This is mainly for unit tests, where we run multiple unit tests in the
        # same Python process, and each unit test has ray init/shutdown. This makes cls's attributes
        # outdated (e.g. the _ray_actor points to a stale actor in the previous ray session).
        try:
            _ = ray.get_actor(
                cls._actor_name
            )  # this raises exception if the actor is stale
        except ValueError:
            cls._ray_actor = None
            cls._synced_to_actor = False

        # First, sync our local functions to the actor
        cls._sync_local_to_actor()

        actor = cls._get_or_create_actor()
        if actor is None:
            return

        available = ray.get(actor.list_available.remote())

        # Sync any new functions from actor to local registry
        for name in available:
            if name not in cls._functions:
                func_serialized = ray.get(actor.get.remote(name))
                if func_serialized is not None:
                    # Deserialize the function
                    try:
                        func = cloudpickle.loads(func_serialized)
                        cls._functions[name] = func
                    except Exception as e:
                        # If deserialization fails, skip this function
                        logger.error(f"Error deserializing {name} from actor: {e}")
                        raise e

    @classmethod
    def register(cls, name: Union[str, StrEnum], func: Callable):
        """Register a function.

        If ray is initialized, this function will get or create a named ray actor (RegistryActor)
        for the registry, and sync the registry to the actor.

        If ray is not initalized, the function will be stored in the local registry only.

        To make sure all locally registered functions are available to all ray processes,
        call sync_with_actor() after ray.init().

        Args:
            name: Name of the function to register. Can be a string or a StrEnum.
            func: Function to register.

        Raises:
            ValueError: If the function is already registered.
        """
        # Convert enum to string if needed
        # note: StrEnum is not cloudpickleable: https://github.com/cloudpipe/cloudpickle/issues/558
        if isinstance(name, StrEnum):
            name = name.value

        if name in cls._functions:
            raise ValueError(f"{cls._function_type} '{name}' already registered")

        # Always store in local registry first
        cls._functions[name] = func

        # Try to sync with Ray actor if Ray is initialized
        if ray.is_initialized():
            actor = cls._get_or_create_actor()
            if actor is not None:
                # Serialize the function using cloudpickle
                func_serialized = cloudpickle.dumps(func)
                ray.get(actor.register.remote(name, func_serialized))

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a function by name.

        If ray is initialized, this function will first sync the local registry with the RegistryActor.
        Then it will return the function if it is found in the registry.

        Args:
            name: Name of the function to get. Can be a string or a StrEnum.

        Returns:
            The function if it is found in the registry.
        """
        # Try to sync with actor first if Ray is available
        if ray.is_initialized():
            cls.sync_with_actor()

        if name not in cls._functions:
            available = list(cls._functions.keys())
            raise ValueError(
                f"Unknown {cls._function_type.lower()} '{name}'. Available: {available}"
            )
        return cls._functions[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered functions."""
        # Try to sync with actor first if Ray is available
        if ray.is_initialized():
            cls.sync_with_actor()
        return list(cls._functions.keys())

    @classmethod
    def unregister(cls, name: Union[str, StrEnum]):
        """Unregister a function. Useful for testing."""
        # Convert enum to string if needed
        if isinstance(name, StrEnum):
            name = name.value

        # Try to sync with actor first to get any functions that might be in the actor but not local
        if ray.is_initialized():
            cls.sync_with_actor()

        # Track if we found the function anywhere
        found_locally = name in cls._functions
        found_in_actor = False

        # Remove from local registry if it exists
        if found_locally:
            del cls._functions[name]

        # Try to remove from Ray actor if Ray is available
        if ray.is_initialized():
            actor = cls._get_or_create_actor()
            if actor is not None:
                # Check if it exists in actor first
                available_in_actor = ray.get(actor.list_available.remote())
                if name in available_in_actor:
                    found_in_actor = True
                    ray.get(actor.unregister.remote(name))

        # Only raise error if the function wasn't found anywhere
        if not found_locally and not found_in_actor:
            raise ValueError(f"{cls._function_type} '{name}' not registered")

    @classmethod
    def reset(cls):
        """Resets the registry (useful for testing purposes)."""
        if ray.is_initialized() and cls._ray_actor is not None:
            try:
                actor = ray.get_actor(
                    cls._actor_name
                )  # this raises exception if the actor is stale
                ray.kill(actor)
            except ValueError:
                pass  # Actor may already be gone
        cls._functions.clear()
        cls._ray_actor = None
        cls._synced_to_actor = False


class AdvantageEstimator(StrEnum):
    GAE = "gae"
    GRPO = "grpo"
    RLOO = "rloo"
    REINFORCE_PP = "reinforce++"


class AdvantageEstimatorRegistry(BaseFunctionRegistry):
    """
    Registry for advantage estimator functions.

    This registry allows users to register custom advantage estimators without modifying
    the skyrl_train package. Custom estimators can be registered by calling
    AdvantageEstimatorRegistry.register() directly or by using the @register_advantage_estimator
    decorator.

    See examples/algorithm/custom_advantage_estimator for a simple example of how to
    register and use custom advantage estimators.
    """

    _actor_name = "advantage_estimator_registry"
    _function_type = "advantage estimator"


class PolicyLossType(StrEnum):
    REGULAR = "regular"
    DUAL_CLIP = "dual_clip"
    GSPO = "gspo"


class PolicyLossRegistry(BaseFunctionRegistry):
    """
    Registry for policy loss functions.

    This registry allows users to register custom policy loss functions without modifying
    the skyrl_train package. Custom functions can be registered by calling
    PolicyLossRegistry.register() directly or by using the @register_policy_loss
    decorator.

    See examples/algorithm/custom_policy_loss for a simple example of how to
    register and use custom policy loss functions.
    """

    _actor_name = "policy_loss_registry"
    _function_type = "policy loss"


def register_advantage_estimator(name: Union[str, AdvantageEstimator]):
    """Decorator to register an advantage estimator function."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        AdvantageEstimatorRegistry.register(name, wrapper)
        return wrapper

    return decorator


def register_policy_loss(name: Union[str, PolicyLossType]):
    """Decorator to register a policy loss function."""

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        PolicyLossRegistry.register(name, wrapper)
        return wrapper

    return decorator


def sync_registries():
    """Sync the registries with the ray actor once ray is initialized"""
    if not ray.is_initialized():
        raise ValueError("Ray is not initialized, cannot sync registries")
    PolicyLossRegistry.sync_with_actor()
    AdvantageEstimatorRegistry.sync_with_actor()
    logger.info("Synced registries to ray actor")


@register_policy_loss(PolicyLossType.REGULAR)
@register_policy_loss(PolicyLossType.DUAL_CLIP)
def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    assert config.policy_loss_type in [
        "regular",
        "dual_clip",
    ], "loss_type must be either 'regular' or 'dual_clip'"
    loss_reduction = config.loss_reduction
    assert loss_reduction in [
        "token_mean",
        "sequence_mean",
        "seq_mean_token_sum_norm",
    ], "loss_reduction must be either 'token_mean', 'sequence_mean', or 'seq_mean_token_sum_norm'"

    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_ratio = (
        masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    )
    clip_pg_losses1 = loss
    if config.policy_loss_type == "dual_clip":
        pg_losses3 = -advantages * config.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    if config.use_tis:
        from loguru import logger as logger_

        logger_.debug(f"Using TIS with dtype: {rollout_logprobs.dtype}")
        # Apply truncated importance sampling -> https://fengyao.notion.site/off-policy-rl
        tis_imp_ratio = torch.exp(old_log_probs - rollout_logprobs)
        tis_imp_ratio = torch.clamp(tis_imp_ratio, max=config.tis_imp_ratio_cap)
        loss = loss * tis_imp_ratio

    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len)
    return loss, clip_ratio


@register_policy_loss(PolicyLossType.GSPO)
def gspo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
    rollout_logprobs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    GSPO (Group Sequence Policy Optimization) policy loss function,
    as proposed in https://arxiv.org/abs/2507.18071.

    This implements sequence-level importance sampling instead of token-level importance sampling.
    The key difference is that importance weights are computed at the sequence level and then
    applied uniformly across all tokens in the sequence. This can lead to more stable training
    dynamics by reducing the variance in clipping behavior within sequences.

    The variant of GSPO used here is GSPO-token, a generalization which allows for token-level
    advantages [equations 14 and 15 in the paper].
    """
    # GSPO must use sequence_mean reduction
    loss_reduction = config.loss_reduction
    if loss_reduction != "sequence_mean":
        # The GSPO paper uses sequence_mean reduction; there's no reason
        # why a user couldn't use token_mean reduction, but
        # it's not clear whether it would be stable or not.
        from loguru import (
            logger as logger_,
        )  # have to do lazy import to avoid pickling error

        logger_.warning(
            f"With GSPO it's recommended to use 'sequence_mean' loss reduction; got {loss_reduction}"
        )

    # Compute log ratios
    log_ratio = log_probs - old_log_probs

    # Key GSPO innovation: sequence-level importance sampling
    # Instead of using per-token ratios, compute sequence-averaged ratios
    log_importance_weights = masked_mean(log_ratio, loss_mask, dim=-1).unsqueeze(-1)

    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_probs - sg[log_probs]
    # note: we put the addition at the end to avoid precision issues,
    # per https://github.com/volcengine/verl/pull/2775#discussion_r2241500280
    log_token_importance_weights = (
        log_probs - log_probs.detach() + log_importance_weights.detach()
    )
    # clip to avoid overflow
    log_token_importance_weights = torch.clamp(log_token_importance_weights, max=10)
    ratio = torch.exp(log_token_importance_weights)

    # Standard PPO surrogate objective with sequence-level importance weights
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)

    # Compute clipping ratio for monitoring
    clip_ratio = (
        masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    )

    loss = reduce_loss(loss, loss_mask, loss_reduction, config.max_seq_len)

    return loss, clip_ratio


def reduce_loss(
    loss: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    loss_reduction: Literal["token_mean", "sequence_mean", "seq_mean_token_sum_norm"],
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    if loss_reduction == "token_mean":
        # sum over *all* valid tokens, divide by total valid-token count
        loss = masked_mean(loss, loss_mask)
    elif loss_reduction == "sequence_mean":
        # per-sequence token-mean (dim=-1), then batch-mean
        loss = masked_mean(loss, loss_mask, dim=-1).mean()
    elif loss_reduction == "seq_mean_token_sum_norm":
        # per-sequence token-sum, normalized by the max sequence length, then batch mean
        # this is the Dr. GRPO loss reduction to avoid length bias by normalizing by a constant
        assert (
            max_seq_len is not None
        ), "max_seq_len must be provided for seq_mean_token_sum_norm loss reduction"
        # NOTE: max_seq_len is computed as cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length by default
        if loss_mask is not None:
            seq_losses = torch.sum(loss * loss_mask, dim=-1) / max_seq_len
        else:
            # If no mask, assume all tokens are valid
            seq_losses = torch.sum(loss, dim=-1) / max_seq_len
        loss = torch.mean(seq_losses)
    else:
        raise ValueError(f"Invalid loss reduction type: {loss_reduction}")
    return loss


# NOTE (erictang000): below ported from verl
@register_advantage_estimator(AdvantageEstimator.REINFORCE_PP)
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_advantage_estimator(AdvantageEstimator.RLOO)
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    This advantage estimator is also used in LOOP (https://arxiv.org/pdf/2502.01600),
    and was originally introduced in "Buy 4 REINFORCE Samples, Get a Baseline for Free!"
    (https://openreview.net/pdf?id=r1lgTGL5DE).

    Args:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    from loguru import logger as logger_

    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                factor = response_num / (response_num - 1)
                scores[i] = (scores[i] - id2mean[index[i]]) * factor
            else:
                # if there's only one response, set the advantage to 0
                logger_.warning(
                    f"Only one response for prompt index {index[i]}, setting advantage to 0"
                )
                scores[i] = 0.0
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_advantage_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: Float[torch.Tensor, "batch_size seqlen"],
    values: Float[torch.Tensor, "batch_size seqlen"],
    response_mask: Float[torch.Tensor, "batch_size seqlen"],
    gamma: float,
    lambd: float,
    **kwargs,
) -> Tuple[
    Float[torch.Tensor, "batch_size seqlen"], Float[torch.Tensor, "batch_size seqlen"]
]:
    """
    Compute advantage and return for GAE.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


@register_advantage_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    grpo_norm_by_std: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Expects:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)
        - epsilon: float
        - grpo_norm_by_std: bool

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    # this assumes response-level rewards
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if grpo_norm_by_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (
                    id2std[index[i]] + epsilon
                )
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_advantages_and_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    adv_estimator: AdvantageEstimator,
    config: DictConfig,
    values: Optional[torch.Tensor] = None,
    grpo_norm_by_std: bool = True,
    gamma=1.0,
    lambd=1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    estimator_func = AdvantageEstimatorRegistry.get(adv_estimator)

    return estimator_func(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        values=values,
        grpo_norm_by_std=grpo_norm_by_std,
        gamma=gamma,
        lambd=lambd,
        config=config,
    )
