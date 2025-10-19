from typing import List, Dict, Any, Union, Callable, Optional, Tuple, TypedDict
from enum import Enum
import ray
from skyrl_train.workers.worker import PPORayActorGroup
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import os
import shutil
from loguru import logger
import glob
import json
import numpy as np
from collections import defaultdict
from skyrl_train.generators.utils import (
    get_metrics_from_generator_output,
    concatenate_generator_outputs,
)
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from transformers import AutoTokenizer
from pathlib import Path

BasicType = Union[int, float, str, bool, type(None)]

GLOBAL_STEP_PREFIX = "global_step_"


class ResumeMode(Enum):
    NONE = "none"
    LATEST = "latest"
    FROM_PATH = "from_path"

    @classmethod
    def _missing_(cls, value):
        if value is None:
            return cls.NONE
        return super()._missing_(value)


def get_node_ids(
    policy_model: PPORayActorGroup,
    critic_model: Optional[PPORayActorGroup],
    ref_model: Optional[PPORayActorGroup],
) -> List[str]:
    """Get the node ids of the policy, critic, and ref models.

    Args:
        policy_model: Policy model actor group
        critic_model: Critic model actor group (Optional)
        ref_model: Ref model actor group (Optional)
    """
    policy_node_ids: List[str] = ray.get(
        policy_model.async_run_ray_method("pass_through", "get_ray_node_id")
    )
    if critic_model is not None:
        critic_node_ids: List[str] = ray.get(
            critic_model.async_run_ray_method("pass_through", "get_ray_node_id")
        )
    else:
        critic_node_ids = []
    if ref_model is not None:
        ref_node_ids: List[str] = ray.get(
            ref_model.async_run_ray_method("pass_through", "get_ray_node_id")
        )
    else:
        ref_node_ids = []

    unique_node_ids = list(set(policy_node_ids + critic_node_ids + ref_node_ids))
    return unique_node_ids


def run_on_each_node(node_ids: List[str], fn: Callable, *args, **kwargs):
    """Simple helper to run a function on each node.

    Args:
        node_ids: List of node ids to run the function on
        fn: Function to run
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
    """
    node_ids = list(set(node_ids))
    task = ray.remote(num_cpus=0.25)(fn)
    refs = []

    for node_id in node_ids:
        node_task = task.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            )
        )
        refs.append(node_task.remote(*args, **kwargs))

    return ray.get(refs)


def extract_step_from_path(path: str) -> int:
    basename = os.path.basename(path)
    if basename.startswith(GLOBAL_STEP_PREFIX):
        return int(basename.split(GLOBAL_STEP_PREFIX)[1])
    return -1


def cleanup_old_checkpoints(
    ckpt_path: str, max_ckpts_to_keep: int, current_global_step: int
):
    """Remove old global_step directories, keeping only the most recent max_ckpts_to_keep"""

    if max_ckpts_to_keep < 0:
        return

    # Find all global_step directories
    pattern = os.path.join(ckpt_path, f"{GLOBAL_STEP_PREFIX}*")
    checkpoint_dirs = glob.glob(pattern)

    # track only valid checkpoints - id <= current_global_step
    checkpoint_dirs = [
        dir
        for dir in checkpoint_dirs
        if extract_step_from_path(dir) <= current_global_step
    ]

    if len(checkpoint_dirs) <= max_ckpts_to_keep:
        logger.info(
            f"Only {len(checkpoint_dirs)} checkpoints found for the current run, no need to cleanup"
        )
        return

    checkpoint_dirs.sort(key=extract_step_from_path, reverse=True)

    logger.info(
        f"Found {len(checkpoint_dirs)} checkpoints for the current run, keeping only the most recent {max_ckpts_to_keep}"
    )

    # Remove old checkpoints
    for old_dir in checkpoint_dirs[max_ckpts_to_keep:]:
        try:
            shutil.rmtree(old_dir)
            logger.info(f"Removed old checkpoint: {old_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {old_dir}: {e}")


def validate_consistency_for_latest_checkpoint(
    root_ckpt_folder: str,
    ckpt_iteration: int,
    checkpoint_path: str,
    latest_checkpoint_file: str,
    save_interval: int,
):
    """Validate that the checkpoint folder is consistent with the latest checkpoint file.

    Asserts that the folder with the highest global step is the latest checkpoint tracked by `latest_checkpoint_file`.
    Otherwise, the folder state is inconsistent and the user should delete other checkpoints.
    """
    global_step_values = [
        extract_step_from_path(p)
        for p in os.listdir(root_ckpt_folder)
        if p.startswith(GLOBAL_STEP_PREFIX)
    ]
    max_global_step_in_folder = max(global_step_values)
    # NOTE (sumanthrh): We allow a checkpoint folder to be `save_interval` steps ahead of the latest checkpoint in `latest_checkpoint_file`. This is because the last checkpoint can be an incomplete checkpoint.
    if max_global_step_in_folder - ckpt_iteration > save_interval:
        max_global_step_in_folder_path = os.path.join(
            root_ckpt_folder, f"{GLOBAL_STEP_PREFIX}{max_global_step_in_folder}"
        )
        raise ValueError(
            f"Inconsistent checkpoint folder. Latest checkpoint file {latest_checkpoint_file} points to {ckpt_iteration}, but the folder has checkpoints with higher global step - Found global steps {max_global_step_in_folder_path}. This is likely because checkpoint {max_global_step_in_folder_path} was created in a previous run while the latest run is at {checkpoint_path}. Please delete/move checkpoints from older runs and try again."
        )


def sanitize_data_source(data_source: str) -> str:
    """Sanitize data source name for use in file paths."""
    if data_source is None:
        return "unknown"
    return data_source.replace("/", "_")


def calculate_per_dataset_metrics(
    concat_generator_outputs: GeneratorOutput,
    concat_uids: List[str],
    concat_data_sources: List[str],
    n_samples_per_prompt: int,
) -> Dict[str, float]:
    """Calculate metrics per data source."""
    eval_metrics = {}

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Calculate metrics for each data source
    for data_source, indices in data_source_indices.items():
        # Extract subset for this data source
        subset_generator_output = {
            key: [value[i] for i in indices]
            for key, value in concat_generator_outputs.items()
            if isinstance(value, list)
        }
        subset_uids = [concat_uids[i] for i in indices]

        # Calculate metrics for this subset
        avg_score, pass_at_n = get_metrics_from_generator_output(
            subset_generator_output, subset_uids
        )

        # Add to eval metrics with proper naming
        sanitized_data_source = sanitize_data_source(data_source)
        eval_metrics[f"eval/{sanitized_data_source}/avg_score"] = avg_score
        eval_metrics[f"eval/{sanitized_data_source}/pass_at_{n_samples_per_prompt}"] = (
            pass_at_n
        )

    return eval_metrics


def dump_per_dataset_eval_results(
    dump_dir_path: Path,
    tokenizer: AutoTokenizer,
    concat_generator_outputs: GeneratorOutput,
    concat_data_sources: List[str],
    concat_all_envs: List[str],
    concat_env_extras: List[Dict[str, Any]],
    eval_metrics: Dict[str, float],
):
    """Dump evaluation results per dataset and overall aggregated results."""

    # Prepare common data
    input_prompts = [
        tokenizer.decode(prompt)
        for prompt in concat_generator_outputs["prompt_token_ids"]
    ]
    output_responses = [
        tokenizer.decode(response)
        for response in concat_generator_outputs["response_ids"]
    ]

    # Group indices by data source
    data_source_indices = {}
    for i, data_source in enumerate(concat_data_sources):
        if data_source is None:
            data_source = "unknown"
        if data_source not in data_source_indices:
            data_source_indices[data_source] = []
        data_source_indices[data_source].append(i)

    # Dump per-dataset files
    for data_source, indices in data_source_indices.items():
        sanitized_data_source = sanitize_data_source(data_source)
        filename = dump_dir_path / f"{sanitized_data_source}.jsonl"

        with open(filename, "w") as f:
            for i in indices:
                entry = {
                    "input_prompt": input_prompts[i],
                    "output_response": output_responses[i],
                    "score": concat_generator_outputs["rewards"][i],
                    "stop_reason": concat_generator_outputs.get(
                        "stop_reasons", [None] * len(input_prompts)
                    )[i],
                    "env_class": concat_all_envs[i],
                    "env_extras": concat_env_extras[i],
                    "data_source": data_source,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped eval data for {data_source} to {filename}")

    # Dump aggregated results file
    aggregated_filename = dump_dir_path / "aggregated_results.jsonl"
    with open(aggregated_filename, "w") as f:
        f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")

    print(f"Dumped aggregated eval metrics to {aggregated_filename}")


class DynamicSamplingState(TypedDict, total=False):
    """Schema for dynamic sampling state dictionary.

    Fields:
        sample_batch_count: Counter for the number of sample batches processed
        collected_generator_output: Accumulated generator output (filter strategy only)
        collected_uids: Accumulated UIDs (filter strategy only)
        num_prompts_in_batch: Number of prompts collected so far (filter strategy only)
    """

    sample_batch_count: int
    collected_generator_output: Optional[GeneratorOutput]
    collected_uids: Optional[List[str]]
    num_prompts_in_batch: Optional[int]


def handle_dynamic_sampling(
    generator_output: GeneratorOutput,
    uids: List[str],
    sampling_config: Dict[str, Any],
    collected_state: Optional[DynamicSamplingState] = None,
) -> Tuple[GeneratorOutput, List[str], bool, Optional[DynamicSamplingState]]:
    """
    Handle dynamic sampling with different strategies (filter, replace).

    filter (used in DAPO) - filter out groups with std == 0 and group size > 1 and resample until we have enough prompts
    replace (used in POLARIS, WebSailor) - replace bad (std == 0) samples with good (std > 0) samples

    Args:
        generator_output: Current batch generator output
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
        collected_state: State for accumulating data across batches (for filter strategy)

    Returns:
        Tuple of (processed_generator_output, processed_uids, keep_sampling, updated_state)
    """
    sampling_type = sampling_config.get("type", None)

    if sampling_type is None:
        return generator_output, uids, False, None

    if sampling_type == "replace":
        # For "replace" strategy, the collected state is not used.
        processed_output, processed_uids, keep_sampling = handle_replace_sampling(
            generator_output, uids, sampling_config
        )
        return processed_output, processed_uids, keep_sampling, collected_state
    elif sampling_type == "filter":
        # For filter strategies, accumulate the generator output and UIDs across batches in collected_state if we are sampling repeatedly.
        return handle_filter_sampling(
            generator_output, uids, sampling_config, collected_state
        )
    else:
        raise ValueError(f"Invalid dynamic sampling type: {sampling_type}")


def handle_replace_sampling(
    generator_output: GeneratorOutput, uids: List[str], sampling_config: Dict[str, Any]
) -> Tuple[GeneratorOutput, List[str], bool]:
    """
    Handle replace sampling strategy based on POLARIS implementation

    Reference: https://github.com/ChenxinAn-fdu/POLARIS/blob/8c82adb16b8e45c1a34f6d0e23e35deb66dd1ae7/verl/verl/trainer/ppo/ray_trainer.py#L995-L1022.

    Args:
        generator_output: Current batch generator output
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
    Returns:
        Tuple of (processed_generator_output, processed_uids, keep_sampling)
    """
    n_samples_per_prompt = sampling_config["n_samples_per_prompt"]
    min_replace_ratio = sampling_config["min_replace_ratio"]

    # Extract rewards and convert to sequence-level if needed
    rewards_list = generator_output["rewards"]
    if rewards_list and isinstance(rewards_list[0], list):
        # Token-level rewards: sum to get sequence rewards
        rewards = np.array([sum(r) for r in rewards_list])
    else:
        rewards = np.array(rewards_list)

    # get mapping of uids to list of indices and metrics
    uid2indices = defaultdict(list)
    uid2metric_vals = defaultdict(list)
    for idx, uid in enumerate(uids):
        uid2indices[uid].append(idx)
        uid2metric_vals[uid].append(rewards[idx])

    # Group by UID and calculate metrics
    uid2metric_std = {}
    for uid, metric_vals in uid2metric_vals.items():
        uid2metric_std[uid] = np.std(metric_vals)

    # Determine good UIDs: those with std > 0 (or group size == 1)
    good_uids = set(
        [
            uid
            for uid, std in uid2metric_std.items()
            if std > 0 or n_samples_per_prompt == 1
        ]
    )
    bad_uids = set(
        [
            uid
            for uid, std in uid2metric_std.items()
            if std == 0 and n_samples_per_prompt > 1
        ]
    )

    logger.info(
        f"Replace sampling: {len(good_uids)} good UIDs out of {len(uid2metric_vals)} total prompts"
    )

    # Check if we have enough good UIDs (more than min_replace_ratio of the batch)
    if len(good_uids) > len(uid2metric_vals) * min_replace_ratio:
        logger.info("============= Dynamic sampling replace ===========")
        logger.info(f"Number of good prompts: {len(good_uids)}")
        logger.info(f"Number of bad prompts: {len(bad_uids)}")

        # Get good uids to replace the bad uids (length of bad uids)
        replacement_uids = get_bad_sample_replacements(
            good_uids, bad_uids
        )  # uids to replace the bad uids
        # get replacement indices
        replacement_indices = []
        for uid in replacement_uids:
            replacement_indices.extend(uid2indices[uid])
        # get bad indices
        bad_indices = []
        for uid in bad_uids:
            bad_indices.extend(uid2indices[uid])

        # Replace bad samples with good ones (modify in place because replacement_idx and bad_idx should not overlap)
        for bad_idx, replacement_idx in zip(bad_indices, replacement_indices):
            generator_output["prompt_token_ids"][bad_idx] = generator_output[
                "prompt_token_ids"
            ][replacement_idx].copy()
            generator_output["response_ids"][bad_idx] = generator_output[
                "response_ids"
            ][replacement_idx].copy()
            replacement_reward = generator_output["rewards"][replacement_idx]
            generator_output["rewards"][bad_idx] = (
                replacement_reward.copy()
                if isinstance(replacement_reward, list)
                else replacement_reward
            )
            generator_output["loss_masks"][bad_idx] = generator_output["loss_masks"][
                replacement_idx
            ].copy()
            if generator_output["stop_reasons"]:
                generator_output["stop_reasons"][bad_idx] = generator_output[
                    "stop_reasons"
                ][replacement_idx]

            if generator_output["rollout_logprobs"]:
                generator_output["rollout_logprobs"][bad_idx] = generator_output[
                    "rollout_logprobs"
                ][replacement_idx]

        # Update UIDs accordingly
        replaced_uids = uids.copy()
        for bad_idx, replacement_idx in zip(bad_indices, replacement_indices):
            replaced_uids[bad_idx] = uids[replacement_idx]

        logger.info(
            f"After replacement - Replaced {len(bad_indices) // n_samples_per_prompt} bad prompts"
        )
        logger.info("==================================================")

        return generator_output, replaced_uids, False
    else:
        logger.warning(
            "===================== Warning (Dynamic sampling replace) ===================="
        )
        logger.warning(
            "In this mini-batch, most training samples receive low variance rewards."
        )
        logger.warning(
            "If you continue to see this warning, please check your data difficulty distribution."
        )
        logger.warning("==================================================")

        return generator_output, uids, True


def handle_filter_sampling(
    generator_output: GeneratorOutput,
    uids: List[str],
    sampling_config: Dict[str, Any],
    collected_state: DynamicSamplingState,
) -> Tuple[GeneratorOutput, List[str], bool, DynamicSamplingState]:
    """
    Handle filter-based sampling strategy (like DAPO).

    Args:
        generator_output: Current batch generator output
        uids: Current batch UIDs
        sampling_config: Configuration dict with sampling parameters
        collected_state: State for accumulating data across batches

    Returns:
        Tuple of (processed_generator_output, processed_uids, keep_sampling, updated_state)
    """
    target_batch_size = sampling_config["train_batch_size"]
    n_samples_per_prompt = sampling_config["n_samples_per_prompt"]

    # Extract rewards from collected output
    rewards_list = generator_output["rewards"]
    if rewards_list and isinstance(rewards_list[0], list):
        # Token-level rewards: sum to get sequence rewards
        rewards = np.array([sum(r) for r in rewards_list])
    else:
        rewards = np.array(rewards_list)

    # Group by UID and calculate standard deviation
    uid2metric_vals = defaultdict(list)
    for uid, reward in zip(uids, rewards):
        uid2metric_vals[uid].append(reward)

    uid2metric_std = {}
    for uid, metric_vals in uid2metric_vals.items():
        uid2metric_std[uid] = np.std(metric_vals)

    # Filter out groups with std == 0 and group size > 1
    kept_uids = [
        uid
        for uid, std in uid2metric_std.items()
        if std > 0 or n_samples_per_prompt == 1
    ]
    kept_uids_set = set(kept_uids)

    # Filter trajectories based on kept UIDs
    kept_traj_idxs = []
    for idx, traj_uid in enumerate(uids):
        if traj_uid in kept_uids_set:
            kept_traj_idxs.append(idx)

    # Apply filtering to generator output
    filtered_output = filter_generator_output(generator_output, kept_traj_idxs)
    filtered_uids = [uids[idx] for idx in kept_traj_idxs]

    if "collected_generator_output" not in collected_state:
        collected_state.update(
            {
                "collected_generator_output": filtered_output,
                "collected_uids": filtered_uids.copy(),
                "num_prompts_in_batch": len(kept_uids),
            }
        )
    else:
        collected_state["collected_generator_output"] = concatenate_generator_outputs(
            [collected_state["collected_generator_output"], filtered_output]
        )
        collected_state["collected_uids"].extend(filtered_uids)
        collected_state["num_prompts_in_batch"] += len(kept_uids)

    # Check if we have enough prompts
    if collected_state["num_prompts_in_batch"] < target_batch_size:
        logger.info("============= Dynamic sampling filter =============")
        logger.info(
            f"Dynamic sampling: {collected_state['num_prompts_in_batch']} < {target_batch_size} prompts"
        )
        logger.info(
            f"Resample batch {collected_state['sample_batch_count']}, continue sampling..."
        )
        logger.info("==================================================")
        return generator_output, uids, True, collected_state
    else:
        logger.info("============= Dynamic sampling filter =============")
        logger.info(
            f"Dynamic sampling: collected {collected_state['num_prompts_in_batch']} >= {target_batch_size} prompts"
        )
        logger.info("==================================================")
        # Truncate to exact batch size if needed
        n_samples_per_prompt = sampling_config.get("n_samples_per_prompt", 1)
        max_trajectories = target_batch_size * n_samples_per_prompt
        final_output = collected_state["collected_generator_output"]
        final_uids = collected_state["collected_uids"]

        if len(final_uids) > max_trajectories:
            final_output = filter_generator_output(
                final_output, list(range(max_trajectories))
            )
            final_uids = final_uids[:max_trajectories]

        return final_output, final_uids, False, None


def get_bad_sample_replacements(good_uids: List[str], bad_uids: List[str]) -> List[str]:
    num_replacements = len(bad_uids)
    num_candidates = len(good_uids)

    if num_candidates >= num_replacements:
        perm = np.random.permutation(num_candidates)
        chosen_replacement_uids = np.array(list(good_uids))[perm[:num_replacements]]
    else:
        indices = np.random.randint(
            low=0, high=num_candidates, size=(num_replacements,)
        )
        chosen_replacement_uids = np.array(list(good_uids))[indices]

    return chosen_replacement_uids


def filter_generator_output(
    output: GeneratorOutput, kept_indices: List[int]
) -> GeneratorOutput:
    """Filter GeneratorOutput based on kept indices."""
    filtered = {
        "prompt_token_ids": [output["prompt_token_ids"][i] for i in kept_indices],
        "response_ids": [output["response_ids"][i] for i in kept_indices],
        "rewards": [output["rewards"][i] for i in kept_indices],
        "loss_masks": [output["loss_masks"][i] for i in kept_indices],
        "stop_reasons": None,
        "rollout_metrics": output.get("rollout_metrics"),
        "rollout_logprobs": (
            [output["rollout_logprobs"][i] for i in kept_indices]
            if output["rollout_logprobs"]
            else None
        ),
    }

    if output.get("stop_reasons"):
        filtered["stop_reasons"] = [output["stop_reasons"][i] for i in kept_indices]

    return filtered


def validate_generator_output(
    input_batch: GeneratorInput, generator_output: GeneratorOutput
):
    """Validate the generator output."""
    if len(generator_output["response_ids"]) <= 0:
        raise RuntimeError("No outputs generated")

    # check that input prompts, response ids, and prompt token ids are all the same length
    num_prompts = len(input_batch["prompts"])
    num_responses = len(generator_output["response_ids"])
    num_prompt_tokens = len(generator_output["prompt_token_ids"])
    assert (
        num_prompts == num_responses
    ), f"Mismatch between prompts ({num_prompts}) and responses ({num_responses})"
    assert (
        num_responses == num_prompt_tokens
    ), f"Mismatch between responses ({num_responses}) and prompt_token_ids ({num_prompt_tokens})"

    # make sure all batch elements have the same length as response_ids (which should be non-zero)
    for key in generator_output:
        if isinstance(generator_output[key], list) and key in [
            "response_ids",
            "loss_masks",
            "rewards",
            "rollout_logprobs",
        ]:
            assert len(generator_output[key]) == len(
                generator_output["response_ids"]
            ), f"Generator output {key} length must be equal to response_ids length, got {len(generator_output[key])} and {len(generator_output['response_ids'])}"

    # make sure that each element of response ids and loss masks are all the same length (and token level rewards if used)
    for i, (response_ids, loss_masks, rewards) in enumerate(
        zip(
            generator_output["response_ids"],
            generator_output["loss_masks"],
            generator_output["rewards"],
        )
    ):
        assert len(response_ids) == len(
            loss_masks
        ), f"Response ids and loss masks must have the same length, for sample {i} got {len(response_ids)} and {len(loss_masks)}"
        if isinstance(rewards, list):
            assert len(rewards) == len(
                response_ids
            ), f"Token rewards and response ids must have the same length, for sample {i} got {len(rewards)} and {len(response_ids)}"

        if generator_output["rollout_logprobs"]:
            assert len(response_ids) == len(
                generator_output["rollout_logprobs"][i]
            ), f"Response ids and rollout logprobs must have the same length, for sample {i} got {len(response_ids)} and {len(generator_output['rollout_logprobs'][i])}"

    # loss masks should be non-zero for at least one element for trainer
    if np.concatenate(generator_output["loss_masks"]).sum() == 0:
        logger.info(
            "WARNING: All outputs are loss masked, which may lead to NaN loss, please check your generation logic!!"
        )
