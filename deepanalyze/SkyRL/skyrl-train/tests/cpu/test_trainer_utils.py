"""
uv run --isolated --extra dev pytest tests/cpu/test_trainer_utils.py
"""

from skyrl_train.utils.trainer_utils import (
    run_on_each_node,
    cleanup_old_checkpoints,
    validate_consistency_for_latest_checkpoint,
    sanitize_data_source,
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    handle_dynamic_sampling,
    handle_replace_sampling,
    handle_filter_sampling,
    filter_generator_output,
    validate_generator_output,
)
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from typing import Union
import ray
import os
import tempfile
import pytest
import re

from unittest.mock import Mock, patch, mock_open
import json

BasicType = Union[int, float, str, bool, type(None)]


def test_run_on_node_local_rank_0():
    def fn(x):
        return x + 1

    all_nodes = [node for node in ray.nodes() if node.get("CPU", 0) > 0]
    # repeat the node ids 4 times to test that the function is called only once per node
    node_ids = [all_nodes[i]["NodeID"] for i in range(len(all_nodes))] * 4
    ret = run_on_each_node(node_ids, fn, 1)
    assert ret == [2] * len(all_nodes)


def setup_mock_ckpts(tmpdir, checkpoint_steps):
    """
    Sets up dummy checkpoint directories.
    """
    # Create dummy checkpoint directories
    for step in checkpoint_steps:
        os.makedirs(os.path.join(tmpdir, f"global_step_{step}"))
    return


def test_cleanup_old_checkpoints():
    """
    Verify that _cleanup_old_checkpoints correctly removes old checkpoints
    while keeping the most recent ones.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=2, current_global_step=11)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        expected_remaining = ["global_step_10", "global_step_11"]

        assert len(remaining_dirs) == 2, "Incorrect number of checkpoints remaining"
        assert (
            remaining_dirs == expected_remaining
        ), "Did not keep the correct (most recent) checkpoints"

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute - remove all checkpoints
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=0, current_global_step=11)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert len(remaining_dirs) == 0, "Cleanup should have removed all checkpoints"

    # 3. Test cleanup with `current_global_step` less than the highest global step in the folder
    # This means that the folder contains checkpoints from a previous run.
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 10, 11]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=2, current_global_step=2)

        remaining_dirs = sorted(os.listdir(tmpdir))
        assert (
            len(remaining_dirs) == 4
        ), "Cleanup should not have removed any checkpoints"


def test_cleanup_does_not_run_when_not_needed():
    """
    Verify that cleanup does not remove any checkpoints if the total number
    is less than or equal to max_ckpts_to_keep.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=5, current_global_step=4)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert (
            len(remaining_dirs) == 4
        ), "Cleanup should not have removed any checkpoints"


def test_cleanup_with_negative_max_checkpoints():
    """
    Verify that cleanup is disabled when max_ckpts_to_keep is -1
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # 2. Execute
        cleanup_old_checkpoints(tmpdir, max_ckpts_to_keep=-1, current_global_step=5)

        # 3. Verify
        remaining_dirs = sorted(os.listdir(tmpdir))
        assert (
            len(remaining_dirs) == 5
        ), "Cleanup should be disabled when max_ckpts_to_keep is -1"


def test_validate_consistency_for_latest_checkpoint():
    """
    Verify that `validate_consistency_for_latest_checkpoint` correctly validates the checkpoint folder.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("5")

        latest_ckpt_path = os.path.join(tmpdir, "global_step_5")
        ckpt_iteration = 5

        # 2. Execute
        validate_consistency_for_latest_checkpoint(
            tmpdir, ckpt_iteration, latest_ckpt_path, latest_ckpt_file, save_interval=1
        )


def test_validate_consistency_for_latest_checkpoint_with_inconsistent_folder():
    """
    Verify that `validate_consistency_for_latest_checkpoint` correctly validates the checkpoint folder.
    """
    # Example 1: `latest_ckpt_global_step.txt` points to a lower global step than the highest global step in the folder
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 2, 3, 4, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # change the latest checkpoint file to point to a lower global step
        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("3")

        latest_ckpt_path = os.path.join(tmpdir, "global_step_3")
        ckpt_iteration = 3
        save_interval = 1

        # 2. Execute
        with pytest.raises(ValueError, match="Inconsistent checkpoint folder"):
            validate_consistency_for_latest_checkpoint(
                tmpdir,
                ckpt_iteration,
                latest_ckpt_path,
                latest_ckpt_file,
                save_interval=save_interval,
            )

    # Example 2: `latest_ckpt_global_step.txt` points to a lower global step but it's within the save interval
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        checkpoint_steps = [1, 3, 5]
        setup_mock_ckpts(tmpdir, checkpoint_steps=checkpoint_steps)

        # change the latest checkpoint file to point to a lower global step
        latest_ckpt_file = os.path.join(tmpdir, "latest_ckpt_global_step.txt")
        with open(latest_ckpt_file, "w") as f:
            f.write("3")

        save_interval = 2
        latest_ckpt_path = os.path.join(tmpdir, "global_step_3")
        ckpt_iteration = 3

        # 2. Execute
        validate_consistency_for_latest_checkpoint(
            tmpdir,
            ckpt_iteration,
            latest_ckpt_path,
            latest_ckpt_file,
            save_interval=save_interval,
        )


def test_sanitize_data_source_none():
    """Test sanitize_data_source with None input."""
    result = sanitize_data_source(None)
    assert result == "unknown"


def test_sanitize_data_source_slash_replacement():
    """Test sanitize_data_source replaces slashes with underscores."""
    result = sanitize_data_source("dataset/with/slashes")
    assert result == "dataset_with_slashes"


def test_sanitize_data_source_normal_string():
    """Test sanitize_data_source with normal string."""
    result = sanitize_data_source("normal_dataset")
    assert result == "normal_dataset"


def test_calculate_per_dataset_metrics_single_source():
    """Test calculate_per_dataset_metrics with single data source."""
    # Create test data
    generator_outputs = {
        "rewards": [0.5, 0.7, 0.9],
        "prompt_token_ids": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "response_ids": [[10, 11], [12, 13], [14, 15]],
    }
    uids = ["uid1", "uid2", "uid3"]
    data_sources = ["dataset1", "dataset1", "dataset1"]

    result = calculate_per_dataset_metrics(generator_outputs, uids, data_sources, 2)

    # Verify results - actual computed values
    # Mean reward: (0.5 + 0.7 + 0.9) / 3 = 0.7
    # Pass@N: all rewards > 0, all unique uids, so 3/3 = 1.0
    assert "eval/dataset1/avg_score" in result
    assert "eval/dataset1/pass_at_2" in result
    assert result["eval/dataset1/avg_score"] == pytest.approx(0.7)
    assert result["eval/dataset1/pass_at_2"] == 1.0


def test_calculate_per_dataset_metrics_multiple_sources():
    """Test calculate_per_dataset_metrics with multiple data sources including None."""
    # Create test data with mixed sources
    generator_outputs = {
        "rewards": [0.5, 0.7, 0.9, 0.4],
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6], [7, 8]],
        "response_ids": [[10, 11], [12, 13], [14, 15], [16, 17]],
    }
    uids = ["uid1", "uid2", "uid3", "uid4"]
    data_sources = ["dataset1", None, "dataset1", None]

    result = calculate_per_dataset_metrics(generator_outputs, uids, data_sources, 2)

    # Verify results for both datasets - actual computed values
    # dataset1: indices 0, 2 -> rewards [0.5, 0.9] -> mean = 0.7, pass@n = 2/2 = 1.0
    # unknown (None): indices 1, 3 -> rewards [0.7, 0.4] -> mean = 0.55, pass@n = 2/2 = 1.0
    assert "eval/dataset1/avg_score" in result
    assert "eval/dataset1/pass_at_2" in result
    assert "eval/unknown/avg_score" in result
    assert "eval/unknown/pass_at_2" in result

    assert result["eval/dataset1/avg_score"] == pytest.approx(0.7)
    assert result["eval/dataset1/pass_at_2"] == 1.0
    assert result["eval/unknown/avg_score"] == pytest.approx(0.55)
    assert result["eval/unknown/pass_at_2"] == 1.0


@patch("builtins.open", new_callable=mock_open)
def test_dump_per_dataset_eval_results_comprehensive(mock_file):
    """Test dump_per_dataset_eval_results comprehensive functionality."""
    # Mock dump directory path
    mock_dump_dir = Mock()
    mock_dump_dir.__truediv__ = Mock(side_effect=lambda x: f"mock_path/{x}")

    # Mock tokenizer
    mock_tokenizer = Mock()
    mock_tokenizer.decode.side_effect = lambda x: f"decoded_{x}"

    # Create test data
    generator_outputs = {
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6]],
        "response_ids": [[10, 11], [12, 13], [14, 15]],
        "rewards": [0.5, 0.7, 0.9],
        "stop_reasons": ["stop1", "stop2", "stop3"],
    }
    data_sources = ["dataset1", None, "dataset1"]
    all_envs = ["env1", "env2", "env3"]
    env_extras = [{"extra1": "val1"}, {"extra2": "val2"}, {"extra3": "val3"}]
    eval_metrics = {"eval/dataset1/avg_score": 0.8, "eval/unknown/avg_score": 0.6}

    # Call the function
    dump_per_dataset_eval_results(
        mock_dump_dir,
        mock_tokenizer,
        generator_outputs,
        data_sources,
        all_envs,
        env_extras,
        eval_metrics,
    )

    # Verify tokenizer was called for decoding
    assert mock_tokenizer.decode.call_count == 6  # 3 prompts + 3 responses

    # Verify files were opened (2 per-dataset files + 1 aggregated file)
    assert mock_file.call_count == 3

    # Verify file writes occurred
    handle = mock_file.return_value
    assert handle.write.call_count > 0

    # Verify JSON structure by checking some write calls contain expected data
    write_calls = [call[0][0] for call in handle.write.call_args_list]
    json_writes = [
        call for call in write_calls if call.strip() and not call.startswith("Dumped")
    ]

    # At least one JSON line should contain our test data
    assert len(json_writes) > 0

    # Parse one of the JSON writes to verify structure
    for write_call in json_writes:
        try:
            data = json.loads(write_call.strip())
            if "input_prompt" in data:
                # This is a per-dataset entry
                assert "output_response" in data
                assert "score" in data
                assert "data_source" in data
                break
        except json.JSONDecodeError:
            continue


def test_handle_dynamic_sampling_null_strategy():
    """Test that null strategy returns input unchanged."""
    generator_output = {
        "prompt_token_ids": [[1, 2, 3], [4, 5, 6]],
        "response_ids": [[7, 8], [9, 10]],
        "rewards": [[1.0, 2.0], [3.0, 4.0]],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
        "rollout_logprobs": [[0.16, 0.4], [0.2, 0.3]],
    }
    uids = ["uid1", "uid2"]
    sampling_config = {"type": None}

    result_output, result_uids, keep_sampling, state = handle_dynamic_sampling(
        generator_output, uids, sampling_config
    )

    assert result_output == generator_output
    assert result_uids == uids
    assert keep_sampling is False
    assert state is None


def test_handle_dynamic_sampling_invalid_strategy():
    """Test that invalid strategy raises ValueError."""
    generator_output = {
        "prompt_token_ids": [[1, 2, 3]],
        "response_ids": [[7, 8]],
        "rewards": [[1.0, 2.0]],
        "loss_masks": [[1, 1]],
        "rollout_logprobs": [[0.16, 0.4]],
    }
    uids = ["uid1"]
    sampling_config = {"type": "invalid_strategy"}

    with pytest.raises(
        ValueError, match="Invalid dynamic sampling type: invalid_strategy"
    ):
        handle_dynamic_sampling(generator_output, uids, sampling_config)


def test_handle_replace_sampling_sufficient_good_samples():
    """Test replace sampling when there are sufficient good samples (>0.3)."""
    # Create test data with some good UIDs (high variance) and some bad UIDs (zero variance)
    generator_output = {
        "prompt_token_ids": [[1, 2], [1, 2], [3, 4], [3, 4], [5, 6], [5, 6]],
        "response_ids": [[13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24]],
        "rewards": [
            1.0,
            2.0,
            1.0,
            1.0,
            3.0,
            4.0,
        ],  # uid1: [1.0, 2.0] (good), uid2: [1.0, 1.0] (bad), uid3: [3.0, 4.0] (good)
        "loss_masks": [[1, 1]] * 6,
        "stop_reasons": ["length"] * 6,
        "rollout_metrics": None,
        "rollout_logprobs": [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.25],
            [0.15, 0.25],
            [0.1, 0.2],
            [0.3, 0.4],
        ],
    }
    uids = ["uid1", "uid1", "uid2", "uid2", "uid3", "uid3"]  # 2 samples per prompt
    sampling_config = {"n_samples_per_prompt": 2, "min_replace_ratio": 0.3}

    result_output, result_uids, keep_sampling = handle_replace_sampling(
        generator_output, uids, sampling_config
    )

    # Should not keep sampling
    assert keep_sampling is False

    # Output should have same structure but with replacements
    assert len(result_output["prompt_token_ids"]) == 6
    assert len(result_output["response_ids"]) == 6
    assert len(result_output["rewards"]) == 6
    assert len(result_output["rollout_logprobs"]) == 6
    assert len(result_uids) == 6

    # Check that bad uid2 samples were replaced with good samples
    uid2_indices = [i for i, uid in enumerate(result_uids) if uid == "uid2"]
    # After replacement, uid2 indices should now contain UIDs from good samples
    assert len(uid2_indices) == 0  # uid2 should be completely replaced


def test_handle_replace_sampling_insufficient_good_samples():
    """Test replace sampling when there are insufficient good samples (<0.3)."""
    generator_output = {
        "prompt_token_ids": [[1, 2], [1, 2], [3, 4], [3, 4]],
        "response_ids": [[9, 10], [11, 12], [13, 14], [15, 16]],
        "rewards": [
            1.0,
            1.0,
            2.0,
            2.0,
        ],  # uid1: [1.0, 1.0] (bad), uid2: [2.0, 2.0] (bad)
        "loss_masks": [[1, 1]] * 4,
        "stop_reasons": ["length"] * 4,
        "rollout_metrics": None,
        "rollout_logprobs": None,
    }
    uids = ["uid1", "uid1", "uid2", "uid2"]  # 2 samples per prompt
    sampling_config = {"n_samples_per_prompt": 2, "min_replace_ratio": 0.3}

    result_output, result_uids, keep_sampling = handle_replace_sampling(
        generator_output, uids, sampling_config
    )

    # Should keep sampling due to insufficient good samples
    assert keep_sampling is True

    # Output should be unchanged
    assert result_output == generator_output
    assert result_uids == uids


def test_handle_replace_sampling_single_sample_per_prompt():
    """Test replace sampling with single sample per prompt (should always be considered good)."""
    generator_output = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8]],
        "rewards": [1.0, 1.0],
        "loss_masks": [[1, 1]] * 2,
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": None,
        "rollout_logprobs": [[0.1, 0.2]],
    }
    uids = ["uid1", "uid2"]
    sampling_config = {"n_samples_per_prompt": 1, "min_replace_ratio": 0.3}

    result_output, result_uids, keep_sampling = handle_replace_sampling(
        generator_output, uids, sampling_config
    )

    # Should not keep sampling (single samples are always considered good)
    assert keep_sampling is False

    # Output should be unchanged since all samples are good
    assert result_output == generator_output
    assert result_uids == uids


def test_handle_replace_sampling_token_level_rewards():
    """Test replace sampling with token-level rewards (should sum to sequence level)."""
    generator_output = {
        "prompt_token_ids": [[1, 2], [1, 2], [3, 4], [3, 4]],
        "response_ids": [[9, 10], [11, 12, 13], [14, 15], [16]],
        "rewards": [
            [1.0, 2.0],
            [3.0, 4.0, 5.0],
            [1.0, 1.0],
            [1.0],
        ],  # Token-level rewards
        "loss_masks": [[1, 1]] * 4,
        "stop_reasons": ["stop"] * 4,
        "rollout_metrics": None,
        "rollout_logprobs": None,
    }
    uids = [
        "uid1",
        "uid1",
        "uid2",
        "uid2",
    ]  # uid1: [3.0, 7.0] (good), uid2: [2.0, 2.0] (bad)
    sampling_config = {"n_samples_per_prompt": 2, "min_replace_ratio": 0.3}

    result_output, result_uids, keep_sampling = handle_replace_sampling(
        generator_output, uids, sampling_config
    )

    # Should not keep sampling (sufficient good samples)
    assert keep_sampling is False

    # Check that replacements occurred
    assert len(result_output["rewards"]) == 4
    assert len(result_uids) == 4


def test_handle_filter_sampling_sufficient_prompts():
    """Test filter sampling when we get sufficient prompts in one batch."""
    generator_output = {
        "prompt_token_ids": [[1, 2], [1, 2], [3, 4], [3, 4]],
        "response_ids": [[9, 10], [11, 12], [13, 14], [15, 16]],
        "rewards": [
            1.0,
            2.0,
            3.0,
            3.0,
        ],  # uid1: [1.0, 2.0] (good), uid2: [3.0, 3.0] (bad)
        "loss_masks": [[1, 1]] * 4,
        "stop_reasons": ["stop"] * 4,
        "rollout_metrics": None,
        "rollout_logprobs": None,
    }
    uids = ["uid1", "uid1", "uid2", "uid2"]
    sampling_config = {
        "train_batch_size": 1,  # Only need 1 prompt
        "n_samples_per_prompt": 2,
        "max_sample_batches": 20,
    }

    result_output, result_uids, keep_sampling, state = handle_filter_sampling(
        generator_output,
        uids,
        sampling_config,
        collected_state={"sample_batch_count": 1},
    )

    # Should not keep sampling (sufficient prompts)
    assert keep_sampling is False
    assert state is None

    # Should only keep the good uid1 samples
    assert len(result_output["prompt_token_ids"]) == 2
    assert len(result_uids) == 2
    assert all(uid == "uid1" for uid in result_uids)


def test_handle_filter_sampling_insufficient_prompts_continue():
    """Test filter sampling when we need to continue sampling."""
    generator_output = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8]],
        "rewards": [1.0, 2.0],  # Only 1 good prompt
        "loss_masks": [[1, 1]] * 2,
        "stop_reasons": ["stop"] * 2,
        "rollout_metrics": None,
        "rollout_logprobs": None,
    }
    uids = ["uid1", "uid1"]
    sampling_config = {
        "train_batch_size": 2,  # Need 2 prompts
        "n_samples_per_prompt": 2,
        "max_sample_batches": 20,
    }

    collected_state = {"sample_batch_count": 1}

    result_output, result_uids, keep_sampling, state = handle_filter_sampling(
        generator_output, uids, sampling_config, collected_state=collected_state
    )

    # Should keep sampling (insufficient prompts)
    assert keep_sampling is True
    assert result_output is generator_output
    assert result_uids is uids
    assert state is not None
    assert state["num_prompts_in_batch"] == 1
    assert state["sample_batch_count"] == 1


def test_handle_filter_sampling_accumulation():
    """Test filter sampling accumulation across multiple batches."""
    # First batch
    generator_output1 = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8]],
        "rewards": [1.0, 2.0],  # Good prompt
        "loss_masks": [[1, 1]] * 2,
        "stop_reasons": ["stop"] * 2,
        "rollout_metrics": None,
        "rollout_logprobs": None,
    }
    uids1 = ["uid1", "uid1"]

    # Second batch
    generator_output2 = {
        "prompt_token_ids": [[9, 10], [11, 12]],
        "response_ids": [[13, 14], [15, 16]],
        "rewards": [3.0, 4.0],  # Another good prompt
        "loss_masks": [[1, 1]] * 2,
        "stop_reasons": ["stop"] * 2,
        "rollout_metrics": None,
        "rollout_logprobs": None,
    }
    uids2 = ["uid2", "uid2"]

    sampling_config = {
        "train_batch_size": 2,  # Need 2 prompts
        "n_samples_per_prompt": 2,
        "max_sample_batches": 20,
    }

    collected_state = {"sample_batch_count": 1}

    # Process first batch
    result1_output, result1_uids, keep_sampling1, state1 = handle_filter_sampling(
        generator_output1, uids1, sampling_config, collected_state=collected_state
    )

    assert keep_sampling1 is True  # Need more prompts
    assert state1["num_prompts_in_batch"] == 1

    # Process second batch
    result2_output, result2_uids, keep_sampling2, state2 = handle_filter_sampling(
        generator_output2, uids2, sampling_config, collected_state=state1
    )

    assert keep_sampling2 is False  # Now have enough prompts
    assert state2 is None
    assert len(result2_output["prompt_token_ids"]) == 4  # Both batches combined
    assert len(result2_uids) == 4


def test_handle_filter_sampling_single_sample_per_prompt():
    """Test filter sampling with single sample per prompt."""
    generator_output = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[5, 6], [7, 8]],
        "rewards": [1.0, 1.0],  # Same rewards but single sample per prompt
        "loss_masks": [[1, 1]] * 2,
        "stop_reasons": ["stop"] * 2,
        "rollout_metrics": None,
        "rollout_logprobs": None,
    }
    uids = ["uid1", "uid2"]  # Different UIDs, single sample each
    sampling_config = {
        "train_batch_size": 2,
        "n_samples_per_prompt": 1,
        "max_sample_batches": 20,
    }

    result_output, result_uids, keep_sampling, state = handle_filter_sampling(
        generator_output,
        uids,
        sampling_config,
        collected_state={"sample_batch_count": 1},
    )

    # Should not keep sampling (single samples are always kept)
    assert keep_sampling is False
    assert state is None
    assert len(result_output["prompt_token_ids"]) == 2
    assert len(result_uids) == 2


def test_filter_generator_output():
    """Test the filter_generator_output utility function."""
    generator_output = {
        "prompt_token_ids": [[1, 2], [3, 4], [5, 6]],
        "response_ids": [[7, 8], [9, 10], [11, 12]],
        "rewards": [1.0, 2.0, 3.0],
        "loss_masks": [[1, 1]] * 3,
        "stop_reasons": ["length", "length", "stop"],
        "rollout_metrics": {"metric": "value"},
        "rollout_logprobs": [[0.16, 0.4], [0.1, 0.2], [0.3, 0.4]],
    }
    kept_indices = [0, 2]  # Keep first and third samples

    filtered = filter_generator_output(generator_output, kept_indices)

    assert filtered["prompt_token_ids"] == [[1, 2], [5, 6]]
    assert filtered["response_ids"] == [[7, 8], [11, 12]]
    assert filtered["rewards"] == [1.0, 3.0]
    assert filtered["loss_masks"] == [[1, 1]] * 2
    assert filtered["stop_reasons"] == ["length", "stop"]
    assert filtered["rollout_metrics"] == {"metric": "value"}
    assert filtered["rollout_logprobs"] == [[0.16, 0.4], [0.3, 0.4]]


def test_validate_generator_output_valid_case():
    """Test validate_generator_output with valid case."""
    input_batch = GeneratorInput(
        prompts=["prompt1", "prompt2", "prompt3"],
        env_classes=["env1", "env2", "env3"],
        env_extras=[{"extra": 1}, {"extra": 2}, {"extra": 3}],
        sampling_params={"temperature": 0.7},
    )

    generator_output = GeneratorOutput(
        prompt_token_ids=[[1, 2, 3, 4], [5, 6], [7, 8, 9]],
        response_ids=[[10, 11, 12], [13, 14], [15]],
        rewards=[[0.5, 0.6, 0.7], [0.8, 0.9], [1.0]],  # Nested list structure
        loss_masks=[[1, 1, 0], [1, 1], [0]],
        stop_reasons=["stop", "length", "stop"],
        rollout_metrics={"metric1": 0.5, "metric2": 0.6},
        rollout_logprobs=None,
    )

    # Should not raise any exceptions
    validate_generator_output(input_batch, generator_output)

    # per trajectory rewards should work too
    generator_output["rewards"] = [0.5, 0.6, 0.7]
    validate_generator_output(input_batch, generator_output)

    # valid rollout logprobs
    generator_output["rollout_logprobs"] = [[0.11, 0.12, 0.13], [0.2, 0.3], [0.4]]
    validate_generator_output(input_batch, generator_output)


def test_validate_generator_output_empty_response_ids():
    """Test validate_generator_output raises RuntimeError when response_ids is empty."""
    input_batch = GeneratorInput(
        prompts=["prompt1"], env_classes=["env1"], env_extras=None, sampling_params=None
    )

    generator_output = GeneratorOutput(
        prompt_token_ids=[[1, 2, 3]],
        response_ids=[],
        rewards=[],
        loss_masks=[],
        stop_reasons=[],
        rollout_logprobs=[],  # Empty response_ids
    )

    with pytest.raises(RuntimeError, match="No outputs generated"):
        validate_generator_output(input_batch, generator_output)


def test_validate_generator_output_mismatched_prompts_responses():
    """Test validate_generator_output raises AssertionError when prompts and response_ids lengths don't match."""
    input_batch = GeneratorInput(
        prompts=["prompt1", "prompt2", "prompt3"],  # 3 prompts
        env_classes=["env1", "env2", "env3"],
        env_extras=None,
        sampling_params=None,
    )

    generator_output = GeneratorOutput(
        prompt_token_ids=[[1, 2], [3, 4]],
        response_ids=[[7, 8], [9, 10]],  # Only 2 responses
        rewards=[0.5, 0.7],
        loss_masks=[[1, 1], [1, 0]],
        stop_reasons=["eos", "eos"],
        rollout_logprobs=None,
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("Mismatch between prompts (3) and responses (2)"),
    ):
        validate_generator_output(input_batch, generator_output)


def test_validate_generator_output_all_loss_masked():
    """Test validate_generator_output logs warning when all outputs are loss masked."""
    input_batch = GeneratorInput(
        prompts=["prompt1", "prompt2"],
        env_classes=["env1", "env2"],
        env_extras=None,
        sampling_params=None,
    )

    generator_output = GeneratorOutput(
        prompt_token_ids=[[1, 2, 3], [4, 5, 6]],
        response_ids=[[7, 8], [9, 10]],
        rewards=[0.5, 0.7],
        loss_masks=[[0, 0], [0, 0]],  # All zeros - completely loss masked
        stop_reasons=["eos", "eos"],
        rollout_logprobs=None,
    )

    # Capture log output to verify warning is issued
    with patch("skyrl_train.utils.trainer_utils.logger") as mock_logger:
        validate_generator_output(input_batch, generator_output)
        mock_logger.info.assert_called_once_with(
            "WARNING: All outputs are loss masked, which may lead to NaN loss, please check your generation logic!!"
        )


def test_validate_generator_output_mismatched_list_lengths():
    """Test validate_generator_output raises AssertionError when generator output lists have mismatched lengths."""
    input_batch = GeneratorInput(
        prompts=["prompt1", "prompt2"],
        env_classes=["env1", "env2"],
        env_extras=None,
        sampling_params=None,
    )

    generator_output = GeneratorOutput(
        prompt_token_ids=[[1, 2, 3], [4, 5, 6]],
        response_ids=[[7, 8], [9, 10]],  # Length 2
        rewards=[0.5, 0.7, 0.9],  # Length 3 - mismatch!
        loss_masks=[[1, 1], [1, 0]],
        stop_reasons=["eos", "eos"],
        rollout_logprobs=None,
    )

    with pytest.raises(
        AssertionError,
        match="Generator output rewards length must be equal to response_ids length",
    ):
        validate_generator_output(input_batch, generator_output)


def test_validate_generator_output_element_length_mismatch():
    """Test validate_generator_output with element length mismatch."""
    input_batch = GeneratorInput(
        prompts=["prompt1", "prompt2", "prompt3"],
        env_classes=["env1", "env2", "env3"],
        env_extras=[{"extra": 1}, {"extra": 2}, {"extra": 3}],
        sampling_params={"temperature": 0.7},
    )

    generator_output = GeneratorOutput(
        prompt_token_ids=[[1, 2, 3, 4], [5, 6], [7, 8, 9]],
        response_ids=[[10, 11, 12], [13, 14], [15]],
        rewards=[[0.5, 0.6, 0.7], [0.8, 0.9], [1.0]],
        loss_masks=[
            [1, 1],
            [1],
            [1, 1],
        ],  # loss masks are not the same length as response ids
        stop_reasons=["stop", "length", "stop"],
        rollout_metrics={"metric1": 0.5, "metric2": 0.6},
        rollout_logprobs=None,
    )

    with pytest.raises(
        AssertionError, match="Response ids and loss masks must have the same length"
    ):
        validate_generator_output(input_batch, generator_output)

    generator_output["loss_masks"] = [[1, 1, 1], [1, 1], [1]]  # add correct loss masks
    generator_output["rewards"] = [
        [0.5, 0.6],
        [0.8],
        [1.0, 2.0],
    ]  # add incorrect rewards
    with pytest.raises(
        AssertionError, match="Token rewards and response ids must have the same length"
    ):
        validate_generator_output(input_batch, generator_output)

    generator_output = GeneratorOutput(
        prompt_token_ids=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        response_ids=[[7, 8], [9, 10], [11, 12]],
        rewards=[0.5, 0.7, -0.1],
        loss_masks=[[1, 1], [1, 0], [1, 1]],
        stop_reasons=["eos", "eos", "length"],
        rollout_logprobs=[
            [0.17, 0.2],
            [0.9],
            [0.1, 0.2],
        ],  # Second entry has length 1 - mismatch !
    )

    with pytest.raises(
        AssertionError,
        match="Response ids and rollout logprobs must have the same length",
    ):
        validate_generator_output(input_batch, generator_output)
