"""
Tests for policy loss functions.

uv run --isolated --extra dev -- pytest tests/cpu/algorithms/test_losses.py
"""

import pytest
import torch
from omegaconf import DictConfig
from skyrl_train.utils.ppo_utils import PolicyLossRegistry, masked_mean


# Adapted a good test from NeMO-RL
def test_policy_loss_dual_clip():
    """Tests dual clipping in PolicyLoss function."""

    device = "cpu"

    # Create test data with a mix of advantages: positive, slightly negative, strongly negative
    advantages = torch.tensor([[1.0, -1.0, -4.0]], device=device)

    # Set up logprobs to test different probability ratios
    old_log_probs = torch.tensor([[-1.0, -1.0, -3.0]], device=device)
    log_probs = torch.tensor([[-1.69315, -1.0, -0.69741]], device=device)  # approx log(0.5)-1, log(1)-1, log(10)-3

    # Create config for dual clipping
    config = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "dual_clip",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    # Create loss function with dual clipping
    loss_fn = PolicyLossRegistry.get("dual_clip")

    # Calculate expected values
    ratio = torch.exp(log_probs - old_log_probs)  # approx [0.5, 1.0, 10.0]
    assert torch.allclose(ratio, torch.tensor([[0.5, 1.0, 10.0]], device=device), rtol=1e-3)

    # Standard PPO clipping
    loss1 = -ratio * advantages  # [0.5, -1.0, -40.0]
    loss2 = -ratio.clamp(1 - 0.2, 1 + 0.2) * advantages  # [0.8, -1.0, -4.8]
    max_loss = torch.maximum(loss1, loss2)  # [0.5, -1.0, -40.0]

    # Dual clipping
    loss3 = -advantages * 3.0  # [-3.0, 3.0, 12.0]
    min_loss = torch.min(loss3, max_loss)  # [-3.0, 1.0, 12.0]

    # For negative advantages, use dual clipped loss
    final_loss = torch.where(advantages < 0, min_loss, max_loss)  # [-0.5, 1.0, 12.0]
    assert torch.allclose(final_loss, torch.tensor([[-0.5, 1.0, 12.0]], device=device), rtol=1e-3)
    expected_loss = final_loss.mean()  # -(-12.5/3) = 4.1667

    # Calculate actual loss
    actual_loss, _ = loss_fn(log_probs=log_probs, old_log_probs=old_log_probs, advantages=advantages, config=config)

    # Verify results
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-3, atol=1e-8)
    # close to hand calculated value
    assert actual_loss.item() == pytest.approx(4.1667, abs=1e-4)


def test_policy_loss_reduction_modes():
    """Tests different loss_reduction modes in PolicyLoss function.

    Note: token_mean and sequence_mean give the same result when all sequences
    have the same length and no mask is applied, but differ when masking creates
    different effective sequence lengths.
    """

    device = "cpu"

    clip_eps_low = 0.2
    clip_eps_high = 0.2

    advantages = torch.tensor(
        [
            [2.0, 2.0, 2.0],  # sequence 1: consistently higher advantages
            [1.0, 1.0, 1.0],  # sequence 2: consistently lower advantages
        ],
        device=device,
    )

    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]], device=device)

    log_probs = torch.tensor(
        [[-1.5, -0.5, -1.2], [-0.8, -1.3, -0.9]],  # ratios ≈ [[0.61, 1.65, 0.83],[1.22, 0.74, 1.11]]
        device=device,
    )

    # Create masks to test sequences with different numbers of valid tokens
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]], device=device)

    # Create configs for different reduction modes
    config_token = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    config_seq = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    # Get loss function
    loss_fn = PolicyLossRegistry.get("regular")

    # Test token_mean without mask
    loss_token_no_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_token)

    # Test token_mean with mask
    loss_token_with_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_token, loss_mask)

    # Test sequence_mean without mask
    loss_seq_no_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq)

    # Test sequence_mean with mask
    loss_seq_with_mask, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq, loss_mask)

    # Manual calculations to verify (using default PolicyLoss parameters)
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages  # clip_eps_low=0.2, clip_eps_high=0.2
    loss_per_token = -torch.min(surr1, surr2)

    # Expected token_mean without mask: mean of all tokens
    expected_token_no_mask = loss_per_token.mean()

    # Expected token_mean with mask: masked mean of all tokens
    expected_token_with_mask = (loss_per_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Expected sequence_mean without mask: mean of sequence means
    expected_seq_no_mask = loss_per_token.mean(dim=1).mean()

    # Expected sequence_mean with mask: mean of masked sequence means
    seq_means_masked = (loss_per_token * loss_mask).sum(dim=1) / (loss_mask.sum(dim=1) + 1e-8)
    expected_seq_with_mask = seq_means_masked.mean()

    # Verify results
    torch.testing.assert_close(loss_token_no_mask, expected_token_no_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_token_with_mask, expected_token_with_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_seq_no_mask, expected_seq_no_mask, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_seq_with_mask, expected_seq_with_mask, rtol=1e-5, atol=1e-8)

    # Verify that the two reduction modes give the same results when sequences have equal length and no mask
    assert torch.allclose(
        loss_token_no_mask, loss_seq_no_mask, rtol=1e-5
    ), "token_mean and sequence_mean should give same results when sequences have equal length and no mask"
    # But they should give different results when mask creates different effective sequence lengths
    assert not torch.allclose(
        loss_token_with_mask, loss_seq_with_mask, rtol=1e-3
    ), "token_mean and sequence_mean with mask should give different results"


def test_policy_loss_reduction_edge_cases():
    """Tests edge cases for loss_reduction modes."""

    device = "cpu"

    # Test with single sequence (should give same result for both modes)
    advantages = torch.tensor([[1.0, -1.0, 2.0]], device=device)
    old_log_probs = torch.tensor([[-1.0, -1.0, -1.0]], device=device)
    log_probs = torch.tensor([[-1.5, -0.5, -1.2]], device=device)

    # Create configs for different reduction modes
    config_token = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    config_seq = DictConfig(
        {
            "eps_clip_low": 0.2,
            "eps_clip_high": 0.2,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "sequence_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )

    # Get loss function
    loss_fn = PolicyLossRegistry.get("regular")

    loss_token, _ = loss_fn(log_probs, old_log_probs, advantages, config_token)
    loss_seq, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq)

    # With single sequence, both modes should give same result
    torch.testing.assert_close(loss_token, loss_seq, rtol=1e-6, atol=1e-8)

    # Test with completely masked sequence
    loss_mask = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    loss_token_masked, _ = loss_fn(log_probs, old_log_probs, advantages, config_token, loss_mask)
    loss_seq_masked, _ = loss_fn(log_probs, old_log_probs, advantages, config_seq, loss_mask)

    # Should handle zero mask gracefully (due to +1e-8 in denominator)
    assert torch.isfinite(loss_token_masked)
    assert torch.isfinite(loss_seq_masked)


def test_gspo_importance_sampling_levels():
    """Tests GSPO policy loss function with sequence-level importance sampling.

    This test focuses on GSPO's key benefit: stabilizing clipping behavior through sequence-level
    importance sampling, which should lead to more consistent training dynamics compared to
    token-level importance sampling in standard PPO.
    """

    device = "cpu"

    clip_eps_low = 0.2
    clip_eps_high = 0.2

    # Create test data with varied sequence lengths and extreme ratios to test clipping stability
    # GSPO's benefit is most apparent with sequences of different lengths and high variance
    advantages = torch.tensor(
        [
            [1.5, 2.0, 1.0, 0.8, 0.5, 0.0, 0.0, 0.0],  # long sequence: 5 valid tokens
            [3.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # short sequence: 2 valid tokens
            [0.5, 0.8, 1.2, 2.5, 0.0, 0.0, 0.0, 0.0],  # medium sequence: 4 valid tokens
        ],
        device=device,
    )

    old_log_probs = torch.tensor(
        [
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        ],
        device=device,
    )

    # Create extreme log probability ratios to trigger significant clipping
    # This tests GSPO's stability benefits under conditions that would cause unstable clipping
    log_probs = torch.tensor(
        [
            [0.2, -2.5, -0.3, 0.1, -1.8, -1.0, -1.0, -1.0],  # high variance within sequence
            [0.8, -0.2, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # extreme ratios (exp(1.8)≈6.0, exp(0.8)≈2.2)
            [-0.5, 0.3, -1.7, 0.4, -1.0, -1.0, -1.0, -1.0],  # mixed extreme values
        ],
        device=device,
    )

    # Create masks for different sequence lengths (key for testing length normalization)
    loss_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],  # 5 tokens
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # 2 tokens
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # 4 tokens
        ],
        device=device,
    )

    # Test standard PPO (token-level importance sampling)
    ppo_config = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "regular",
            "loss_reduction": "token_mean",
            "max_seq_len": 4,
            "use_tis": False,
        }
    )
    ppo_loss_fn = PolicyLossRegistry.get("regular")
    loss_token, _ = ppo_loss_fn(log_probs, old_log_probs, advantages, ppo_config, loss_mask)

    # Test GSPO (sequence-level importance sampling)
    gspo_config = DictConfig(
        {
            "eps_clip_low": clip_eps_low,
            "eps_clip_high": clip_eps_high,
            "clip_ratio_c": 3.0,
            "policy_loss_type": "gspo",
            "loss_reduction": "sequence_mean",  # GSPO recommended reduction
            "max_seq_len": 4,
            "use_tis": False,
        }
    )
    gspo_loss_fn = PolicyLossRegistry.get("gspo")
    loss_sequence, _ = gspo_loss_fn(log_probs, old_log_probs, advantages, gspo_config, loss_mask)

    # Manual calculation for token-level (standard PPO)
    log_ratio = log_probs - old_log_probs
    ratio_token = log_ratio.exp()
    surr1_token = ratio_token * advantages
    surr2_token = ratio_token.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    loss_per_token_token = -torch.min(surr1_token, surr2_token)
    expected_token = (loss_per_token_token * loss_mask).sum() / (loss_mask.sum() + 1e-8)

    # Calculate token-level clipping ratio
    is_clipped_token = (-surr2_token > -surr1_token) & (loss_mask.bool())
    clip_ratio_token = is_clipped_token.float().sum() / loss_mask.sum()

    # Manual calculation for sequence-level (GSPO)
    # First compute sequence-level importance weights (key GSPO innovation)
    log_importance_weights_seq = masked_mean(log_ratio, loss_mask, dim=-1).unsqueeze(-1)

    # GSPO uses stop gradients: s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_probs - sg[log_probs]
    ratio_sequence = torch.exp(log_importance_weights_seq.detach() + log_probs - log_probs.detach())
    surr1_sequence = ratio_sequence * advantages
    surr2_sequence = ratio_sequence.clamp(1 - clip_eps_low, 1 + clip_eps_high) * advantages
    loss_per_token_sequence = -torch.min(surr1_sequence, surr2_sequence)
    # GSPO uses sequence_mean reduction
    expected_sequence = masked_mean(loss_per_token_sequence, loss_mask, dim=-1).mean()

    # Calculate sequence-level clipping ratio
    is_clipped_sequence = (-surr2_sequence > -surr1_sequence) & (loss_mask.bool())
    clip_ratio_sequence = is_clipped_sequence.float().sum() / loss_mask.sum()

    # Verify loss calculations
    torch.testing.assert_close(loss_token, expected_token, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(loss_sequence, expected_sequence, rtol=1e-5, atol=1e-8)

    # Core GSPO benefit test: Different clipping behavior
    # GSPO should produce different clipping patterns due to sequence-level importance sampling
    assert not torch.allclose(
        clip_ratio_token, clip_ratio_sequence, rtol=1e-2
    ), f"Clipping ratios should differ: token={clip_ratio_token:.4f} vs sequence={clip_ratio_sequence:.4f}"

    # Test stability: sequence-level should smooth out extreme per-token variations
    # Check that sequence-level ratios have lower variance within each sequence
    token_ratio_variance = torch.var(ratio_token * loss_mask, dim=-1).mean()
    sequence_ratio_variance = torch.var(ratio_sequence * loss_mask, dim=-1).mean()

    # The key insight: GSPO should reduce within-sequence variance by using sequence-averaged ratios
    assert sequence_ratio_variance < token_ratio_variance, (
        f"GSPO should reduce ratio variance: sequence={sequence_ratio_variance:.4f} < "
        f"token={token_ratio_variance:.4f}"
    )

    # Token-level and sequence-level should give different results due to different importance weighting
    assert not torch.allclose(
        loss_token, loss_sequence, rtol=1e-3
    ), f"Loss values should differ: token={loss_token:.6f} vs sequence={loss_sequence:.6f}"

    # Test length normalization effect: sequences with different lengths should be handled more uniformly
    # This is a key stability benefit of GSPO mentioned in the paper
    seq_lengths = loss_mask.sum(dim=-1)  # [5, 2, 4]

    # In GSPO, the sequence-level importance weights should be the same across all tokens in a sequence
    # This should make the treatment more uniform across different sequence lengths
    for seq_idx in range(log_importance_weights_seq.shape[0]):
        seq_len = int(seq_lengths[seq_idx])
        if seq_len > 1:
            # All importance weights within a sequence should be identical (GSPO property)
            seq_weights = log_importance_weights_seq[seq_idx, :seq_len]
            assert torch.allclose(
                seq_weights, seq_weights[0], rtol=1e-6
            ), f"GSPO should have uniform importance weights within sequence {seq_idx}"
