import torch
from unittest.mock import patch
from skyrl_train.distributed.ulysses.monkey_patch import _ulysses_flash_attention_forward


def test_basic_forward_no_parallel():
    """Test basic forward pass without sequence parallelism."""
    batch_size = 2
    seq_len = 4
    n_heads = 8
    head_dim = 64

    # Create input tensors
    query_states = torch.randn(batch_size, seq_len, n_heads, head_dim)
    key_states = torch.randn(batch_size, seq_len, n_heads, head_dim)
    value_states = torch.randn(batch_size, seq_len, n_heads, head_dim)
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    # Mock the original flash attention forward
    with patch("skyrl_train.distributed.ulysses.monkey_patch._flash_attention_forward") as mock_fa:
        mock_fa.return_value = torch.randn_like(query_states)

        # Mock sequence parallel size to be 1
        with patch(
            "skyrl_train.distributed.ulysses.monkey_patch.get_ulysses_sequence_parallel_world_size", return_value=1
        ):
            _ulysses_flash_attention_forward(query_states, key_states, value_states, position_ids=position_ids)

            # Verify the original flash attention was called with correct inputs
            mock_fa.assert_called_once()
            args, kwargs = mock_fa.call_args
            assert torch.equal(args[0], query_states)
            assert torch.equal(args[1], key_states)
            assert torch.equal(args[2], value_states)
            assert torch.equal(kwargs["position_ids"], position_ids)


# NOTE (sumanthrh): This test is extremely mocking heavy.
# Extensive mocking is fine here because this unit test is simply checking output metadata i.e tensor shapes
def test_sequence_parallel_forward():
    """Test forward pass with sequence parallelism."""
    batch_size = 2
    seq_len = 8
    n_heads = 8
    head_dim = 64
    sp_size = 2

    # Create input tensors (already sharded for sequence parallel)
    query_states = torch.randn(batch_size, seq_len // sp_size, n_heads, head_dim)
    key_states = torch.randn(batch_size, seq_len // sp_size, n_heads // sp_size, head_dim)
    value_states = torch.randn(batch_size, seq_len // sp_size, n_heads // sp_size, head_dim)
    position_ids = torch.arange(seq_len // sp_size).unsqueeze(0).repeat(batch_size, 1)

    # Mock the original flash attention forward
    with patch("skyrl_train.distributed.ulysses.monkey_patch._flash_attention_forward") as mock_fa:
        mock_fa.return_value = torch.randn(batch_size, seq_len, n_heads // sp_size, head_dim)

        # Mock sequence parallel size and communication functions
        with (
            patch(
                "skyrl_train.distributed.ulysses.monkey_patch.get_ulysses_sequence_parallel_world_size",
                return_value=sp_size,
            ),
            patch(
                "skyrl_train.distributed.ulysses.monkey_patch.gather_seq_scatter_heads"
            ) as mock_gather_seq_scatter_heads,
            patch(
                "skyrl_train.distributed.ulysses.monkey_patch.gather_heads_scatter_seq"
            ) as mock_gather_heads_scatter_seq,
            patch("torch.distributed.all_gather") as mock_all_gather,
        ):

            # Mock the gather/scatter operations
            mock_gather_seq_scatter_heads.side_effect = lambda x, **kwargs: x[:, :, : n_heads // sp_size, :].repeat(
                1, sp_size, 1, 1
            )
            mock_gather_heads_scatter_seq.side_effect = lambda x, **kwargs: x[:, : seq_len // sp_size, :, :].repeat(
                1, 1, sp_size, 1
            )
            # Do nothing - output list remains as is
            mock_all_gather.side_effect = lambda output_list, input_tensor, **kwargs: None

            output = _ulysses_flash_attention_forward(query_states, key_states, value_states, position_ids=position_ids)

            # Verify the communication operations were called
            assert mock_gather_seq_scatter_heads.call_count == 3  # Called for q, k, v
            assert mock_gather_heads_scatter_seq.call_count == 1  # Called for output
            assert mock_all_gather.call_count == 1  # Called for position_ids
            assert output.shape == (batch_size, seq_len // sp_size, n_heads, head_dim)
            # Verify the original flash attention was called
            mock_fa.assert_called_once()
            args, kwargs = mock_fa.call_args
            assert args[0].shape == (batch_size, seq_len, n_heads // sp_size, head_dim)
            assert args[1].shape == (batch_size, seq_len, n_heads // sp_size, head_dim)
            assert args[2].shape == (batch_size, seq_len, n_heads // sp_size, head_dim)
            assert kwargs["position_ids"].shape == (batch_size, seq_len)
