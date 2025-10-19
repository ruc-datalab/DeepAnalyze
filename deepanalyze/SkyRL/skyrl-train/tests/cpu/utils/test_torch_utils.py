from skyrl_train.utils.torch_utils import chunked_cross_entropy_from_log_probs, chunked_entropy_from_logits
import torch
import math


def test_chunked_cross_entropy_from_logprobs():
    # Define a small log-probability tensor (batch_size=2, seqlen=3, vocab_size=4)
    logits = [
        [
            [1.0, 2.0, 3.0, 4.0],  # example 1, token 1
            [1.0, 0.0, 0.0, 0.0],  # token 2
            [0.0, 0.0, 0.0, 0.0],
        ],  # token 3 (uniform)
        [
            [0.0, 0.0, 0.0, 0.0],  # example 2, token 1 (uniform)
            [1.0, 2.0, 3.0, 4.0],  # token 2
            [4.0, 3.0, 2.0, 1.0],
        ],  # token 3
    ]
    logits = torch.tensor(logits, dtype=torch.float32)
    logprobs_BSV = torch.log_softmax(logits, dim=-1)  # shape: (2, 3, 4)

    result_BS = chunked_cross_entropy_from_log_probs(logprobs_BSV)

    # For uniform logprobs (all zeros before softmax), entropy should be log(vocab_size) = log(4)
    expected_uniform_entropy = math.log(4.0)  # ≈ 1.386

    assert torch.allclose(result_BS[0, 2], torch.tensor(expected_uniform_entropy), atol=1e-4)
    assert torch.allclose(result_BS[1, 0], torch.tensor(expected_uniform_entropy), atol=1e-4)


def test_chunked_entropy_from_logits():
    # Define a small log-probability tensor (batch_size=2, seqlen=3, vocab_size=4)
    logits = [
        [
            [1.0, 2.0, 3.0, 4.0],  # example 1, token 1
            [1.0, 0.0, 0.0, 0.0],  # token 2
            [0.0, 0.0, 0.0, 0.0],
        ],  # token 3 (uniform)
        [
            [0.0, 0.0, 0.0, 0.0],  # example 2, token 1 (uniform)
            [1.0, 2.0, 3.0, 4.0],  # token 2
            [4.0, 3.0, 2.0, 1.0],
        ],  # token 3
    ]
    logits = torch.tensor(logits, dtype=torch.float32)

    result_BS = chunked_entropy_from_logits(logits)

    # For uniform logprobs (all zeros before softmax), entropy should be log(vocab_size) = log(4)
    expected_uniform_entropy = math.log(4.0)  # ≈ 1.386

    assert torch.allclose(result_BS[0, 2], torch.tensor(expected_uniform_entropy), atol=1e-4)
    assert torch.allclose(result_BS[1, 0], torch.tensor(expected_uniform_entropy), atol=1e-4)
