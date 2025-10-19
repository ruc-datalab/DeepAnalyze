from skyrl_train.models import Actor
import torch
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer
import ray
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
from skyrl_train.workers.worker import set_ulysses_sequence_parallel_group, apply_monkey_patch

MODEL_NAME = "llamafactory/tiny-random-Llama-3"
TOKENIZER_NAME = "unsloth/Meta-Llama-3.1-8B"


def get_dummy_inputs():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    input_without_padding = [10, 14, 200, 3, 40, tokenizer.eos_token_id]
    input_ids = [tokenizer.pad_token_id] * 3 + input_without_padding + [tokenizer.pad_token_id]
    attention_mask = [0] * 3 + [1] * len(input_without_padding) + [0]
    num_actions = 4
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask).unsqueeze(0)
    input_without_padding = torch.tensor(input_without_padding).unsqueeze(0)
    return input_ids, attention_mask, num_actions, input_without_padding


@patch("skyrl_train.models.logprobs_from_logits")
def test_actor_model_fwd_with_sample_packing(mock_logprobs_from_logits):
    model = Actor(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=True,
        bf16=False,
        sequence_parallel_size=1,
        use_sample_packing=True,
    )
    model.model.eval()
    model.model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    input_ids, attention_mask, num_actions, input_without_padding = get_dummy_inputs()
    actual_actions = input_ids[:, -num_actions:]
    # mock the actual forward
    input_ids, attention_mask, input_without_padding = (
        input_ids.to("cuda"),
        attention_mask.to("cuda"),
        input_without_padding.to("cuda"),
    )
    # forward is with packed sequence length
    model.model.forward = MagicMock(return_value={"logits": torch.randn(1, 6, 1000).to("cuda")})
    # just roll the inputs over for return value. This means that the model is correctly predicting the next token.
    # The last token will be the prediction beyond the eos token, just set to pad token id
    mock_return_value = torch.roll(input_without_padding, shifts=-1, dims=-1)
    mock_return_value[:, -1] = tokenizer.pad_token_id
    mock_logprobs_from_logits.return_value = mock_return_value

    action_log_probs = model(input_ids, num_actions, attention_mask)
    expected_log_probs = actual_actions.to("cuda")
    assert torch.equal(
        action_log_probs, expected_log_probs
    ), f"Expected log probs to be {expected_log_probs} but got {action_log_probs}"


@patch("skyrl_train.models.logprobs_from_logits")
def test_actor_model_fwd_without_sample_packing(mock_logprobs_from_logits):
    model = Actor(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=True,
        bf16=False,
        sequence_parallel_size=1,
        use_sample_packing=False,
    )
    model.model.eval()
    model.model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    input_ids, attention_mask, num_actions, input_without_padding = get_dummy_inputs()
    actual_actions = input_ids[:, -num_actions:]
    # mock the actual forward
    input_ids, attention_mask, input_without_padding = (
        input_ids.to("cuda"),
        attention_mask.to("cuda"),
        input_without_padding.to("cuda"),
    )
    # forward is with packed sequence length
    model.model.forward = MagicMock(return_value={"logits": torch.randn(1, 10, 1000).to("cuda")})
    # just roll the inputs over. This means that the model is correctly predicting the next token.
    mock_return_value = torch.roll(input_ids, shifts=-1, dims=-1)
    # The last token will be the prediction beyond the eos token, just set to pad token id
    mock_return_value[:, -1] = tokenizer.pad_token_id
    mock_logprobs_from_logits.return_value = mock_return_value

    action_log_probs = model(input_ids, num_actions, attention_mask)
    expected_log_probs = actual_actions.to("cuda")
    assert torch.equal(
        action_log_probs, expected_log_probs
    ), f"Expected log probs to be {expected_log_probs} but got {action_log_probs}"


@ray.remote(num_gpus=1)
class ActorTask:
    def __init__(self, rank, world_size, sequence_parallel_size):
        self.rank = rank
        self.world_size = world_size
        self.sequence_parallel_size = sequence_parallel_size

        # Initialize distributed environment
        dist.init_process_group(backend="nccl", init_method="tcp://localhost:23456", world_size=world_size, rank=rank)

        # Create model with sequence parallelism
        self.model = Actor(
            pretrain_or_model=MODEL_NAME,
            use_flash_attention_2=True,
            bf16=True,
            sequence_parallel_size=sequence_parallel_size,
            use_sample_packing=True,
        )
        group = _get_default_group()
        set_ulysses_sequence_parallel_group(group)
        apply_monkey_patch(model=self.model.model, ulysses_sp_size=self.sequence_parallel_size, use_parent_class=False)
        self.model.model.eval()
        self.model.model.to("cuda")

    def forward(self, input_ids, attention_mask, num_actions):
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        return self.model(input_ids, num_actions, attention_mask)

    def cleanup(self):
        dist.destroy_process_group()


def test_actor_model_fwd_with_sequence_parallelism(ray_init_fixture):

    # Create input sequence
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    input_ids = [
        [tokenizer.pad_token_id] * 3 + [10, 14, 200, 3, 40, tokenizer.eos_token_id] + [tokenizer.pad_token_id],
        [tokenizer.pad_token_id, 12, 13, 220, 1000, 3, 40, tokenizer.eos_token_id] + [tokenizer.pad_token_id] * 2,
    ]
    attention_mask = [[0] * 3 + [1] * 6 + [0], [0] + [1] * 7 + [0] * 2]
    num_actions = 4

    # Convert to tensors
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)

    # First run without sequence parallelism
    model_no_sp = Actor(
        pretrain_or_model=MODEL_NAME,
        use_flash_attention_2=True,
        bf16=True,
        sequence_parallel_size=1,
        use_sample_packing=True,
    )
    model_no_sp.model.eval()
    model_no_sp.model.to("cuda:0")

    # Forward pass without sequence parallelism
    with torch.no_grad():
        output_no_sp = model_no_sp(input_ids.to("cuda:0"), num_actions, attention_mask.to("cuda:0"))

    # Now run with sequence parallelism using Ray
    world_size = 2
    sequence_parallel_size = 2

    # Create Ray tasks
    actors = [ActorTask.remote(rank, world_size, sequence_parallel_size) for rank in range(world_size)]

    # Run forward pass with sequence parallelism
    outputs_sp = ray.get([actor.forward.remote(input_ids, attention_mask, num_actions) for actor in actors])

    # Verify outputs match
    # Since we're using sequence parallelism, each GPU processes half the sequence
    # and the outputs should be gathered and match the non-parallel output
    for i, output_sp in enumerate(outputs_sp):
        assert torch.allclose(
            output_sp, output_no_sp
        ), f"Outputs with sequence parallelism don't match outputs without sequence parallelism for rank {i}"

    # Cleanup
    ray.get([actor.cleanup.remote() for actor in actors])
    ray.shutdown()
