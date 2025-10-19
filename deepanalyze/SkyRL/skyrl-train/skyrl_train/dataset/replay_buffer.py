"""
Replay buffer utilities.
"""

# NOTE (sumanthrh): These are replay buffer utilities from OpenRLHF.
# While currently unused, we may need them in the future for more involved off-policy algorithms.
import copy
import random
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from jaxtyping import Float, Integer

import torch
import torch.nn.functional as F


BasicType = Union[int, float, str, bool]


def to(tensor: Union[torch.Tensor, List[torch.Tensor], BasicType], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    else:
        return tensor


def pin_memory(tensor: Union[torch.Tensor, List[torch.Tensor], BasicType]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.pin_memory()
    else:
        return tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions/ response length.
    """

    sequences: Integer[torch.Tensor, "batch seq_len"]
    action_log_probs: Float[torch.Tensor, "batch response_len"]
    base_action_log_probs: Optional[Float[torch.Tensor, "batch response_len"]]
    values: Optional[Float[torch.Tensor, "batch response_len"]]
    returns: Optional[Float[torch.Tensor, "batch response_len"]]
    advantages: Optional[Float[torch.Tensor, "batch response_len"]]
    attention_mask: Optional[Integer[torch.LongTensor, "batch seq_len"]]
    loss_mask: Optional[Integer[torch.LongTensor, "batch response_len"]]
    action_mask: Optional[Integer[torch.Tensor, "batch response_len"]]
    rollout_logprobs: Optional[Float[torch.Tensor, "batch response_len"]]
    num_actions: int
    info: Optional[dict]
    kl: Optional[Float[torch.Tensor, "batch response_len"]] = None
    metadata: Optional[Dict[str, Any]] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = to(self.base_action_log_probs, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.returns is not None:
            self.returns = to(self.returns, device)
        if self.advantages is not None:
            self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = to(self.attention_mask, device)
        if self.loss_mask is not None:
            self.loss_mask = to(self.loss_mask, device)
        if self.action_mask is not None:
            self.action_mask = to(self.action_mask, device)
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = to(self.rollout_logprobs, device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        if self.values is not None:
            self.values = pin_memory(self.values)
        if self.returns is not None:
            self.returns = pin_memory(self.returns)
        if self.advantages is not None:
            self.advantages = pin_memory(self.advantages)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.loss_mask is not None:
            self.loss_mask = self.loss_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = self.rollout_logprobs.pin_memory()
        return self


@dataclass
class BufferItem:
    """BufferItem is an item of experience data.

    Shapes of each tensor:
    sequences: (S)
    action_log_probs: (A)
    base_action_log_probs: (A)
    values: (1)
    returns: (1)
    advatanges: (1)
    attention_mask: (S)
    loss_mask: (A)
    action_mask: (A)

    "A" is the number of actions.
    """

    sequences: Integer[torch.Tensor, "seq_len"]  # noqa: F821
    action_log_probs: Float[torch.Tensor, "response_len"]  # noqa: F821
    base_action_log_probs: Optional[Float[torch.Tensor, "response_len"]]  # noqa: F821
    values: Optional[Float[torch.Tensor, "response_len"]]  # noqa: F821
    returns: Optional[Float[torch.Tensor, "response_len"]]  # noqa: F821
    advantages: Optional[Float[torch.Tensor, "response_len"]]  # noqa: F821
    attention_mask: Optional[Integer[torch.LongTensor, "seq_len"]]  # noqa: F821
    loss_mask: Optional[Integer[torch.LongTensor, "response_len"]]  # noqa: F821
    action_mask: Optional[Integer[torch.Tensor, "response_len"]]  # noqa: F821
    num_actions: int
    info: Optional[dict]

    def to_json(self) -> dict:
        def _to_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {k: _to_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_to_json(x) for x in obj]
            else:
                return obj

        return {k: _to_json(v) for k, v in self.__dict__.items()}


def split_experience_batch(experience: Experience) -> List[BufferItem]:
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "loss_mask",
        "action_mask",
        "num_actions",
    )
    if len(experience.sequences.shape) == 1:
        # no padding
        batch_size = 1
        for key in keys:
            value = getattr(experience, key)
            if value is not None:
                if isinstance(value, torch.Tensor):
                    value = value.unsqueeze(0)
                setattr(experience, key, value)
    else:
        batch_size = len(experience.sequences)
    batch_kwargs = [{} for _ in range(batch_size)]

    for key in keys:
        value = getattr(experience, key)
        if value is None:
            for i in range(batch_size):
                batch_kwargs[i][key] = None
            continue
        vals = value
        if isinstance(vals, torch.Tensor):
            vals = torch.unbind(vals)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            if isinstance(v, torch.Tensor):
                v = v.clone()
            batch_kwargs[i][key] = v

    for i in range(batch_size):
        batch_kwargs[i]["info"] = {}
    for k, v in experience.info.items():
        vals = torch.unbind(v)
        assert batch_size == len(vals)
        for i, vv in enumerate(vals):
            if isinstance(vv, torch.Tensor):
                assert (
                    vv.numel() == 1
                ), f"info[{k}] must be a scalar tensor, but got {vv.shape}"
                vv = vv.item()
            batch_kwargs[i]["info"][k] = vv

    items = [BufferItem(**kwargs) for kwargs in batch_kwargs]
    return items


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left"
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(0) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(0)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding))
    return torch.stack(padded_sequences, dim=0)


def make_experience_batch(items: List[BufferItem]) -> Experience:
    kwargs = {}
    keys = (
        "sequences",
        "action_log_probs",
        "base_action_log_probs",
        "values",
        "returns",
        "advantages",
        "attention_mask",
        "loss_mask",
        "action_mask",
        "num_actions",
    )
    for key in keys:
        vals = [getattr(item, key) for item in items]
        # NOTE (sumanthrh): Assumes list of Tensors
        batch_data = vals if vals[0] is not None else None
        kwargs[key] = batch_data

    kwargs["info"] = {}
    for key in items[0].info.keys():
        vals = torch.tensor([item.info[key] for item in items])
        kwargs["info"][key] = vals
    return Experience(**kwargs)


def remove_padding_in_sequences(items):
    for item in items:
        seq, act_log_prob, base_act_log_prob, value, ret, adv, att_mask, act_mask = (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        )
        right_pad = (1 - act_mask.long()).sum()
        right_pad = None if right_pad == 0 else -right_pad

        # left_pad for seq and att_mask
        left_pad = att_mask.long().argmax()
        (
            item.sequences,
            item.action_log_probs,
            item.base_action_log_probs,
            item.values,
            item.returns,
            item.advantages,
            item.attention_mask,
            item.action_mask,
        ) = (
            seq[left_pad:right_pad],
            act_log_prob[:right_pad],
            base_act_log_prob[:right_pad],
            value[:right_pad] if value is not None else None,
            ret[:right_pad] if ret is not None else None,
            adv[:right_pad] if adv is not None else None,
            att_mask[left_pad:right_pad] if att_mask is not None else None,
            act_mask[:right_pad] if act_mask is not None else None,
        )
    return items


class NaiveReplayBuffer(ABC):
    """Naive replay buffer class. It stores experience.

    Args:
        sample_batch_size (int): Batch size when sampling.
        limit (int, optional): Limit of number of experience samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Whether to offload experience to cpu when sampling. Defaults to True.
    """

    def __init__(
        self,
        sample_batch_size: int,
        limit: int = 0,
        cpu_offload: bool = True,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.target_device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.items: List[BufferItem] = []

    @torch.no_grad()
    def split_to_n_batches(
        self, n_batches: int, drop_last: bool = False
    ) -> List["NaiveReplayBuffer"]:
        assert n_batches > 0
        if not drop_last:
            batch_size = (len(self.items) + n_batches - 1) // n_batches
        else:
            batch_size = len(self.items) // n_batches
        if len(self.items) < batch_size * n_batches:
            # padding
            padding_size = batch_size * n_batches - len(self.items)
            padding_items = random.choices(self.items, k=padding_size)
            self.items.extend(padding_items)
        bfs = []
        items = copy.deepcopy(self.items)
        random.shuffle(items)
        for i in range(n_batches):
            bf = NaiveReplayBuffer(
                sample_batch_size=self.sample_batch_size,
                limit=self.limit,
                cpu_offload=self.cpu_offload,
            )
            bf.items = items[i * batch_size : (i + 1) * batch_size]
            bfs.append(bf)
        return bfs

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch)
        return experience
