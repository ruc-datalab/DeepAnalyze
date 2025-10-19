"""Defines interfaces for training data."""

from typing import TypedDict, Dict, Any, List, Optional, Generic, TypeVar
import torch
from jaxtyping import Float, Integer
import pickle
import io

DictType = TypeVar("DictType")


# NOTE (sumanthrh): This is inspired by `TensorDict` but is much simpler.
class TensorBatch(dict, Generic[DictType]):
    """Base class for training batches

    This defines a generic container for a batch of training data (inputs or outputs).
    Consists of a dictionary of tensors along with some metadata.
    """

    metadata: Optional[Dict[str, Any]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._batch_size = None
        self._device = None
        self._check_consistency()

    def select(
        self, keys: List[str], metadata_keys: Optional[List[str]] = None
    ) -> "TensorBatch[DictType]":
        """Select a subset of the data batch.

        Args:
            keys: The keys to select
            metadata_keys: The metadata keys to select

        Returns:
            A new `TensorBatch` object with the selected keys and metadata
        """
        selected_batch_data = {}
        for key in keys:
            selected_batch_data[key] = self[key]
        selected_metadata = {}
        if metadata_keys is None:
            selected_metadata = self.metadata
        else:
            selected_metadata = {}
            for key in metadata_keys:
                selected_metadata[key] = self.metadata[key]
        new_batch = self.__class__(selected_batch_data)
        new_batch.metadata = selected_metadata
        return new_batch

    def _check_consistency(self):
        """Check consistency of all present fields"""
        keys = list(self.keys())
        if len(keys) == 0:
            return

        batch_size = len(self[keys[0]])
        self._batch_size = batch_size
        for key in keys:
            value = self[key]
            if value is None:
                continue
            self._device = value.device if self._device is None else self._device
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"Field {key} must be a tensor, got {type(value)}")
            if len(value) != batch_size:
                raise ValueError(f"Batch size mismatch in {key}")
            if value.device != self._device:
                raise ValueError(
                    f"Device mismatch in {key}. Expected {self._device}, got {value.device}"
                )

    def __getitem__(self, index) -> "TensorBatch[DictType]":
        if isinstance(index, slice):
            return self.slice(index.start, index.stop, index.step)
        elif isinstance(index, int):
            return self.slice(index, index + 1)
        else:
            return super().__getitem__(index)

    def __setitem__(self, key: str, value: Optional[torch.Tensor]) -> None:
        if value is None:
            super().__setitem__(key, value)
            return

        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Field {key} must be a tensor, got {type(value)}")

        if (
            hasattr(self, "_batch_size")
            and self._batch_size is not None
            and len(value) != self._batch_size
        ):
            raise ValueError(
                f"Batch size mismatch in {key}. Expected tensor to be of size {self._batch_size}, got {len(value)}."
            )

        super().__setitem__(key, value)

        if hasattr(self, "_batch_size") and self._batch_size is None:
            self._batch_size = len(value)

    def to(
        self,
        device: torch.device = None,
        dtype: torch.dtype = None,
        *,
        non_blocking: bool = False,
    ) -> "TensorBatch":
        """Move tensors to device and/or cast to dtype.

        Args:
            device: The device to move the tensors to
            dtype: The dtype to cast the tensors to
            non_blocking: Whether the operation should be non-blocking
        """
        for key, value in self.items():
            if value is None:
                continue
            assert isinstance(
                value, torch.Tensor
            ), f"Field {key} must be a tensor, got {type(value)}"
            self[key] = value.to(device, dtype, non_blocking=non_blocking)
        return self

    def contiguous(self) -> "TensorBatch":
        """Make the tensors contiguous"""
        for key, value in self.items():
            if value is None:
                continue
            # some of these asserts are not needed, but it's kept for type safety
            assert isinstance(
                value, torch.Tensor
            ), f"Field {key} must be a tensor, got {type(value)}"
            self[key] = value.contiguous()
        return self

    @property
    def batch_size(self) -> int:
        """Batch size for the tensors"""
        return self._batch_size

    @property
    def device(self) -> torch.device:
        """Get the device for the tensors"""
        return self._device

    def __getstate__(self):
        """Serialize the `TensorBatch` object for pickle protocol"""
        self.contiguous()
        if self._device is not None:
            assert self._device == torch.device(
                "cpu"
            ), "Tensors must be on CPU before serialization"
        batch_dict = {}
        for key, value in self.items():
            buffer = io.BytesIO()
            torch.save(value, buffer)
            batch_dict[key] = buffer.getvalue()

        return {
            "batch_dict": batch_dict,
            "batch_size": self._batch_size,
            "device": self._device,
            "metadata": self.metadata,
        }

    def __setstate__(self, state):
        """Deserialize the `TensorBatch` object and load it into memory"""
        for key, value in state["batch_dict"].items():
            buffer = io.BytesIO(value)
            self[key] = torch.load(buffer)

        self._batch_size = state["batch_size"]
        self._device = state["device"]
        self.metadata = state["metadata"]
        self._check_consistency()
        return self

    def repeat(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat` (and `numpy.tile`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            else:
                assert isinstance(
                    value, torch.Tensor
                ), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat(repeats)
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def repeat_interleave(self, repeats: int):
        """Repeat entries in the data batch a specified number of times.

        This is similar to `torch.repeat_interleave` (and `numpy.repeat`). `metadata` is not repeated.

        Args:
            repeats: The number of times to repeat the data batch

        Returns:
            A new `TensorBatch` object with the data repeated
        """
        new_batch = {}
        for key, value in self.items():
            if value is None:
                new_batch[key] = value
            else:
                assert isinstance(
                    value, torch.Tensor
                ), f"Field {key} must be a tensor, got {type(value)}"
                new_batch[key] = value.repeat_interleave(repeats)
        new_batch = self.__class__(new_batch)
        new_batch.metadata = self.metadata
        return new_batch

    def chunk(self, chunk_size: int) -> List["TensorBatch[DictType]"]:
        """Split into smaller chunks"""
        chunks = []
        for i in range(0, self.batch_size, chunk_size):
            chunk_data = {}
            for key, value in self.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        chunk_data[key] = value[i : i + chunk_size]
                    else:
                        raise ValueError(
                            f"Unsupported type {type(value)} for key {key}"
                        )
                else:
                    # `None` values are not chunked
                    chunk_data[key] = value
            chunk = self.__class__(chunk_data)
            chunk.metadata = self.metadata
            chunks.append(chunk)
        return chunks

    def slice(self, start: int, end: int, step: int = 1) -> "TensorBatch[DictType]":
        """Slice the data batch.

        Args:
            start: The start index
            end: The end index
            step: The step size

        Returns:
            A new `TensorBatch` object with the view of the specified slice.
        """
        slice_obj = slice(start, end, step)
        sliced_data = {}
        for key, value in self.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    sliced_data[key] = value[slice_obj]
                else:
                    raise ValueError(f"Unsupported type {type(value)} for key {key}")
            else:
                # `None` values are not sliced
                sliced_data[key] = value
        sliced_batch = self.__class__(sliced_data)
        sliced_batch.metadata = self.metadata
        return sliced_batch

    def save(self, path: str):
        """Save the data to a pickle file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def load(self, path: str):
        """Load the data from a pickle file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def cat(cls, shards: List["TensorBatch[DictType]"]) -> "TensorBatch[DictType]":
        """Concatenate shards.

        Args:
            shards: The list of `TensorBatch` objects to cat

        Returns:
            A new `TensorBatch` object with the concatenated data
        """
        cat_data = {}
        assert len(shards) > 0, "Cannot cat an empty list of shards"
        for key, value in shards[0].items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    cat_data[key] = torch.cat([shard[key] for shard in shards])
                else:
                    raise ValueError(f"Unsupported type {type(value)} for key {key}")
            else:
                # `None` values are not cat'd
                cat_data[key] = value
        metadata = shards[0].metadata
        cat_batch = cls(cat_data)
        cat_batch.metadata = metadata
        return cat_batch

    def __len__(self) -> int:
        """Length of the batch.

        Note that this is the same as the batch size rather than the number of keys in the batch.
        """
        return self._batch_size

    def __eq__(self, other: Any) -> bool:
        """Check if two `TensorBatch` objects are equal"""
        if not isinstance(other, TensorBatch):
            return False
        if self.metadata != other.metadata:
            return False
        if len(self) != len(other):
            return False
        if len(self.items()) != len(other.items()):
            return False
        for k, v in self.items():
            if k not in other or not torch.equal(v, other[k]):
                return False
        return True

    def __str__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return f"TensorBatch(batch_size={self.batch_size}, device={self.device}, metadata={self.metadata}), items={self.items()}"

    def __repr__(self) -> str:
        """String representation of the `TensorBatch` object"""
        return self.__str__()


class TrainingInput(TypedDict, total=False):
    """Schema for training input batch"""

    sequences: Integer[torch.Tensor, "batch_size seq_len"]
    attention_mask: Integer[torch.Tensor, "batch_size seq_len"]
    loss_mask: Integer[torch.Tensor, "batch_size seq_len"]
    response_mask: Integer[torch.Tensor, "batch_size seq_len"]
    action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    base_action_log_probs: Float[torch.Tensor, "batch_size seq_len"]
    values: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    returns: Float[torch.Tensor, "batch_size seq_len"]
    advantages: Float[torch.Tensor, "batch_size seq_len"]
    kl: Float[torch.Tensor, "batch_size seq_len"]
    rm_rewards: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    custom_rewards: Optional[Float[torch.Tensor, "batch_size seq_len"]]
    rollout_logprobs: Optional[Float[torch.Tensor, "batch_size seq_len"]]


class TrainingInputBatch(TensorBatch[TrainingInput]):
    """Training input data"""

    pass


class TrainingOutputBatch(TensorBatch[Dict[str, torch.Tensor]]):
    """Training output data"""

    pass
