import pytest
import torch
from skyrl_train.training_batch import TensorBatch
import pickle
import ray
import numpy as np


def test_train_batch_initialization():
    # Test basic initialization
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)

    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "response_mask": response_mask,
        }
    )
    assert isinstance(data, TensorBatch)
    assert data.batch_size == batch_size
    assert torch.equal(data["sequences"], sequences)
    assert torch.equal(data["attention_mask"], attention_mask)


def test_train_batch_validation():
    # Test validation of batch sizes
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size - 1, seq_len)  # Different size

    with pytest.raises(ValueError, match="Batch size mismatch"):
        batch = TensorBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
            }
        )
        TensorBatch(batch=batch, metadata={})


def test_train_batch_chunk():
    batch_size = 4
    seq_len = 3
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    chunks = data.chunk(2)
    assert len(chunks) == 2
    assert chunks[0].batch_size == 2
    assert chunks[1].batch_size == 2
    assert chunks[0]["sequences"].shape == (2, seq_len)
    assert chunks[1]["sequences"].shape == (2, seq_len)


def test_train_batch_slice():
    batch_size = 4
    seq_len = 3
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    sliced = data.slice(1, 3)
    assert len(sliced) == 2
    assert sliced["sequences"].shape == (2, seq_len)
    assert sliced["attention_mask"].shape == (2, seq_len)


def test_train_batch_to_dtype():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = None
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    data.to(dtype=torch.float16)
    assert data["sequences"].dtype == torch.float16
    assert data["attention_mask"] is None


def test_train_batch_select():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    metadata = {"info": "test", "extra": "data"}

    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }
    )
    data.metadata = metadata

    selected = data.select(["sequences", "attention_mask"], ["info"])
    assert "sequences" in selected
    assert "attention_mask" in selected
    assert "loss_mask" not in selected
    assert "info" in selected.metadata
    assert "extra" not in selected.metadata


def test_train_batch_cat():
    batch_size = 3
    seq_len = 4
    sequences1 = torch.randn(batch_size, seq_len)
    attention_mask1 = torch.ones(batch_size, seq_len)
    data1 = TensorBatch(
        {
            "sequences": sequences1,
            "attention_mask": attention_mask1,
        }
    )
    sequences2 = torch.randn(batch_size, seq_len)
    attention_mask2 = torch.ones(batch_size, seq_len)
    data2 = TensorBatch(
        {
            "sequences": sequences2,
            "attention_mask": attention_mask2,
        }
    )

    concatenated = data1.cat([data1, data2])
    assert len(concatenated) == 2 * batch_size
    assert concatenated["sequences"].shape == (2 * batch_size, seq_len)
    assert concatenated["attention_mask"].shape == (2 * batch_size, seq_len)


def test_train_batch_pickle():
    # Test pickle serialization
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )
    metadata = {"info": "test"}
    data.metadata = metadata

    # Serialize
    pickled = pickle.dumps(data)

    # Deserialize
    unpickled = pickle.loads(pickled)

    # Verify all components are preserved
    assert len(unpickled) == len(data)
    assert all(torch.equal(unpickled[k], data[k]) for k in data.keys())
    assert unpickled.metadata == data.metadata


def test_train_batch_setitem():
    batch_size = 3
    seq_len = 4
    sequences = torch.randn(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len)
    data = TensorBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )

    # Test setting tensor
    new_sequences = torch.randn(batch_size, seq_len)
    data["sequences"] = new_sequences
    assert torch.equal(data["sequences"], new_sequences)

    # Test invalid tensor shape
    with pytest.raises(ValueError, match="Batch size mismatch"):
        data["sequences"] = torch.randn(batch_size + 1, seq_len)

    # Test invalid types
    # 1. string
    with pytest.raises(ValueError, match="must be a tensor"):
        data["sequences"] = "invalid"
    # 2. numpy array
    with pytest.raises(ValueError, match="must be a tensor"):
        data["sequences"] = np.zeros((batch_size, seq_len))


def test_train_batch_ray_serialization():
    data = TensorBatch(
        **{"a": torch.tensor([1.2, 2.4, 3.6, 4.8]), "b": torch.tensor([4, 5, 6, 7])},
    )
    data.metadata = {"hello": "world"}

    def _task(inp: TensorBatch):
        assert inp == data

    _inp_ray = ray.put(data)
    ray.remote(_task).remote(_inp_ray)


def test_train_batch_repeat():
    batch = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
    data = TensorBatch(**batch)
    data.metadata = {"d": 1, "e": "test"}
    repeated = data.repeat(2)
    assert len(repeated) == 6
    assert torch.equal(repeated["a"], torch.tensor([1, 2, 3, 1, 2, 3]))
    assert torch.equal(repeated["b"], torch.tensor([4, 5, 6, 4, 5, 6]))
    assert repeated.metadata == {"d": 1, "e": "test"}


def test_train_batch_repeat_interleave():
    batch = {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])}
    data = TensorBatch(**batch)
    data.metadata = {"c": "test"}
    repeated = data.repeat_interleave(2)
    assert len(repeated) == 6
    assert torch.equal(repeated["a"], torch.tensor([1, 1, 2, 2, 3, 3]))
    assert torch.equal(repeated["b"], torch.tensor([4, 4, 5, 5, 6, 6]))
    assert repeated.metadata == {"c": "test"}


def test_train_batch_get_item():
    batch = {"a": torch.tensor([1, 2, 3, 4]), "b": torch.tensor([4, 5, 6, 7])}
    data = TensorBatch(**batch)
    data.metadata = {"c": "test"}
    new_data = data[:2]
    assert torch.equal(new_data["a"], torch.tensor([1, 2]))
    assert torch.equal(new_data["b"], torch.tensor([4, 5]))
