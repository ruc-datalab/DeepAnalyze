from skyrl_train.training_batch import TrainingInputBatch
import torch


def test_data_to_device():
    batch = {
        "sequences": torch.tensor([1, 2, 3]),
        "attention_mask": torch.tensor([4, 5, 6]),
    }
    data = TrainingInputBatch(batch)
    # in-place
    data.to(device="cuda")
    assert data["sequences"].device.type == "cuda"
    assert data["attention_mask"].device.type == "cuda"
