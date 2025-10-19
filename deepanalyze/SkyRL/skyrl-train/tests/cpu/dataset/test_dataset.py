import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset
from skyrl_train.dataset import PromptDataset


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.side_effect = lambda x, add_generation_prompt: x
    return tokenizer


@pytest.fixture
def sample_dataset():
    # 3 samples: one too long, two valid
    data = {
        "prompt": [
            "short prompt",  # length 13
            "a" * 120,  # length 120
            "b" * 200,  # length 200 (to be filtered out if max len < 200)
        ],
        "answer": ["a1", "a2", "a3"],
    }
    return Dataset.from_dict(data)


@patch("datasets.load_dataset")
def test_prompt_dataset_filtering(mock_load_dataset, mock_tokenizer, sample_dataset):
    mock_load_dataset.return_value = {"train": sample_dataset}

    dataset = PromptDataset(
        data_files=["dummy1.parquet"],
        tokenizer=mock_tokenizer,
        max_prompt_length=150,  # should exclude third item
        num_workers=1,
        prompt_key="prompt",
        env_class_key="env_class",
    )

    # Only first two prompts should remain
    assert len(dataset) == 2
    messages, env, extra = dataset[0]
    assert env is None
    assert messages == "short prompt"
    assert extra == {"answer": "a1"}


def test_collate_fn():
    dataset = PromptDataset.__new__(PromptDataset)  # Bypass __init__
    sample_data = [("prompt 1", "env", {"answer": "a1"}), ("prompt 2", "env", {"answer": "a2"})]
    expected = [
        {"prompt": "prompt 1", "env_class": "env", "env_extras": {"answer": "a1"}},
        {"prompt": "prompt 2", "env_class": "env", "env_extras": {"answer": "a2"}},
    ]

    output = dataset.collate_fn(sample_data)
    assert output == expected
