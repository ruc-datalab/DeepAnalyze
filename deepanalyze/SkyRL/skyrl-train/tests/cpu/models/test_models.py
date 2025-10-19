import torch
from flash_attn.bert_padding import pad_input, unpad_input
import pytest


@pytest.fixture
def input_ids():
    return torch.tensor(
        [[0, 0, 10, 12, 13, 14, 15, 17, 0, 0, 0], [0, 14, 6, 4, 50, 20, 18, 19, 21, 22, 23]], dtype=torch.long
    )


@pytest.fixture
def attention_mask():
    return torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)


@pytest.fixture
def position_ids(attention_mask):
    pos_ids = torch.cumsum(attention_mask, dim=1) - 1
    pos_ids.masked_fill_(attention_mask == 0, 1)
    return pos_ids


def test_flash_attention_sequence_packing(input_ids, attention_mask, position_ids):
    input_ids_packed, nnz_indices, _, _, _ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_packed = input_ids_packed.transpose(0, 1)
    position_ids_packed, _, _, _, _ = unpad_input(position_ids.unsqueeze(-1), attention_mask=attention_mask)
    position_ids_packed = position_ids_packed.transpose(0, 1)

    # (1, nnz)
    expected_input_ids_packed = torch.tensor(
        [[10, 12, 13, 14, 15, 17, 14, 6, 4, 50, 20, 18, 19, 21, 22, 23]], dtype=torch.long
    )
    # (1, nnz)
    expected_position_ids_packed = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    # (nnz,)
    expected_nnz_indices = torch.tensor([2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], dtype=torch.int64)

    assert torch.equal(expected_input_ids_packed, input_ids_packed)
    assert torch.equal(expected_position_ids_packed, position_ids_packed)
    assert torch.equal(nnz_indices, expected_nnz_indices)


def test_flash_attention_sequence_unpacking(input_ids, attention_mask, position_ids):
    # pack
    input_ids_packed, nnz_indices, _, _, _ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_packed = input_ids_packed.transpose(0, 1)
    position_ids_packed, _, _, _, _ = unpad_input(position_ids.unsqueeze(-1), attention_mask=attention_mask)
    position_ids_packed = position_ids_packed.transpose(0, 1)

    # unpack
    # add padding back - postprocess logits to be compatible with original tensors
    batch_size, seqlen = attention_mask.shape
    # (nnz, 1) -> (batch_size, seqlen, 1)
    unpacked_input_ids = pad_input(
        input_ids_packed.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
    )
    unpacked_position_ids = pad_input(
        position_ids_packed.transpose(0, 1), indices=nnz_indices, batch=batch_size, seqlen=seqlen
    )
    unpacked_input_ids = unpacked_input_ids.squeeze(-1)
    unpacked_position_ids = unpacked_position_ids.squeeze(-1)

    assert torch.equal(unpacked_input_ids, input_ids)
    # mask out the attention mask because the padding value used can differ
    assert torch.equal(unpacked_position_ids * attention_mask, position_ids * attention_mask)
