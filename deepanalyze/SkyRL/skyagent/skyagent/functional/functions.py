import logging

logger = logging.getLogger(__name__)


def compute_position_id_with_mask(mask):
    return torch.clip(torch.cumsum(mask, dim=-1) - 1, min=0, max=None)


def convert_right_padding_to_left(
    tokenizer, input_ids, attention_mask, device, max_len=None
):
    """
    Converts right-padded tensors to left-padded tensors with optional custom length.

    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        input_ids (torch.Tensor): Right-padded input IDs tensor of shape [batch_size, seq_length]
        attention_mask (torch.Tensor): Right-padded attention mask tensor of shape [batch_size, seq_length]
        device: The device to place the new tensors on
        max_len (int, optional): The desired maximum length of the returned tensors.
                                If None, uses the original sequence length.

    Returns:
        tuple: (left_padded_input_ids, left_padded_attention_mask)
    """
    batch_size, orig_seq_length = input_ids.size()

    # Use original length if max_len is not specified
    seq_length = max_len if max_len is not None else orig_seq_length

    # Create new tensors with the desired size
    left_padded_input_ids = torch.full(
        (batch_size, seq_length),
        tokenizer.pad_token_id,
        dtype=input_ids.dtype,
        device=device,
    )
    left_padded_attention_mask = torch.zeros(
        (batch_size, seq_length), dtype=attention_mask.dtype, device=device
    )

    for i in range(batch_size):
        # Get the non-padded length of this sequence
        seq_len = attention_mask[i].sum().item()

        # Trim sequence if it's longer than max_len
        if seq_len > seq_length:
            logger.warning(f"Trimming sequence length from {seq_len} to {seq_length}")
            seq_len = seq_length

        # Calculate the offset for left padding
        offset = seq_length - seq_len

        # Copy the non-padded tokens to the end
        left_padded_input_ids[i, offset:] = input_ids[i, :seq_len]
        left_padded_attention_mask[i, offset:] = (
            1  # Set attention mask for non-padding tokens
        )

    return left_padded_input_ids, left_padded_attention_mask


def pad_to_max_length_right(tokenizer, encodings, max_length, device):
    """
    Pads tokenizer outputs to a specific maximum length with configurable padding side.

    Args:
        tokenizer: The tokenizer object with pad_token_id attribute
        encodings (dict): Dictionary containing 'input_ids', 'attention_mask', and optionally 'assistant_masks'
        max_length (int): The desired maximum length to pad to
        device: The device to place the tensors on

    Returns:
        dict: Dictionary with padded tensors for 'input_ids', 'attention_mask', and 'assistant_masks' if present
    """
    batch_size = len(encodings["input_ids"])

    # Initialize output tensors
    padded_input_ids = torch.full(
        (batch_size, max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
        device=device,
    )
    padded_attention_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )
    padded_assistant_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.long, device=device
    )

    # Fill tensors with actual values
    num_trimmed = 0
    for i in range(batch_size):
        seq_len = (
            encodings["attention_mask"][i].sum().item()
            if isinstance(encodings["attention_mask"][i], torch.Tensor)
            else sum(encodings["attention_mask"][i])
        )
        # Trim if longer than max_length
        actual_len = min(seq_len, max_length)
        if seq_len > max_length:
            logger.warning(
                f"Trimming sequence length from {seq_len} to {actual_len} for batch item {i}"
            )
            num_trimmed += 1

        # Right padding - copy sequence data to the beginning
        padded_input_ids[i, :actual_len] = torch.tensor(
            encodings["input_ids"][i][:actual_len], device=device
        )
        padded_attention_mask[i, :actual_len] = torch.tensor(
            encodings["attention_mask"][i][:actual_len], device=device
        )
        padded_assistant_mask[i, :actual_len] = torch.tensor(
            encodings["assistant_masks"][i][:actual_len], device=device
        )

    logger.info(
        f"Trimmed {num_trimmed*100 / max(batch_size, 1)}% of samples in the batch of size {batch_size}"
    )
    return padded_input_ids, padded_attention_mask, padded_assistant_mask
