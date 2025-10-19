"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_utils.py
"""

import pytest
from skyrl_train.generators.utils import apply_overlong_filtering


@pytest.mark.parametrize(
    "loss_masks,response_ids,eos_token_id,expected_masks",
    [
        # Test case 1: All responses end with eos token - masks should remain unchanged
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 4], [8, 9, 4]],  # All end with eos_token_id=4
            4,
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
        ),
        # Test case 2: No responses end with eos token - all masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1]],
            [[1, 2, 3, 5], [5, 6, 7, 8], [8, 9, 10]],  # None end with eos_token_id=4
            4,
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0]],
        ),
        # Test case 3: Mixed responses - only non-eos ending masks should be zeroed
        (
            [[1, 1, 0, 1], [0, 1, 1, 1], [1, 0, 1, 0, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 9, 10, 11, 4]],  # First and third end with eos_token_id=4
            4,
            [[1, 1, 0, 1], [0, 0, 0, 0], [1, 0, 1, 0, 1]],
        ),
        # Test case 4: Empty responses should be zeroed
        (
            [[1, 1], [1, 0, 1], [0, 1, 1, 1]],
            [[], [1, 2, 3], [4, 5, 6, 7]],  # Empty, no eos, no eos (eos_token_id=4)
            4,
            [[0, 0], [0, 0, 0], [0, 0, 0, 0]],
        ),
        # Test case 5: Empty lists
        ([], [], 4, []),
        # Test case 6: Different eos token id
        (
            [[1, 1], [1, 0, 1], [0, 1, 1, 1]],
            [[1, 2], [3, 4, 99], [5, 6, 7, 99]],  # Second and third end with eos_token_id=99
            99,
            [[0, 0], [1, 0, 1], [0, 1, 1, 1]],
        ),
    ],
)
def test_apply_overlong_filtering(loss_masks, response_ids, eos_token_id, expected_masks):
    """
    Test the apply_overlong_filtering function which implements DAPO Overlong Filtering.

    This function should zero-out every token's mask whenever the response does not end
    with the eos token id (i.e. truncated), while leaving other masks unchanged.
    """
    result = apply_overlong_filtering(loss_masks, response_ids, eos_token_id)

    assert result == expected_masks, f"Expected {expected_masks}, but got {result}"

    # Verify that the original inputs are not modified (immutability check)
    assert len(result) == len(loss_masks), "Result should have same length as input"

    # Check that each individual mask is processed correctly
    for i, (original_mask, response, expected_mask) in enumerate(zip(loss_masks, response_ids, expected_masks)):
        if len(response) == 0 or response[-1] != eos_token_id:
            # Should be all zeros with same length as original
            assert result[i] == [0] * len(original_mask), f"Mask {i} should be all zeros for truncated response"
        else:
            # Should be unchanged
            assert result[i] == original_mask, f"Mask {i} should be unchanged for response ending with eos token"


def test_apply_overlong_filtering_immutability():
    """
    Test that apply_overlong_filtering doesn't modify the original input lists.
    """
    original_loss_masks = [[1, 1, 0, 1], [0, 1, 1]]
    original_response_ids = [[1, 2, 3, 4], [5, 6, 7]]  # First ends with eos=4, second doesn't
    eos_token_id = 4

    # Create copies to compare against later
    loss_masks_copy = [mask[:] for mask in original_loss_masks]  # Deep copy of lists
    response_ids_copy = [response[:] for response in original_response_ids]  # Deep copy of lists

    result = apply_overlong_filtering(original_loss_masks, original_response_ids, eos_token_id)

    # Verify original inputs are unchanged
    assert original_loss_masks == loss_masks_copy, "Original loss_masks should not be modified"
    assert original_response_ids == response_ids_copy, "Original response_ids should not be modified"

    # Verify result is correct
    expected = [[1, 1, 0, 1], [0, 0, 0]]  # Second mask zeroed due to not ending with eos
    assert result == expected, f"Expected {expected}, got {result}"


@pytest.mark.parametrize(
    "loss_masks,response_ids",
    [
        # Test case 1: More loss_masks than response_ids
        ([[1, 1], [0, 1]], [[1, 2]]),
        # Test case 2: More response_ids than loss_masks
        ([[1, 1]], [[1, 2], [3, 4]]),
        # Test case 3: Empty loss_masks but non-empty response_ids
        ([], [[1, 2]]),
        # Test case 4: Non-empty loss_masks but empty response_ids
        ([[1, 0]], []),
    ],
)
def test_apply_overlong_filtering_length_mismatch_assertion(loss_masks, response_ids):
    """
    Test that apply_overlong_filtering raises AssertionError when loss_masks and response_ids
    have different lengths.
    """
    eos_token_id = 4
    with pytest.raises(AssertionError, match="loss_masks and response_ids must have the same length"):
        apply_overlong_filtering(loss_masks, response_ids, eos_token_id)
