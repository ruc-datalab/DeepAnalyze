import skyrl_gym
import pytest
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "output, ground_truth, expected",
    [
        ("The answer is #### 42", "42", 1.0),
        ("The answer is #### 42", "43", 0.0),
        # answer is not in the expected format
        ("The answer is 42", "42", 0.0),
    ],
)
def test_compute_score(output, ground_truth, expected):
    env = skyrl_gym.make(
        "gsm8k",
        env_config=DictConfig({"env_class": "gsm8k"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": ground_truth}},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(output)
    assert step_output["reward"] == expected
