import pytest
import skyrl_gym
import json
from omegaconf import DictConfig


@pytest.mark.parametrize(
    "model_response, tests, expected_reward",
    [
        # Correct code: second largest index
        (
            """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

if __name__ == "__main__":
    main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                    {"input": "3\n3 2 1\n", "output": "2\n", "testtype": "stdin"},
                ]
            ),
            1.0,
        ),
        # Wrong logic: returns index of largest
        (
            """```python
def main():
    N = int(input())
    A = list(map(int, input().split()))
    print(A.index(max(A)) + 1)

if __name__ == "__main__":
    main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
        # Missing main() call — runtime error
        (
            """```python
def main():
    A = list(map(int, input().split()))
    B = sorted(A, reverse=True)
    second = B[1]
    print(A.index(second) + 1)

# forgot to call main()
```""",
            json.dumps(
                [
                    {"input": "4\n8 2 5 1\n", "output": "3\n", "testtype": "stdin"},
                ]
            ),
            0.0,
        ),
    ],
)
def test_compute_score(model_response, tests, expected_reward):
    env = skyrl_gym.make(
        "lcb",
        env_config=DictConfig({"env_class": "lcb"}),
        extras={"reward_spec": {"method": "rule", "ground_truth": tests}},
    )
    # Skip init() since it's not used in this test
    step_output = env.step(model_response)
    assert step_output["reward"] == expected_reward
