from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
import re


class MultiplyEnv(BaseTextEnv):
    """
    Environment for multiplication.
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert (
            "ground_truth" in extras["reward_spec"]
        ), "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        self.max_turns = extras["max_turns"] if "max_turns" in extras else 5

    def _parse_action(self, action: str) -> str:
        match = re.search(r"\\boxed\{([^}]+)\}", action)
        return match.group(1) if match else None

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        answer = self._parse_action(action)

        is_correct = (
            answer is not None and answer.strip() == str(self.ground_truth).strip()
        )
        found_boxed = answer is not None

        done = self.turns >= self.max_turns or is_correct
        if is_correct:
            reward = 1.0
        elif found_boxed:
            reward = 0.5
        else:
            reward = 0.0

        if done:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=True,
                metadata={"parsed_answer": answer},
            )

        if answer is not None:
            # Agent provided an answer but it's incorrect
            feedback = f"Your answer '{answer}' is incorrect. Please try again."
        else:
            # Agent didn't provide a boxed answer
            feedback = "Please provide your answer in the format \\boxed{your_answer}."

        new_obs = {"role": "user", "content": feedback}

        return BaseTextEnvStepOutput(
            observations=[new_obs],
            reward=0.0,
            done=False,
            metadata={"parsed_answer": answer},
        )
