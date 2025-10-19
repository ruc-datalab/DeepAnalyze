from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict
from skyrl_gym.envs.lcb.livecodebench import compute_score
import json
from omegaconf import DictConfig


class LCBEnv(BaseTextEnv):
    """
    Environment for LiveCodeBench execution environment.
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.tests = json.loads(extras["reward_spec"]["ground_truth"])

    def _get_reward(self, action: str) -> float:
        return compute_score(action, self.tests)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        parsed_code, reward = compute_score(action, self.tests)

        # RL on LCB w/ single-turn
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={"parsed_code": parsed_code})
