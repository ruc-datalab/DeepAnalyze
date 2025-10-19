from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any
from omegaconf import DictConfig


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

    def _get_reward(self, action: str) -> float:
        return utils.compute_score(action, self.ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)

        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
