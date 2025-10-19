from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any
from typing import Dict
from omegaconf import DictConfig
from openai import OpenAI
import os
import re

PROMPT = """
You are a strict math evaluation assistant.

Compare the following **gold** and **predicted** math solutions. Your job is to determine if the predicted solution is mathematically correct and if the predicted solution ends with a line of the form:

#### <number>

You must only give a score of "1" if:
- The final line of the predicted solution **ends with `#### <number>`**, and
- The number **matches the final answer in the gold solution** exactly.

Instructions:
- You may provide internal reasoning or explanation before giving your final judgment.
- Your final judgment must appear as a separate line at the end of your response, in the format:

### Final Score: 1

or

### Final Score: 0

Do not include any explanation after the final score.
"""


class GSM8kLLMJudgeEnv(BaseTextEnv):
    """
    Example implementtion of GSM8k environment with LLM as judge.

    Use LLM as judge to evaluate the answer similarity with the ground truth.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert (
            "ground_truth" in extras["reward_spec"]
        ), "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        # Set up OpenAI client
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("`OPENAI_API_KEY` must be set for Llm as a judge env")
        self.llm_judge_client = OpenAI(
            base_url=env_config.base_url, api_key=openai_api_key
        )
        self.model = env_config.model

    def _get_reward(self, action: str) -> float:
        message = (
            PROMPT
            + f"\n\nGOLD SOLUTION:\n{self.ground_truth}\n\nPREDICTED SOLUTION:\n{action}\n\nAnswer:"
        )

        try:
            response = self.llm_judge_client.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": message}]
            )
            reply = response.choices[0].message.content.strip()

            # Try to parse score from "### Final Score: x"
            match = re.search(r"### Final Score:\s*([01](?:\.0)?)", reply)
            if match:
                return float(match.group(1))

            # Fallback: raw "1" or "0"
            if reply.strip() in {"1", "0"}:
                return float(reply.strip())

            print(f"Unrecognized reward output: {reply}")
            return 0.0

        except Exception as e:
            print(f"LLM Judge error: {type(e).__name__}: {e}")
            return 0.0

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True
        reward = self._get_reward(action)

        return BaseTextEnvStepOutput(
            observations=[], reward=reward, done=done, metadata={}
        )
