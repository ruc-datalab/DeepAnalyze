from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
from skyrl_gym.envs.search.utils import compute_score
from skyrl_gym.tools import SearchToolGroup
import re
from typing import Dict, Optional, List
from omegaconf import DictConfig


class SearchEnv(BaseTextEnv):
    """
    Environment for Search execution tasks.

    Based on Verl + Search-R1 integration
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Initialize the tools
        # name is hardcoded to "SearchToolGroup", with tool name "search"
        self.tool_group = SearchToolGroup(
            search_url=env_config.search_url,
            topk=env_config.topk,
            timeout=env_config.timeout,
            log_requests=env_config.log_requests,
        )
        self.init_tool_groups([self.tool_group])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> List[Optional[str]]:
        match = None
        if "<search>" in action and "</search>" in action:
            match = re.search(r"<search>(.*?)</search>", action, re.DOTALL)
        return [match.group(1)] if match else [None]

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            return compute_score(chat_history_str, self.ground_truth)
        else:
            # No reward for intermediate steps for Search tasks
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<answer>" in action and "</answer>" in action

    def _postprocess_action(self, action: str) -> str:
        if "</search>" in action:
            return action.split("</search>")[0] + "</search>"
        elif "</answer>" in action:
            return action.split("</answer>")[0] + "</answer>"
        else:
            return action

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_input: Any) -> str:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_input)

        return "\n<information>" + tool_output + "</information>\n"

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        action = self._postprocess_action(action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(
                observations=[], reward=reward, done=done, metadata={}, postprocessed_action=action
            )

        try:
            query = self._parse_action(action)
            observation = self._execute_tool("SearchToolGroup", "search", query)
        except Exception as e:
            error = str(e)
            observation = None

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": "SearchToolGroup",
            "tool_name": "search",
            "tool_input": query,
        }

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
            postprocessed_action=action,
        )
