from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Tuple, Any
from skyrl_gym.tools import SearchToolGroup, PythonCodeExecutorToolGroup
from skyrl_gym.envs.gsm8k import utils
import re
from typing import Dict
from omegaconf import DictConfig


class SearchCodeEnv(BaseTextEnv):
    """
    Environment that calls multiple tools
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 2

        # Initialize the tools
        self.init_tool_groups([SearchToolGroup(), PythonCodeExecutorToolGroup()])

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []

    def _parse_action(self, action: str) -> Tuple[str, str, Any]:
        """
        Parses the action string to detect a tool call in the format:
        <tool><tool_name>...</tool_name></tool>, where tool_name can be one of the
        predefined tool names in the tool group (e.g., search, python).
        Returns the tool group name, tool name, and corresponding tool input.
        """
        tool_block_match = re.search(r"<tool>(.*?)</tool>", action, re.DOTALL)
        if not tool_block_match:
            raise ValueError("No <tool>...</tool> block found in action string.")

        tool_content = tool_block_match.group(1).strip()
        inner_tag_match = re.search(r"<(\w+)>(.*?)</\1>", tool_content, re.DOTALL)
        if not inner_tag_match:
            raise ValueError("No valid inner tool tag found inside <tool> block.")

        tool_name = inner_tag_match.group(1)
        tool_input = inner_tag_match.group(2).strip()

        # Lookup the tool by name from tool group
        if tool_name not in self.tool_to_toolgroup:
            raise ValueError(f"Tool '{tool_name}' not found in any registered tool group.")
        tool_group_name = self.tool_to_toolgroup[tool_name]

        return tool_group_name, tool_name, [tool_input]

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "\n".join([item["content"] for item in self.chat_history])
            return utils.compute_score(chat_history_str, self.ground_truth)
        else:
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<solution>" in action and "</solution>" in action

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Step 1: parse action to get the corresponding tool group, tool name, and tool input
        Step 2: execute the tool and get observation (as a string)
        Step 3: get reward based on the action and observation
        Step 4: determine `done` based on the action and observation
        """
        self.turns += 1
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})

        try:
            tool_group_name, tool_name, tool_input = self._parse_action(action)
            observation = self._execute_tool(tool_group_name, tool_name, tool_input)
        except Exception as e:
            error = str(e)
            observation = None
            tool_group_name = None
            tool_name = None
            tool_input = ""

        # Wrap the observation properly as a message
        if observation:
            new_obs = {"role": "user", "content": observation}
        elif error:
            # Give error as observation if any
            new_obs = {"role": "user", "content": error}
        else:
            new_obs = None

        info = {
            "tool_group": tool_group_name,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tools": [tg.get_tool_names() for tg in self.tool_groups],
        }

        # Update chat history
        if new_obs:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(observations=[new_obs] if new_obs else [], reward=reward, done=done, metadata=info)
