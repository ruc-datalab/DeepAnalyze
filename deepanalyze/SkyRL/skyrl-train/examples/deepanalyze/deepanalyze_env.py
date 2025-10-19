from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)
from typing import Tuple, Any
from examples.deepanalyze.sql_tool import SQLCodeExecutorToolGroup
from examples.deepanalyze.python_tool import PythonCodeExecutorToolGroup
import re
from examples.deepanalyze.utils import *
import os
from typing import Dict
from omegaconf import DictConfig
from openai import OpenAI
import random
from transformers import AutoTokenizer
import numpy as np


class DeepAnalyzeEnv(BaseTextEnv):
    """
    Environment for one SQL execution task.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Initialize the environment
        # assert "db_id" in extras, "db_id field is required"
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "data" in extras, "data field is required"

        self.extras = extras
        self.workspace_id = str(extras.get("workspace_id", ""))
        self.workspace = env_config.workspace
        self.reward_spec = extras["reward_spec"]
        self.task = extras["data"]

        api_key = env_config.api_key
        base_url = env_config.base_url
        self.llm_judgement_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.llm_judgement_model = env_config.llm_judgement_model

        if self.workspace_id:

            self.workspace = os.path.join(
                self.workspace,
                self.task,
            )
            self.workspace = os.path.join(
                self.workspace,
                self.workspace_id,
            )
            if not os.path.exists(self.workspace):
                raise FileNotFoundError(f"workspace not found at: {self.workspace}")

        # override parent
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 30

        # Initialize the tools
        self.tool_group = PythonCodeExecutorToolGroup(workspace=self.workspace)
        self.init_tool_groups([self.tool_group])

        # Chat history
        # Dict[str, str]: role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []
        self.code_pass = []

    def _parse_action(self, action: str) -> Tuple[str, str, Any]:
        """
        Parse action string to return tool name and corresponding arguments.

        Expected: <Code>...</Code>
        """
        code_match = re.search(r"<Code>(.*?)</Code>", action, re.DOTALL)
        if code_match:

            code_content = code_match.group(1).strip()
            md_match = re.search(r"```(?:python)?(.*?)```", code_content, re.DOTALL)
            tool_input = md_match.group(1).strip() if md_match else code_content
        else:
            tool_input = None

        # NOTE: hard code
        # NOTE (shu): in the future imagine can use different tools here
        # Format <tool>tool_name</tool><input>tool_input</input>
        tool_group_name = self.tool_group.get_name()
        tool_name = self.tool_group.get_tool_names()[0]
        return tool_group_name, tool_name, (tool_input,)

    def _get_reward(self, action: str, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["content"] for item in self.chat_history])
            rewards = {}
            if (
                "<Analyze>" not in chat_history_str
                or "</Analyze>" not in chat_history_str
                or "<Answer>" not in chat_history_str
                or "</Answer>" not in chat_history_str
            ):
                return -1.0
            if self.reward_spec["method"] == "qa":
                reward_acc = compute_tableqa_score_single(
                    chat_history_str, self.reward_spec["ground_truth"]
                )
                reward_analysis = llm_as_judgement_analyze(
                    chat_history_str,
                    self.reward_spec["ground_truth"],
                    self.extras["input_seq"],
                    self.llm_judgement_client,
                    self.llm_judgement_model,
                )

                rewards["qa_acc_reward"] = reward_acc
                rewards["qa_analysisc_reward"] = reward_analysis

                reward = sum(rewards.values()) / len(rewards)

            elif self.reward_spec["method"] in ["datatask", "openresearch"]:
                if not check_valid_code_block(chat_history_str):
                    return -1.0
                if isinstance(self.reward_spec["function"], str):
                    self.reward_spec["function"] = [self.reward_spec["function"]]

                for func in self.reward_spec["function"]:
                    if func in globals():
                        func_obj = globals()[func]
                        single_reward = func_obj(
                            chat_history_str,
                            self.reward_spec["ground_truth"],
                            self.extras["input_seq"],
                            self.llm_judgement_client,
                            self.llm_judgement_model,
                        )

                        if isinstance(single_reward, dict):
                            rewards.update(single_reward)
                        else:
                            rewards.update(
                                {f"{self.reward_spec['function']}": single_reward}
                            )

                if len(self.code_pass) > 0:
                    reward_code_acc = sum(self.code_pass) / len(self.code_pass)
                    rewards[f"{self.reward_spec['method']}_code_reward"] = (
                        reward_code_acc
                    )

                if self.reward_spec["method"] == "openresearch":
                    turns = 0
                    for item in self.chat_history:
                        if item["role"] == "assistant":
                            turns += 1
                    turn_reward = min(turns / 10, 1)

                    rewards[f"{self.reward_spec['method']}_turns_reward"] = turn_reward

                reward = sum(rewards.values()) / len(rewards)
            else:
                reward = 0

            return reward
        else:
            return 0

    def _is_done(self, action: str) -> bool:
        if self.turns >= self.max_turns:
            return True
        return "<Answer>" in action and "</Answer>" in action

    def _postprocess_action(self, action: str) -> str:
        if "</Answer>" in action:
            return action.split("</Answer>")[0] + "</Answer>"
        elif "</Code>" in action:
            return action.split("</Code>")[0] + "</Code>"
        else:
            return action

    def step(self, action: str) -> BaseTextEnvStepOutput:

        self.turns += 1

        action = self._postprocess_action(action)
        # print("self.turns",self.turns,"\n",action)
        self.chat_history.append({"role": "assistant", "content": action})

        error = None
        done = self._is_done(action)
        reward = self._get_reward(action, done)

        if done:
            print("DONEDONEDONEDONEDONE")
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=done,
                metadata={},
                postprocessed_action=action,
            )

        observation = None
        tool_group_name = None
        tool_name = None
        tool_input = ""

        if self.workspace_id:
            try:
                tool_group_name, tool_name, tool_input = self._parse_action(action)
                observation = self._execute_tool(tool_group_name, tool_name, tool_input)

            except Exception as e:
                error = str(e)
                observation = None
                tool_group_name = None
                tool_name = None
                tool_input = ""
                print("execute error", error)

        if observation:
            new_obs = {
                "role": "execute",
                "content": f"\n<Execute>\n{observation}\n<Execute>",
            }

            if "[Error]:" in observation:
                self.code_pass.append(0)
            else:
                self.code_pass.append(1)
        else:
            new_obs = None
            self.code_pass.append(0)

        info = {
            "tool_group": tool_group_name,
            "tool_name": tool_name,
            "tool_input": tool_input,
        }
        # Update chat history
        if new_obs is not None:
            self.chat_history.append(new_obs)

        return BaseTextEnvStepOutput(
            observations=[new_obs] if new_obs else [],
            reward=reward,
            done=done,
            metadata=info,
            postprocessed_action=action,
        )
