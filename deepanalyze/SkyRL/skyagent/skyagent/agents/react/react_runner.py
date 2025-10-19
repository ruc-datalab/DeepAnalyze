import os
import importlib
from tokenize import TokenInfo
from typing import Type
from typing import Dict, Any, List, Optional

import pandas as pd
from omegaconf import OmegaConf
from loguru import logger

from skyagent.agents.react.react_agent import ReActAgent
from skyagent.integrations.base import _import_object
from skyagent.dispatcher.async_utils import call_sync_from_async
from skyagent.config.configuration_utils import TrajectoryConfig, get_field_from_config

from skyagent.tools.base import BaseTool
from skyagent.agents.base import AgentRunner, BaseTrajectory


class ReaActTrajectory(BaseTrajectory):
    async def initialize_trajectory(self):
        pass

    async def generate_trajectory(self) -> None:
        data = self.data
        instance_id = data['instance_id'] if data['instance_id'] else self.cfg.instance_id
        instance = pd.Series(data["instance"])
        # self.agent = ReActAgent(traj_config=self.cfg, infer_engine=self.infer_engine, tokenizer=self.tokenizer)
        self.agent: ReActAgent = self.agent_cls(
            traj_config=self.cfg,
            infer_engine=self.infer_engine,
            tokenizer=self.tokenizer,
        )
        
        # sys + user messages
        instruction = self.task.get_instruction(instance) 
        
        finish_reason, result = await self.agent.run(instruction)
        self.result = {
            'instance_id': instance_id,
            'trajectory_id': self.cfg.trajectory_id,
            'messages': self.agent.get_messages(),
            'results': result,
            'finish_reason': finish_reason,
        }

    async def evaluate_trajectory(self) -> None:
        instance_id = self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id
        data = self.data
        instance_id = data['instance_id'] if data['instance_id'] else self.cfg.instance_id
        instance = pd.Series(data["instance"])
        result = self.result.get('results')

        try:
            eval_result = await self.task.evaluate_result(
                result,
                instance,
                data["data_source"],
                instance_id,
                trajectory_id,
            )
            self.result['reward'] = eval_result
        except Exception as e:
            self.result['reward'] = 0
            self.result['eval_error'] = str(e)