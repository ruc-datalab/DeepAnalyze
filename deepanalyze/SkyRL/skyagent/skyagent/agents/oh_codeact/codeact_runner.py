from typing import Any, Dict, Callable
import os
import importlib
import copy
from omegaconf import OmegaConf
import pandas as pd
import traceback

from skyagent.agents.oh_codeact.codeact_agent import OHCodeActAgent
from skyagent.integrations.base import BACKEND_REGISTRY, build_backend, build_generator_input, build_generator_output, _import_object
from skyagent.dispatcher.dispatchers import DISPATCHER_REGISTRY
from skyagent.dispatcher.async_utils import call_sync_from_async, call_async_from_sync
from skyagent.config.configuration_utils import TASK_CONFIG_REGISTRY, TrajectoryConfig, get_field_from_config
from skyagent.functional.chat_template import chat_template, chat_template_qwen3_thinking
from skyagent.agents.base import AgentRunner, BaseTrajectory, TrajectoryResult, TrajectoryConfig, AsyncInferBackend, AutoTokenizer, RuntimeConfig, BaseTask

from openhands.core.main import create_runtime, run_controller
from openhands.core.config import AgentConfig
from openhands.controller.state.state import State
from openhands.events.action import (
    Action,
    MessageAction,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.exceptions import (
    AgentRuntimeBuildError,
    AgentRuntimeDisconnectedError,
    AgentRuntimeError,
    AgentRuntimeNotFoundError,
    AgentRuntimeNotReadyError,
    AgentRuntimeTimeoutError,
    AgentRuntimeUnavailableError,
)

from dataclasses import dataclass

from skyagent.tasks.swebench.utils import SWEBenchTask

@dataclass
class TaskHandle:
    instance_id: str
    trajectory_id: int
    batch_id: int


def is_fatal_evaluation_error(error: str | None) -> bool:
    if not error:
        return False

    FATAL_EXCEPTIONS = [
        AgentRuntimeError,
        AgentRuntimeBuildError,
        AgentRuntimeTimeoutError,
        AgentRuntimeUnavailableError,
        AgentRuntimeNotReadyError,
        AgentRuntimeDisconnectedError,
        AgentRuntimeNotFoundError,
        ConnectionError,
    ]

    if any(exception.__name__ in error for exception in FATAL_EXCEPTIONS):
        logger.error(f'Fatal evaluation error detected: {error}')
        return True

    return False

def codeact_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> 42 </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'No function call detected.\n'
        'If you think you have solved the task, please use the "finish" tool to finish the interaction.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'You should at least take one actions.\n'
            )
    return msg


class CodeActTrajectory(BaseTrajectory):
    def __init__(self, cfg: TrajectoryConfig, data: Dict[str, Any], infer_engine: AsyncInferBackend, tokenizer: AutoTokenizer, task: SWEBenchTask) -> None:
        super().__init__(cfg, data, infer_engine, tokenizer, task)
        assert isinstance(task, SWEBenchTask)

    async def initialize_trajectory(self):
        """Initialize the runtime for a specific agent."""
        # only swebench task supported, redundant but makes linter happy
        assert isinstance(self.task, SWEBenchTask)

        batch_id = self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id

        # data = self._get_data(data)
        data = self.data
        instance_id = data['instance_id'] if data['instance_id'] else batch_id
        instance = pd.Series(data["instance"])
        data_source = data["data_source"]
        self.agent = OHCodeActAgent(traj_config=self.cfg, infer_engine=self.infer_engine, tokenizer=self.tokenizer)
        
        init_successful = False
        try:
            
            runtime = await self.task.initialize_runtime(instance, data_source, self.agent.config, self.cfg.max_iterations)

            app_config = self.task.get_config(instance, data_source, self.agent.config, self.cfg.max_iterations)
            
            # Store the runtime and instruction
            self.agent.runtime = runtime
            self.agent.instruction = self.task.get_instruction(instance, data_source)
            self.agent.app_config = app_config

            init_successful = True 
            logger.info(f"Successfully initialized runtime for instance {instance_id}")
        except Exception as e:
            logger.error(f"Failed to initialize runtime for instance {instance_id}: {str(e)}")
            self.agent.runtime = None
            
            return_val = {
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'messages': [],
                'state': None,
                'results': None,
                'error': str(e),
                'finish': False,
                'finish_reason': 'error_initialization',
            }

            self.result = return_val
        finally:
            if not init_successful: 
                logger.info(f"Init failed. Running cleanup for init agent task for instance {instance_id}, trajectory {trajectory_id}")
                if 'runtime' in locals() and runtime:
                    runtime.event_stream.close()
                    runtime.close() 

    async def generate_trajectory(self) -> None:
        # only swebench task supported, redundant but makes linter happy
        assert isinstance(self.task, SWEBenchTask)

        data = self.data
        instance_id = data['instance_id'] if data['instance_id'] else self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id
        instance = pd.Series(data["instance"])
        data_source = data["data_source"]
        agent = self.agent
        runtime = agent.runtime
        state = None

        try:
            if not runtime:
                raise Exception(f"Runtime not initialized for instance {instance_id}, trajectory {trajectory_id}")
            
            state = await run_controller(
                config=agent.app_config,
                initial_user_action=agent.instruction,
                runtime=runtime,
                agent=agent,
                fake_user_response_fn=codeact_user_response,
            )
            if state and is_fatal_evaluation_error(state.last_error):
                raise Exception('Fatal error: ' + state.last_error)

            final_messages = agent.get_final_messages(state)
            result = await call_sync_from_async(self.task.complete_runtime, runtime, instance, data_source)

            finish, finish_reason = agent._is_last_action_finish(state)
            if state and state.last_error:
                if "RuntimeError: Agent reached maximum iteration in headless mode" in state.last_error:
                    finish_reason = 'max_iterations_reached'
                elif "Agent got stuck in a loop" in state.last_error:
                    finish_reason = 'stuck_in_a_loop'
            
            if "finish_reason" in result:
                finish_reason = result["finish_reason"]

            return_val = TrajectoryResult({
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'messages': final_messages,
                'state': state,
                'results': result,
                'error': state.last_error if state and state.last_error else None,
                'finish': finish,
                'finish_reason': finish_reason,
            })
        except Exception as e:
            logger.error(f"Run error {instance_id}: {e}")
            logger.debug(f"Full Traceback: {traceback.format_exc()}")
            final_messages = agent.get_final_messages(state) if state else []
            if not final_messages or len(final_messages) == 0:
                logger.debug(f'Final messages are non-existent (or empty) for instance {instance_id}, trajectory {trajectory_id}')
            return_val = TrajectoryResult({
                'instance_id': instance_id,
                'trajectory_id': trajectory_id,
                'messages': final_messages,
                'state': state,
                'results': None,
                'error': str(e),
                'finish': False,
                'finish_reason': 'error_runtime',
            })
        finally:
            logger.info(f"Running cleanup for run agent task for instance {instance_id}, trajectory {trajectory_id}")
            self._cleanup_agent()

        self.result = return_val

    async def evaluate_trajectory(self) -> None:
        # only swebench task supported, redundant but makes linter happy
        assert isinstance(self.task, SWEBenchTask)

        batch_id = self.cfg.instance_id
        trajectory_id = self.cfg.trajectory_id
        data = self.data
        instance_id = data['instance_id'] if data['instance_id'] else batch_id
        instance = pd.Series(data["instance"])
        data_source = data["data_source"]
        
        try:
            # TODO: Why does "result" have a "results" entry? we should flatten
            results = self.result.get('results', None)
            if not results:
                raise Exception(f"No results found for instance {instance_id}, trajectory {trajectory_id}")
            if "reward" in results:
                self.result['reward'] = results['reward']
                return

            eval_results = await self.task.evaluate_result(instance, results, instance_id, trajectory_id, data_source)
            self.result['reward'] = eval_results
            
            logger.info(f"Successfully evaluated instance {instance_id}, trajectory {trajectory_id} with reward {eval_results}")
        except Exception as e:
            logger.error(f"Failed to evaluate traj {trajectory_id} for instance {instance_id}: {str(e)}")
            self.result['reward'] = False
            self.result['eval_error'] = str(e)
            self.result['finish_reason'] = 'error_evaluation' if not "No git patch found" in str(e) else None

    def _cleanup_agent(self):
        try:
            self.agent.close()
        except Exception as e:
            logger.warning(f"Error closing agent {self.cfg.instance_id}, trajectory {self.cfg.trajectory_id}: {str(e)}")
