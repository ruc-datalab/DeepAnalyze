from typing import Any, List, Optional
import json
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from skyagent.functional.function_calling import convert_str_to_completion_format
from skyagent.functional.chat_template import get_templates_path
from skyagent.config.configuration_utils import TrajectoryConfig
from skyagent.integrations.base import AsyncInferBackend
from skyagent.dispatcher.async_utils import call_sync_from_async, call_async_from_sync

import openhands.agenthub.codeact_agent.function_calling as codeact_function_calling
from openhands.agenthub.codeact_agent import CodeActAgent
from openhands.controller.agent import Agent
from openhands.controller.state.state import State, AgentState
from openhands.core.config import LLMConfig, AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message
from openhands.core.exceptions import (
    AgentStuckInLoopError,
    FunctionCallNotExistsError,
    FunctionCallValidationError,
    LLMContextWindowExceedError,
    LLMMalformedActionError,
    LLMNoActionError,
    LLMResponseError,
)
from openhands.events.event import Event
from openhands.events.action import (
    Action,
    AgentFinishAction,
    MessageAction,
)
from openhands.memory.condenser.condenser import Condensation, View

from openhands.llm.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)
from openhands.llm.llm import LLM
from openhands.core.config.condenser_config import (
    NoOpCondenserConfig,
)


class OHCodeActAgent(CodeActAgent):
    """
    An online implementation of CodeActAgent that leverages infer's asynchronous capabilities
    for a single agent instance.
    """
    
    def __init__(
        self,
        traj_config: TrajectoryConfig,
        infer_engine: Optional[AsyncInferBackend] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """
        Initialize a single OHCodeActAgent instance.
        """
        # dummy value to let openhands tracks the name
        llm = LLM(LLMConfig(model="dummy", 
                            max_message_chars=32768))

        self.agent_config = AgentConfig(
            enable_jupyter=traj_config.tools.get('enable_jupyter', False),
            enable_browsing=traj_config.tools.get('enable_browsing', False),
            enable_llm_editor=traj_config.tools.get('enable_llm_editor', False),
            enable_editor=traj_config.tools.get('enable_editor', False),
            enable_think=traj_config.tools.get('enable_think', False),
            condenser=NoOpCondenserConfig(),
            enable_prompt_extensions=traj_config.tools.get('enable_prompt_extensions', False),
        )
        super().__init__(llm, self.agent_config)
        
        self.tokenizer = tokenizer
        self.max_prompt_length = traj_config.max_prompt_length
        self.step_count = 0
        self.infer_engine = infer_engine
        self.sampling_params = traj_config.sampling_params
        
        # Store instance and trajectory IDs separately
        self.instance_id = traj_config.instance_id
        self.trajectory_id = traj_config.trajectory_id
        self.qwen3_enable_thinking = traj_config.qwen3_enable_thinking
        self.qwen3_acc_thinking = traj_config.qwen3_acc_thinking

        # will be set in _initialize_runtime_for_agent
        self.runtime = None
        self.instruction = None
        self.app_config = None

        self.agent_id = uuid4().hex

    def close(self):
        """Close the agent runtime."""
        if self.runtime:
            # remove all threads in event stream
            self.runtime.event_stream.close()
            self.runtime.close()

    def step(self, state: State) -> Action:
        """Generate a response using batched infer."""
        self.step_count += 1
        print(f"instance id {self.instance_id}, trajectory {self.trajectory_id}, step {self.step_count}")
        if self.pending_actions:
            return self.pending_actions.popleft()

        # if we're done, go back
        latest_user_message = state.get_last_user_message()
        if latest_user_message and latest_user_message.content.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        messages = self.llm.format_messages_for_llm(messages)
        messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, self.tools, add_in_context_learning_example=False
                )
        
        try:
            if self.qwen3_acc_thinking:
                # Use the Qwen3 thinking mode chat template
                assert self.qwen3_enable_thinking, "Qwen3 thinking mode should for accumulating thinking."
                chat_template = get_templates_path() / "qwen3_acc_thinking.jinja2"
                input_ids = self.tokenizer.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=True, 
                    enable_thinking=self.qwen3_enable_thinking,
                    chat_template=chat_template.read_text()
                )
            else:
                input_ids = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, enable_thinking=self.qwen3_enable_thinking
                )
            if len(input_ids) >= self.max_prompt_length:
                return AgentFinishAction(thought="CONTEXT_WINDOW_EXCEEDED")

            response_str, stop_reason = call_async_from_sync(self.infer_engine.async_generate_ids, 
                                                input_ids=input_ids, 
                                                sampling_params=self.sampling_params,
                                                request_id=self.agent_id)
            print(f"instance id {self.instance_id}, trajectory {self.trajectory_id}, stop reason {stop_reason}, response {response_str} ")
            
            if not response_str:
                # If we got an empty response (possible error), return a message action
                self.pending_actions.append(
                    MessageAction(
                        content="I encountered an error processing your request. Let's try again.",
                    )
                )
            else:
                # Convert to actions
                message = [
                    {
                        'role': 'assistant',
                        'content': response_str,
                    }
                ]
                fn_call_messages = convert_non_fncall_messages_to_fncall_messages(
                    message, self.tools
                )
                actions = codeact_function_calling.response_to_actions(
                    convert_str_to_completion_format(fn_call_messages),
                    mcp_tool_names=list(self.mcp_tools.keys())
                )
                print(f"Take action: {[type(action) for action in actions]}")
                
                for action in actions:
                    self.pending_actions.append(action)
                if stop_reason == "length":
                    self.pending_actions.append(AgentFinishAction(thought="CONTEXT_WINDOW_EXCEEDED"))
        
        except (
            LLMMalformedActionError,
            LLMNoActionError,
            LLMResponseError,
            FunctionCallValidationError,
            FunctionCallNotExistsError,
        ):
            raise

        except Exception as e:
            logger.error(f"Error in agent step: {str(e)}")
            # Handle errors gracefully by creating a message action
            self.pending_actions.append(
                MessageAction(
                    content=f"An error: {str(e)} encountered. Please try a different approach.",
                )
            )
        
        # Return the first pending action
        if not self.pending_actions:
            # Fallback in case of empty actions
            return AgentFinishAction()
            
        return self.pending_actions.popleft()
    
    def get_final_messages(self, state: State) -> List[Message]:
        """Get the final messages for this agent."""
        condensed_history: list[Event] = []
        match self.condenser.condensed_history(state):
            case View(events=events):
                condensed_history = events

            case Condensation(action=condensation_action):
                return condensation_action

        logger.debug(
            f'Processing {len(condensed_history)} events from a total of {len(state.history)} events'
        )

        initial_user_message = self._get_initial_user_message(state.history)
        messages = self._get_messages(condensed_history, initial_user_message)
        messages = self.llm.format_messages_for_llm(messages)
        messages = convert_fncall_messages_to_non_fncall_messages(
                    messages, self.tools, add_in_context_learning_example=False
                )

        import wandb
        current_step = wandb.run.step if wandb.run else 1
        run_name = wandb.run.name if wandb.run else "no_run"
        logger.info(f"Detected run name: {run_name}")
        if  (current_step == 1) or (current_step % 5 == 0) :
            instance_dir = Path(f"/mnt/shared_storage/trace/{run_name}/step{current_step}") / str(self.instance_id) / str(self.trajectory_id)
            instance_dir.mkdir(exist_ok=True, parents=True)

            # Generate a unique filename using a timestamp with microsecond resolution
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            trace_file = instance_dir / f"trace_{timestamp}.json"

            with open(trace_file, "w") as f:
                result_json = json.dumps(messages, default=lambda x: str(x))
                f.write(result_json)

        return messages
    
    def _is_last_action_finish(self, state: State):
        finish_reason = None
        if state and state.history:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            if isinstance(last_action, AgentFinishAction):
                finish_reason = last_action.thought if last_action.thought == "CONTEXT_WINDOW_EXCEEDED" else "FINISH_TOOL"
                return True, finish_reason
        return False, finish_reason
    

Agent.register('OHCodeActAgent', OHCodeActAgent)

