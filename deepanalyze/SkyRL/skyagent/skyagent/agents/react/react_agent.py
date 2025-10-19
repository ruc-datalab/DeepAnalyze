import json
from typing import Any, List, Dict

from skyagent.functional.function_calling import convert_str_to_completion_format
from skyagent.config.configuration_utils import TrajectoryConfig
from skyagent.integrations.base import AsyncInferBackend
from skyagent.tools.base import TOOL_REGISTRY
from skyagent.dispatcher.async_utils import call_sync_from_async

from openhands.llm.fn_call_converter import (
    convert_fncall_messages_to_non_fncall_messages,
    convert_non_fncall_messages_to_fncall_messages,
)

class ReActAgent:
    def __init__(
        self,
        traj_config: TrajectoryConfig,
        infer_engine: AsyncInferBackend,
        tokenizer: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.infer_engine = infer_engine
        self.sampling_params = traj_config.sampling_params

        self.max_prompt_length = traj_config.max_prompt_length
        self.qwen3_enable_thinking = traj_config.qwen3_enable_thinking

        self.instance_id = traj_config.instance_id
        self.trajectory_id = traj_config.trajectory_id
        self.max_iterations = traj_config.max_iterations

        self.step_count = 0
        self.messages: List[dict] = []
        self.tools = {}
        self.tool_params = []

        self._register_tools(traj_config.tools.enabled)

    def _register_tools(self, tools: List[str]) -> None:
        """Register a list of tool instances."""
        print(f"[Register Tools] {tools}")
        for name in tools:
            if name not in TOOL_REGISTRY:
                raise ValueError(f"Unknown tool '{name}'. Must be one of: {list(TOOL_REGISTRY)}")
            tool = TOOL_REGISTRY[name]()
            self.tools[tool.name] = tool
            self.tool_params.append(tool.get_tool_param())

    async def step(self):
        done = False
        finish_reason = None
        
        self.step_count += 1
        # print(f"[Agent Step {self.step_count}] instance={self.instance_id} traj={self.trajectory_id}")

        formatted_messages = convert_fncall_messages_to_non_fncall_messages(
            self.messages, self.tool_params, add_in_context_learning_example=False
        )
        # print(f"[Agent Step {self.step_count}] Formatted messages: {formatted_messages}, messages: {self.messages}")

        input_ids = self.tokenizer.apply_chat_template(
            formatted_messages,
            add_generation_prompt=True,
            tokenize=True,
            enable_thinking=self.qwen3_enable_thinking,
        )

        if len(input_ids) >= self.max_prompt_length:
            # raise ValueError(
            #     f"Input length {len(input_ids)} exceeds max prompt length {self.max_prompt_length}. "
            #     "Please reduce the input size or increase the max prompt length."
            # )
            # For now, we will just stop the agent if the input length exceeds the max prompt length.
            print(f"[Agent Step] Input length {len(input_ids)} exceeds max prompt length {self.max_prompt_length}. Stopping agent.")
            done = True
            finish_reason = "CONTEXT_WINDOW_EXCEEDED"
            return done, finish_reason, None

        try:
            response_str, stop_reason = await self.infer_engine.async_generate_ids(
                input_ids=input_ids,
                sampling_params=self.sampling_params,
            )

            assistant_msg = {"role": "assistant", "content": response_str}
            self.messages.append(assistant_msg)

            fncall_messages = convert_non_fncall_messages_to_fncall_messages(
                [assistant_msg], self.tool_params
            )

            # if no tools provided, we can just return the response
            if not self.tools:
                # print(f"[Agent Step {self.step_count}] No tools provided, returning response.")
                done = True
                finish_reason = "FINISH"
                return done, finish_reason, response_str
            
            tool_call = fncall_messages[0].get("tool_calls", [None])[0]
            if tool_call is None:
                self.messages.append({
                    "role": "user",
                    "content": "No tool call found in the response. Use the finish tool if you want to complete the task.",
                })
                return done, finish_reason, None
            tool_name = tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("function", {}).get("arguments")
            if tool_name not in self.tools:
                self.messages.append({
                    "role": "user",
                    "content": json.dumps({"error": f"Tool '{tool_name}' not found."}),
                })
                return done, finish_reason, response_str

            tool = self.tools[tool_name]
            print(f"[Tool Dispatch] Calling tool: {tool_name} with args: {tool_args}")
            output = await call_sync_from_async(
                tool.call,
                tool_args,
            )
            # append observations to messages
            try:
                self.messages.append({
                    "role": "tool",
                    "content": json.dumps(output),
                    "tool_call_id": tool_call.get("id"),
                    })
            except Exception as e:
                print(f"[Agent Step Error] Error appending tool output to messages: {str(e)}")
                self.messages.append({
                    "role": "tool",
                    "content": json.dumps({"error": str(e)}),
                    "tool_call_id": tool_call.get("id"),
                })
            # if it is a finish tool, we can stop the agent
            if tool_name == "finish":
                print("[Agent Step] Finish tool called. Stopping agent.")
                done = True
                finish_reason = "FINISH_TOOL"
                return done, finish_reason, output
            
            # For non-finish tools, continue the agent loop
            return done, finish_reason, response_str

        except Exception as e:
            print(f"[Agent Step Error] Error during step: {str(e)}")
            done = True
            finish_reason = f"error: {str(e)}"
            return done, finish_reason, None
    
    async def run(self, instruction: List[Dict]) -> List[str]:
        """Run the agent till the end with the provided user input."""
        self._init_message(instruction)
        result = None
        finish_reason = None
        while self.step_count < self.max_iterations:
            try:
                done, finish_reason, result = await self.step()
                if done:
                    break
            except Exception as e:
                finish_reason = f"error: {str(e)}"
                print(f"[Agent Run Error] Exception during step: {str(e)}")
                break
        else:            # If we exit the loop without hitting a break, it means we reached max iterations
            finish_reason = "MAX_ITERATIONS"
        
        return finish_reason, result

    def get_messages(self) -> List[dict]:
        return convert_fncall_messages_to_non_fncall_messages(
            self.messages, self.tool_params, add_in_context_learning_example=False
        )
        # return self.messages
    
    def _init_message(self, instruction: List[Dict]) -> None:
        """Initialize the agent's message history with the provided instruction."""
        self.messages = []
        if not isinstance(instruction, list):
            raise ValueError("Instruction must be a list of messages.")
        
        for msg in instruction:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError("Each message must be a dictionary with 'role' and 'content'.")
            self.messages.append(msg)

if __name__ == "__main__":
    # Example usage for testing
    from skyagent.config.configuration_utils import TrajectoryConfig
    from skyagent.integrations.openai import OpenAIBackend, OpenAIBackendConfig
    from transformers import AutoTokenizer
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm import AsyncEngineArgs
    import asyncio

    # Load tokenizer and model
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define trajectory configuration
    traj_config = TrajectoryConfig(
        instance_id="test_instance",
        trajectory_id="test_trajectory",
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 2048,
        },
        max_prompt_length=12048,
        qwen3_enable_thinking=True,
        tools=["finish", "code_interpreter"],
        max_iterations=5,
        agent_cls="skyagent.agents.react.ReActAgent",  # Use ReActAgent for testing
    )

    backend_config = OpenAIBackendConfig(
        model_name=model_name,
        # change this to your desired url and port
        api_url="http://localhost:8000" 
    )
    # TODO: model_name need not be in config
    infer_engine = OpenAIBackend(infer_engine=None, cfg=backend_config)

    # Create the ReAct agent
    # Test for with tools
    agent = ReActAgent(
        traj_config=traj_config,
        infer_engine=infer_engine,
        tokenizer=tokenizer,
    )

    # Define a sample instruction
    instruction = [{'content': 'Please reason step by step, and put your final answer within \\boxed{}.', 'role': 'system'}, {'content': 'Points $A,B,C,D,E$ and $F$ lie, in that order, on $\\overline{AF}$, dividing it into five segments, each of length 1. Point $G$ is not on line $AF$. Point $H$ lies on $\\overline{GD}$, and point $J$ lies on $\\overline{GF}$. The line segments $\\overline{HC}, \\overline{JE},$ and $\\overline{AG}$ are parallel. Find $HC/JE$.', 'role': 'user'}]

    # Run the agent
    finish_reason, result = asyncio.run(agent.run(instruction))

    print(agent.get_messages())
    print(f"Finish Reason: {finish_reason}")
    print(f"Result: {result}")
