from .base import BaseTool, register_tool
from typing import Union


@register_tool("finish")
class FinishTool(BaseTool):
    name = "finish"
    description = (
        "Signals the completion of the current task or conversation.\n\n"
        "Use this tool when:\n"
        "- You have successfully completed the requested task\n"
        "- You cannot proceed further due to technical limitations or missing information\n\n"
        "The answer field should include the final answer to the problem (follow the required format) if an answer is required by the problem.\n"
    )
    parameters = {
        "type": "object",
        "required": ["answer"],
        "properties": {
            "answer": {
                "type": "string",
                "description": "Final message summarizing the task or containing the answer.",
            },
        },
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        # If the parameters are valid, we can proceed to finish the task.
        answer = params.get("answer", "")
        return answer
