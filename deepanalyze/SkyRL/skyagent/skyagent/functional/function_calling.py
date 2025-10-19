# Conversion utility function
def convert_str_to_completion_format(fn_call_messages):
    # from types import SimpleNamespace
    from litellm import ModelResponse

    role = fn_call_messages[0]["role"]
    response_str = fn_call_messages[0]["content"]
    tool_calls = fn_call_messages[0].get("tool_calls", None)

    return ModelResponse(
        choices=[
            {
                "index": 0,
                "message": {
                    "content": response_str,
                    "role": role,
                    "tool_calls": tool_calls,
                    "function_calling": None,
                },
            }
        ]
    )
