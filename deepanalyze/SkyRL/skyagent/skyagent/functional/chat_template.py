# this is for the tokenizer.apply_chat_template to be able to generate assistant masks directly
# todo: this is a hack, we should find a better way to do this
chat_template = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{{message['content'] + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\n'}}"
    "{% endif %}"
    "{% endfor %}"
)

# chat template for qwen3 thinking mode to remove think tokens similar to generation phase
chat_template_qwen3_thinking = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{% set full_content = message['content'] %}"
    "{% set mycontent = message['content'] %}"
    "{% set is_last_message = loop.last and messages[-1]['role'] == 'assistant' %}"
    "{% if '</think>' in full_content and not is_last_message %}"
    "{% set mycontent = full_content.split('</think>')[-1].lstrip('\n') %}"
    "{% endif %}"
    "{{mycontent + '<|im_end|>'}}"
    "{% endgeneration %}"
    "{{'\n'}}"
    "{% endif %}"
    "{% endfor %}"
)


def get_templates_path():
    from pathlib import Path

    # Get the current file's directory
    current_file_path = Path(__file__).resolve()
    templates_path = current_file_path.parent / "templates"

    # Ensure the templates directory exists
    if not templates_path.exists():
        raise FileNotFoundError(f"Templates directory does not exist: {templates_path}")

    return templates_path
