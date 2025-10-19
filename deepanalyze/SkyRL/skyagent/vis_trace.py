import json
from pathlib import Path
from typing import List
import gradio as gr
from fire import Fire


def load_conversations(file_path: Path) -> List[List[dict]]:
    conversations = []

    if file_path.suffix == ".jsonl":
        # Handle JSONL files (one conversation per line)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    conversation = json.loads(line)
                    # Each line should be an array of messages
                    if isinstance(conversation, list):
                        conversations.append(conversation)
                    else:
                        # If it's a single message, wrap it in a list
                        conversations.append([conversation])
    else:
        # Handle JSON files
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                # Check if it's a list of conversations or a single conversation
                if data and isinstance(data[0], list):
                    # List of conversations (each is an array of messages)
                    conversations = data
                elif data and isinstance(data[0], dict) and "role" in data[0]:
                    # Single conversation (array of messages)
                    conversations = [data]
                else:
                    # Unknown format, treat as single conversation
                    conversations = [data]
            else:
                conversations = [[data]]

    return conversations


def build_interface(trace_dir: str = "./trace/14b", port: int = 9781):
    trace_root = Path(trace_dir)
    all_json_files = sorted([*trace_root.rglob("*.jsonl"), *trace_root.rglob("*.json")])

    if not all_json_files:
        raise FileNotFoundError(f"No .json or .jsonl files found in {trace_root}")

    file_options = [str(p.relative_to(trace_root)) for p in all_json_files]
    conversations_cache = {}

    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 OpenAI Message Trace Viewer")
        gr.Markdown(
            "Select a trace file and conversation index to visualize the conversation."
        )

        with gr.Row():
            file_dropdown = gr.Dropdown(
                label="Select trace file", choices=file_options, value=file_options[0]
            )
            index_slider = gr.Number(
                label="Conversation index", value=0, precision=0, minimum=0
            )

        chatbot = gr.Chatbot(label="Chat Trace", height=600)
        status_text = gr.Textbox(label="Status", interactive=False)

        def process_content(content):
            """Process different content formats and extract readable text"""
            import html

            if isinstance(content, list):
                content_str = ""
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            content_str += part.get("text", "")
                        else:
                            content_str += str(part)
                    else:
                        content_str += str(part)
                result = content_str
            elif isinstance(content, dict):
                if "text" in content:
                    result = content["text"]
                else:
                    result = str(content)
            elif content is None:
                result = ""
            else:
                result = str(content)

            # Escape HTML characters so tags like <think>, <function> are displayed as text
            return html.escape(result)

        def load_and_show(file_name: str, idx: int):
            try:
                if not file_name:
                    return [], "Please select a file"

                file_path = trace_root / file_name
                if file_name not in conversations_cache:
                    conversations_cache[file_name] = load_conversations(file_path)

                conversations = conversations_cache[file_name]

                if not conversations:
                    return [], "No conversations found in file"

                idx = int(idx)
                if not (0 <= idx < len(conversations)):
                    return [], f"Index {idx} out of range (0-{len(conversations)-1})"

                messages = conversations[idx]

                if not isinstance(messages, list):
                    return [], f"Invalid conversation format at index {idx}"

                chat_pairs = []
                i = 0

                while i < len(messages):
                    msg = messages[i]

                    if not isinstance(msg, dict):
                        i += 1
                        continue

                    role = msg.get("role", "unknown")
                    content = process_content(msg.get("content", ""))

                    if role == "system":
                        # System messages stand alone
                        chat_pairs.append([f"[SYSTEM]", content])

                    elif role == "user":
                        # Look for the next assistant message
                        assistant_content = ""
                        if i + 1 < len(messages):
                            next_msg = messages[i + 1]
                            if (
                                isinstance(next_msg, dict)
                                and next_msg.get("role") == "assistant"
                            ):
                                assistant_content = process_content(
                                    next_msg.get("content", "")
                                )
                                i += 1  # Skip the assistant message since we processed it

                        chat_pairs.append([content, assistant_content])

                    elif role in {"assistant", "tool"}:
                        # Standalone assistant/tool message (no preceding user message)
                        chat_pairs.append([f"[{role.upper()}]", content])

                    else:
                        chat_pairs.append([f"[{role.upper()}]", content])

                    i += 1

                status = f"Loaded conversation {idx}/{len(conversations)-1} from {file_name} - {len(chat_pairs)} message pairs, {len(messages)} total messages"
                return chat_pairs, status

            except Exception as e:
                import traceback

                error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
                return [], error_msg

        def update_file_selection(file_name: str):
            # Reset index when file changes
            if file_name:
                try:
                    file_path = trace_root / file_name
                    conversations = load_conversations(file_path)
                    max_idx = len(conversations) - 1 if conversations else 0
                    return (
                        gr.Number(value=0, maximum=max_idx),
                        f"File loaded: {len(conversations)} conversations",
                    )
                except Exception as e:
                    return gr.Number(value=0), f"Error loading file: {str(e)}"
            return gr.Number(value=0), ""

        # Event handlers
        file_dropdown.change(
            update_file_selection,
            inputs=[file_dropdown],
            outputs=[index_slider, status_text],
        )

        file_dropdown.change(
            load_and_show,
            inputs=[file_dropdown, index_slider],
            outputs=[chatbot, status_text],
        )

        index_slider.change(
            load_and_show,
            inputs=[file_dropdown, index_slider],
            outputs=[chatbot, status_text],
        )

        # Load initial conversation
        demo.load(
            load_and_show,
            inputs=[file_dropdown, index_slider],
            outputs=[chatbot, status_text],
        )

    demo.launch(server_port=port, share=True)


def main(trace_dir="./14b", port=9781):
    build_interface(trace_dir, port)


if __name__ == "__main__":
    Fire(main)
