from __future__ import annotations

import re
from dataclasses import dataclass


ACTION_TAGS = ("Analyze", "Understand", "Code", "Execute", "Answer", "File")
MODEL_ACTION_TAGS = frozenset({"Analyze", "Understand", "Code", "Answer"})
ACTION_TAG_PATTERN = "|".join(ACTION_TAGS)
ACTION_OPEN_RE = re.compile(rf"<({ACTION_TAG_PATTERN})>")
ACTION_CLOSE_RE = re.compile(rf"</({ACTION_TAG_PATTERN})>")
PYTHON_FENCE_RE = re.compile(r"```(?:python|py|python3)?\s*\r?\n", re.IGNORECASE)


class ProtocolValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ActionSection:
    tag: str
    body: str
    start: int
    end: int


def mask_backticked_content(content: str) -> str:
    raw = content or ""
    chars = list(raw)
    cursor = 0
    while cursor < len(raw):
        if raw[cursor] != "`":
            cursor += 1
            continue
        tick_count = 1
        while cursor + tick_count < len(raw) and raw[cursor + tick_count] == "`":
            tick_count += 1
        delimiter = "`" * tick_count
        end_index = raw.find(delimiter, cursor + tick_count)
        if end_index == -1:
            cursor += tick_count
            continue
        end_index += tick_count
        for index in range(cursor, end_index):
            chars[index] = " "
        cursor = end_index
    return "".join(chars)


def parse_actions(content: str) -> list[ActionSection]:
    raw = content or ""
    masked = mask_backticked_content(raw)
    actions: list[ActionSection] = []
    cursor = 0

    while True:
        match = ACTION_OPEN_RE.search(masked, cursor)
        if match is None:
            if masked[cursor:].strip():
                raise ProtocolValidationError("text outside structured action blocks")
            break
        if masked[cursor : match.start()].strip():
            raise ProtocolValidationError("text outside structured action blocks")

        tag = match.group(1)
        close_tag = f"</{tag}>"
        close_index = masked.find(close_tag, match.end())
        if close_index == -1:
            raise ProtocolValidationError(f"incomplete <{tag}> action")

        nested_match = ACTION_OPEN_RE.search(masked, match.end(), close_index)
        if nested_match is not None:
            raise ProtocolValidationError(
                f"nested <{nested_match.group(1)}> action inside <{tag}>"
            )
        mismatched_close = ACTION_CLOSE_RE.search(masked, match.end(), close_index)
        if mismatched_close is not None:
            raise ProtocolValidationError(
                f"mismatched </{mismatched_close.group(1)}> inside <{tag}>"
            )

        body = raw[match.end() : close_index].strip()
        if not body:
            raise ProtocolValidationError(f"empty <{tag}> action")
        end = close_index + len(close_tag)
        actions.append(ActionSection(tag=tag, body=body, start=match.start(), end=end))
        cursor = end

    if not actions:
        raise ProtocolValidationError("no structured action blocks")
    return actions


def validate_model_actions(content: str) -> list[ActionSection]:
    actions = parse_actions(content)
    unsupported = [action.tag for action in actions if action.tag not in MODEL_ACTION_TAGS]
    if unsupported:
        raise ProtocolValidationError(
            f"model emitted system-owned action <{unsupported[0]}>"
        )

    terminal_actions = [action for action in actions if action.tag in {"Code", "Answer"}]
    if len(terminal_actions) != 1:
        raise ProtocolValidationError("exactly one terminal <Code> or <Answer> action is required")
    if actions[-1] != terminal_actions[0]:
        raise ProtocolValidationError("terminal <Code> or <Answer> action must be last")
    return actions


def normalize_model_output(content: str) -> tuple[str, list[ActionSection]]:
    """将常见格式偏差整理为规范动作块，不猜测或补全截断代码。"""
    raw = (content or "").strip()
    try:
        return raw, validate_model_actions(raw)
    except ProtocolValidationError:
        pass

    masked = mask_backticked_content(raw)
    complete_sections: list[ActionSection] = []
    occupied: list[tuple[int, int]] = []
    cursor = 0
    while match := ACTION_OPEN_RE.search(masked, cursor):
        tag = match.group(1)
        close_tag = f"</{tag}>"
        close_index = masked.find(close_tag, match.end())
        if close_index == -1:
            if tag == "Code":
                raise ProtocolValidationError("incomplete <Code> action")
            if tag == "Answer":
                body = raw[match.end() :].strip()
                if body:
                    complete_sections.append(
                        ActionSection(tag=tag, body=body, start=match.start(), end=len(raw))
                    )
                    occupied.append((match.start(), len(raw)))
                break
            cursor = match.end()
            continue
        end = close_index + len(close_tag)
        body = raw[match.end() : close_index].strip()
        if body:
            complete_sections.append(
                ActionSection(tag=tag, body=body, start=match.start(), end=end)
            )
        occupied.append((match.start(), end))
        cursor = end

    def clean_control_tags(body: str) -> str:
        return ACTION_CLOSE_RE.sub("", ACTION_OPEN_RE.sub("", body)).strip()

    terminals = [
        section for section in complete_sections if section.tag in {"Code", "Answer"}
    ]
    if terminals:
        terminal = terminals[-1]
        prefix = [
            ActionSection(
                section.tag,
                clean_control_tags(section.body),
                section.start,
                section.end,
            )
            for section in complete_sections
            if section.start < terminal.start and section.tag in {"Analyze", "Understand"}
            and clean_control_tags(section.body)
        ]
        if terminal.tag == "Answer":
            terminal = ActionSection(
                terminal.tag,
                clean_control_tags(terminal.body),
                terminal.start,
                terminal.end,
            )
    else:
        if complete_sections:
            final_section = complete_sections[-1]
            terminal = ActionSection(
                "Answer",
                clean_control_tags(final_section.body),
                final_section.start,
                final_section.end,
            )
            prefix = [
                ActionSection(
                    section.tag,
                    clean_control_tags(section.body),
                    section.start,
                    section.end,
                )
                for section in complete_sections[:-1]
                if section.tag in {"Analyze", "Understand"}
                and clean_control_tags(section.body)
            ]
        else:
            terminal_tag = "Code" if PYTHON_FENCE_RE.search(raw) else "Answer"
            terminal = ActionSection(terminal_tag, raw, 0, len(raw))
            prefix = []

    outside_parts: list[str] = []
    outside_cursor = 0
    for start, end in occupied:
        if start >= terminal.start:
            break
        text = raw[outside_cursor:start].strip()
        if text:
            outside_parts.append(text)
        outside_cursor = max(outside_cursor, end)
    trailing_prefix = raw[outside_cursor : terminal.start].strip()
    if trailing_prefix:
        outside_parts.append(trailing_prefix)
    if outside_parts:
        prefix.insert(0, ActionSection("Analyze", "\n\n".join(outside_parts), 0, 0))

    normalized_sections = [*prefix, terminal]
    normalized = "\n".join(
        f"<{section.tag}>{section.body}</{section.tag}>"
        for section in normalized_sections
    )
    return normalized, validate_model_actions(normalized)


def contains_completed_action(content: str, tag: str) -> bool:
    if tag not in ACTION_TAGS:
        return False
    masked = mask_backticked_content(content or "")
    return f"</{tag}>" in masked


def find_completed_action_end(
    content: str,
    tags: tuple[str, ...] = ("Code", "Answer"),
) -> int | None:
    """返回内容中第一个完整终止动作的边界位置。"""
    masked = mask_backticked_content(content or "")
    boundaries = []
    for tag in tags:
        if tag not in ACTION_TAGS:
            continue
        close_tag = f"</{tag}>"
        close_index = masked.find(close_tag)
        if close_index >= 0:
            boundaries.append(close_index + len(close_tag))
    return min(boundaries) if boundaries else None
