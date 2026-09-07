"""Project-scoped vLLM server compatibility for rejected tool-call history.

Codex returns malformed function calls together with a tool error so the model
can recover. vLLM 0.24 parses those arguments again when rendering history and
rejects the entire next request. Preserve rejected calls as assistant text;
never repair arguments, execute them, or modify valid calls/tool results.
"""

from __future__ import annotations

import json
import logging
import runpy

logger = logging.getLogger(__name__)


def preserve_invalid_tool_history(messages: list[dict]) -> int:
    count = 0
    for message in messages:
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list):
            continue
        valid = []
        rejected = []
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else None
            if isinstance(arguments, str) and arguments:
                try:
                    json.loads(arguments)
                except json.JSONDecodeError:
                    rejected.append(json.dumps(call, ensure_ascii=False))
                    count += 1
                    continue
            valid.append(call)
        if rejected:
            content = message.get("content")
            raw_text = "\n[Previous invalid tool calls; raw records]\n" + "\n".join(rejected)
            if content is None or isinstance(content, str):
                message["content"] = (content or "") + raw_text
            elif isinstance(content, list):
                # Responses history may retain OpenAI content blocks. Preserve
                # their types/order; append text instead of stringifying blocks.
                message["content"] = [*content, {"type": "text", "text": raw_text}]
            else:
                raise TypeError("Expected text or content blocks for invalid tool history")
            message["tool_calls"] = valid
    return count


def install_history_compatibility() -> None:
    from vllm.entrypoints import chat_utils

    original = chat_utils._postprocess_messages

    def postprocess(messages):
        count = preserve_invalid_tool_history(messages)
        if count:
            logger.warning("Preserved %d invalid historical tool calls as text", count)
        original(messages)

    chat_utils._postprocess_messages = postprocess


if __name__ == "__main__":
    install_history_compatibility()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
