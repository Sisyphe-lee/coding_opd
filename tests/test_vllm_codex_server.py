import json
import pytest
from copy import deepcopy

from coding_opd.vllm_codex_server import preserve_invalid_tool_history


@pytest.mark.parametrize("content", [None, "", [], [{"type": "text", "text": "reasoning"}],
    [{"type": "text", "text": "first"}, {"type": "image_url", "image_url": {"url": "test"}}]])
def test_invalid_history_accepts_content_blocks_without_losing_content(content):
    call = {"id": "bad", "type": "function", "function": {
        "name": "exec_command", "arguments": '{"cmd":',
    }}
    original = deepcopy(content)
    messages = [{"role": "assistant", "content": content, "tool_calls": [call]},
                {"role": "tool", "tool_call_id": "bad", "content": "parse error"}]
    tool = deepcopy(messages[1])
    assert preserve_invalid_tool_history(messages) == 1
    result = messages[0]["content"]
    if isinstance(content, list):
        assert result[:-1] == original
        assert content == original
        assert result[-1]["type"] == "text"
        text = result[-1]["text"]
    else:
        text = result
    assert json.dumps(call, ensure_ascii=False) in text
    assert messages[1] == tool
    assert preserve_invalid_tool_history(messages) == 0


def test_invalid_history_preserves_raw_call_and_error_without_repair():
    call = {"id": "bad", "type": "function", "function": {
        "name": "exec_command", "arguments": '{"cmd": "cd /app && ls -',
    }}
    good = {"id": "good", "type": "function", "function": {
        "name": "exec_command", "arguments": '{"cmd": "pwd"}',
    }}
    messages = [
        {"role": "assistant", "content": "", "tool_calls": [good, call]},
        {"role": "tool", "tool_call_id": "bad", "content": "failed to parse function arguments"},
    ]
    error = deepcopy(messages[1])
    assert preserve_invalid_tool_history(messages) == 1
    assert messages[0]["tool_calls"] == [good]
    assert json.dumps(call, ensure_ascii=False) in messages[0]["content"]
    assert messages[1] == error
    assert preserve_invalid_tool_history(messages) == 0


def test_valid_calls_and_plain_messages_are_unchanged():
    messages = [{"role": "assistant", "content": "hello", "tool_calls": [
        {"type": "function", "function": {"name": "exec_command", "arguments": '{"cmd":"pwd"}'}},
    ]}, {"role": "user", "content": "continue"}]
    original = deepcopy(messages)
    assert preserve_invalid_tool_history(messages) == 0
    assert messages == original
