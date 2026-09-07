from __future__ import annotations

import pytest

from coding_opd.r2e_task import ConciseEditFileConfig, cap_tool_result
from coding_opd.swesmith_task import (
    SWESmithTaskConfig,
    _backfill_problem_statement,
    _checkout_command,
    _prepare_evaluation_command,
    expected_test_files,
    score_test_statuses,
)


def _metadata() -> dict:
    return {
        "instance_id": "owner__repo.1234.func_basic__abcd1234",
        "repo": "swesmith/owner__repo.1234",
        "FAIL_TO_PASS": ["tests/test_a.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_a.py::test_old", "tests/test_b.py::test_other"],
    }


def test_swesmith_scoring_requires_f2p_and_p2p() -> None:
    metadata = _metadata()
    result = score_test_statuses(
        metadata,
        {
            "tests/test_a.py::test_fix": "PASSED",
            "tests/test_a.py::test_old": "PASSED",
            "tests/test_b.py::test_other": "XFAIL",
        },
        0,
    )
    assert result["resolved"] is True
    assert result["reward"] == 1.0

    failed = score_test_statuses(metadata, {"tests/test_a.py::test_fix": "FAILED"}, 1)
    assert failed["resolved"] is False
    assert failed["partial_score"] == 0.0


def test_swesmith_evaluation_restores_hidden_tests() -> None:
    metadata = _metadata()
    assert expected_test_files(metadata) == ["tests/test_a.py", "tests/test_b.py"]
    command = _prepare_evaluation_command(metadata)
    assert "HEAD~1" not in command
    assert "coding-opd-swesmith-bug-base" in command
    assert "git checkout HEAD -- tests/test_a.py tests/test_b.py" in command


def test_swesmith_checkout_rejects_shell_metacharacters() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _checkout_command("task; touch /tmp/no")


def test_swesmith_runner_backfills_old_parquet_metadata() -> None:
    kwargs = {
        "raw_prompt": [{"role": "user", "content": "Fix the regression"}],
        "tools_kwargs": {"task": {"name": "swe_smith", "metadata": {"instance_id": "task-1"}}},
    }

    updated = _backfill_problem_statement(kwargs)

    assert updated["tools_kwargs"]["task"]["metadata"]["problem_statement"] == "Fix the regression"
    assert "problem_statement" not in kwargs["tools_kwargs"]["task"]["metadata"]


def test_swesmith_evaluation_defaults_on_but_can_be_disabled() -> None:
    assert SWESmithTaskConfig(sandbox={"provider": "coding_opd_podman"}).run_evaluation is True
    assert (
        SWESmithTaskConfig(sandbox={"provider": "coding_opd_podman"}, run_evaluation=False).run_evaluation
        is False
    )


def test_concise_editor_output_cap_is_configurable() -> None:
    assert ConciseEditFileConfig().max_output_chars == 10_000
    assert ConciseEditFileConfig(max_output_chars=8_000).max_output_chars == 8_000
    with pytest.raises(ValueError):
        ConciseEditFileConfig(max_output_chars=1_000)


def test_concise_editor_caps_long_observations_without_changing_status() -> None:
    from uni_agent.tools.base import ToolResult

    original = ToolResult(text="x" * 20, status="format_error")
    clipped = cap_tool_result(original, 10)

    assert clipped.text.startswith("x" * 10)
    assert "response clipped" in clipped.text
    assert clipped.status == "format_error"
    assert cap_tool_result(original, 20) is original
