import asyncio
import json
from types import SimpleNamespace

import pytest

from coding_opd import profiling, r2e_task
from coding_opd.opd_framework import OPDGatewayAgentFramework
from uni_agent.framework.framework import GatewayAgentFramework


def records(path):
    return [json.loads(line) for file in path.glob("*.jsonl") for line in file.read_text().splitlines()]


def test_concurrent_spans_keep_identity_and_cancellation(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_OPD_PROFILE_DIR", str(tmp_path))

    async def job(uid):
        with profiling.profile_span("session", uid=uid):
            await asyncio.sleep(0)
            with profiling.profile_span("child"):
                await asyncio.sleep(0)
                if uid == "cancelled":
                    raise asyncio.CancelledError()

    async def run():
        return await asyncio.gather(job("ok"), job("cancelled"), return_exceptions=True)

    results = asyncio.run(run())
    assert isinstance(results[1], asyncio.CancelledError)
    rows = records(tmp_path)
    assert len(rows) == 4
    assert all(row["duration_s"] >= 0 and row["start_ns"] > 0 for row in rows)
    assert {(row["uid"], row["status"]) for row in rows} == {
        ("ok", "ok"), ("cancelled", "CancelledError")
    }
    assert profiling._context.get() == {}


def test_disabled_and_unwritable_profiling_preserve_result(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_OPD_PROFILE_DIR", "")
    with profiling.profile_span("disabled"):
        pass
    assert not list(tmp_path.iterdir())
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("existing")
    monkeypatch.setenv("CODING_OPD_PROFILE_DIR", str(not_a_directory))
    with profiling.profile_span("unwritable"):
        result = 42
    assert result == 42
    assert not_a_directory.read_text() == "existing"


def test_model_client_preserves_request_output_and_exceptions(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_OPD_PROFILE_DIR", str(tmp_path))
    output = SimpleNamespace(token_ids=[9, 8])
    kwargs = {"request_id": "session-1", "prompt_ids": [1, 2, 3], "sampling_params": {"temperature": 0.6}}
    captured = []

    class Client:
        async def generate(self, **request):
            captured.append(request)
            if request["request_id"] == "fail":
                raise RuntimeError("original failure")
            return output

    client = profiling.ProfiledLLMClient(Client(), "rollout")
    assert asyncio.run(client.generate(**kwargs)) is output
    assert captured == [kwargs]
    with pytest.raises(RuntimeError, match="original failure"):
        asyncio.run(client.generate(**{**kwargs, "request_id": "fail"}))
    rows = records(tmp_path)
    assert rows[0]["session_id"] == "session-1"
    assert (rows[0]["input_tokens"], rows[0]["output_tokens"]) == (3, 2)
    assert rows[1]["status"] == "RuntimeError"
    assert all("prompt_ids" not in row and "sampling_params" not in row for row in rows)


def test_tool_context_crosses_runner_and_thread(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_OPD_PROFILE_DIR", str(tmp_path))

    async def task(**kwargs):
        def work():
            with profiling.profile_span("thread_work"):
                return 7
        return await asyncio.to_thread(work)

    monkeypatch.setattr(r2e_task, "_run_task", task)
    assert asyncio.run(r2e_task.run_r2e_task(tools_kwargs={"_trace_identity": {
        "uid": "u1", "session_id": "s1", "global_steps": 64,
    }})) == 7
    assert all(row["uid"] == "u1" and row["step"] == 64 and row["session_id"] == "s1"
               for row in records(tmp_path))


def test_teacher_finishes_before_enqueue_and_slot_wait_is_separate(monkeypatch, tmp_path):
    monkeypatch.setenv("CODING_OPD_PROFILE_DIR", str(tmp_path))
    framework = object.__new__(OPDGatewayAgentFramework)
    framework._distillation_enabled = True
    calls = []

    async def teacher(*args):
        await asyncio.sleep(0.01)
        calls.append("teacher")

    async def write(self, **kwargs):
        assert calls == ["teacher"]
        calls.append("write")

    async def acquire(self, **kwargs):
        await asyncio.sleep(0.01)
        return await self._run_agent_episode(**kwargs)

    async def episode(self, **kwargs):
        return "unchanged"

    monkeypatch.setattr(framework, "_attach_teacher_logprobs", teacher)
    monkeypatch.setattr(GatewayAgentFramework, "_write_session_trajectories_to_tq", write)
    monkeypatch.setattr(GatewayAgentFramework, "_run_agent_episode_with_concurrency_limit", acquire)
    monkeypatch.setattr(GatewayAgentFramework, "_run_agent_episode", episode)

    async def run():
        with profiling.profile_span("prompt", uid="u1", step=3):
            assert await framework._run_agent_episode_with_concurrency_limit(
                session_index=0, runner_config=SimpleNamespace(runner_kwargs={}), sample_fields={},
            ) == "unchanged"
            await framework._write_session_trajectories_to_tq(
                uid="u1", session_index=0, trajectories=[SimpleNamespace(extra_fields={})], sample_fields={},
                global_steps=3, partition_id="train",
            )

    asyncio.run(run())
    rows = {row["event"]: row for row in records(tmp_path)}
    assert rows["agent_episode"]["start_ns"] - rows["session_wait_and_run"]["start_ns"] >= 5_000_000
    assert rows["tq_write"]["start_ns"] > rows["teacher_score"]["start_ns"]
    assert all(row["uid"] == "u1" and row["step"] == 3 for row in rows.values())
