import json
import asyncio
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

import pytest
from uni_agent.tasks import TaskConfigResolver

from coding_opd.codex_agent import (
    _instruction,
    _summarize_jsonl,
    resolve_auto_compact_limit,
    summarize_codex_artifacts,
)
from coding_opd.codex_eval_entrypoint import _load_records, _safe_name
from coding_opd import codex_eval_entrypoint as entrypoint
from coding_opd.deepswe_task import DeepSWETask as DeepSWETask  # noqa: F401
from coding_opd.eval_data import select_canary_rows


def test_explicit_canary_retry_selection_is_validated():
    rows = [{"extra_info": {"task_id": key}} for key in ("a", "b", "c")]
    assert entrypoint._select_tasks(rows, ["c", "a"]) == [rows[0], rows[2]]
    assert entrypoint._select_tasks(rows, []) == rows
    with pytest.raises(ValueError, match="Unknown"):
        entrypoint._select_tasks(rows, ["missing"])
    with pytest.raises(ValueError, match="Duplicate"):
        entrypoint._select_tasks(rows, ["a", "a"])


def test_codex_jsonl_summary_preserves_usage_and_agent_message() -> None:
    output = "\n".join(
        [
            '{"type":"thread.started","thread_id":"abc"}',
            '{"type":"item.completed","item":{"type":"command_execution"}}',
            '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
            '{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}',
            "not-json",
        ]
    )
    transcript, info = _summarize_jsonl(output)
    assert transcript == [{"role": "assistant", "content": "done"}]
    assert info["thread_id"] == "abc"
    assert info["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert info["event_counts"]["item.command_execution"] == 1
    assert info["parse_errors"] == 1


def test_codex_instruction_keeps_single_user_prompt_exact() -> None:
    assert _instruction([{"role": "user", "content": "task text"}]) == "task text"


def test_auto_compact_threshold_is_explicit_and_bounded() -> None:
    assert resolve_auto_compact_limit(262144) == 235929
    assert resolve_auto_compact_limit(272000) == 244800
    assert resolve_auto_compact_limit(262144, 210000) == 210000
    for context, limit in [(262144, 260000), (262144, 0), (1000, None)]:
        with pytest.raises(ValueError):
            resolve_auto_compact_limit(context, limit)


def test_session_compaction_and_context_usage_diagnostics(tmp_path: Path) -> None:
    session = tmp_path / "codex-home/sessions/rollout.jsonl"
    session.parent.mkdir(parents=True)
    events = [
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "last_token_usage": {"total_tokens": 262144}}}},
        {"type": "compacted", "payload": {"message": "summary"}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": {
            "last_token_usage": {"total_tokens": 5000}}}},
        {"type": "event_msg", "payload": {"type": "token_count", "info": None}},
        None,
    ]
    session.write_text("\n".join(map(json.dumps, events)) + "\ntruncated")
    result = summarize_codex_artifacts(tmp_path)
    assert result["compaction_count"] == 1
    assert result["max_request_total_tokens"] == 262144
    assert result["last_request_total_tokens"] == 5000
    assert result["session_parse_errors"] == 2


def test_codex_jsonl_ignores_valid_json_non_events():
    output = '\n'.join(['"diagnostic"', 'null', '42', 'true', '[]',
        '{"type":"thread.started","thread_id":"valid"}',
        '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}'])
    transcript, info = _summarize_jsonl(output)
    assert info["parse_errors"] == 5
    assert info["thread_id"] == "valid"
    assert transcript == [{"role": "assistant", "content": "done"}]


def test_codex_artifacts_survive_in_live_mounted_log_dir(tmp_path: Path) -> None:
    raw_path = tmp_path / "codex-cli.jsonl"
    raw_path.write_text(
        '{"type":"thread.started","thread_id":"thread-1"}\n', encoding="utf-8"
    )
    session_path = tmp_path / "codex-home" / "sessions" / "2026" / "rollout.jsonl"
    session_path.parent.mkdir(parents=True)
    session_path.write_text('{"type":"session_meta"}\n', encoding="utf-8")

    summary = summarize_codex_artifacts(tmp_path)

    assert summary["thread_id"] == "thread-1"
    assert summary["raw_jsonl_bytes"] == raw_path.stat().st_size
    assert summary["session_files"] == [str(session_path)]
    assert summary["session_bytes"] == session_path.stat().st_size


def test_resume_records_are_locked_to_run_fingerprint(tmp_path: Path) -> None:
    task_id = "owner/repo:task/one"
    path = tmp_path / f"{_safe_name(task_id)}.json"
    path.write_text(
        json.dumps({"task_id": task_id, "run_fingerprint": "old"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="different run configuration"):
        _load_records(
            tmp_path,
            expected_task_ids={task_id},
            run_fingerprint="new",
        )


def test_resource_resume_preserves_zero_and_trailing_record(tmp_path):
    source = tmp_path / "source"
    (source / "records").mkdir(parents=True)
    old = dict(repo_git_sha="a" * 40, task_config_sha256="c" * 64, protocol="verified", model_path="student",
               sampling_config={"temperature": 0.6}, backend_count=4, concurrency=32,
               tasks_per_replica=8, serving_config=dict(gpu_groups=[[0], [1], [2], [3]],
               replica_count=4, tasks_per_replica=8, total_task_concurrency=32,
               max_num_seqs=8, tensor_parallel_size=1))
    fingerprint = entrypoint._fingerprint(old)
    records = [{"task_id": key, "score": score, "run_fingerprint": fingerprint}
               for key, score in [("zero", 0), ("pass", 1)]]
    for record in records:
        (source / "records" / (record["task_id"] + ".json")).write_text(json.dumps(record))
    payload = dict(run_identity=old, run_fingerprint=fingerprint, tasks=records[:1],
                   wall_seconds=100)
    result = source / "result.json"
    result.write_text(json.dumps(payload))
    args = SimpleNamespace(resume_from_result=result, result_path=tmp_path / "new.json",
                           resume_source_git_sha="a" * 40)
    new = deepcopy(old)
    new.update(repo_git_sha="b" * 40, backend_count=2, concurrency=16, task_cpu_threads=2)
    new["serving_config"].update(gpu_groups=[[0], [1]], replica_count=2,
                                  total_task_concurrency=16)
    loaded, provenance = entrypoint._resume_records(args, new, {"zero", "pass", "pending"})
    assert loaded == {r["task_id"]: r for r in records}
    assert provenance["inherited_tasks"] == 2
    assert loaded["zero"]["score"] == 0
    assert json.loads(result.read_text()) == payload
    args.resume_source_git_sha = None
    with pytest.raises(ValueError, match="reviewed full SHA"):
        entrypoint._resume_records(args, new, {"zero", "pass"})
    args.resume_source_git_sha = "a" * 40
    for key, value in [("sampling_config", {"temperature": 1.0}),
                       ("model_path", "teacher"), ("protocol", "other")]:
        incompatible = deepcopy(new)
        incompatible[key] = value
        with pytest.raises(ValueError, match="evaluation semantics"):
            entrypoint._resume_records(args, incompatible, {"zero", "pass"})
    with pytest.raises(ValueError, match="unexpected task record"):
        entrypoint._resume_records(args, new, {"zero"})


    changed = deepcopy(new)
    changed["task_config_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="evaluation semantics"):
        entrypoint._resume_records(args, changed, {"zero", "pass"})
    args.resume_source_task_config_sha256 = "wrong"
    with pytest.raises(ValueError, match="reviewed hash"):
        entrypoint._resume_records(args, changed, {"zero", "pass"})
    args.resume_source_task_config_sha256 = "c" * 64
    loaded, provenance = entrypoint._resume_records(args, changed, {"zero", "pass"})
    assert loaded == {r["task_id"]: r for r in records}
    assert provenance["reviewed_source_task_config_sha256"] == "c" * 64
    assert json.loads(result.read_text()) == payload


def test_deepswe_codex_config_is_network_isolated_and_uses_256k() -> None:
    path = Path(__file__).parents[1] / "configs" / "deepswe_codex.yaml"
    resolver = TaskConfigResolver.from_file(str(path))
    config = resolver.defaults_by_name["deep_swe"]
    run_args = config["sandbox"]["sandbox_kwargs"]["run_args"]
    assert run_args[run_args.index("--network") + 1] == "none"
    assert config["agent"]["context_window"] == 262_144
    assert config["agent"]["reasoning_effort"] == "xhigh"
    assert config["agent"]["reasoning_summary"] == "auto"


def test_deepswe_launcher_pins_official_runtime_semantics() -> None:
    root = Path(__file__).parents[1]
    launcher = (root / "scripts" / "run_deepswe_codex_eval.sh").read_text()
    task = (root / "src" / "coding_opd" / "deepswe_task.py").read_text()

    for setting in (
        "--enable-prefix-caching",
        "--mamba-cache-mode align",
        "--generation-config vllm",
        "--override-generation-config",
        "--max-num-seqs",
        "--language-model-only",
        "--tool-call-parser qwen3_coder",
        "--reasoning-parser qwen3",
    ):
        assert setting in launcher
    assert "git diff --binary {base_commit} HEAD" in task
    assert "git add -A" not in task


def test_verified_quick_launcher_uses_frozen64_not_random_canary():
    root = Path(__file__).parents[1]
    launcher = (root / "scripts/run_deepswe_codex_eval.sh").read_text()
    assert 'canary|quick|full)' in launcher
    assert 'DATA_PATH="${FULL_ROOT}/quick.parquet"' in launcher
    assert '!= "swebench_verified_64"' in launcher
    assert 'manifest["splits"]["quick"]["count"] != 64' in launcher
    assert 'EVAL_ROOT:-' in launcher
    materializer = (root / "scripts/materialize_swebench_verified.py").read_text()
    assert 'default=Path("configs/dataset_manifests/swebench_verified_64.json")' in materializer


def test_canary_seed_matches_pier_shuffle_then_truncate() -> None:
    rows = [{"id": value} for value in range(10)]
    selected = select_canary_rows(rows, count=5, sample_seed=0)
    assert [row["id"] for row in selected] == [7, 8, 1, 5, 3]
    assert rows == [{"id": value} for value in range(10)]


@pytest.mark.parametrize("broken_summary", [False, True])
def test_concurrent_tasks_do_not_share_log_mounts(tmp_path, monkeypatch, broken_summary) -> None:
    resolver = TaskConfigResolver.from_file(
        str(Path(__file__).parents[1] / "configs" / "deepswe_codex.yaml")
    )
    original = deepcopy(resolver.defaults_by_name)
    configs = []

    def fake_task(config):
        configs.append(config)

        async def run():
            await asyncio.sleep(0)
            return SimpleNamespace(reward=1, finished=True, extra_info={})

        return SimpleNamespace(run=run)

    monkeypatch.setattr(entrypoint, "get_task", fake_task)
    if broken_summary:
        def fail_summary(path):
            raise OSError("artifact read failed")
        monkeypatch.setattr(entrypoint, "summarize_codex_artifacts", fail_summary)

    async def exercise():
        queue = asyncio.Queue()
        for i in range(3):
            queue.put_nowait(entrypoint.Backend(str(i), f"/model-{i}.sock"))
        semaphore = asyncio.Semaphore(3)
        await asyncio.gather(*(
            entrypoint._run_one(
                {"prompt": [], "extra_info": {
                    "task_id": f"task-{i}",
                    "tools_kwargs": {"task": {"name": "deep_swe"}},
                }},
                resolver=resolver, model_name="test", log_dir=tmp_path / "logs",
                records_dir=tmp_path / "records", backend_queue=queue,
                semaphore=semaphore, run_fingerprint="test",
            ) for i in range(3)
        ))

    asyncio.run(exercise())
    records = [json.loads(p.read_text()) for p in (tmp_path / "records").glob("*.json")]
    assert len(records) == 3
    assert all(r["score"] == 1 and r["status"] == "success" for r in records)
    if broken_summary:
        assert all("summary_error" in r["agent_artifacts"] for r in records)
    mounts = []
    for config in configs:
        task_mounts = [arg for arg in config["sandbox"]["sandbox_kwargs"]["run_args"]
                       if arg.endswith(":/opt/coding-opd/agent-logs:rw")]
        assert len(task_mounts) == 1
        mounts.extend(task_mounts)
    assert len(set(mounts)) == 3
    assert resolver.defaults_by_name == original
