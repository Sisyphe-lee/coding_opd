from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from coding_opd.dataset_selection import id_sha256
from coding_opd.deepswe_task import parse_deepswe_reward
from coding_opd.eval_data import (
    agent_images_from_rows,
    convert_verified_row,
    load_deepswe_task_ids,
    materialize_deepswe_rows,
    portable_deepswe_verifier_dockerfile,
    select_verified_manifest_rows,
)
from coding_opd.eval_entrypoint import task_runner_fqn


def _verified_row(task_id: str) -> dict:
    return {
        "instance_id": task_id,
        "repo": "org/repo",
        "problem_statement": "Fix the bug",
        "base_commit": "a" * 40,
        "patch": "secret gold patch",
        "test_patch": "secret test patch",
        "version": "1",
        "FAIL_TO_PASS": ["test_new"],
        "PASS_TO_PASS": ["test_old"],
    }


def test_verified_runtime_uses_exact_frozen_quick_order() -> None:
    rows = [_verified_row("task-b"), _verified_row("task-a")]
    quick_ids = ["task-a"]
    manifest = {
        "source": {"full_count": 2},
        "task_id_sha256": id_sha256(quick_ids),
        "tasks": [{"task_id": task_id} for task_id in quick_ids],
    }
    quick, full = select_verified_manifest_rows(rows, manifest)
    assert [row["instance_id"] for row in quick] == quick_ids
    assert [row["instance_id"] for row in full] == ["task-b", "task-a"]
    converted = convert_verified_row(quick[0], split="quick")
    assert "secret gold patch" not in json.dumps(converted["prompt"])
    assert converted["extra_info"]["task_id"] == "task-a"


def test_deepswe_materialization_validates_pinned_json_and_separates_verifier(tmp_path: Path) -> None:
    official = {"n_tasks": 1, "rows": [{"id": "task-one"}]}
    raw = json.dumps(official, separators=(",", ":")).encode()
    official_path = tmp_path / "tasks.json"
    official_path.write_bytes(raw)
    manifest = {
        "source": {"official_full_count": 1, "official_tasks_sha256": hashlib.sha256(raw).hexdigest()}
    }
    assert load_deepswe_task_ids(official_path, manifest) == ["task-one"]

    task_dir = tmp_path / "tasks" / "task-one"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Implement the feature", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        """
[metadata]
task_id = "task-one"
language = "python"
repository_url = "https://example.test/repo"
base_commit_hash = "abcdef1"
[environment]
docker_image = "example/agent:v1"
[verifier]
environment_mode = "separate"
""".strip(),
        encoding="utf-8",
    )
    rows = materialize_deepswe_rows(tmp_path / "tasks", ["task-one"], split="quick")
    task = rows[0]["extra_info"]["tools_kwargs"]["task"]
    assert task["sandbox"]["image"] == "example/agent:v1"
    assert task["metadata"]["verifier_image"].startswith("coding-opd/deepswe-verifier:")
    assert task["metadata"]["instruction"] == "Implement the feature"


def test_deepswe_tasks_json_hash_mismatch_is_fatal(tmp_path: Path) -> None:
    path = tmp_path / "tasks.json"
    path.write_text('{"n_tasks": 0, "rows": []}', encoding="utf-8")
    manifest = {"source": {"official_full_count": 1, "official_tasks_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="hash mismatch"):
        load_deepswe_task_ids(path, manifest)


def test_deepswe_reward_is_binary() -> None:
    assert parse_deepswe_reward('{"reward": 1, "partial": 0.5}')["reward"] == 1.0
    with pytest.raises(ValueError, match="binary reward"):
        parse_deepswe_reward('{"reward": 0.5}')


@pytest.mark.parametrize(
    ("task_name", "expected"),
    [
        ("deep_swe", "coding_opd.deepswe_task.run_deepswe_task"),
        ("swe_bench", "coding_opd.swe_bench_task.run_swe_bench_task"),
    ],
)
def test_external_eval_uses_project_runner_for_distributed_registration(
    task_name: str, expected: str
) -> None:
    assert task_runner_fqn([{"name": task_name}]) == expected


def test_external_eval_rejects_unknown_or_mixed_task_runners() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        task_runner_fqn([{"name": "unknown"}])
    with pytest.raises(ValueError, match="exactly one"):
        task_runner_fqn([{"name": "deep_swe"}, {"name": "swe_bench"}])


def test_agent_image_manifest_requires_one_unique_image_per_task() -> None:
    rows = [convert_verified_row(_verified_row("task-a"), split="full")]
    assert agent_images_from_rows(rows) == ["swebench/sweb.eval.x86_64.task-a"]
    with pytest.raises(ValueError, match="unique agent image"):
        agent_images_from_rows(rows * 2)


def test_deepswe_portable_dockerfile_removes_only_redundant_chmod(tmp_path: Path) -> None:
    context = tmp_path / "tests"
    context.mkdir()
    test_script = context / "test.sh"
    test_script.write_text("#!/bin/sh\n", encoding="utf-8")
    test_script.chmod(0o755)
    dockerfile = "FROM example/base\nCOPY test.sh /tests/test.sh\nRUN chmod +x /tests/test.sh\n"
    (context / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    assert portable_deepswe_verifier_dockerfile(context) == (
        "FROM example/base\nCOPY test.sh /tests/test.sh\n"
    )


def test_deepswe_portable_dockerfile_rejects_non_executable_source(tmp_path: Path) -> None:
    context = tmp_path / "tests"
    context.mkdir()
    (context / "test.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (context / "Dockerfile").write_text("RUN chmod +x /tests/test.sh\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must already be executable"):
        portable_deepswe_verifier_dockerfile(context)


def test_quick_first_evaluation_runs_complement_once_and_resumes(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from coding_opd import eval_entrypoint as evaluation

    samples = [{"extra_info": {"task_id": name}} for name in ("a", "b", "c")]
    assert evaluation.evaluation_stages(samples, ["b"]) == [[samples[1]], [samples[0], samples[2]]]
    with pytest.raises(ValueError, match="contained"):
        evaluation.evaluation_stages(samples, ["missing"])
    config_file = tmp_path / "config"
    config_file.write_text("unchanged")
    args = SimpleNamespace(result_path=tmp_path / "result.json", task_config=config_file,
                           runtime_manifest=config_file, model_path=tmp_path / "model",
                           data_path=tmp_path / "full.parquet", served_model_name="student",
                           n=1, quick_first=True, split="full")
    manifest = {"benchmark": "test", "splits": {"quick": {"task_ids": ["b"]}}}
    calls, scores = [], {}

    class Adapter:
        def generate_sequences_and_wait(self, prompts):
            rows, uids = prompts
            calls.append([row["extra_info"]["task_id"] for row in rows])
            scores.update({uid: [1.0] for uid in uids})

    monkeypatch.setattr(evaluation.LLMServerManager, "create", lambda **kw: SimpleNamespace(get_client=lambda: None))
    monkeypatch.setattr(evaluation.OPDAgentFrameworkRolloutAdapter, "create", lambda **kw: Adapter())
    monkeypatch.setattr(evaluation, "_build_prompts", lambda rows, uids: (rows, uids))
    monkeypatch.setattr(evaluation, "_read_scores", lambda uids: (
        {uid: scores[uid] for uid in uids if uid in scores}, {uid: "success" for uid in uids}))
    monkeypatch.setattr(evaluation.time, "sleep", lambda _: None)
    evaluation._evaluate(args, None, samples, manifest)
    assert calls == [["b"], ["a", "c"]]
    assert json.loads(args.result_path.read_text())["completed_tasks"] == 3
    assert json.loads((tmp_path / "quick_result.json").read_text())["num_tasks"] == 1
    evaluation._evaluate(args, None, samples, manifest)
    assert calls == [["b"], ["a", "c"]]
