import subprocess
import asyncio
from pathlib import Path

import pytest
from uni_agent.tasks import TaskConfigResolver, get_task

from coding_opd.swebench_codex_task import collect_patch_command, installed_image_test_spec
from coding_opd.codex_eval_entrypoint import _protocol
from coding_opd.codex_eval_entrypoint import _run
from types import SimpleNamespace


def test_verified_collects_committed_uncommitted_and_new_files(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()
    git("init", "-q")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    (tmp_path / "source.py").write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "base")
    # Canonical images commit their chmod changes before any model sees them.
    (tmp_path / "source.py").chmod(0o755)
    git("add", ".")
    git("commit", "-qm", "image preparation")
    base = git("rev-parse", "HEAD")
    (tmp_path / "source.py").write_text("committed\n")
    git("commit", "-qam", "agent commit")
    (tmp_path / "source.py").write_text("uncommitted\n")
    (tmp_path / "new.py").write_text("new source\n")
    command = collect_patch_command(base).replace(
        "git config --global --add safe.directory /testbed; ", ""
    )
    patch = subprocess.check_output(["bash", "-c", command], cwd=tmp_path, text=True)
    assert "+uncommitted" in patch and "+new source" in patch and "-original" in patch
    assert "old mode" not in patch
    with pytest.raises(ValueError):
        collect_patch_command("HEAD; echo bad")


def test_verified_codex_prompt_excludes_hidden_metadata():
    resolver = TaskConfigResolver.from_file(str(Path(__file__).parents[1] / "configs/swebench_codex.yaml"))
    config = resolver.resolve({"name": "swe_bench_codex", "sandbox": {"image": "example:latest"},
                               "metadata": {"problem_statement": "Fix public issue",
                                            "patch": "SECRET_GOLD", "test_patch": "SECRET_TEST"}},
                              runtime_model={"model_name": "Qwen3.5-9B"})
    task = get_task(config)
    prompt = str(task.config.prompt)
    assert "Fix public issue" in prompt
    assert "SECRET" not in prompt
    assert "no git commit is required" in prompt
    assert task.config.agent_timeout == 2700
    assert task.config.eval_timeout == 1800


def test_verified_protocol_is_not_deepswe():
    assert "Verified" in _protocol(SimpleNamespace(benchmark="swebench_verified"))
    assert "DeepSWE" in _protocol(SimpleNamespace())


def test_verified_scheduler_accepts_verified_defaults():
    args = SimpleNamespace(
        task_config=Path(__file__).parents[1] / "configs/swebench_codex.yaml",
        context_window=262144, reasoning_effort="xhigh", reasoning_summary="auto",
        auto_compact_token_limit=235929,
        concurrency=2, tasks_per_replica=2, model_socket=["/example.sock"],
    )
    result = asyncio.run(_run(args, [], records={}, runtime_manifest={}, identity={},
                             run_fingerprint="test", started_at=0))
    assert result == {}


def test_installed_image_spec_matches_official_eval_script(monkeypatch):
    from swebench.harness.test_spec import test_spec as official
    sample = dict(instance_id="django__django-12345", repo="django/django", version="3.2",
                  base_commit="a" * 40, test_patch="", FAIL_TO_PASS='["test_fix"]', PASS_TO_PASS="[]")
    monkeypatch.setattr(official, "make_repo_script_list", lambda *a: [])
    monkeypatch.setattr(official, "make_env_script_list", lambda *a: [])
    expected = official.make_test_spec(sample)
    actual = installed_image_test_spec(sample)
    assert actual.eval_script == expected.eval_script
    assert actual.FAIL_TO_PASS == expected.FAIL_TO_PASS
    assert actual.PASS_TO_PASS == expected.PASS_TO_PASS
