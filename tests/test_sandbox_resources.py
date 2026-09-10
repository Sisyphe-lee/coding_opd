import pytest

from coding_opd.sandbox_resources import THREAD_ENV_VARS, thread_limit_args, verifier_resource_args


def test_thread_limits_reach_grader_without_agent_mounts():
    limits = thread_limit_args(2)
    assert len(limits) == 2 * len(THREAD_ENV_VARS)
    args = ["--volume", "/agent:/agent:rw", "--env", "SECRET=hidden", *limits]
    assert verifier_resource_args(args) == limits
    assert verifier_resource_args(["--env", "OMP_NUM_THREADS=bad"]) == []


def test_invalid_thread_limit():
    with pytest.raises(ValueError):
        thread_limit_args(0)
