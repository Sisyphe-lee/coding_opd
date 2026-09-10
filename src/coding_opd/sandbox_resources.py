"""Resource-only limits shared by evaluation agents and isolated graders."""

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "LOKY_MAX_CPU_COUNT",
)


def thread_limit_args(threads: int) -> list[str]:
    if threads < 1:
        raise ValueError("task CPU threads must be positive")
    return [arg for key in THREAD_ENV_VARS for arg in ("--env", f"{key}={threads}")]


def verifier_resource_args(run_args: list[str]) -> list[str]:
    """Carry only numeric thread limits across; never agent mounts or credentials."""
    result = []
    for index, arg in enumerate(run_args[:-1]):
        if arg != "--env":
            continue
        key, sep, value = run_args[index + 1].partition("=")
        if sep and key in THREAD_ENV_VARS and value.isdecimal() and int(value) > 0:
            result.extend(["--env", f"{key}={value}"])
    return result
