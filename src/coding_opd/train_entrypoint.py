"""Coding OPD trainer entry point with project-specific kernel setup."""

from __future__ import annotations

from coding_opd.qwen35_kernels import install_qwen35_fla_kernels
from coding_opd.verl_optimizations import (
    install_isolated_vllm_compile_cache,
    install_pure_distillation_fast_path,
    install_separate_async_standalone_memory_budget,
)


def _project_task_runner(main_ppo):
    """Install project optimizations inside the remote veRL task runner."""

    import ray

    # Ray's decorator keeps the original user class as the second class in the
    # generated actor's MRO.  Subclass it so veRL remains responsible for the
    # complete runner lifecycle while this project owns only its fast paths.
    base = main_ppo.TaskRunnerV1.__ray_metadata__.modified_class.__mro__[1]

    @ray.remote
    class CodingOPDTaskRunnerV1(base):
        def run(self, config):
            install_pure_distillation_fast_path()
            install_isolated_vllm_compile_cache()
            install_separate_async_standalone_memory_budget()
            return super().run(config)

    return CodingOPDTaskRunnerV1


def main() -> None:
    install_qwen35_fla_kernels()
    # Import instead of runpy so we can swap in a thin TaskRunner subclass.
    # The optimization must be installed in the remote Ray actor process, not
    # only in this launcher process.
    import verl.trainer.main_ppo as main_ppo

    main_ppo.TaskRunnerV1 = _project_task_runner(main_ppo)
    main_ppo.main()


if __name__ == "__main__":
    main()
