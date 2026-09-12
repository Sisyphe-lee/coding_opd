"""Coding OPD trainer entry point with project-specific kernel setup."""

from __future__ import annotations

import os
from pathlib import Path

from coding_opd.async_prefetch import install_checkpoint_safe_prefetch
from coding_opd.rollout_config import configure_coding_rollout
from coding_opd.qwen35_kernels import install_qwen35_fla_kernels
from coding_opd.parallel_startup import install_parallel_startup
from coding_opd.verl_optimizations import (
    install_isolated_vllm_compile_cache,
    install_pure_distillation_fast_path,
    install_separate_async_standalone_memory_budget,
)


def _save_run_config(config) -> None:
    path = os.environ.get("CODING_OPD_RUN_CONFIG_PATH")
    if not path:
        return
    from omegaconf import OmegaConf

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    OmegaConf.save(config=config, f=temporary, resolve=True)
    temporary.replace(destination)


def _load_wandb_api_key(config) -> None:
    loggers = config.trainer.logger
    if "wandb" not in ([loggers] if isinstance(loggers, str) else loggers):
        return
    path = os.environ.get("CODING_OPD_WANDB_API_KEY_FILE")
    if path and "WANDB_API_KEY" not in os.environ:
        os.environ["WANDB_API_KEY"] = Path(path).read_text().strip()


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
            configure_coding_rollout(config)
            _load_wandb_api_key(config)
            _save_run_config(config)
            install_pure_distillation_fast_path()
            install_isolated_vllm_compile_cache()
            install_separate_async_standalone_memory_budget()
            install_parallel_startup()
            install_checkpoint_safe_prefetch(config)
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
