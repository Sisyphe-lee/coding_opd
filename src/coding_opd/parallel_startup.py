"""Initialize independent Teacher and rollout GPUs alongside the Actor."""

from __future__ import annotations

import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("verl.coding_opd.parallel_startup")
logger.setLevel(logging.INFO)


def _timed(stage, function, *args, **kwargs):
    start = time.monotonic()
    logger.info("STARTUP_BEGIN stage=%s", stage)
    result = function(*args, **kwargs)
    logger.info("STARTUP_END stage=%s elapsed_s=%.3f", stage, time.monotonic() - start)
    return result


def _parallel_initialize(base_setup, standalone_setup):
    # These constructors launch Ray workers on disjoint resource pools. veRL's
    # auto_await creates the standalone thread's own asyncio event loop.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="opd-startup") as executor:
        standalone = executor.submit(_timed, "standalone_rollout", standalone_setup)
        _timed("actor_hybrid_restore_and_teacher_join", base_setup)
        return standalone.result()  # Propagate failures before connecting/syncing workers.


def _initialize_with_teacher(trainer, base_setup, teacher_factory, teacher_role, convert):
    if not trainer.use_teacher_policy:
        return base_setup()
    original_init = trainer._init_resource_pool_mgr
    previous_override = trainer.__dict__.get("_init_resource_pool_mgr")
    teacher = None
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="opd-teacher-startup") as executor:
        def init_pools():
            original_init()
            pools = trainer.resource_pool_manager
            create = pools.create_resource_pool

            def create_and_start_teacher():
                nonlocal teacher
                pools.create_resource_pool = create
                create()
                # Materialize the original pools before either thread uses them;
                # never allocate a second Teacher pool or extra GPUs.
                for pool in pools.resource_pool_dict.values():
                    pool.get_placement_groups(device_name=trainer.config.trainer.device)
                config = copy.deepcopy(trainer.config)
                teacher = executor.submit(
                    _timed, "teacher", teacher_factory,
                    config=config, resource_pool=pools.get_resource_pool(teacher_role),
                )
                # Skip only the later synchronous Teacher constructor. The
                # distillation config stays enabled for all Actor workers.
                trainer.use_teacher_policy = False

            pools.create_resource_pool = create_and_start_teacher

        trainer._init_resource_pool_mgr = init_pools
        try:
            result = base_setup()
        finally:
            trainer.use_teacher_policy = True
            if previous_override is None:
                trainer.__dict__.pop("_init_resource_pool_mgr", None)
            else:
                trainer._init_resource_pool_mgr = previous_override
        if teacher is None:
            raise RuntimeError("Base setup did not initialize the Teacher resource pool")
        trainer.teacher_model_manager = teacher.result()
        trainer.distillation_config = convert(trainer.config.distillation)
        return result


def install_parallel_startup() -> None:
    """Overlap independent service creation; keep resume and initial sync intact.

    Install after the standalone memory-budget adapter. This small replacement
    mirrors the pinned separate-async _setup, changing only its first two stages.
    """
    from verl.trainer.ppo.v1 import trainer_base as base
    from verl.trainer.ppo.v1 import trainer_separate_async as module

    original = module.PPOTrainerSeparateAsync._setup
    if getattr(original, "_coding_opd_parallel_startup", False):
        return

    def setup(self):
        rollout = self.config.actor_rollout_ref.rollout
        # Limit the early replica-rank calculation to our validated single-node,
        # TP=1 vLLM topology. Other configurations keep upstream startup.
        if (
            rollout.name != "vllm"
            or rollout.disaggregation.enabled
            or self.config.trainer.nnodes != 1
            or rollout.nnodes != 1
            or rollout.tensor_model_parallel_size != 1
            or rollout.data_parallel_size != 1
            or rollout.pipeline_model_parallel_size != 1
        ):
            return original(self)
        start = time.monotonic()
        start_rank = self.config.trainer.n_gpus_per_node
        config = copy.deepcopy(self.config)
        self.standalone_server_manager = _parallel_initialize(
            lambda: _initialize_with_teacher(
                self, lambda: module.PPOTrainer._setup(self),
                base.MultiTeacherModelManager, base.Role.TeacherModel, base.omega_conf_to_dataclass,
            ),
            lambda: module.LLMServerManager.create(config=config, start_rank=start_rank),
        )
        if len(self.llm_server_manager.rollout_replicas) != start_rank:
            raise RuntimeError("Parallel startup replica ranks differ from hybrid replicas")

        # Unchanged upstream post-setup wiring. Initial weight synchronization
        # still happens in on_init_end(), after checkpoint restoration.
        if rollout.prometheus.enable:
            addresses = self.llm_server_manager.server_addresses + self.standalone_server_manager.server_addresses
            module.update_prometheus_config(rollout.prometheus, addresses, rollout.name)
        checkpoint_config = module.omega_conf_to_dataclass(rollout.checkpoint_engine)
        self.standalone_checkpoint_manager = module.CheckpointEngineManager(
            config=checkpoint_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.standalone_server_manager.get_replicas(),
        )
        self.current_mode = module.HybridEngineMode.ROLLOUT
        self.add_replicas_to_balancer()
        logger.info("STARTUP_END stage=trainer_setup elapsed_s=%.3f", time.monotonic() - start)

    setup._coding_opd_parallel_startup = True
    module.PPOTrainerSeparateAsync._setup = setup
    logger.info("Enabled independent Teacher and standalone rollout startup alongside Actor setup")
