from threading import Event
from types import SimpleNamespace

import pytest

from coding_opd.parallel_startup import _parallel_initialize
from coding_opd.parallel_startup import install_parallel_startup
from coding_opd.parallel_startup import _initialize_with_teacher


def test_startup_overlaps_and_joins_before_return():
    standalone_started, base_done = Event(), Event()
    manager = object()

    def standalone():
        standalone_started.set()
        assert base_done.wait(5), "base setup must overlap standalone startup"
        return manager

    def base():
        assert standalone_started.wait(5)
        base_done.set()

    assert _parallel_initialize(base, standalone) is manager


def test_standalone_failure_prevents_successful_setup():
    def fail():
        raise RuntimeError("server failed")

    with pytest.raises(RuntimeError, match="server failed"):
        _parallel_initialize(lambda: None, fail)


def test_base_failure_is_propagated_and_background_thread_is_joined():
    started, done = Event(), Event()

    def standalone():
        started.set()
        done.set()

    def base():
        assert started.wait(5)
        raise RuntimeError("restore failed")

    with pytest.raises(RuntimeError, match="restore failed"):
        _parallel_initialize(base, standalone)
    assert done.is_set()


def test_setup_preserves_restore_and_post_setup_wiring(monkeypatch):
    from verl.trainer.ppo import v1

    started, restored = Event(), Event()
    standalone = SimpleNamespace(get_replicas=lambda: ["standalone"])
    rollout = SimpleNamespace(
        name="vllm", disaggregation=SimpleNamespace(enabled=False), nnodes=1,
        tensor_model_parallel_size=1, data_parallel_size=1, pipeline_model_parallel_size=1,
        prometheus=SimpleNamespace(enable=False), checkpoint_engine="nccl-config",
    )
    config = SimpleNamespace(
        trainer=SimpleNamespace(nnodes=1, n_gpus_per_node=4),
        actor_rollout_ref=SimpleNamespace(rollout=rollout),
    )

    def base_setup(trainer):
        assert started.wait(5)
        trainer.llm_server_manager = SimpleNamespace(rollout_replicas=[0, 1, 2, 3])
        trainer.actor_rollout_wg = "restored-actor"
        restored.set()

    def create(*, config: object, start_rank):
        assert start_rank == 4
        config.actor_rollout_ref.rollout.name = "copy-only"
        started.set()
        assert restored.wait(5)
        return standalone

    def checkpoint_manager(**kwargs):
        assert restored.is_set()
        assert kwargs == {"config": "nccl-config", "actor_wg": "restored-actor", "replicas": ["standalone"]}
        return "checkpoint-manager"

    class Trainer:
        def _setup(self):
            self.fallback = True

        def add_replicas_to_balancer(self):
            assert self.standalone_server_manager is standalone
            assert self.standalone_checkpoint_manager == "checkpoint-manager"
            self.registered = True

    module = SimpleNamespace(
        PPOTrainerSeparateAsync=Trainer, PPOTrainer=SimpleNamespace(_setup=base_setup),
        LLMServerManager=SimpleNamespace(create=create), CheckpointEngineManager=checkpoint_manager,
        omega_conf_to_dataclass=lambda c: c, HybridEngineMode=SimpleNamespace(ROLLOUT="rollout"),
    )
    monkeypatch.setattr(v1, "trainer_separate_async", module, raising=False)
    install_parallel_startup()
    patched = Trainer._setup
    install_parallel_startup()
    assert Trainer._setup is patched
    trainer = Trainer()
    trainer.use_teacher_policy = False
    trainer.config = config
    trainer._setup()
    assert trainer.registered and trainer.current_mode == "rollout"
    assert rollout.name == "vllm"
    rollout.tensor_model_parallel_size = 2
    trainer._setup()
    assert trainer.fallback


@pytest.mark.parametrize("teacher_fails", [False, True])
def test_teacher_overlaps_actor_and_uses_original_pool(teacher_fails):
    teacher_started, actor_restored = Event(), Event()
    pool = SimpleNamespace(get_placement_groups=lambda **kwargs: None)
    manager = object()

    class Trainer:
        use_teacher_policy = True
        config = SimpleNamespace(
            trainer=SimpleNamespace(device="cuda"), distillation={"enabled": True},
        )

        def _init_resource_pool_mgr(self):
            self.resource_pool_manager = SimpleNamespace(
                create_resource_pool=lambda: None,
                resource_pool_dict={"teacher": pool},
                get_resource_pool=lambda role: pool,
            )

    trainer = Trainer()

    def teacher_factory(*, config, resource_pool):
        assert resource_pool is pool
        assert config is not trainer.config
        teacher_started.set()
        assert actor_restored.wait(5), "Teacher must not block Actor initialization"
        if teacher_fails:
            raise RuntimeError("teacher startup failed")
        return manager

    def base_setup():
        trainer._init_resource_pool_mgr()
        trainer.resource_pool_manager.create_resource_pool()
        assert teacher_started.wait(5)
        assert not trainer.use_teacher_policy  # No duplicate synchronous constructor.
        assert trainer.config.distillation["enabled"]  # Actor still gets distillation.
        trainer.teacher_model_manager = None
        trainer.distillation_config = None
        actor_restored.set()

    def run():
        _initialize_with_teacher(trainer, base_setup, teacher_factory, "teacher", lambda x: x)

    if teacher_fails:
        with pytest.raises(RuntimeError, match="teacher startup failed"):
            run()
    else:
        run()
        assert trainer.teacher_model_manager is manager
        assert trainer.distillation_config["enabled"]
    assert trainer.use_teacher_policy
    assert "_init_resource_pool_mgr" not in trainer.__dict__
