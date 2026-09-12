import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

from omegaconf import OmegaConf

from coding_opd.train_entrypoint import _load_wandb_api_key, _project_task_runner, _save_run_config
from coding_opd.verl_optimizations import (
    _replica_compile_cache_dir,
    _safe_fsdp2_deferred_gradient_sync,
    _standalone_rollout_config,
    install_persistent_nccl_sender_buffers,
    is_pure_direct_distillation,
)


def _config(*, enabled=True, use_task_rewards=False, use_policy_gradient=False):
    return SimpleNamespace(
        distillation=SimpleNamespace(
            enabled=enabled,
            distillation_loss=SimpleNamespace(
                use_task_rewards=use_task_rewards,
                use_policy_gradient=use_policy_gradient,
            ),
        )
    )


def test_identifies_pure_direct_distillation() -> None:
    assert is_pure_direct_distillation(_config())
    assert not is_pure_direct_distillation(_config(enabled=False))
    assert not is_pure_direct_distillation(_config(use_task_rewards=True))
    assert not is_pure_direct_distillation(_config(use_policy_gradient=True))


def test_project_task_runner_preserves_verl_runner_lifecycle() -> None:
    import verl.trainer.main_ppo as main_ppo

    runner = _project_task_runner(main_ppo)
    actor_class = runner.__ray_metadata__.modified_class
    assert any(base.__name__ == "TaskRunnerV1" for base in actor_class.__mro__)


def test_save_run_config_writes_resolved_yaml(tmp_path, monkeypatch) -> None:
    destination = tmp_path / "config.yaml"
    monkeypatch.setenv("CODING_OPD_RUN_CONFIG_PATH", str(destination))
    config = OmegaConf.create({"model": {"path": "/models/student"}, "copy": "${model.path}"})

    _save_run_config(config)

    saved = OmegaConf.load(destination)
    assert saved["copy"] == "/models/student"


def test_load_wandb_api_key_from_shared_file(tmp_path, monkeypatch) -> None:
    key_file = tmp_path / "wandb_api_key"
    key_file.write_text("secret-key\n")
    monkeypatch.setenv("CODING_OPD_WANDB_API_KEY_FILE", str(key_file))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    _load_wandb_api_key(OmegaConf.create({"trainer": {"logger": ["console", "wandb"]}}))

    assert os.environ["WANDB_API_KEY"] == "secret-key"


def test_vllm_compile_cache_is_stable_and_replica_isolated(tmp_path) -> None:
    server_a = SimpleNamespace(
        model_config=SimpleNamespace(path="/models/student"), replica_rank=0, node_rank=0
    )
    server_b = SimpleNamespace(
        model_config=SimpleNamespace(path="/models/student"), replica_rank=1, node_rank=0
    )
    first = _replica_compile_cache_dir(server_a, str(tmp_path))
    assert first == _replica_compile_cache_dir(server_a, str(tmp_path))
    assert first != _replica_compile_cache_dir(server_b, str(tmp_path))


def test_standalone_rollout_config_uses_independent_memory_budget() -> None:
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "gpu_memory_utilization": 0.8,
                    "standalone_gpu_memory_utilization": 0.6,
                }
            }
        }
    )

    standalone = _standalone_rollout_config(config)

    assert standalone is not config
    assert standalone.actor_rollout_ref.rollout.gpu_memory_utilization == 0.6
    assert config.actor_rollout_ref.rollout.gpu_memory_utilization == 0.8


def test_persistent_nccl_sender_buffers_preserve_consumer_cleanup(monkeypatch) -> None:
    @dataclass
    class WorkerMetadata:
        node_id: str
        master: object = None

    @dataclass
    class MasterMetadata:
        zmq_ip: str
        zmq_port: int
        multi_sender: bool

    class FakeNCCLCheckpointEngine:
        prepare_calls = 0
        finalize_calls = 0

        def prepare(self):
            type(self).prepare_calls += 1
            self.send_buf = object()
            self.recv_buf = object()
            return WorkerMetadata(node_id="original")

        def finalize(self):
            type(self).finalize_calls += 1
            self.send_buf = None
            self.recv_buf = None

    fake_module = ModuleType("verl.checkpoint_engine.nccl_checkpoint_engine")
    fake_module.NCCLCheckpointEngine = FakeNCCLCheckpointEngine
    fake_module.MasterMetadata = MasterMetadata
    fake_module.WorkerMetadata = WorkerMetadata
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)

    install_persistent_nccl_sender_buffers()
    patched_prepare = FakeNCCLCheckpointEngine.prepare
    patched_finalize = FakeNCCLCheckpointEngine.finalize
    install_persistent_nccl_sender_buffers()
    assert FakeNCCLCheckpointEngine.prepare is patched_prepare
    assert FakeNCCLCheckpointEngine.finalize is patched_finalize

    sender = SimpleNamespace(
        rank=0,
        num_senders=4,
        rebuild_group=False,
        send_buf=object(),
        recv_buf=object(),
        is_master=False,
        get_node_id=lambda: "actor-node",
    )
    send_buf = sender.send_buf
    recv_buf = sender.recv_buf
    patched_finalize(sender)
    metadata = patched_prepare(sender)
    assert sender.send_buf is send_buf
    assert sender.recv_buf is recv_buf
    assert metadata.node_id == "actor-node"

    consumer = object.__new__(FakeNCCLCheckpointEngine)
    consumer.rank = 4
    consumer.num_senders = 4
    consumer.rebuild_group = False
    consumer.send_buf = object()
    consumer.recv_buf = object()
    patched_finalize(consumer)
    assert consumer.send_buf is None
    assert consumer.recv_buf is None

    excluded_actor = SimpleNamespace(
        rank=-1,
        num_senders=1,
        rebuild_group=False,
        send_buf=object(),
        recv_buf=object(),
    )
    patched_finalize(excluded_actor)
    assert excluded_actor.send_buf is None
    assert excluded_actor.recv_buf is None

    assert FakeNCCLCheckpointEngine.prepare_calls == 0
    assert FakeNCCLCheckpointEngine.finalize_calls == 1


def test_fsdp2_deferred_sync_ignores_unused_wrapped_module(tmp_path) -> None:
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    class ConditionalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.used = torch.nn.Linear(4, 4)
            self.unused = torch.nn.Linear(4, 4)

        def forward(self, inputs):
            return self.used(inputs)

    def build_model(mesh):
        torch.manual_seed(42)
        model = ConditionalModel()
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        fully_shard(model.used, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        fully_shard(model.unused, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=False)
        return model

    def optimizer_step(model, inputs, *, defer_first_micro_batch):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        optimizer.zero_grad()
        for index in range(2):
            sync_context = (
                _safe_fsdp2_deferred_gradient_sync(model)
                if defer_first_micro_batch and index == 0
                else nullcontext()
            )
            with sync_context:
                model(inputs).sum().backward()
        optimizer.step()
        return [parameter.full_tensor().detach().clone() for parameter in model.parameters()]

    rendezvous = tmp_path / "fsdp2-rendezvous"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        mesh = init_device_mesh("cpu", (1,))
        inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
        baseline = optimizer_step(build_model(mesh), inputs, defer_first_micro_batch=False)
        deferred = optimizer_step(build_model(mesh), inputs, defer_first_micro_batch=True)
        for baseline_parameter, deferred_parameter in zip(baseline, deferred, strict=True):
            torch.testing.assert_close(baseline_parameter, deferred_parameter, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
