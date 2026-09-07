"""Project-owned fast paths layered on top of the pinned veRL checkout."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@contextmanager
def _safe_fsdp2_deferred_gradient_sync(module):
    """Defer sync only for FSDP2 modules executed by this micro-batch.

    PyTorch 2.11's deferred-sync path assumes every FSDP parameter group has
    materialized an unsharded parameter. Hybrid or otherwise conditional
    models can leave some wrapped modules unused, causing post-backward to
    access a missing ``_unsharded_param``. Forward hooks let us use only the
    public FSDP2 synchronization API while leaving unused groups untouched.
    """

    from torch.distributed.fsdp import FSDPModule

    touched: dict[int, FSDPModule] = {}
    handles = []

    def defer_executed_module(fsdp_module, _args, _output):
        fsdp_module.set_requires_gradient_sync(False, recurse=False)
        touched[id(fsdp_module)] = fsdp_module

    for candidate in module.modules():
        if isinstance(candidate, FSDPModule):
            handles.append(candidate.register_forward_hook(defer_executed_module))

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for fsdp_module in touched.values():
            fsdp_module.set_requires_gradient_sync(True, recurse=False)


def install_safe_fsdp2_deferred_gradient_sync() -> None:
    """Make veRL's FSDP2 gradient accumulation safe on PyTorch 2.11.

    The optimization remains controlled by veRL's
    ``use_no_sync_for_gradient_accumulation`` setting. FSDP1 and synchronized
    FSDP2 execution continue through veRL's original implementation.
    """

    from verl.workers.engine.fsdp import transformer_impl
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    current = FSDPEngine._gradient_sync_context
    if getattr(current, "_coding_opd_safe_fsdp2_deferred_sync", False):
        return

    original = current

    @contextmanager
    def _gradient_sync_context(self, *, is_last_micro_batch: bool):
        defer_sync = getattr(
            self.engine_config,
            "use_no_sync_for_gradient_accumulation",
            True,
        )
        if (
            is_last_micro_batch
            or not defer_sync
            or transformer_impl.fsdp_version(self.module) != 2
        ):
            with original(self, is_last_micro_batch=is_last_micro_batch):
                yield
            return

        with _safe_fsdp2_deferred_gradient_sync(self.module):
            yield

    _gradient_sync_context._coding_opd_safe_fsdp2_deferred_sync = True
    _gradient_sync_context._coding_opd_original = original
    FSDPEngine._gradient_sync_context = _gradient_sync_context
    logger.info("Installed PyTorch 2.11-safe FSDP2 deferred-gradient synchronization")


def _get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def is_pure_direct_distillation(config: Any) -> bool:
    """Return whether PPO old-policy data cannot affect the configured loss."""

    distillation = _get(config, "distillation")
    loss = _get(distillation, "distillation_loss")
    return bool(
        _get(distillation, "enabled", False)
        and not _get(loss, "use_task_rewards", True)
        and not _get(loss, "use_policy_gradient", True)
    )


def install_pure_distillation_fast_path() -> None:
    """Skip old-policy logprob plumbing for direct-backprop distillation.

    K3 still computes the current student's exact sampled-token logprob during
    the actor forward pass.  What is skipped here is the separate rollout
    policy logprob/old-policy anchor used only by PPO-style losses.
    """

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    current = PPOTrainer._compute_old_log_prob
    if getattr(current, "_coding_opd_pure_distillation_fast_path", False):
        return

    original = current

    def _compute_old_log_prob(self, batch, metrics):
        if is_pure_direct_distillation(self.config):
            metrics["perf/skipped_old_log_prob"] = 1.0
            return batch
        return original(self, batch, metrics)

    _compute_old_log_prob._coding_opd_pure_distillation_fast_path = True
    _compute_old_log_prob._coding_opd_original = original
    PPOTrainer._compute_old_log_prob = _compute_old_log_prob
    logger.info("Installed pure direct-distillation old-logprob fast path")


def _standalone_rollout_config(config: Any) -> Any:
    """Copy a trainer config and apply its standalone-only vLLM budget."""

    rollout = _get(_get(config, "actor_rollout_ref"), "rollout")
    standalone = _get(rollout, "standalone_gpu_memory_utilization")
    if standalone is None:
        return config

    standalone_config = copy.deepcopy(config)
    standalone_config.actor_rollout_ref.rollout.gpu_memory_utilization = standalone
    return standalone_config


def install_separate_async_standalone_memory_budget() -> None:
    """Honor the standalone rollout memory budget in veRL V1 separate-async.

    The pinned veRL config exposes ``standalone_gpu_memory_utilization``, but
    V1's separate-async trainer passes the shared hybrid config unchanged when
    it constructs its dedicated rollout manager.  Patch only that module's
    manager reference so hybrid replicas retain their original budget.
    """

    import verl.trainer.ppo.v1.trainer_separate_async as trainer_module

    current = trainer_module.LLMServerManager
    if getattr(current, "_coding_opd_standalone_memory_budget", False):
        return

    class CodingOPDStandaloneLLMServerManager(current):
        _coding_opd_standalone_memory_budget = True

        def __init__(self, config, worker_group=None, *args, **kwargs):
            if worker_group is None:
                config = _standalone_rollout_config(config)
            super().__init__(config, worker_group, *args, **kwargs)

    trainer_module.LLMServerManager = CodingOPDStandaloneLLMServerManager
    logger.info("Installed V1 separate-async standalone rollout memory budget")


def install_persistent_nccl_sender_buffers() -> None:
    """Reuse NCCL transport buffers without flushing the actor allocator.

    The pinned NCCL checkpoint engine recreates two full-size buffers and calls
    ``torch.cuda.empty_cache()`` on every worker after every weight update.  A
    persistent process group does not need that cleanup on actor workers, and
    flushing their FSDP/compile cache can take tens of seconds.  Rollout
    consumers retain the original cleanup because vLLM needs the released
    transport memory when its KV cache is restored.
    """

    from verl.checkpoint_engine.nccl_checkpoint_engine import (
        MasterMetadata,
        NCCLCheckpointEngine,
        WorkerMetadata,
    )

    current_prepare = NCCLCheckpointEngine.prepare
    if getattr(current_prepare, "_coding_opd_persistent_sender_buffers", False):
        return

    original_prepare = current_prepare
    original_finalize = NCCLCheckpointEngine.finalize

    def prepare(self):
        if getattr(self, "send_buf", None) is None or getattr(self, "recv_buf", None) is None:
            return original_prepare(self)

        master = (
            MasterMetadata(zmq_ip=self.ip, zmq_port=self.listen_port, multi_sender=self.multi_sender)
            if self.is_master
            else None
        )
        return WorkerMetadata(node_id=self.get_node_id(), master=master)

    def finalize(self):
        rank = getattr(self, "rank", None)
        num_senders = getattr(self, "num_senders", None)

        # Preserve upstream's complete cleanup for rollout consumers, rebuilt
        # groups, and any unexpected call before the worker role is known.
        if (
            self.rebuild_group
            or rank is None
            or num_senders is None
            or rank >= num_senders
        ):
            return original_finalize(self)

        # rank -1 is an actor omitted from a single-sender group.  Its buffers
        # are unused, but a global cache flush would still evict unrelated actor
        # allocations.  Active senders keep both buffers for the next update.
        if rank < 0:
            self.send_buf = None
            self.recv_buf = None
        return None

    prepare._coding_opd_persistent_sender_buffers = True
    prepare._coding_opd_original = original_prepare
    finalize._coding_opd_persistent_sender_buffers = True
    finalize._coding_opd_original = original_finalize
    NCCLCheckpointEngine.prepare = prepare
    NCCLCheckpointEngine.finalize = finalize
    logger.info("Enabled persistent NCCL buffers on actor sender workers")


def _replica_compile_cache_dir(server: Any, root: str) -> str:
    model_path = str(_get(server.model_config, "path", "unknown-model"))
    model_key = hashlib.sha256(model_path.encode()).hexdigest()[:16]
    replica_rank = int(getattr(server, "replica_rank", 0))
    node_rank = int(getattr(server, "node_rank", 0))
    return str(Path(root) / model_key / f"replica-{replica_rank}" / f"node-{node_rank}")


def install_isolated_vllm_compile_cache() -> None:
    """Give every vLLM replica a stable, non-overlapping compile cache.

    veRL disables the vLLM compile cache globally because concurrent replicas
    can corrupt a shared rank-0 cache.  Stable per-model/per-replica paths avoid
    that collision and allow later launches to reuse compiled artifacts.
    """

    root = os.getenv("CODING_OPD_VLLM_COMPILE_CACHE_ROOT", "")
    if not root or os.getenv("VLLM_DISABLE_COMPILE_CACHE", "1") != "0":
        return

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    current = vLLMHttpServer._preprocess_engine_kwargs
    if getattr(current, "_coding_opd_isolated_compile_cache", False):
        return

    original = current

    def _preprocess_engine_kwargs(self, engine_kwargs):
        original(self, engine_kwargs)
        compilation_config = engine_kwargs.get("compilation_config") or {}
        if isinstance(compilation_config, str):
            compilation_config = json.loads(compilation_config)
        else:
            compilation_config = dict(compilation_config)
        cache_dir = _replica_compile_cache_dir(self, root)
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        compilation_config["cache_dir"] = cache_dir
        engine_kwargs["compilation_config"] = compilation_config

    _preprocess_engine_kwargs._coding_opd_isolated_compile_cache = True
    _preprocess_engine_kwargs._coding_opd_original = original
    vLLMHttpServer._preprocess_engine_kwargs = _preprocess_engine_kwargs
    logger.info("Enabled isolated vLLM compile caches under %s", root)
