"""One-batch rollout lookahead that drains before veRL saves a checkpoint."""

from __future__ import annotations

import logging

from coding_opd.profiling import profile_span

logger = logging.getLogger(__name__)


def install_checkpoint_safe_prefetch(config) -> None:
    """Overlap standalone rollout with Actor updates, retaining veRL's sampler.

    Submit the next batch after the first mini-batch is sampled, then skip the
    normal submission at the next step. At save/final steps, consume the queued
    batch without another prefetch. Thus the dataloader checkpoint cannot run
    ahead of training even when TransferQueue has no checkpoint API.
    """
    if config.trainer.v1.trainer_mode != "separate_async" or config.skip.rollout_tq.enable:
        return
    async_config = config.trainer.v1.separate_async

    from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

    cls = PPOTrainerSeparateAsync
    if async_config.get("first_checkpoint_step", -1) > 0 and not getattr(
        cls.on_step_end, "_coding_opd_first_checkpoint", False
    ):
        original_step_end = cls.on_step_end

        def on_step_end(self):
            first_step = self.config.trainer.v1.separate_async.first_checkpoint_step
            save_freq = self.config.trainer.save_freq
            # Regular/final saves already happened in fit(). Save the early
            # checkpoint at the same boundary, before weight synchronization.
            if (
                self.global_steps == first_step
                and self.global_steps < self.total_training_steps
                and save_freq > 0
                and self.global_steps % save_freq != 0
            ):
                self._save_checkpoint()
            return original_step_end(self)

        on_step_end._coding_opd_first_checkpoint = True
        cls.on_step_end = on_step_end

    if not async_config.get("checkpoint_safe_prefetch", False):
        return
    if async_config.num_warmup_batches != 0:
        raise ValueError("checkpoint_safe_prefetch requires num_warmup_batches=0")
    if config.trainer.v1.sampler.max_off_policy_threshold < 2:
        raise ValueError("checkpoint_safe_prefetch requires max_off_policy_threshold >= 2")

    import transfer_queue as tq
    if getattr(cls._add_batch_to_generate, "_coding_opd_prefetch", False):
        return
    original_add = cls._add_batch_to_generate
    original_sample_end = cls.on_sample_end
    original_save = cls._save_checkpoint

    def _add_batch_to_generate(self):
        if getattr(self, "_coding_opd_batch_prefetched", False):
            self._coding_opd_batch_prefetched = False
            return
        return original_add(self)

    def on_sample_end(self):
        original_sample_end(self)
        save_freq = self.config.trainer.save_freq
        first_step = self.config.trainer.v1.separate_async.get("first_checkpoint_step", -1)
        boundary = self.global_steps >= self.total_training_steps or (
            save_freq > 0 and (self.global_steps % save_freq == 0 or self.global_steps == first_step)
        )
        if self.local_trigger_step == 0 and not boundary:
            with profile_span("rollout_prefetch", step=self.global_steps):
                original_add(self)
            self._coding_opd_batch_prefetched = True

    def _save_checkpoint(self):
        # Consumed trajectory payloads remain until the end of this step; only
        # prompt markers represent fetched rows not yet included in the model.
        items = (tq.kv_list("train") or {}).get("train", {})
        outstanding = sum(bool(tags.get("is_prompt", False)) for tags in items.values())
        if outstanding or getattr(self, "_coding_opd_batch_prefetched", False):
            raise RuntimeError(f"Refusing checkpoint with {outstanding} unconsumed prefetched prompts")
        logger.info("Prefetch drained before checkpoint at step %s", self.global_steps)
        return original_save(self)

    _add_batch_to_generate._coding_opd_prefetch = True
    cls._add_batch_to_generate = _add_batch_to_generate
    cls.on_sample_end = on_sample_end
    cls._save_checkpoint = _save_checkpoint
    logger.info("Enabled one-batch rollout prefetch with checkpoint-boundary draining")
