import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from coding_opd.async_prefetch import install_checkpoint_safe_prefetch


@pytest.fixture
def harness(monkeypatch, tmp_path):
    import transfer_queue as tq
    from verl.trainer.ppo.v1 import trainer_separate_async as module
    from verl.trainer.ppo.v1.replay_buffer import ReplayBufferAsync

    store = {}
    evicted = set()
    slow_rows = set()
    failed_rows = set()
    active = []

    def kv_list(partition_id=None):
        # Delayed completion lets newer batches overtake an older running row.
        for uid, tags in store.items():
            if tags.get("is_prompt") and int(uid) in slow_rows and active[0].global_steps >= 3:
                tags["status"] = "finished"
        return {"train": copy.deepcopy(store)}

    def kv_clear(*, partition_id, keys):
        for key in keys:
            tags = store.pop(key, {})
            if tags.get("is_prompt") and (
                tags["status"] == "failure" or active[0].global_steps - tags["global_steps"] + 1 > 2
            ):
                evicted.add(int(key))

    monkeypatch.setattr(tq, "kv_list", kv_list)
    monkeypatch.setattr(tq, "kv_clear", kv_clear)

    class Trainer(module.PPOTrainerSeparateAsync):
        def __init__(self, save_freq=2, total=5):
            self.config = OmegaConf.create({
                "data": {"train_batch_size": 32},
                "skip": {"rollout_tq": {"enable": False}},
                "trainer": {
                    "save_freq": save_freq, "default_local_dir": str(tmp_path), "default_hdfs_dir": None,
                    "v1": {"trainer_mode": "separate_async", "separate_async": {
                        "num_warmup_batches": 0, "checkpoint_safe_prefetch": True,
                    }, "sampler": {"max_off_policy_threshold": 2}},
                },
                "actor_rollout_ref": {"actor": {"checkpoint": {}}},
            })
            self.train_dataloader = StatefulDataLoader(range(4096), batch_size=1)
            self.train_dataloader_it = None
            self.global_steps = 1
            self.total_training_steps = total
            self.parameter_sync_step = 2
            self.trainer_mode = "separate_async"
            self.use_critic = False
            self.trained = []
            self.actor_overlap = []
            self.dispatched = []
            self.saved = []
            self._step_sample_wait_seconds = 0.0
            self.replay_buffer = ReplayBufferAsync(
                trainer_mode="separate_async", trainer_config=self.config.trainer,
                max_off_policy_threshold=2, max_off_policy_strategy="drop", sampler_kwargs={},
                refill_fn=self._add_prompts_to_generate, poll_interval=0,
            )
            self.actor_rollout_wg = SimpleNamespace(save_checkpoint=lambda *args, **kwargs: None)
            self.checkpoint_callback = SimpleNamespace(on_save=lambda **kwargs: self.saved.append(self.global_steps))
            active[:] = [self]

        def _add_prompts_to_generate(self, count):
            if self.train_dataloader_it is None:
                self.train_dataloader_it = iter(self.train_dataloader)
            for _ in range(count):
                row = int(next(self.train_dataloader_it).item())
                self.dispatched.append(row)
                status = "failure" if row in failed_rows else "running" if row in slow_rows else "finished"
                store[str(row)] = {"is_prompt": True, "global_steps": self.global_steps, "status": status}
                store[f"{row}_0_0"] = {"row": row}
            return count

        def _add_batch_to_generate(self):
            self._add_prompts_to_generate(32)

        def _wait_for_sampleable_and_switch(self):
            return {}  # GPU ownership changes are outside this CPU scheduling test.

        def on_step_end(self):
            pass  # The early-save hook precedes the GPU weight synchronization.

        def _step_once(self, metrics, timing_raw, sample_batch_size):
            batch, _ = self.replay_buffer.sample(self.global_steps, "train", sample_batch_size)
            self._sample_start = 0
            self.on_sample_end()
            # Represents Actor execution after the real sampler and sample-end hook.
            self.actor_overlap.append(sum(bool(t.get("is_prompt")) for t in store.values()))
            self.trained.extend(int(key.split("_")[0]) for key in batch.keys)
            return batch

        def run_step(self):
            batch = self.step({}, {})
            boundary = self.global_steps == self.total_training_steps or (
                self.config.trainer.save_freq > 0 and self.global_steps % self.config.trainer.save_freq == 0
            )
            if self.config.trainer.save_freq > 0 and boundary:
                self._save_checkpoint()  # Actual veRL dataloader save path.
            self.on_step_end()
            kv_clear(partition_id="train", keys=batch.keys)
            self.global_steps += 1

    monkeypatch.setattr(module, "PPOTrainerSeparateAsync", Trainer)
    trainer = Trainer()
    install_checkpoint_safe_prefetch(trainer.config)
    return SimpleNamespace(
        trainer=trainer, cls=Trainer, store=store, evicted=evicted, slow=slow_rows, failed=failed_rows,
    )


@pytest.mark.parametrize("save_freq", [1, 2, 3, 64, -1])
def test_prefetch_overlaps_actor_and_drains_at_save_and_final_steps(harness, save_freq):
    trainer = harness.trainer
    trainer.config.trainer.save_freq = save_freq
    for step in range(1, 6):
        trainer.run_step()
        boundary = step == 5 or (save_freq > 0 and step % save_freq == 0)
        assert len(harness.store) == (0 if boundary else 64)  # Prompt + trajectory per queued row.
        assert trainer.actor_overlap[-1] == (0 if boundary else 32)
    assert sorted(trainer.trained) == list(range(160))
    assert trainer.dispatched == list(range(160))


@pytest.mark.parametrize("with_failures", [False, True])
def test_real_sampler_and_dataloader_resume_do_not_skip_prefetched_rows(harness, tmp_path, with_failures):
    before = harness.trainer
    before.config.trainer.save_freq = 3
    if with_failures:
        harness.slow.add(0)
        harness.failed.add(7)
    for _ in range(3):
        before.run_step()
    assert not harness.store
    saved = torch.load(tmp_path / "global_step_3/data.pt", weights_only=False)
    after = harness.cls(save_freq=3)
    after.train_dataloader.load_state_dict(saved)
    after.global_steps = 4
    after.run_step()
    after.run_step()
    trained = before.trained + after.trained
    assert len(trained) == len(set(trained)) == 160
    assert sorted(trained + list(harness.evicted)) == list(range(160 + len(harness.evicted)))
    assert harness.evicted == ({0, 7} if with_failures else set())
    assert not harness.store


@pytest.mark.parametrize("status", ["pending", "running", "finished", "failure"])
def test_checkpoint_refuses_any_unconsumed_prompt(harness, tmp_path, status):
    harness.store["999"] = {"is_prompt": True, "global_steps": 1, "status": status}
    with pytest.raises(RuntimeError, match="unconsumed prefetched prompts"):
        harness.trainer._save_checkpoint()
    assert not (tmp_path / "global_step_1").exists()


def test_installation_is_idempotent_and_rejects_unsafe_settings(harness):
    cls, config = harness.cls, harness.trainer.config
    original = cls.on_sample_end
    install_checkpoint_safe_prefetch(config)
    assert cls.on_sample_end is original
    config.trainer.v1.separate_async.num_warmup_batches = 1
    with pytest.raises(ValueError, match="num_warmup_batches=0"):
        install_checkpoint_safe_prefetch(config)
    config.trainer.v1.separate_async.num_warmup_batches = 0
    config.trainer.v1.sampler.max_off_policy_threshold = 1
    with pytest.raises(ValueError, match="max_off_policy_threshold"):
        install_checkpoint_safe_prefetch(config)


@pytest.mark.parametrize("first_step,total,expected", [(16, 66, [16, 64, 66]), (16, 16, [16]), (64, 65, [64, 65])])
def test_early_checkpoint_drains_once_without_changing_regular_cadence(harness, first_step, total, expected):
    trainer = harness.trainer
    trainer.total_training_steps = total
    trainer.config.trainer.save_freq = 64
    trainer.config.trainer.v1.separate_async.first_checkpoint_step = first_step
    install_checkpoint_safe_prefetch(trainer.config)
    for step in range(1, total + 1):
        trainer.run_step()
        if step in expected:
            assert not harness.store
    assert trainer.saved == expected
    assert sorted(trainer.trained) == list(range(32 * total))
