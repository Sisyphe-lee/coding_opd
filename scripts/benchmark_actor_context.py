"""Bounded four-GPU Actor benchmark; synthetic tokens, no rollout or saved weights.

Run with torchrun --nproc-per-node=4. Match the formal Actor kernels/FSDP flags.
Lengths are actual single sequences, not sums of short packed trajectories.
"""
import argparse
import json
import os
import time

import torch
import torch.distributed as dist

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead


def synthetic_k3_loss(model_output, data, dp_group):
    logp = model_output["log_probs"].values()
    # Synthetic detached teacher scores avoid a Teacher server and K3 saturation
    # on random tokens. Exercise nonzero sampled-token K3 gradients, not quality.
    teacher_logp = logp.detach() + 0.1
    delta = teacher_logp - logp
    mask = data["loss_mask"].values()
    scale = tu.get_non_tensor_data(data, "dp_size", 1) / tu.get_non_tensor_data(data, "batch_num_tokens", 1)
    return ((delta.exp() - delta - 1) * mask).sum() * scale, {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--local-batch", type=int, default=8)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    model = HFModelConfig(
        path=args.model, external_lib="coding_opd.qwen35_kernel_plugin",
        enable_gradient_checkpointing=args.checkpointing,
        use_remove_padding=True, use_fused_kernels=True,
        fused_kernel_options={"impl_backend": "triton"},
    )
    engine = FSDPEngineWithLMHead(
        model_config=model,
        engine_config=FSDPEngineConfig(
            strategy="fsdp2", reshard_after_forward=False,
            use_no_sync_for_gradient_accumulation=False,
            use_torch_compile=True, use_remove_padding=True, use_fused_kernels=True,
            pad_to_length=True, pad_to_length_bucket=1024,
        ),
        optimizer_config=FSDPOptimizerConfig(lr=1e-6, total_training_steps=args.steps),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    # Generate the same 32K fixture in each variant, then take the requested prefix.
    generator = torch.Generator().manual_seed(42 + rank)
    ids = [torch.randint(1000, 20000, (32768,), generator=generator)[:args.tokens]
           for _ in range(args.local_batch)]
    masks = [torch.ones(args.tokens) for _ in ids]
    for mask in masks:
        mask[:1024] = 0
        mask[-1] = 0
    nested = lambda values: torch.nested.as_nested_tensor(values, layout=torch.jagged)
    data = tu.get_tensordict({
        "input_ids": nested(ids),
        "position_ids": nested([torch.arange(args.tokens) for _ in ids]),
        "loss_mask": nested(masks),
    }, {
        "temperature": 1.0, "use_remove_padding": True, "use_fused_kernels": True,
        "use_dynamic_bsz": True, "max_token_len_per_gpu": args.tokens,
        "calculate_entropy": False,
    })
    if rank == 0:
        print("BENCH_READY " + json.dumps(vars(args)), flush=True)
    for step in range(args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with engine.train_mode():
            output = engine.train_batch(data, synthetic_k3_loss)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        stats = torch.tensor([
            elapsed, torch.cuda.max_memory_allocated() / 2**30,
            torch.cuda.max_memory_reserved() / 2**30,
        ], device="cuda", dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
        if rank == 0:
            seconds, allocated, reserved = stats.tolist()
            print("BENCH_RESULT " + json.dumps({
                "step": step, "warmup": step == 0, "tokens_per_sequence": args.tokens,
                "checkpointing": args.checkpointing, "global_batch": args.local_batch * dist.get_world_size(),
                "seconds": seconds, "tokens_per_second": args.tokens * args.local_batch * dist.get_world_size() / seconds,
                "peak_allocated_gib": allocated, "peak_reserved_gib": reserved,
                "grad_norm": float(output["metrics"]["grad_norm"]),
            }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
