#!/usr/bin/env python3
"""Bounded synthetic-code serving benchmark; not an evaluation/quality result.

One process per configuration. Keep target dtype and sampling fixed. Cold and
warm-prefix timings include prefill/scheduling, not just GPU decode time.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mtp", type=int, choices=[0, 1, 3], default=0)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 49152])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--output-tokens", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing results")
    from vllm import LLM, SamplingParams

    kwargs = dict(
        model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len, max_num_seqs=max(args.concurrency),
        max_num_batched_tokens=8192, gpu_memory_utilization=0.8,
        attention_backend="FLASH_ATTN", gdn_prefill_backend="triton",
        enable_prefix_caching=True, mamba_cache_mode="align",
        language_model_only=True, seed=0, disable_log_stats=False,
        generation_config="vllm",
        compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
    )
    if args.mtp:
        kwargs["speculative_config"] = {
            "method": "mtp", "num_speculative_tokens": args.mtp,
            # Draft attention has an independent backend; target FLASH_ATTN
            # alone still lets it select FlashInfer and hit CUDA 12.8 JIT.
            "attention_backend": "FLASH_ATTN",
        }
    result = {
        "arguments": {**vars(args), "output": str(args.output)},
        "engine": kwargs,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "caveat": "Synthetic repeated repository code; hot-prefix batch wall throughput, not DeepSWE or pure decode throughput. ignore_eos forces equal output lengths.",
        "measurements": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, default=str) + "\n")
        temporary.replace(args.output)

    save()
    started = time.perf_counter()
    llm = LLM(**kwargs)
    result["startup_seconds"] = time.perf_counter() - started
    tokenizer = llm.get_tokenizer()
    code = Path(__file__).read_text()
    chunk = tokenizer.encode(code, add_special_tokens=False)
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0,
        max_tokens=args.output_tokens, ignore_eos=True, seed=0,
    )
    for context in args.contexts:
        for concurrency in args.concurrency:
            prompts = []
            for index in range(concurrency):
                prefix = tokenizer.encode(
                    f"Request {context}/{concurrency}/{index}. Review the following Python code.\n",
                    add_special_tokens=False,
                )
                suffix = tokenizer.encode("\nExplain bugs and write an improved implementation:\n", add_special_tokens=False)
                body_length = context - len(prefix) - len(suffix)
                if body_length < 1:
                    raise ValueError("Context too small")
                tokens = prefix + (chunk * (body_length // len(chunk) + 1))[:body_length] + suffix
                prompts.append({"prompt_token_ids": tokens})
            for repeat in range(3):
                start = time.perf_counter()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                elapsed = time.perf_counter() - start
                count = sum(len(output.outputs[0].token_ids) for output in outputs)
                row = dict(context=context, concurrency=concurrency,
                           phase="cold_warmup" if repeat == 0 else "warm_prefix",
                           repeat=repeat, seconds=elapsed, output_tokens=count,
                           aggregate_tokens_per_second=count / elapsed,
                           tokens_per_second_per_gpu=count / elapsed / args.tp)
                result["measurements"].append(row)
                result["metrics_snapshot"] = [dataclasses.asdict(metric) for metric in llm.get_metrics()]
                save()
                print(json.dumps(row), flush=True)
    result["complete"] = True
    save()


if __name__ == "__main__":
    main()
