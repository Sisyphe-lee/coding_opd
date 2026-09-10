# Uni-Agent reference evaluation

The reference entry point is `scripts/run_external_eval.sh swebench_verified`,
which now defaults to `configs/uni_agent_react_reference.yaml`. The paused
training configuration remains `configs/coding_react.yaml`; do not restart
training against that older solver and call it aligned with this evaluation.
After validating the reference solver, use that validated recipe for the next
training setup, including a separately measured long-context memory budget.

The reference agent and prompt are copied from the pinned Uni-Agent
`examples/quickstart/inference/task_config_react.yaml`: upstream `react`,
`str_replace_editor`, `stateful_shell`, `submit`, 65,536 context tokens,
100 turns, temperature 0.8 and top-p 0.9. There is no independent per-turn
response cap and no custom repetition stopping in this reproduction run.
A regression test checks the solver and prompt against that pinned source.

The official benchmark table reports Qwen3.5-9B at 53.8 on Verified-500 with
64K/100 turns, but explicitly says Qwen3.5 uses task-specific sampling.
The generic quickstart sampling is therefore a documented approximation,
not evidence of exact reproduction of that row. Verified-64 is an initial
panel; compare the published score only with the full 500-task result.
Sources: https://uni-agent.readthedocs.io/en/latest/benchmark/inference.html
and the pinned upstream quickstart configuration.

Local infrastructure differences are explicit: network-isolated Podman,
8 GiB address-space limit per command, OPENBLAS_NUM_THREADS=1, a 45-minute
agent timeout, and the existing isolated official SWE-bench patch grader
(30-minute grading timeout). The framework session deadline covers both
phases plus ten minutes of overhead. We retain the image's task environment.

Before evaluation, provide tmux offline:

```bash
.venv/bin/python scripts/prepare_tmux_runtime.py /personal/coding_opd_runtime/tools/tmux
```

The bundle contains host tmux and its dynamic loader/dependencies with a hash
manifest. Only tmux uses that loader; task programs keep their original image
libraries. The config mounts this bundle read-only, allowing the unmodified
upstream stateful shell to operate without network package installation.
Validate cross-call cwd, exports and shell functions in a real task container
before starting model inference. Do not replace a bundle during an active run.

Use a fresh result directory: historical 16K/24-turn results are not reusable
under this recipe. The evaluator also checks task-config hashes on resume.
Student and Teacher each use four GPUs; quick-first full evaluation runs the
64-task panel, then the other 436 tasks without repeating the panel.

For a context-only ablation, set `MAX_CONTEXT_TOKENS=262144`. The launcher
writes the effective task config into the new run directory and hashes it with
the results. It preserves the 100-turn limit, sampling, tools, prompt and
45-minute agent deadline. Engine max_model_len equals this total context
budget; it must not add the initial prompt again beyond the model's native
262,144-token limit. On a free eight-GPU node, Teacher-only evaluation can use
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`, `TENSOR_PARALLEL_SIZE=1` and
`EVAL_CONCURRENCY=32` (eight replicas, same four sessions per GPU as the
four-GPU/16-session reference). Use `QUICK_FIRST=true EVAL_TIER=full`.
