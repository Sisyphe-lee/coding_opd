from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_upstreams_exist() -> None:
    for path in ("third_party/uni-agent", "third_party/verl", "third_party/R2E-Gym"):
        assert (ROOT / path).is_dir()


def test_runtime_artifacts_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text()
    for entry in ("models/", "checkpoints/", "outputs/", ".venv/"):
        assert entry in ignore


def test_podman_uses_native_overlay_store() -> None:
    wrapper = (ROOT / "scripts" / "podman_sandbox").read_text()
    assert "/workspaces/coding-opd-podman-overlay" in wrapper
    assert "/run/coding-opd-podman-overlay" in wrapper
    assert "CODING_OPD_PODMAN_DRIVER:-overlay" in wrapper


def test_swesmith_debug_launcher_uses_swesmith_runner() -> None:
    for name in ("run_swesmith_opd_debug.sh", "run_swesmith_opd_async_smoke.sh"):
        launcher = (ROOT / "scripts" / name).read_text()
        assert "coding_opd.swesmith_task.run_swesmith_task" in launcher
        assert "configs/swesmith_react.yaml" in launcher

    actor_perf = (ROOT / "scripts" / "run_swesmith_actor_perf.sh").read_text()
    assert "SKIP_ROLLOUT_TQ=true" in actor_perf
    assert "gc64k" in actor_perf


def test_async_smoke_keeps_teacher_kv_budget_conservative() -> None:
    launcher = (ROOT / "scripts" / "run_r2e_opd_async_smoke.sh").read_text()
    assert 'TEACHER_GPU_MEMORY_UTILIZATION:-0.5' in launcher
    assert 'ACTOR_GPUS:-4' in launcher
    assert 'TEACHER_GPUS:-2' in launcher
    assert 'MAX_CONCURRENT_SESSIONS:-32' in launcher
    assert 'TEACHER_MAX_NUM_SEQS:-16' in launcher
    assert 'ENABLE_GRADIENT_CHECKPOINTING:-false' in launcher
    assert 'HYBRID_ROLLOUT_ENABLE_SWITCH:-true' in launcher
    assert 'CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7' in launcher
    assert 'TRAIN_BATCH_SIZE:-32' in launcher
    assert 'PPO_MINI_BATCH_SIZE:-16' in launcher
    assert 'PARAMETER_SYNC_STEP:-2' in launcher
    assert 'MAX_OFF_POLICY_THRESHOLD:-2' in launcher
    assert 'ROLLOUT_BUDGET:-64' in launcher
    assert 'ACTOR_USE_LIGER:-false' in launcher
    assert 'ACTOR_FUSED_ADAMW:-false' in launcher
    assert 'ACTOR_FORWARD_PREFETCH:-false' in launcher
    assert 'ACTOR_NO_SYNC_GRAD_ACCUMULATION:-false' in launcher

    swesmith_launcher = (ROOT / "scripts" / "run_swesmith_opd_async_smoke.sh").read_text()
    assert "debug_repeat2.parquet" in swesmith_launcher


def test_r2e_budget_launchers_inherit_optimized_async_defaults() -> None:
    for name in ("run_r2e_opd_pilot.sh", "run_r2e_opd_train.sh"):
        launcher = (ROOT / "scripts" / name).read_text()
        assert "run_r2e_opd_async_smoke.sh" in launcher


def test_checkpoint_resume_launcher_uses_frozen_r2e128_and_two_phases() -> None:
    launcher = (ROOT / "scripts" / "run_r2e128_checkpoint_resume.sh").read_text()
    assert "datasets/r2e_train_128" in launcher
    assert "RESUME_MODE=disable" in launcher
    assert "RESUME_MODE=resume_path" in launcher
    assert "ASYNC_WARMUP_BATCHES=0" in launcher
    assert "validate_training_checkpoint.py" in launcher
    assert "hf_model" in launcher


def test_external_eval_launcher_uses_only_authoritative_benchmarks() -> None:
    launcher = (ROOT / "scripts" / "run_external_eval.sh").read_text()
    assert "swebench_verified" in launcher
    assert "deepswe" in launcher
    assert "coding_opd_eval_v2" in launcher
    assert "RAY_ADDRESS must point to a dedicated Coding OPD Ray head" in launcher
    assert 'EVAL_TIER="${EVAL_TIER:-full}"' in launcher
    assert 'VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/var/lib/coding-opd-vllm-eval-cache}"' in launcher
    assert 'configs/coding_react.yaml' in launcher
    assert '--vllm-cache-root "${VLLM_CACHE_ROOT}"' in launcher
    assert "summarize_external_eval.py" in launcher
    assert "swesmith" not in launcher.lower()


def test_oci_importer_propagates_manifest_validation_failures() -> None:
    importer = (ROOT / "scripts" / "import_oci_manifest_images.sh").read_text()
    assert "required-images.XXXXXX" in importer
    assert "mapfile -t images <\"$image_list\"" in importer
    assert "mapfile -t images < <(" not in importer


def test_external_eval_canary_is_explicitly_non_reportable() -> None:
    materializer = (ROOT / "scripts" / "materialize_external_eval_canary.py").read_text()
    assert "coding-opd-external-eval-canary-v1" in materializer
    assert '"is_canary": True' in materializer
    assert 'default="full"' in materializer


def test_primary_docs_separate_frozen_r2e_data_from_swesmith_perf_fixture() -> None:
    for name in ("README.md", "AGENTS.md"):
        document = (ROOT / name).read_text()
        assert "run_swesmith_opd_async_smoke.sh" in document
        assert "batch 32" in document
        assert "parameter_sync_step=2" in document
        assert "docs/datasets.md" in document
        assert "R2E-128" in document
        assert "R2E-512" in document

    assert (ROOT / "docs" / "datasets.md").is_file()
    assert not (ROOT / "docs" / "evaluation_protocol.md").exists()


def test_smoke_launcher_exposes_checkpoint_sync_backend() -> None:
    launcher = (ROOT / "scripts" / "run_r2e_opd_smoke.sh").read_text()
    assert 'CHECKPOINT_ENGINE_BACKEND="${CHECKPOINT_ENGINE_BACKEND:-nccl}"' in launcher
    assert "checkpoint_engine.backend=${CHECKPOINT_ENGINE_BACKEND}" in launcher
    assert "engine_kwargs.delta_sharded.encoding=${DELTA_SHARDED_ENCODING}" in launcher
    assert "engine_kwargs.nccl.multi_sender=${NCCL_CHECKPOINT_MULTI_SENDER}" in launcher
    assert "checkpoint_engine.custom_backend_module=coding_opd.nccl_checkpoint_plugin" in launcher

    async_launcher = (ROOT / "scripts" / "run_r2e_opd_async_smoke.sh").read_text()
    assert 'UPDATE_WEIGHTS_BUCKET_MEGABYTES:-8192' in async_launcher
    assert 'PPO_MAX_TOKEN_LEN_PER_GPU:-24576' in async_launcher
    assert 'NCCL_CHECKPOINT_MULTI_SENDER:-true' in async_launcher


def test_trainer_loads_qwen35_kernel_plugin_in_actor_workers() -> None:
    launcher = (ROOT / "scripts" / "run_r2e_opd_smoke.sh").read_text()
    assert "actor_rollout_ref.model.external_lib=coding_opd.qwen35_kernel_plugin" in launcher
    assert "disable_custom_all_reduce=True" in launcher


def test_trainer_exposes_actor_perf_and_rollout_replay_controls() -> None:
    launcher = (ROOT / "scripts" / "run_r2e_opd_smoke.sh").read_text()
    for setting in (
        "EXPERIMENT_NAME",
        "ACTOR_USE_TORCH_COMPILE",
        "ACTOR_USE_FUSED_KERNELS",
        "ACTOR_FUSED_KERNELS_BACKEND",
        "ACTOR_RESHARD_AFTER_FORWARD",
        "ACTOR_NO_SYNC_GRAD_ACCUMULATION",
        "ROLLOUT_CUDAGRAPH_MODE",
        "ROLLOUT_CUDAGRAPH_CAPTURE_SIZES",
        "TEACHER_CUDAGRAPH_MODE",
        "TEACHER_CUDAGRAPH_CAPTURE_SIZES",
        "TEACHER_MAX_NUM_BATCHED_TOKENS",
        "ROLLOUT_CALCULATE_LOG_PROBS",
        "VLLM_DISABLE_COMPILE_CACHE",
        "CODING_OPD_VLLM_COMPILE_CACHE_ROOT",
        "ACTOR_SEQUENCE_PARALLEL_SIZE",
        "ACTOR_PAD_TO_LENGTH",
        "ACTOR_PAD_TO_LENGTH_BUCKET",
        "SKIP_ROLLOUT_TQ",
        "ROLLOUT_CACHE_DIR",
        "ROLLOUT_CACHE_ACTION",
        "ROLLOUT_CACHE_STEPS",
        "TEACHER_ENFORCE_EAGER",
    ):
        assert setting in launcher
    assert 'skip.rollout_tq.enable="${SKIP_ROLLOUT_TQ}"' in launcher
    assert "runtime_env.env_vars.CODING_OPD_DEVICE_PEAK_TFLOPS" in launcher
    assert "runtime_env.env_vars.VLLM_USE_FLASHINFER_SAMPLER" in launcher
