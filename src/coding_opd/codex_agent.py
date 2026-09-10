"""Codex CLI agent for network-isolated external-evaluation containers."""

from __future__ import annotations

import json
import logging
import shlex
import time
from pathlib import Path
from typing import Any

from pydantic import Field
from uni_agent.agents.base import Agent, AgentConfig, AgentResult
from uni_agent.agents.registry import register_agent
from uni_agent.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class CodexCliConfig(AgentConfig):
    name: str = "coding_opd_codex"
    codex_binary: str = "/opt/coding-opd/codex"
    proxy_script: str = "/opt/coding-opd/unix_socket_http_proxy.py"
    model_socket: str = "/opt/coding-opd/model/qwen.sock"
    api_port: int = Field(default=8000, ge=1, le=65535)
    context_window: int = Field(default=262_144, ge=4096)
    auto_compact_token_limit: int | None = Field(default=None, ge=1)
    reasoning_effort: str = Field(default="xhigh", pattern="^(minimal|low|medium|high|xhigh)$")
    reasoning_summary: str = Field(default="auto", pattern="^(auto|concise|detailed|none)$")
    log_dir: str | None = None
    container_log_dir: str = "/opt/coding-opd/agent-logs"


def resolve_auto_compact_limit(context_window: int, requested: int | None = None) -> int:
    """Pin Codex's 90% default; reject overrides it would silently clamp."""
    if context_window < 4096:
        raise ValueError("context_window must be at least 4096")
    maximum = context_window * 9 // 10
    if requested is not None and not 1 <= requested <= maximum:
        raise ValueError(f"auto_compact_token_limit must be between 1 and {maximum}")
    return maximum if requested is None else requested


def _instruction(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = str(message.get("role") or "user")
        if role == "user" and not parts:
            parts.append(content)
        else:
            parts.append(f"[{role}]\n{content}")
    if not parts:
        raise ValueError("coding_opd_codex requires at least one non-empty text message")
    return "\n\n".join(parts)


def _summarize_jsonl(output: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transcript: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    usage: dict[str, Any] = {}
    thread_id: str | None = None
    parse_errors = 0
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if not isinstance(event, dict):
            # stdout/stderr can contain valid JSON scalar/list diagnostics, not events.
            parse_errors += 1
            continue
        event_type = str(event.get("type") or "unknown")
        counts[event_type] = counts.get(event_type, 0) + 1
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
        elif event_type == "turn.completed" and isinstance(event.get("usage"), dict):
            usage = dict(event["usage"])
        item = event.get("item")
        if event_type == "item.completed" and isinstance(item, dict):
            item_type = str(item.get("type") or "unknown")
            counts[f"item.{item_type}"] = counts.get(f"item.{item_type}", 0) + 1
            if item_type == "agent_message" and isinstance(item.get("text"), str):
                transcript.append({"role": "assistant", "content": item["text"]})
    return transcript, {
        "event_counts": counts,
        "parse_errors": parse_errors,
        "thread_id": thread_id,
        "usage": usage,
    }


def summarize_codex_artifacts(log_dir: Path) -> dict[str, Any]:
    """Return audit metadata for the live-mounted Codex output and sessions."""
    raw_path = log_dir / "codex-cli.jsonl"
    output = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.exists() else ""
    _, summary = _summarize_jsonl(output)
    session_files = sorted((log_dir / "codex-home" / "sessions").rglob("*.jsonl"))
    compactions = 0
    max_request_tokens = 0
    last_request_tokens = 0
    session_parse_errors = 0
    for path in session_files:
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    session_parse_errors += 1
                    continue
                if not isinstance(event, dict):
                    session_parse_errors += 1
                    continue
                # Count durable history replacements, not start/progress messages.
                if event.get("type") == "compacted":
                    compactions += 1
                payload = event.get("payload")
                if event.get("type") != "event_msg" or not isinstance(payload, dict):
                    continue
                if payload.get("type") != "token_count":
                    continue
                info = payload.get("info")
                usage = info.get("last_token_usage") if isinstance(info, dict) else None
                total = usage.get("total_tokens") if isinstance(usage, dict) else None
                if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
                    last_request_tokens = total
                    max_request_tokens = max(max_request_tokens, total)
    summary.update(
        {
            "compaction_count": compactions,
            "max_request_total_tokens": max_request_tokens,
            "last_request_total_tokens": last_request_tokens,
            "session_parse_errors": session_parse_errors,
            "raw_jsonl_path": str(raw_path),
            "raw_jsonl_bytes": raw_path.stat().st_size if raw_path.exists() else 0,
            "session_files": [str(path) for path in session_files],
            "session_bytes": sum(path.stat().st_size for path in session_files),
        }
    )
    return summary


@register_agent("coding_opd_codex")
class CodexCliAgent(Agent):
    """Run the official Codex CLI against a Responses API over a mounted UDS."""

    config_model = CodexCliConfig

    async def run(
        self,
        *,
        sandbox: Sandbox,
        messages: list[dict[str, Any]],
        workdir: str | None = None,
    ) -> AgentResult:
        cfg: CodexCliConfig = self.config  # type: ignore[assignment]
        compact_limit = resolve_auto_compact_limit(cfg.context_window, cfg.auto_compact_token_limit)
        model_name = cfg.model.model_name
        if not model_name:
            raise ValueError("coding_opd_codex: agent.model.model_name is required")

        if not cfg.log_dir:
            raise ValueError("coding_opd_codex: agent.log_dir is required for durable trajectories")
        host_log_dir = Path(cfg.log_dir)
        host_log_dir.mkdir(parents=True, exist_ok=True)
        container_log_dir = cfg.container_log_dir.rstrip("/")
        codex_home = f"{container_log_dir}/codex-home"
        raw_output = f"{container_log_dir}/codex-cli.jsonl"
        pid_path = f"{container_log_dir}/codex.pid"
        api_base = f"http://127.0.0.1:{cfg.api_port}/v1"
        config_toml = f'''model_provider = "coding_opd_local"
model_context_window = {cfg.context_window}
model_auto_compact_token_limit = {compact_limit}
model_supports_reasoning_summaries = true
model_reasoning_summary = "{cfg.reasoning_summary}"
approval_policy = "never"
sandbox_mode = "danger-full-access"

[model_providers.coding_opd_local]
name = "Coding OPD local vLLM"
base_url = "{api_base}"
wire_api = "responses"
env_key = "CODING_OPD_LOCAL_API_KEY"
supports_websockets = false
'''
        await sandbox.write_file(f"{codex_home}/config.toml", config_toml)
        await sandbox.write_file(
            f"{codex_home}/auth.json",
            json.dumps({"OPENAI_API_KEY": cfg.model.api_key or "EMPTY"}),
        )

        proxy_log = "/tmp/coding-opd-model-proxy.log"
        proxy_start = await sandbox.exec_shell(
            "set -eu; "
            f"python3 {shlex.quote(cfg.proxy_script)} "
            f"--unix-socket {shlex.quote(cfg.model_socket)} --port {cfg.api_port} "
            f">{shlex.quote(proxy_log)} 2>&1 </dev/null & echo $!",
            timeout=30,
            workdir=workdir,
        )
        if proxy_start.exit_code != 0:
            raise RuntimeError(f"failed to start model socket proxy: {proxy_start.stderr}")

        ready = await sandbox.exec(
            [
                "python3",
                "-c",
                (
                    "import socket,time; "
                    f"deadline=time.time()+20; addr=('127.0.0.1',{cfg.api_port}); "
                    "\nwhile True:\n"
                    " try:\n  s=socket.create_connection(addr,1); s.close(); break\n"
                    " except OSError:\n"
                    "  if time.time() >= deadline: raise\n"
                    "  time.sleep(.2)"
                ),
            ],
            timeout=30,
            workdir=workdir,
        )
        if ready.exit_code != 0:
            proxy_tail = await sandbox.exec_shell(
                f"tail -n 40 {shlex.quote(proxy_log)}", timeout=10, workdir=workdir
            )
            raise RuntimeError(
                "model socket proxy did not become ready: "
                f"{ready.stderr or ready.stdout}; {proxy_tail.stdout or proxy_tail.stderr}"
            )

        env = {
            "CODEX_HOME": codex_home,
            "CODING_OPD_LOCAL_API_KEY": cfg.model.api_key or "EMPTY",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "NO_PROXY": "127.0.0.1,localhost",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
        }
        command = [
            cfg.codex_binary,
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model",
            model_name,
            "--json",
            "--enable",
            "unified_exec",
            "-c",
            f"model_reasoning_effort={cfg.reasoning_effort}",
            "-c",
            f"model_reasoning_summary={cfg.reasoning_summary}",
            "--",
            _instruction(messages),
        ]
        begin = time.perf_counter()
        completed = None
        needs_kill = True
        try:
            launch = (
                "set -eu; "
                f"mkdir -p {shlex.quote(container_log_dir)}; "
                f"setsid {shlex.join(command)} >{shlex.quote(raw_output)} 2>&1 </dev/null & "
                f"codex_pid=$!; printf '%s\\n' \"$codex_pid\" >{shlex.quote(pid_path)}; "
                'wait "$codex_pid"'
            )
            completed = await sandbox.exec_shell(launch, workdir=workdir, env=env)
            needs_kill = False
        finally:
            # asyncio cancellation cannot stop subprocess.run's worker thread.  Kill
            # the exact Codex process group before the task snapshots its patch.
            try:
                cleanup = ""
                if needs_kill:
                    cleanup = (
                        "if [ -s {pid} ]; then pid=$(cat {pid}); "
                        "kill -- -\"$pid\" 2>/dev/null || kill \"$pid\" 2>/dev/null || true; fi; "
                    ).format(pid=shlex.quote(pid_path))
                cleanup += f"rm -f {shlex.quote(f'{codex_home}/auth.json')}"
                await sandbox.exec_shell(cleanup, timeout=30, workdir=workdir)
            except Exception:
                logger.exception("failed to clean up the Codex process group")
        if completed is None:
            raise RuntimeError("Codex CLI did not produce a completion result")
        wall_seconds = time.perf_counter() - begin
        output_path = host_log_dir / "codex-cli.jsonl"
        output = output_path.read_text(encoding="utf-8", errors="replace")
        transcript, info = _summarize_jsonl(output)
        info.update(
            {
                "codex_exit_code": completed.exit_code,
                "codex_stderr_tail": completed.stderr[-4000:],
                "codex_version": await self._version(sandbox, cfg, workdir),
                "wall_seconds": wall_seconds,
                "reasoning_effort": cfg.reasoning_effort,
                "reasoning_summary": cfg.reasoning_summary,
                "context_window": cfg.context_window,
                "auto_compact_token_limit": compact_limit,
            }
        )
        info.update(summarize_codex_artifacts(host_log_dir))
        info["context_limit_reached"] = info["max_request_total_tokens"] >= cfg.context_window
        if completed.exit_code != 0:
            logger.warning(
                "Codex CLI exited with %s after %.1fs: %s",
                completed.exit_code,
                wall_seconds,
                completed.stderr[-1000:],
            )
        return AgentResult(
            transcript=transcript,
            info=info,
            finished=completed.exit_code == 0,
        )

    @staticmethod
    async def _version(
        sandbox: Sandbox,
        cfg: CodexCliConfig,
        workdir: str | None,
    ) -> str:
        result = await sandbox.exec([cfg.codex_binary, "--version"], timeout=30, workdir=workdir)
        return (result.stdout or result.stderr).strip()
