"""llama-server lifecycle: launch with the flags this design needs, or connect to a running one."""

import asyncio
import logging
import os
import signal
import subprocess
import time

from .backend import LlamaClient
from .config import Settings

logger = logging.getLogger(__name__)


def backend_command(settings: Settings) -> list[str]:
    if not settings.model:
        raise ValueError("LLAMAJEV_MODEL / --model is required to launch llama-server")
    command = [
        settings.llama_server,
        "-m", settings.model,
        "--host", "127.0.0.1",
        "--port", str(settings.backend_port),
        "-np", str(settings.n_slots),
        "-c", str(settings.n_ctx),
        "-ngl", str(settings.n_gpu_layers),
        "--cache-ram", str(settings.cache_ram_mib),
        "--ctx-checkpoints", "32",
        "--checkpoint-min-step", "0",
        "--reasoning-budget", "0",
    ]
    if settings.mmproj:
        command += ["--mmproj", settings.mmproj]
    return command


def start_backend(settings: Settings) -> subprocess.Popen:
    command = backend_command(settings)
    logger.info("Starting llama-server: %s", " ".join(command))
    return subprocess.Popen(command, start_new_session=True)


def stop_backend(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


async def wait_ready(backend: LlamaClient, process: subprocess.Popen | None, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"llama-server exited with status {process.returncode}")
        if await backend.health():
            return
        await asyncio.sleep(0.5)
    raise TimeoutError(f"llama-server was not ready within {timeout:g}s")
