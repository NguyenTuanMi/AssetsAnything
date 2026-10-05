"""Process supervisor for model-specific worker environments."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BackendCapabilities, ReconstructionRequest, ReconstructionResult


@dataclass(frozen=True)
class WorkerConfig:
    name: str
    port: int
    python: str
    startup_timeout: float = 120.0
    shutdown_timeout: float = 30.0


class ModelSupervisor:
    """Owns worker processes and enforces one active GPU worker."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self._lock = threading.RLock()
        self._active_name: str | None = None
        self._process: subprocess.Popen | None = None
        self._configs = self._build_configs()

    @property
    def active_name(self) -> str | None:
        with self._lock:
            return self._active_name if self._process_is_alive() else None

    def _build_configs(self) -> dict[str, WorkerConfig]:
        defaults = {
            "trellis": (8200, "TRELLIS_WORKER_PYTHON"),
            "trellis2": (8202, "TRELLIS2_WORKER_PYTHON"),
            "pixal3d": (8204, "PIXAL3D_WORKER_PYTHON"),
        }
        configs = {}
        for name, (default_port, python_env) in defaults.items():
            configs[name] = WorkerConfig(
                name=name,
                port=int(os.environ.get(f"{name.upper()}_WORKER_PORT", default_port)),
                python=os.environ.get(python_env, sys.executable),
                startup_timeout=float(
                    os.environ.get(f"{name.upper()}_WORKER_STARTUP_TIMEOUT", "120")
                ),
                shutdown_timeout=float(
                    os.environ.get(f"{name.upper()}_WORKER_SHUTDOWN_TIMEOUT", "30")
                ),
            )
        return configs

    def capabilities(self) -> dict[str, dict[str, Any]]:
        return {name: {"port": config.port, "python": config.python}
                for name, config in self._configs.items()}

    def reconstruct(
        self,
        backend_name: str,
        request: ReconstructionRequest,
        capabilities: BackendCapabilities,
    ) -> ReconstructionResult:
        if backend_name not in self._configs:
            raise ValueError(f"No worker configuration for backend {backend_name!r}")
        if not request.images:
            raise ValueError("At least one image is required.")
        if len(request.images) > 1 and not capabilities.supports_multi_image:
            raise ValueError(f"Backend {backend_name!r} does not support multiple images.")

        with self._lock:
            self._ensure_worker(backend_name)
            config = self._configs[backend_name]
            with tempfile.TemporaryDirectory(prefix=f"assetsanything_{backend_name}_") as staging:
                input_paths = []
                for index, image in enumerate(request.images):
                    path = Path(staging) / f"input_{index}.png"
                    image.convert("RGBA").save(path, format="PNG")
                    input_paths.append(str(path))

                payload = {
                    "input_paths": input_paths,
                    "output_path": str(Path(request.output_path).resolve()),
                    "options": request.options,
                }
                response = self._request(config, "/generate", payload, timeout=3600)

        return ReconstructionResult(
            glb_path=response["glb_path"],
            backend_name=response["backend"],
            num_images_used=response["num_images_used"],
            raw_meta=response.get("raw_meta", {}),
        )

    def _ensure_worker(self, backend_name: str) -> None:
        if self._active_name == backend_name and self._process_is_alive():
            return

        if self._active_name is not None:
            self._stop_active_worker()

        config = self._configs[backend_name]
        command = [
            config.python,
            "-m",
            "inference_backend.worker",
            "--backend",
            backend_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(config.port),
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.root_dir), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        print(f"[supervisor] starting {backend_name} worker on port {config.port}")
        self._process = subprocess.Popen(
            command,
            cwd=self.root_dir,
            env=env,
            start_new_session=True,
        )
        self._active_name = backend_name
        try:
            self._wait_until_ready(config)
        except Exception:
            self._stop_active_worker()
            raise

    def _wait_until_ready(self, config: WorkerConfig) -> None:
        deadline = time.monotonic() + config.startup_timeout
        last_error = "worker did not respond"
        while time.monotonic() < deadline:
            if not self._process_is_alive():
                raise RuntimeError(f"{config.name} worker exited during startup")
            try:
                response = self._request(config, "/health", None, timeout=2)
                if response.get("status") == "ok":
                    return
            except Exception as error:
                last_error = str(error)
            time.sleep(0.25)
        raise RuntimeError(f"Timed out waiting for {config.name} worker: {last_error}")

    def _request(
        self,
        config: WorkerConfig,
        path: str,
        payload: dict[str, Any] | None,
        timeout: float,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{config.port}{path}",
            data=data,
            headers={"Content-Type": "application/json"} if data else {},
            method="POST" if data else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{config.name} worker returned HTTP {error.code}: {detail}") from error

    def _process_is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _stop_active_worker(self) -> None:
        process = self._process
        name = self._active_name
        if process is None:
            self._active_name = None
            return

        print(f"[supervisor] stopping {name} worker")
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=self._configs[name].shutdown_timeout)
            except subprocess.TimeoutExpired:
                print(f"[supervisor] force-stopping {name} worker")
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            except ProcessLookupError:
                pass

        self._process = None
        self._active_name = None

    def shutdown(self) -> None:
        with self._lock:
            self._stop_active_worker()
