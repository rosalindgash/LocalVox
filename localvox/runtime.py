from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from localvox.storage import app_data_root

RUNTIME_ID = "openvoice-v2"


@dataclass(slots=True)
class RuntimeManifest:
    runtime_id: str = RUNTIME_ID
    python_executable: str = ""
    worker_script: str = ""
    checkpoints_dir: str = ""
    openvoice_dir: str = ""
    melo_ready: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    ready: bool
    message: str


class RuntimeManager:
    """Owns LocalVox's private speech-runtime configuration.

    The desktop package and the ML runtime are intentionally separate. This
    lets LocalVox repair or replace an engine without forcing users to reinstall
    the desktop app itself.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (app_data_root() / "runtime" / RUNTIME_ID)
        self.manifest_path = self.root / "runtime.json"

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "downloads").mkdir(exist_ok=True)
        (self.root / "models").mkdir(exist_ok=True)

    def load(self) -> RuntimeManifest:
        if not self.manifest_path.exists():
            return RuntimeManifest()
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return RuntimeManifest(**data)

    def save(self, manifest: RuntimeManifest) -> None:
        self.ensure_directories()
        self.manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2),
            encoding="utf-8",
        )

    def configured_python(self) -> Path | None:
        override = os.getenv("LOCALVOX_OPENVOICE_PYTHON", "").strip()
        if override:
            return Path(override)
        value = self.load().python_executable.strip()
        return Path(value) if value else None

    def configured_worker(self) -> Path | None:
        override = os.getenv("LOCALVOX_OPENVOICE_WORKER", "").strip()
        if override:
            return Path(override)
        value = self.load().worker_script.strip()
        return Path(value) if value else None

    def status(self) -> RuntimeStatus:
        python = self.configured_python()
        worker = self.configured_worker()
        if python is None and worker is None:
            return RuntimeStatus(False, "OpenVoice runtime is not installed")
        if python is None or not python.exists():
            return RuntimeStatus(False, "OpenVoice runtime Python is missing")
        if worker is None or not worker.exists():
            return RuntimeStatus(False, "OpenVoice worker is missing")
        return RuntimeStatus(True, "OpenVoice V2 runtime ready")

    def configure_existing(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        checkpoints_dir: Path | None = None,
        openvoice_dir: Path | None = None,
        melo_ready: bool = True,
    ) -> RuntimeManifest:
        manifest = RuntimeManifest(
            python_executable=str(python_executable),
            worker_script=str(worker_script),
            checkpoints_dir=str(checkpoints_dir or ""),
            openvoice_dir=str(openvoice_dir or ""),
            melo_ready=melo_ready,
        )
        self.save(manifest)
        return manifest
