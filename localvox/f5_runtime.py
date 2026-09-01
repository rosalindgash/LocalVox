from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from localvox.storage import app_data_root

F5_RUNTIME_ID = "f5-tts-onnx"


@dataclass(slots=True)
class F5RuntimeManifest:
    runtime_id: str = F5_RUNTIME_ID
    python_executable: str = ""
    worker_script: str = ""
    model_dir: str = ""
    asr_model: str = "Systran/faster-whisper-tiny.en"


@dataclass(frozen=True, slots=True)
class F5RuntimeStatus:
    ready: bool
    message: str


class F5RuntimeManager:
    """Own LocalVox's private F5-TTS ONNX runtime configuration."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (app_data_root() / "runtime" / F5_RUNTIME_ID)
        self.manifest_path = self.root / "runtime.json"

    def ensure_directories(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "downloads").mkdir(exist_ok=True)
        (self.root / "models").mkdir(exist_ok=True)
        (self.root / "cache").mkdir(exist_ok=True)

    def load(self) -> F5RuntimeManifest:
        if not self.manifest_path.exists():
            return F5RuntimeManifest()
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return F5RuntimeManifest(**data)

    def save(self, manifest: F5RuntimeManifest) -> None:
        self.ensure_directories()
        self.manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2),
            encoding="utf-8",
        )

    def configured_python(self) -> Path | None:
        override = os.getenv("LOCALVOX_F5_PYTHON", "").strip()
        if override:
            return Path(override)
        value = self.load().python_executable.strip()
        return Path(value) if value else None

    def configured_worker(self) -> Path | None:
        override = os.getenv("LOCALVOX_F5_WORKER", "").strip()
        if override:
            return Path(override)
        value = self.load().worker_script.strip()
        return Path(value) if value else None

    def configured_model_dir(self) -> Path | None:
        override = os.getenv("LOCALVOX_F5_MODEL_DIR", "").strip()
        if override:
            return Path(override)
        value = self.load().model_dir.strip()
        return Path(value) if value else None

    def status(self) -> F5RuntimeStatus:
        python = self.configured_python()
        worker = self.configured_worker()
        model_dir = self.configured_model_dir()
        if python is None and worker is None and model_dir is None:
            return F5RuntimeStatus(False, "F5-TTS runtime is not installed")
        if python is None or not python.exists():
            return F5RuntimeStatus(False, "F5-TTS runtime Python is missing")
        if worker is None or not worker.exists():
            return F5RuntimeStatus(False, "F5-TTS worker is missing")
        if model_dir is None or not model_dir.exists():
            return F5RuntimeStatus(False, "F5-TTS ONNX model is missing")
        return F5RuntimeStatus(True, "F5-TTS ONNX runtime ready")

    def configure_existing(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        model_dir: Path,
        asr_model: str = "Systran/faster-whisper-tiny.en",
    ) -> F5RuntimeManifest:
        manifest = F5RuntimeManifest(
            python_executable=str(python_executable),
            worker_script=str(worker_script),
            model_dir=str(model_dir),
            asr_model=asr_model,
        )
        self.save(manifest)
        return manifest
