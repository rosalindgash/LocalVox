from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from localvox.engines.base import EngineStatus, TTSEngine
from localvox.storage import VoiceProfile


class OpenVoiceEngine(TTSEngine):
    """Adapter boundary for an OpenVoice V2 worker.

    The desktop app stays independent from the speech runtime. A worker script
    can live in a separate Python environment and is configured with the
    LOCALVOX_OPENVOICE_WORKER environment variable.
    """

    id = "openvoice-v2"
    display_name = "OpenVoice V2"

    def __init__(self) -> None:
        self.worker = os.getenv("LOCALVOX_OPENVOICE_WORKER", "").strip()

    def status(self) -> EngineStatus:
        if not self.worker:
            return EngineStatus(False, "OpenVoice worker not installed yet")
        path = Path(self.worker)
        if not path.exists():
            return EngineStatus(False, f"Configured worker not found: {path}")
        return EngineStatus(True, "OpenVoice V2 worker ready")

    def generate(self, *, voice: VoiceProfile, text: str, output_path: Path) -> Path:
        status = self.status()
        if not status.available:
            raise RuntimeError(status.message)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            self.worker,
            "--reference",
            voice.reference_audio,
            "--text",
            text,
            "--output",
            str(output_path),
        ]
        if voice.transcript:
            cmd.extend(["--transcript", voice.transcript])
        subprocess.run(cmd, check=True)
        return output_path
