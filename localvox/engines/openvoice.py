from __future__ import annotations

import os
import subprocess
from pathlib import Path

from localvox.engines.base import EngineStatus, TTSEngine
from localvox.runtime import RuntimeManager
from localvox.storage import VoiceProfile


class OpenVoiceEngine(TTSEngine):
    """Adapter boundary for the managed OpenVoice V2 runtime."""

    id = "openvoice-v2"
    display_name = "OpenVoice V2"

    def __init__(self) -> None:
        self.runtime = RuntimeManager()

    def status(self) -> EngineStatus:
        status = self.runtime.status()
        return EngineStatus(status.ready, status.message)

    def generate(self, *, voice: VoiceProfile, text: str, output_path: Path) -> Path:
        status = self.status()
        if not status.available:
            raise RuntimeError(status.message)

        python = self.runtime.configured_python()
        worker = self.runtime.configured_worker()
        manifest = self.runtime.load()
        if python is None or worker is None:
            raise RuntimeError("OpenVoice runtime is incomplete")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(python),
            str(worker),
            "--reference",
            voice.reference_audio,
            "--text",
            text,
            "--output",
            str(output_path),
        ]
        if voice.transcript:
            cmd.extend(["--transcript", voice.transcript])

        env = os.environ.copy()
        if manifest.checkpoints_dir:
            env["LOCALVOX_OPENVOICE_CHECKPOINTS"] = manifest.checkpoints_dir
        env["LOCALVOX_RUNTIME_ROOT"] = str(self.runtime.root)
        env["PYTHONUTF8"] = "1"
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                "OpenVoice could not generate narration."
                + (f"\n\n{detail}" if detail else "")
            ) from exc
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("OpenVoice completed without creating a WAV file.")
        return output_path
