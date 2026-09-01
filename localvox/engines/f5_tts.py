from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

from localvox.engines.base import EngineStatus, TTSEngine
from localvox.f5_runtime import F5RuntimeManager
from localvox.storage import VoiceProfile


class F5TTSEngine(TTSEngine):
    """Adapter for LocalVox's managed F5-TTS ONNX runtime."""

    id = "f5-tts-onnx"
    display_name = "F5-TTS ONNX"

    def __init__(self) -> None:
        self.runtime = F5RuntimeManager()

    def status(self) -> EngineStatus:
        status = self.runtime.status()
        return EngineStatus(status.ready, status.message)

    def profile_cache_directory(self, voice: VoiceProfile) -> Path:
        readable_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", voice.slug).strip("-.")
        readable_slug = readable_slug or "voice"
        identity = hashlib.sha256(voice.slug.encode("utf-8")).hexdigest()[:12]
        return (
            self.runtime.root
            / "cache"
            / "voice-profiles"
            / f"{readable_slug}-{identity}"
        )

    def generate(self, *, voice: VoiceProfile, text: str, output_path: Path) -> Path:
        status = self.status()
        if not status.available:
            raise RuntimeError(status.message)

        python = self.runtime.configured_python()
        worker = self.runtime.configured_worker()
        model_dir = self.runtime.configured_model_dir()
        manifest = self.runtime.load()
        if python is None or worker is None or model_dir is None:
            raise RuntimeError("F5-TTS runtime is incomplete")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(python),
            str(worker),
            "--reference",
            voice.reference_audio,
            "--text",
            text,
            "--output",
            str(output_path),
            "--profile-cache",
            str(self.profile_cache_directory(voice)),
        ]
        if voice.transcript:
            command.extend(["--transcript", voice.transcript])

        env = os.environ.copy()
        env["LOCALVOX_F5_MODEL_DIR"] = str(model_dir)
        env["LOCALVOX_F5_RUNTIME_ROOT"] = str(self.runtime.root)
        env["LOCALVOX_F5_ASR_MODEL"] = manifest.asr_model
        env["PYTHONUTF8"] = "1"
        try:
            subprocess.run(
                command,
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
                "F5-TTS could not generate narration."
                + (f"\n\n{detail}" if detail else "")
            ) from exc
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("F5-TTS completed without creating a WAV file.")
        return output_path
