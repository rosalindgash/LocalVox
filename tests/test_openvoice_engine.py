import subprocess
from pathlib import Path

import pytest

from localvox.engines.openvoice import OpenVoiceEngine
from localvox.runtime import RuntimeManager
from localvox.storage import VoiceProfile


def configured_engine(tmp_path: Path) -> tuple[OpenVoiceEngine, VoiceProfile, Path]:
    manager = RuntimeManager(tmp_path / "runtime")
    python = manager.root / "python" / "python.exe"
    worker = manager.root / "openvoice_v2_worker.py"
    checkpoints = manager.root / "models" / "checkpoints_v2"
    reference = tmp_path / "reference.wav"
    for path in (python, worker, reference):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"present")
    manager.configure_existing(
        python_executable=python,
        worker_script=worker,
        checkpoints_dir=checkpoints,
    )
    engine = OpenVoiceEngine()
    engine.runtime = manager
    profile = VoiceProfile("test", "Test", str(reference))
    return engine, profile, checkpoints


def test_generate_uses_private_runtime_and_checks_wav(monkeypatch, tmp_path: Path):
    engine, profile, checkpoints = configured_engine(tmp_path)
    output = tmp_path / "output.wav"
    call = {}

    def fake_run(command, **kwargs):
        call.update(command=command, **kwargs)
        output.write_bytes(b"RIFF-data")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert engine.generate(voice=profile, text="Hello", output_path=output) == output
    assert call["env"]["LOCALVOX_OPENVOICE_CHECKPOINTS"] == str(checkpoints)
    assert call["env"]["LOCALVOX_RUNTIME_ROOT"] == str(engine.runtime.root)
    assert call["capture_output"] is True


def test_generate_surfaces_worker_error(monkeypatch, tmp_path: Path):
    engine, profile, _checkpoints = configured_engine(tmp_path)

    def fail(_command, **_kwargs):
        raise subprocess.CalledProcessError(1, "worker", stderr="model failed")

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="model failed"):
        engine.generate(
            voice=profile,
            text="Hello",
            output_path=tmp_path / "output.wav",
        )
