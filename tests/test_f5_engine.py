import subprocess
from pathlib import Path

import pytest

from localvox.engines.f5_tts import F5TTSEngine
from localvox.f5_runtime import F5RuntimeManager
from localvox.storage import VoiceProfile


def configured_engine(tmp_path: Path):
    manager = F5RuntimeManager(tmp_path / "runtime")
    python = manager.root / "python" / "python.exe"
    worker = manager.root / "f5_tts_onnx_worker.py"
    model_dir = manager.root / "models" / "f5-v1-cpu-f32"
    reference = tmp_path / "reference.wav"
    model_dir.mkdir(parents=True)
    for path in (python, worker, reference):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"present")
    manager.configure_existing(
        python_executable=python,
        worker_script=worker,
        model_dir=model_dir,
    )
    engine = F5TTSEngine()
    engine.runtime = manager
    profile = VoiceProfile("test", "Test", str(reference), "Reference words")
    return engine, profile, model_dir


def test_f5_generate_passes_reference_transcript_and_private_runtime(
    monkeypatch, tmp_path: Path
):
    engine, profile, model_dir = configured_engine(tmp_path)
    output = tmp_path / "output.wav"
    call = {}

    def fake_run(command, **kwargs):
        call.update(command=command, **kwargs)
        output.write_bytes(b"RIFF-data")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert engine.generate(voice=profile, text="Hello", output_path=output) == output
    assert call["env"]["LOCALVOX_F5_MODEL_DIR"] == str(model_dir)
    assert call["env"]["LOCALVOX_F5_RUNTIME_ROOT"] == str(engine.runtime.root)
    assert call["command"][-2:] == ["--transcript", "Reference words"]
    cache_index = call["command"].index("--profile-cache")
    cache_dir = Path(call["command"][cache_index + 1])
    assert cache_dir.parent == engine.runtime.root / "cache" / "voice-profiles"
    assert cache_dir.name.startswith("test-")


def test_f5_generate_surfaces_worker_error(monkeypatch, tmp_path: Path):
    engine, profile, _model_dir = configured_engine(tmp_path)

    def fail(_command, **_kwargs):
        raise subprocess.CalledProcessError(1, "worker", stderr="onnx failed")

    monkeypatch.setattr(subprocess, "run", fail)

    with pytest.raises(RuntimeError, match="onnx failed"):
        engine.generate(
            voice=profile,
            text="Hello",
            output_path=tmp_path / "output.wav",
        )
