from pathlib import Path

from localvox.f5_runtime import F5RuntimeManager


def test_f5_runtime_manifest_round_trip(tmp_path: Path):
    manager = F5RuntimeManager(tmp_path / "runtime")
    python = manager.root / "python" / "python.exe"
    worker = manager.root / "f5_tts_onnx_worker.py"
    model_dir = manager.root / "models" / "f5-v1-cpu-f32"
    python.parent.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    python.write_bytes(b"python")
    worker.write_bytes(b"worker")

    manager.configure_existing(
        python_executable=python,
        worker_script=worker,
        model_dir=model_dir,
    )

    manifest = manager.load()
    assert manifest.runtime_id == "f5-tts-onnx"
    assert Path(manifest.model_dir) == model_dir
    assert manager.status().ready


def test_f5_runtime_reports_missing_model(tmp_path: Path):
    manager = F5RuntimeManager(tmp_path / "runtime")
    python = tmp_path / "python.exe"
    worker = tmp_path / "worker.py"
    python.write_bytes(b"python")
    worker.write_bytes(b"worker")
    manager.configure_existing(
        python_executable=python,
        worker_script=worker,
        model_dir=tmp_path / "missing-model",
    )

    status = manager.status()

    assert not status.ready
    assert "model" in status.message.lower()
