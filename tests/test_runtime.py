from pathlib import Path

from localvox.runtime import RuntimeManager, RuntimeManifest


def test_runtime_reports_not_installed(tmp_path: Path):
    manager = RuntimeManager(tmp_path / "runtime")

    status = manager.status()

    assert not status.ready
    assert "not installed" in status.message.lower()


def test_runtime_manifest_round_trip(tmp_path: Path):
    manager = RuntimeManager(tmp_path / "runtime")
    python = tmp_path / "python.exe"
    worker = tmp_path / "worker.py"
    python.write_text("", encoding="utf-8")
    worker.write_text("", encoding="utf-8")

    manager.configure_existing(
        python_executable=python,
        worker_script=worker,
        checkpoints_dir=tmp_path / "checkpoints_v2",
    )

    loaded = manager.load()
    assert loaded.runtime_id == "openvoice-v2"
    assert Path(loaded.python_executable) == python
    assert Path(loaded.worker_script) == worker
    assert manager.status().ready


def test_runtime_detects_missing_worker(tmp_path: Path):
    manager = RuntimeManager(tmp_path / "runtime")
    python = tmp_path / "python.exe"
    python.write_text("", encoding="utf-8")
    manager.save(
        RuntimeManifest(
            python_executable=str(python),
            worker_script=str(tmp_path / "missing.py"),
        )
    )

    status = manager.status()

    assert not status.ready
    assert "worker" in status.message.lower()
