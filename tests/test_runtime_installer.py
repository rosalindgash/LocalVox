from pathlib import Path

from localvox.runtime import RuntimeManager
from localvox.runtime_installer import RuntimeInstaller


def test_runtime_installer_configures_managed_runtime(monkeypatch, tmp_path: Path):
    manager = RuntimeManager(tmp_path / "runtime")
    installer = RuntimeInstaller(manager)
    progress: list[str] = []
    python = manager.root / "python" / "python.exe"
    worker = manager.root / "openvoice_v2_worker.py"
    checkpoints = manager.root / "models" / "checkpoints_v2"
    openvoice_source = manager.root / "sources" / "OpenVoice"
    melo_source = manager.root / "sources" / "MeloTTS"

    def fake_python(_report):
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("", encoding="utf-8")
        return python

    def fake_source(name, _url, _commit, _report):
        path = openvoice_source if name == "OpenVoice" else melo_source
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fake_checkpoints(_report):
        checkpoints.mkdir(parents=True, exist_ok=True)
        return checkpoints

    def fake_worker():
        worker.write_text("# worker", encoding="utf-8")
        return worker

    monkeypatch.setattr("localvox.runtime_installer.platform.system", lambda: "Windows")
    monkeypatch.setattr(installer, "_ensure_python", fake_python)
    monkeypatch.setattr(installer, "_ensure_source", fake_source)
    monkeypatch.setattr(installer, "_ensure_checkpoints", fake_checkpoints)
    monkeypatch.setattr(installer, "_install_worker", fake_worker)
    monkeypatch.setattr(
        installer,
        "_health_check",
        lambda _python, _worker, _checkpoints: None,
    )
    monkeypatch.setattr(installer, "_run", lambda _python, *_args, **_kwargs: None)

    installer.install(progress.append)

    manifest = manager.load()
    assert manifest.python_executable == str(python)
    assert manifest.worker_script == str(worker)
    assert manifest.checkpoints_dir == str(checkpoints)
    assert manifest.openvoice_dir == str(openvoice_source)
    assert manifest.melo_ready is True
    assert manager.status().ready is True
    assert progress[-1] == "OpenVoice V2 is ready."


def test_health_check_runs_installed_worker_with_private_runtime_env(
    monkeypatch, tmp_path: Path
):
    manager = RuntimeManager(tmp_path / "runtime")
    installer = RuntimeInstaller(manager)
    python = manager.root / "python" / "python.exe"
    worker = manager.root / "openvoice_v2_worker.py"
    checkpoints = manager.root / "models" / "checkpoints_v2"
    calls = []

    monkeypatch.setattr(
        installer,
        "_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    installer._health_check(python, worker, checkpoints)

    args, kwargs = calls[0]
    assert args == (python, str(worker), "--health-check")
    assert kwargs["env"]["LOCALVOX_OPENVOICE_CHECKPOINTS"] == str(checkpoints)
    assert kwargs["env"]["LOCALVOX_RUNTIME_ROOT"] == str(manager.root)
