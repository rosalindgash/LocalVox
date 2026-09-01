import zipfile
from pathlib import Path

from localvox.f5_runtime import F5RuntimeManager
from localvox.f5_runtime_installer import (
    GET_PIP_REVISION,
    PYTHON_VERSION,
    F5RuntimeInstaller,
)


def test_f5_installer_configures_runtime(monkeypatch, tmp_path: Path):
    manager = F5RuntimeManager(tmp_path / "runtime")
    installer = F5RuntimeInstaller(manager)
    python = manager.root / "python" / "python.exe"
    worker = manager.root / "f5_tts_onnx_worker.py"
    model_dir = manager.root / "models" / "f5-v1-cpu-f32"
    progress = []

    def fake_python(_report):
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_bytes(b"python")
        return python

    def fake_models(_report):
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def fake_worker():
        worker.write_bytes(b"worker")
        return worker

    monkeypatch.setattr(
        "localvox.f5_runtime_installer.platform.system", lambda: "Windows"
    )
    monkeypatch.setattr(installer, "_ensure_python", fake_python)
    monkeypatch.setattr(installer, "_ensure_models", fake_models)
    monkeypatch.setattr(installer, "_install_worker", fake_worker)
    monkeypatch.setattr(installer, "_health_check", lambda *_args: None)
    monkeypatch.setattr(installer, "_run", lambda *_args, **_kwargs: None)

    installer.install(progress.append)

    manifest = manager.load()
    assert manifest.python_executable == str(python)
    assert manifest.worker_script == str(worker)
    assert manifest.model_dir == str(model_dir)
    assert manager.status().ready
    assert progress[-1] == "F5-TTS ONNX is ready."


def test_f5_health_check_runs_worker_with_private_environment(
    monkeypatch, tmp_path: Path
):
    manager = F5RuntimeManager(tmp_path / "runtime")
    installer = F5RuntimeInstaller(manager)
    python = manager.root / "python" / "python.exe"
    worker = manager.root / "f5_tts_onnx_worker.py"
    model_dir = manager.root / "models" / "f5-v1-cpu-f32"
    calls = []
    monkeypatch.setattr(
        installer,
        "_run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    installer._health_check(python, worker, model_dir)

    args, kwargs = calls[0]
    assert args == (python, str(worker), "--health-check")
    assert kwargs["env"]["LOCALVOX_F5_MODEL_DIR"] == str(model_dir)
    assert kwargs["env"]["LOCALVOX_F5_RUNTIME_ROOT"] == str(manager.root)


def test_f5_python_bootstrap_uses_registry_free_embeddable_bundle(
    monkeypatch, tmp_path: Path
):
    manager = F5RuntimeManager(tmp_path / "runtime")
    manager.ensure_directories()
    installer = F5RuntimeInstaller(manager)
    archive = manager.root / "downloads" / (f"python-{PYTHON_VERSION}-embed-amd64.zip")
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("python.exe", b"private python")
        bundle.writestr("python311._pth", "python311.zip\n.\n#import site\n")
    get_pip = manager.root / "downloads" / f"get-pip-{GET_PIP_REVISION}.py"
    get_pip.write_text("# pinned package bootstrap", encoding="utf-8")
    runs = []

    monkeypatch.setattr(installer, "_verify_sha256", lambda *_args: None)
    monkeypatch.setattr(installer, "_run", lambda *args, **_kwargs: runs.append(args))
    monkeypatch.setattr(installer, "_python_has_pip", lambda _python: True)

    python = installer._ensure_python(lambda _message: None)

    assert python == manager.root / "python" / "python.exe"
    path_config = (manager.root / "python" / "python311._pth").read_text()
    assert "import site" in path_config
    assert "Lib\\site-packages" in path_config
    assert runs == [(python, str(get_pip))]
