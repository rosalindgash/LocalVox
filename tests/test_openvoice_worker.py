import os
from pathlib import Path

from localvox.workers.openvoice_v2_worker import configure_runtime_environment


def test_worker_uses_private_runtime_caches(monkeypatch, tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    monkeypatch.setenv("LOCALVOX_RUNTIME_ROOT", str(runtime_root))
    for name in (
        "HF_HOME",
        "NLTK_DATA",
        "TORCH_HOME",
        "HF_HUB_DISABLE_SYMLINKS_WARNING",
    ):
        monkeypatch.delenv(name, raising=False)

    configure_runtime_environment()

    assert os.environ["HF_HOME"] == str(runtime_root / "cache" / "huggingface")
    assert os.environ["NLTK_DATA"] == str(runtime_root / "cache" / "nltk")
    assert os.environ["TORCH_HOME"] == str(runtime_root / "cache" / "torch")
    assert os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] == "1"
