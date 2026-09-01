from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from localvox.f5_runtime import F5RuntimeManager

ProgressCallback = Callable[[str], None]

PYTHON_VERSION = "3.11.9"
PYTHON_ARCHIVE_URL = (
    "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
)
PYTHON_ARCHIVE_SHA256 = (
    "009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b"
)
GET_PIP_REVISION = "f6f644156f23dfe9acc06e7b9ca75eee311f2e37"
GET_PIP_URL = (
    "https://raw.githubusercontent.com/pypa/get-pip/"
    f"{GET_PIP_REVISION}/public/get-pip.py"
)
GET_PIP_SHA256 = "fb24e693bab954209a063d90953621412ccad4a500905a726286e038f508ddf6"

F5_ONNX_REVISION = "7cfd12d27f0f7bad554c667fe2ebde9328106129"
F5_ONNX_ARCHIVE_URL = (
    "https://huggingface.co/H5N1AIDS/F5-TTS-ONNX/resolve/"
    f"{F5_ONNX_REVISION}/CPU_F32.zip?download=true"
)
F5_ONNX_ARCHIVE_SHA256 = (
    "09c3b2cf37fc51db6d42cafc5d257b457f389209decb8341dedba446d647f0d5"
)
F5_VOCAB_REVISION = "84e5a410d9cead4de2f847e7c9369a6440bdfaca"
F5_VOCAB_URL = (
    "https://huggingface.co/SWivid/F5-TTS/resolve/"
    f"{F5_VOCAB_REVISION}/F5TTS_v1_Base/vocab.txt?download=true"
)
F5_VOCAB_SHA256 = "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
F5_MODEL_SPECS = {
    "F5_Preprocess.onnx": (
        68_549_853,
        "e71d3ed14e90ba3fc1e83560512e86a771e25b6f0be7789b2cf53a5c9ba5617d",
    ),
    "F5_Transformer.onnx": (
        1_321_718_282,
        "c63aeb96953ccae551df6717d892f73a8408e59f1a6fe8edc2aed8063bd195b8",
    ),
    "F5_Decode.onnx": (
        62_550_703,
        "a16fea891beb4889b47e5987b80c841bb996beaccc1d1fe27a1f9323a089a6da",
    ),
    "vocab.txt": (13_800, F5_VOCAB_SHA256),
}
F5_ASR_MODEL = "Systran/faster-whisper-tiny.en"
F5_RUNTIME_DEPS = (
    "numpy==1.26.4",
    "onnxruntime==1.22.1",
    "av==15.1.0",
    "faster-whisper==1.2.0",
    "huggingface-hub==0.36.0",
    "jieba==0.42.1",
    "pypinyin==0.54.0",
)


class F5RuntimeInstallError(RuntimeError):
    """Raised when the managed F5-TTS ONNX runtime cannot be installed."""


class F5RuntimeInstaller:
    """Install and repair LocalVox's private F5-TTS ONNX runtime."""

    engine_id = "f5-tts-onnx"
    display_name = "F5-TTS ONNX"

    def __init__(self, manager: F5RuntimeManager | None = None) -> None:
        self.manager = manager or F5RuntimeManager()

    def install(self, progress: ProgressCallback | None = None) -> None:
        if platform.system() != "Windows":
            raise F5RuntimeInstallError(
                "Automatic F5-TTS installation currently supports Windows only."
            )

        report = progress or (lambda _message: None)
        self.manager.ensure_directories()

        report("Preparing F5-TTS voice engine…")
        python = self._ensure_python(report)

        report("Installing CPU ONNX speech dependencies…")
        self._run(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--only-binary=av",
            *F5_RUNTIME_DEPS,
        )

        report("Preparing F5-TTS ONNX model files…")
        model_dir = self._ensure_models(report)

        report("Installing LocalVox F5-TTS worker…")
        worker = self._install_worker()

        report("Checking F5-TTS and local transcription…")
        self._health_check(python, worker, model_dir)

        self.manager.configure_existing(
            python_executable=python,
            worker_script=worker,
            model_dir=model_dir,
            asr_model=F5_ASR_MODEL,
        )
        report("F5-TTS ONNX is ready.")

    def _ensure_python(self, report: ProgressCallback) -> Path:
        python_dir = self.manager.root / "python"
        python = python_dir / "python.exe"
        if python.exists() and self._python_has_pip(python):
            return python

        downloads = self.manager.root / "downloads"
        archive = downloads / (f"python-{PYTHON_VERSION}-embed-amd64.zip")
        if not archive.exists():
            report(f"Downloading Python {PYTHON_VERSION} runtime…")
            self._download(PYTHON_ARCHIVE_URL, archive)
        self._verify_sha256(archive, PYTHON_ARCHIVE_SHA256)

        report(f"Preparing private Python {PYTHON_VERSION} runtime…")
        if python_dir.exists():
            shutil.rmtree(python_dir)
        python_dir.mkdir(parents=True)
        self._extract_archive(archive, python_dir)
        if not python.exists():
            raise F5RuntimeInstallError(
                f"Python archive did not contain {python.name}."
            )

        path_configuration = python_dir / "python311._pth"
        if not path_configuration.exists():
            raise F5RuntimeInstallError(
                "Embedded Python path configuration is missing."
            )
        lines = path_configuration.read_text(encoding="utf-8").splitlines()
        lines = ["import site" if line == "#import site" else line for line in lines]
        if "Lib\\site-packages" not in lines:
            lines.append("Lib\\site-packages")
        path_configuration.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

        get_pip = downloads / f"get-pip-{GET_PIP_REVISION}.py"
        if not get_pip.exists():
            report("Downloading pinned Python package bootstrap…")
            self._download(GET_PIP_URL, get_pip)
        self._verify_sha256(get_pip, GET_PIP_SHA256)
        self._run(python, str(get_pip))
        if not self._python_has_pip(python):
            raise F5RuntimeInstallError("Private Python package bootstrap failed.")
        return python

    @staticmethod
    def _python_has_pip(python: Path) -> bool:
        result = subprocess.run(
            [str(python), "-m", "pip", "--version"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0

    def _ensure_models(self, report: ProgressCallback) -> Path:
        downloads = self.manager.root / "downloads"
        archive = downloads / f"CPU_F32-{F5_ONNX_REVISION}.zip"
        if not archive.exists():
            report("Downloading F5-TTS ONNX CPU model (about 1.3 GB)…")
            self._download(F5_ONNX_ARCHIVE_URL, archive)
        self._verify_sha256(archive, F5_ONNX_ARCHIVE_SHA256)

        model_dir = self.manager.root / "models" / "f5-v1-cpu-f32"
        marker = model_dir / ".localvox-model"
        if not self._model_complete(model_dir):
            if model_dir.exists():
                shutil.rmtree(model_dir)
            model_dir.mkdir(parents=True)
            self._extract_archive(archive, model_dir)

        vocab = model_dir / "vocab.txt"
        if not vocab.exists() or self._sha256(vocab) != F5_VOCAB_SHA256:
            report("Downloading F5-TTS vocabulary…")
            self._download(F5_VOCAB_URL, vocab)
        self._verify_sha256(vocab, F5_VOCAB_SHA256)

        if not self._model_complete(model_dir):
            raise F5RuntimeInstallError("F5-TTS ONNX model files are incomplete.")
        for name, (_size, expected_sha256) in F5_MODEL_SPECS.items():
            self._verify_sha256(model_dir / name, expected_sha256)
        marker.write_text(F5_ONNX_REVISION, encoding="utf-8")
        return model_dir

    @staticmethod
    def _model_complete(model_dir: Path) -> bool:
        return all(
            (model_dir / relative).exists()
            and (model_dir / relative).stat().st_size == expected_size
            for relative, (expected_size, _sha256) in F5_MODEL_SPECS.items()
        )

    @staticmethod
    def _extract_archive(archive: Path, destination: Path) -> None:
        destination_resolved = destination.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if (
                    target != destination_resolved
                    and destination_resolved not in target.parents
                ):
                    raise F5RuntimeInstallError(
                        f"Unsafe path in F5-TTS model archive: {member.filename}"
                    )
            bundle.extractall(destination)

    def _install_worker(self) -> Path:
        source = Path(__file__).resolve().parent / "workers" / "f5_tts_onnx_worker.py"
        if not source.exists():
            raise F5RuntimeInstallError("Bundled F5-TTS worker is missing.")
        worker = self.manager.root / "f5_tts_onnx_worker.py"
        shutil.copy2(source, worker)
        return worker

    def _health_check(self, python: Path, worker: Path, model_dir: Path) -> None:
        env = self._runtime_environment(model_dir)
        self._run(python, str(worker), "--health-check", env=env)

    def _runtime_environment(self, model_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env["LOCALVOX_F5_MODEL_DIR"] = str(model_dir)
        env["LOCALVOX_F5_RUNTIME_ROOT"] = str(self.manager.root)
        env["LOCALVOX_F5_ASR_MODEL"] = F5_ASR_MODEL
        env["PYTHONUTF8"] = "1"
        return env

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_sha256(cls, path: Path, expected: str) -> None:
        actual = cls._sha256(path)
        if actual.casefold() != expected.casefold():
            raise F5RuntimeInstallError(
                f"Downloaded file failed SHA-256 verification: {path.name}"
            )

    @staticmethod
    def _download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "LocalVox/0.1"},
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as handle,
            ):
                shutil.copyfileobj(response, handle)
            if partial.stat().st_size == 0:
                raise F5RuntimeInstallError(f"Downloaded file is empty: {url}")
            partial.replace(destination)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            raise F5RuntimeInstallError(
                f"Download failed with HTTP {exc.code}: {url}"
            ) from exc
        except urllib.error.URLError as exc:
            partial.unlink(missing_ok=True)
            raise F5RuntimeInstallError(
                f"Could not download {url}: {exc.reason}"
            ) from exc
        except (OSError, F5RuntimeInstallError):
            partial.unlink(missing_ok=True)
            raise

    @staticmethod
    def _run(
        python: Path,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> None:
        command = [str(python), *args]
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
            rendered = " ".join(command)
            raise F5RuntimeInstallError(
                f"Command failed:\n{rendered}" + (f"\n\n{detail}" if detail else "")
            ) from exc
