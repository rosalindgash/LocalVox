from __future__ import annotations

import platform
import shutil
import subprocess
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from localvox.runtime import RuntimeManager

ProgressCallback = Callable[[str], None]

PYTHON_VERSION = "3.9.13"
PYTHON_INSTALLER_URL = (
    "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe"
)
OPENVOICE_COMMIT = "74a1d147b17a8c3092dd5430504bd83ef6c7eb23"
MELOTTS_COMMIT = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
OPENVOICE_ARCHIVE_URL = (
    f"https://github.com/myshell-ai/OpenVoice/archive/{OPENVOICE_COMMIT}.zip"
)
MELOTTS_ARCHIVE_URL = (
    f"https://github.com/myshell-ai/MeloTTS/archive/{MELOTTS_COMMIT}.zip"
)
CHECKPOINTS_URL = (
    "https://myshell-public-repo-host.s3.amazonaws.com/"
    "openvoice/checkpoints_v2_0417.zip"
)


class RuntimeInstallError(RuntimeError):
    """Raised when the managed OpenVoice runtime cannot be installed."""


class RuntimeInstaller:
    """Installs and repairs LocalVox's private OpenVoice V2 runtime."""

    def __init__(self, manager: RuntimeManager | None = None) -> None:
        self.manager = manager or RuntimeManager()

    def install(self, progress: ProgressCallback | None = None) -> None:
        if platform.system() != "Windows":
            raise RuntimeInstallError(
                "Automatic voice-engine installation currently supports Windows only."
            )

        report = progress or (lambda _message: None)
        self.manager.ensure_directories()

        report("Preparing LocalVox voice engine…")
        python = self._ensure_python(report)

        report("Installing CPU speech dependencies…")
        self._run(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--index-url",
            "https://download.pytorch.org/whl/cpu",
            "torch==2.5.1",
            "torchaudio==2.5.1",
        )

        report("Installing OpenVoice…")
        openvoice_source = self._ensure_source(
            "OpenVoice",
            OPENVOICE_ARCHIVE_URL,
            OPENVOICE_COMMIT,
            report,
        )
        self._run(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(openvoice_source),
        )

        report("Installing MeloTTS…")
        melo_source = self._ensure_source(
            "MeloTTS",
            MELOTTS_ARCHIVE_URL,
            MELOTTS_COMMIT,
            report,
        )
        self._run(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            str(melo_source),
        )

        report("Installing pronunciation data…")
        self._run(python, "-m", "unidic", "download")

        report("Downloading OpenVoice model files…")
        checkpoints = self._ensure_checkpoints(report)

        report("Installing LocalVox speech worker…")
        worker = self._install_worker()

        report("Checking voice engine…")
        self._health_check(python)

        self.manager.configure_existing(
            python_executable=python,
            worker_script=worker,
            checkpoints_dir=checkpoints,
            openvoice_dir=openvoice_source,
            melo_ready=True,
        )
        report("OpenVoice V2 is ready.")

    def _ensure_python(self, report: ProgressCallback) -> Path:
        python_dir = self.manager.root / "python"
        python = python_dir / "python.exe"
        if python.exists():
            return python

        installer = self.manager.root / "downloads" / f"python-{PYTHON_VERSION}-amd64.exe"
        if not installer.exists():
            report(f"Downloading Python {PYTHON_VERSION} runtime…")
            self._download(PYTHON_INSTALLER_URL, installer)

        report(f"Installing private Python {PYTHON_VERSION} runtime…")
        command = [
            str(installer),
            "/quiet",
            "InstallAllUsers=0",
            f"TargetDir={python_dir}",
            "Include_doc=0",
            "Include_test=0",
            "Include_launcher=0",
            "InstallLauncherAllUsers=0",
            "AssociateFiles=0",
            "Shortcuts=0",
            "PrependPath=0",
            "Include_pip=1",
        ]
        subprocess.run(command, check=True)
        if not python.exists():
            raise RuntimeInstallError(
                f"Python installer completed but {python} was not created."
            )

        self._run(python, "-m", "pip", "install", "--upgrade", "pip")
        return python

    def _ensure_source(
        self,
        name: str,
        url: str,
        commit: str,
        report: ProgressCallback,
    ) -> Path:
        sources = self.manager.root / "sources"
        sources.mkdir(parents=True, exist_ok=True)
        target = sources / name
        marker = target / ".localvox-source"
        if target.exists() and marker.exists() and marker.read_text().strip() == commit:
            return target

        archive = self.manager.root / "downloads" / f"{name}-{commit}.zip"
        if not archive.exists():
            report(f"Downloading {name}…")
            self._download(url, archive)

        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)

        with zipfile.ZipFile(archive) as bundle:
            members = bundle.namelist()
            if not members:
                raise RuntimeInstallError(f"{name} archive is empty.")
            top_level = members[0].split("/", 1)[0]
            bundle.extractall(sources)
        extracted = sources / top_level
        if not extracted.exists():
            raise RuntimeInstallError(f"Could not unpack {name}.")
        shutil.move(str(extracted), str(target))
        marker.write_text(commit, encoding="utf-8")
        return target

    def _ensure_checkpoints(self, report: ProgressCallback) -> Path:
        models = self.manager.root / "models"
        checkpoints = models / "checkpoints_v2"
        required = (
            checkpoints / "converter" / "config.json",
            checkpoints / "converter" / "checkpoint.pth",
            checkpoints / "base_speakers" / "ses",
        )
        if all(path.exists() for path in required):
            return checkpoints

        archive = self.manager.root / "downloads" / "checkpoints_v2_0417.zip"
        if not archive.exists():
            report("Downloading OpenVoice V2 checkpoints…")
            self._download(CHECKPOINTS_URL, archive)

        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(models)

        if not all(path.exists() for path in required):
            missing = ", ".join(str(path) for path in required if not path.exists())
            raise RuntimeInstallError(
                f"OpenVoice checkpoints are incomplete after extraction: {missing}"
            )
        return checkpoints

    def _install_worker(self) -> Path:
        source = Path(__file__).resolve().parent / "workers" / "openvoice_v2_worker.py"
        if not source.exists():
            raise RuntimeInstallError("Bundled OpenVoice worker is missing.")
        worker = self.manager.root / "openvoice_v2_worker.py"
        shutil.copy2(source, worker)
        return worker

    @staticmethod
    def _health_check(python: Path) -> None:
        code = (
            "import torch; import openvoice; import melo; "
            "from openvoice.api import ToneColorConverter; "
            "from melo.api import TTS"
        )
        try:
            subprocess.run(
                [str(python), "-c", code],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeInstallError(
                "OpenVoice health check failed"
                + (f": {detail}" if detail else ".")
            ) from exc

    @staticmethod
    def _download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists():
            partial.unlink()
        try:
            urllib.request.urlretrieve(url, partial)
            if partial.stat().st_size == 0:
                raise RuntimeInstallError(f"Downloaded file is empty: {url}")
            partial.replace(destination)
        except (OSError, RuntimeInstallError):
            partial.unlink(missing_ok=True)
            raise

    @staticmethod
    def _run(python: Path, *args: str) -> None:
        subprocess.run([str(python), *args], check=True)
