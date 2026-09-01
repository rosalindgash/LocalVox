from __future__ import annotations

import os
import platform
import shutil
import subprocess
import urllib.error
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

# The historical S3 checkpoint ZIP used by OpenVoice has intermittently returned
# 403/404 responses. MyShell also publishes the same V2 model files from its
# official Hugging Face repository, so LocalVox downloads the pinned files there
# instead of depending on the brittle ZIP endpoint.
OPENVOICE_V2_MODEL_REVISION = "fd981100305a0e4291f93a9ad169c6d9f7bed54a"
OPENVOICE_V2_MODEL_BASE_URL = (
    "https://huggingface.co/myshell-ai/OpenVoiceV2/resolve/"
    f"{OPENVOICE_V2_MODEL_REVISION}"
)
OPENVOICE_V2_MODEL_FILES = (
    "converter/config.json",
    "converter/checkpoint.pth",
    "base_speakers/ses/en-au.pth",
    "base_speakers/ses/en-br.pth",
    "base_speakers/ses/en-default.pth",
    "base_speakers/ses/en-india.pth",
    "base_speakers/ses/en-newest.pth",
    "base_speakers/ses/en-us.pth",
    "base_speakers/ses/es.pth",
    "base_speakers/ses/fr.pth",
    "base_speakers/ses/jp.pth",
    "base_speakers/ses/kr.pth",
    "base_speakers/ses/zh.pth",
)

OPENVOICE_RUNTIME_DEPS = (
    "numpy==1.23.5",
    "librosa==0.9.1",
    "soundfile>=0.12",
    "eng_to_ipa==0.0.2",
    "inflect==7.0.0",
    "unidecode==1.3.7",
    "pypinyin==0.50.0",
    "cn2an==0.5.22",
    "jieba==0.42.1",
    # g2p_en 2.1.0 downloads the pre-NLTK-3.9 perceptron resource name.
    "nltk==3.8.1",
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

        report("Installing OpenVoice runtime dependencies…")
        self._run(
            python,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            *OPENVOICE_RUNTIME_DEPS,
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
            "--no-deps",
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

        # MeloTTS installs both unidic and unidic-lite. On Windows, mecab-python3
        # prefers the full `unidic` package when it is importable, but that package
        # does not contain a dictionary until `python -m unidic download` succeeds.
        # The downloader itself uses a Unix-only multiprocessing "fork" context.
        # Removing the empty full package lets mecab-python3 fall back to the
        # bundled unidic-lite dictionary, which MeloTTS's Japanese frontend also
        # documents as its required dictionary.
        report("Selecting Windows pronunciation dictionary…")
        self._run(
            python,
            "-m",
            "pip",
            "uninstall",
            "--yes",
            "unidic",
        )

        report("Checking pronunciation data…")
        self._ensure_pronunciation_data(python)

        report("Downloading OpenVoice model files…")
        checkpoints = self._ensure_checkpoints(report)

        report("Installing LocalVox speech worker…")
        worker = self._install_worker()

        report("Checking voice engine…")
        self._health_check(python, worker, checkpoints)

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
            if self._is_installable_source(target):
                return target
            shutil.rmtree(target)

        archive = self.manager.root / "downloads" / f"{name}-{commit}.zip"
        if not archive.exists():
            report(f"Downloading {name}…")
            self._download(url, archive)

        if target.exists():
            shutil.rmtree(target)

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
        if not self._is_installable_source(target):
            raise RuntimeInstallError(
                f"{name} source archive did not contain setup.py or pyproject.toml at its root."
            )
        marker.write_text(commit, encoding="utf-8")
        return target

    @staticmethod
    def _is_installable_source(path: Path) -> bool:
        return (path / "pyproject.toml").exists() or (path / "setup.py").exists()

    def _ensure_pronunciation_data(self, python: Path) -> None:
        env = os.environ.copy()
        env["NLTK_DATA"] = str(self.manager.root / "cache" / "nltk")
        env["PYTHONUTF8"] = "1"
        code = (
            "import unidic_lite; from pathlib import Path; "
            "p=Path(unidic_lite.DICDIR); "
            "assert p.exists() and (p/'dicrc').exists(), "
            "f'UniDic Lite dictionary missing: {p}'; "
            "import MeCab; tagger=MeCab.Tagger(); tagger.parse('test'); "
            "import nltk, os; target=os.environ['NLTK_DATA']; "
            "resources=('averaged_perceptron_tagger','cmudict'); "
            "assert all(nltk.download(item, download_dir=target, quiet=True) "
            "for item in resources), 'Could not install NLTK pronunciation data'"
        )
        self._run(python, "-c", code, env=env)

    def _ensure_checkpoints(self, report: ProgressCallback) -> Path:
        checkpoints = self.manager.root / "models" / "checkpoints_v2"
        missing = [
            relative
            for relative in OPENVOICE_V2_MODEL_FILES
            if not (checkpoints / relative).exists()
        ]
        if not missing:
            return checkpoints

        checkpoints.mkdir(parents=True, exist_ok=True)
        total = len(missing)
        for index, relative in enumerate(missing, start=1):
            report(f"Downloading OpenVoice model file {index} of {total}…")
            destination = checkpoints / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"{OPENVOICE_V2_MODEL_BASE_URL}/{relative}?download=true"
            self._download(url, destination)

        still_missing = [
            relative
            for relative in OPENVOICE_V2_MODEL_FILES
            if not (checkpoints / relative).exists()
            or (checkpoints / relative).stat().st_size == 0
        ]
        if still_missing:
            raise RuntimeInstallError(
                "OpenVoice V2 model files are incomplete after download: "
                + ", ".join(still_missing)
            )
        return checkpoints

    def _install_worker(self) -> Path:
        source = Path(__file__).resolve().parent / "workers" / "openvoice_v2_worker.py"
        if not source.exists():
            raise RuntimeInstallError("Bundled OpenVoice worker is missing.")
        worker = self.manager.root / "openvoice_v2_worker.py"
        shutil.copy2(source, worker)
        return worker

    def _health_check(
        self,
        python: Path,
        worker: Path,
        checkpoints: Path,
    ) -> None:
        env = os.environ.copy()
        env["LOCALVOX_OPENVOICE_CHECKPOINTS"] = str(checkpoints)
        env["LOCALVOX_RUNTIME_ROOT"] = str(self.manager.root)
        env["PYTHONUTF8"] = "1"
        self._run(python, str(worker), "--health-check", env=env)

    @staticmethod
    def _download(url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        if partial.exists():
            partial.unlink()
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
                raise RuntimeInstallError(f"Downloaded file is empty: {url}")
            partial.replace(destination)
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeInstallError(
                f"Download failed with HTTP {exc.code}: {url}"
            ) from exc
        except urllib.error.URLError as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeInstallError(
                f"Could not download {url}: {exc.reason}"
            ) from exc
        except (OSError, RuntimeInstallError):
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
            raise RuntimeInstallError(
                f"Command failed:\n{rendered}"
                + (f"\n\n{detail}" if detail else "")
            ) from exc
