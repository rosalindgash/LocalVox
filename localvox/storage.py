from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from platformdirs import user_data_dir

APP_NAME = "LocalVox"
APP_AUTHOR = "LocalVox"
logger = logging.getLogger(__name__)


def app_data_root() -> Path:
    root = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    root.mkdir(parents=True, exist_ok=True)
    return root


def voices_root() -> Path:
    path = app_data_root() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def projects_root() -> Path:
    path = app_data_root() / "projects"
    path.mkdir(parents=True, exist_ok=True)
    return path


def outputs_root() -> Path:
    path = app_data_root() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(slots=True)
class VoiceProfile:
    slug: str
    name: str
    reference_audio: str
    transcript: str = ""
    engine: str = "f5-tts-onnx"
    preset: str = "conversational"

    @property
    def directory(self) -> Path:
        return voices_root() / self.slug

    @property
    def metadata_path(self) -> Path:
        return self.directory / "profile.json"

    def save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> VoiceProfile:
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def slugify(value: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "-" for c in value.strip())
    return "-".join(part for part in cleaned.split("-") if part) or "voice"


def create_voice_profile(
    name: str, source_audio: Path, transcript: str = ""
) -> VoiceProfile:
    slug = slugify(name)
    directory = voices_root() / slug
    index = 2
    while directory.exists():
        slug = f"{slugify(name)}-{index}"
        directory = voices_root() / slug
        index += 1
    directory.mkdir(parents=True)
    target = directory / f"reference{source_audio.suffix.lower() or '.wav'}"
    shutil.copy2(source_audio, target)
    profile = VoiceProfile(
        slug=slug,
        name=name.strip(),
        reference_audio=str(target),
        transcript=transcript.strip(),
    )
    profile.save()
    return profile


def list_voice_profiles() -> list[VoiceProfile]:
    profiles: list[VoiceProfile] = []
    for metadata in voices_root().glob("*/profile.json"):
        try:
            profiles.append(VoiceProfile.load(metadata))
        except (OSError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable voice profile %s: %s", metadata, exc)
    return sorted(profiles, key=lambda p: p.name.lower())
