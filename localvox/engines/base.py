from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from localvox.storage import VoiceProfile


@dataclass(slots=True)
class EngineStatus:
    available: bool
    message: str


class TTSEngine(ABC):
    id: str
    display_name: str

    @abstractmethod
    def status(self) -> EngineStatus:
        raise NotImplementedError

    @abstractmethod
    def generate(self, *, voice: VoiceProfile, text: str, output_path: Path) -> Path:
        raise NotImplementedError
