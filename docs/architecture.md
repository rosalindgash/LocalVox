# Architecture

LocalVox separates the Windows desktop product from speech engines.

- `localvox/ui/`: desktop experience.
- `localvox/storage.py`: persistent named voice profiles and managed local data paths.
- `localvox/engines/`: adapters for local voice-cloning engines.
- `workers/`: speech-runtime worker processes kept outside the UI process.
- `localvox/projects/`: project/session persistence planned for v0.2.
- `localvox/audio/`: chunking, normalization, joining and export planned for v0.2.

## Engine boundary

Each engine implements `TTSEngine.status()` and `TTSEngine.generate()`.
The desktop application does not require a speech model to be imported into the GUI process. This keeps model/runtime failures isolated and allows LocalVox to support multiple engines later.

A saved voice profile owns its managed reference-audio path. After enrollment, the user should never have to locate or upload the original sample again.

## OpenVoice V2 worker

The first worker uses OpenVoice V2 for tone-color transfer and MeloTTS for the base speech signal. Its model runtime is intentionally separate from the desktop application's Python environment because upstream speech libraries may require different Python/PyTorch versions.

The worker accepts a reference path, text, and output path from the LocalVox engine adapter. Checkpoints remain external third-party artifacts and keep their upstream licenses.

## Privacy

Application data is stored under the operating system's LocalVox user-data directory. LocalVox itself requires no account, telemetry, or cloud upload.
