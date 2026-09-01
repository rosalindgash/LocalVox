# LocalVox managed runtime

LocalVox keeps the desktop application separate from the speech-model runtime.

This is deliberate. Speech libraries have heavier and more fragile dependency trees than the UI. A private managed runtime lets LocalVox repair or replace a speech engine without forcing the user to reinstall the desktop application.

## v0.1 runtime contracts

Each engine owns a separate directory below `runtime/<engine-id>/` and a separate manifest. Installing or repairing one engine does not modify saved voice profiles or the other engine.

### F5-TTS ONNX (recommended)

The F5 runtime provides:

- an official CPython 3.11 embeddable distribution with a pinned package bootstrap;
- CPU-only ONNX Runtime and binary PyAV for in-process audio decoding;
- a pinned three-graph F5-TTS v1 CPU FP32 bundle and pinned official vocabulary;
- faster-whisper tiny.en for optional local reference transcription;
- the LocalVox F5 worker.

The installer verifies the Python archive, package bootstrap, model archive, every extracted ONNX graph, and the vocabulary with SHA-256. Its health check loads the ASR model and all three graphs, then validates the pinned graph contract: 24 kHz PCM16 input/output, 256-sample hop, 100 mel channels, `int32` tokenizer IDs, `int64` duration, all input/output names and tensor shapes, 32 denoising steps, and an already-scaled `int16` decoder output.

For a blank profile transcript, the worker prepares a maximum ten-second speech window and transcribes it locally once. The selected PCM and transcript are cached under `runtime/f5-tts-onnx/cache/voice-profiles/<profile-key>/`. The cache fingerprint includes the reference file bytes, supplied transcript, cache schema, and pinned model revision, so it is reused for ordinary narration and invalidated when any relevant input changes. Automatically derived text is not written into the saved profile.

Developer overrides:

- `LOCALVOX_F5_PYTHON`
- `LOCALVOX_F5_WORKER`
- `LOCALVOX_F5_MODEL_DIR`

### OpenVoice V2 (fallback)

The OpenVoice V2 runtime must provide:

- a private Python executable;
- the LocalVox OpenVoice worker script;
- OpenVoice V2 and its dependencies;
- MeloTTS for base speech generation;
- OpenVoice V2 checkpoints;
- UniDic data required by MeloTTS.

LocalVox records those locations in `runtime.json` under the application's local data directory. Environment variables remain available only as developer overrides:

- `LOCALVOX_OPENVOICE_PYTHON`
- `LOCALVOX_OPENVOICE_WORKER`

## User experience

The public installer should expose one action: **Install Voice Engine**.

The user should not need to install Python, Git, Conda, Docker, WSL, PyTorch, MeloTTS, OpenVoice, or model checkpoints manually.

Each installation flow:

1. create the private LocalVox runtime directory;
2. install a pinned private Python runtime known to work with the selected engine;
3. install pinned CPU dependencies;
4. download and verify the selected engine's models;
5. install any engine-specific language data;
6. install the LocalVox worker;
7. run a health check;
8. write `runtime.json` only after the health check succeeds.

A failed or interrupted installation must be retryable without damaging saved voice profiles or generated audio.

## Upstream compatibility

OpenVoice's official documentation currently describes a Python 3.9 environment for its developer installation and lists Windows installation as community-supported rather than part of its primary Linux instructions. LocalVox therefore treats the exact runtime as a pinned application dependency rather than assuming the user's existing Python installation is compatible.

F5-TTS is deliberately CPU-first. LocalVox does not install or select CUDA, DirectML, Conda, Docker, or WSL providers.
