# LocalVox managed runtime

LocalVox keeps the desktop application separate from the speech-model runtime.

This is deliberate. Speech libraries have heavier and more fragile dependency trees than the UI. A private managed runtime lets LocalVox repair or replace a speech engine without forcing the user to reinstall the desktop application.

## v0.1 runtime contract

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

## User experience target

The public installer should expose one action: **Install Voice Engine**.

The user should not need to install Python, Git, Conda, Docker, WSL, PyTorch, MeloTTS, OpenVoice, or model checkpoints manually.

The installation flow will:

1. create the private LocalVox runtime directory;
2. install a pinned Python runtime that is known to work with the engine;
3. install pinned OpenVoice/MeloTTS dependencies;
4. download and verify OpenVoice V2 checkpoints;
5. install UniDic data;
6. install the LocalVox worker;
7. run a health check;
8. write `runtime.json` only after the health check succeeds.

A failed or interrupted installation must be retryable without damaging saved voice profiles or generated audio.

## Upstream compatibility

OpenVoice's official documentation currently describes a Python 3.9 environment for its developer installation and lists Windows installation as community-supported rather than part of its primary Linux instructions. LocalVox therefore treats the exact runtime as a pinned application dependency rather than assuming the user's existing Python installation is compatible.

## Future engines

The runtime manager is intentionally engine-specific below `runtime/<engine-id>/`. A future DirectML/ONNX engine can live alongside OpenVoice V2 without changing saved voice-profile semantics.
