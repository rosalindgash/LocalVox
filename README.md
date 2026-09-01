# LocalVox

**Save your voice once. Type what you want it to say. Everything stays on your PC.**

LocalVox is a free, source-available, noncommercial Windows desktop application for persistent local voice profiles and scripted narration.

> **Status:** v0.1 development build. The desktop UI, persistent voice library, and two private, CPU-first Windows voice runtimes are implemented. F5-TTS ONNX is the recommended cloning engine; OpenVoice V2 remains available as a fallback.

## Product goals

- Create a named voice once and reuse it later without re-uploading the source audio.
- Generate narration from pasted scripts.
- Keep voice samples, scripts, and outputs local by default.
- Require no account, subscription, telemetry, Docker, or WSL.
- Treat CPU-only Windows machines as supported, not as an afterthought.
- Keep model/runtime details behind an engine interface.

## Current development UI

1. Launch LocalVox.
2. Click **+ Add Voice**.
3. Name the voice and choose a reference recording.
4. LocalVox copies that recording into its managed voice library.
5. The saved voice remains available after the app restarts.
6. Select **F5-TTS ONNX**, install its private runtime, paste a script, and generate.

## Development setup

Requires Python 3.11+.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .[dev]
localvox
```

Run tests:

```powershell
pytest
```

## Voice engines

LocalVox v0.1 uses an engine boundary rather than bundling model weights into the desktop package:

- **F5-TTS ONNX** is the recommended higher-fidelity engine. Its Windows installer downloads a pinned CPU model and registry-free private Python runtime. A blank reference transcript is generated locally once and cached with the prepared reference per voice profile.
- **OpenVoice V2** is the working fallback engine for existing profiles and lower-resource compatibility.

Both runtimes live under LocalVox's application-data directory and require no user-installed Python, Git, FFmpeg, Conda, Docker, WSL, CUDA, or compiler. See `docs/runtime.md` for the runtime contract and `docs/third-party.md` for upstream licenses. LocalVox does not relicense third-party code or model weights.

## License

LocalVox is licensed under the **PolyForm Noncommercial License 1.0.0**.

You may use, study, modify, and redistribute LocalVox for permitted noncommercial purposes under that license. Commercial use is not permitted.

See `LICENSE` and <https://polyformproject.org/licenses/noncommercial/1.0.0/>.

## Voice consent

Only create voice profiles from your own voice or from a speaker who has explicitly authorized you to clone their voice. Do not use LocalVox for fraud, impersonation, harassment, or deceptive representation.
