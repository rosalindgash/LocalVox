# LocalVox

**Save your voice once. Type what you want it to say. Everything stays on your PC.**

LocalVox is a free, source-available, noncommercial Windows desktop application for persistent local voice profiles and scripted narration.

> **Status:** v0.1 development scaffold. The desktop UI, persistent voice library, storage layer, and speech-engine interface are implemented. The OpenVoice V2 worker/runtime still needs to be packaged and wired into the Windows release.

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
6. Paste a script and generate once an engine worker is installed.

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

## OpenVoice V2 adapter

LocalVox v0.1 uses an engine boundary rather than bundling model weights into the desktop package. Set `LOCALVOX_OPENVOICE_WORKER` to the path of the LocalVox OpenVoice worker script once installed.

The upstream OpenVoice project and model/runtime retain their own licenses and notices. LocalVox does not relicense third-party model weights.

## License

LocalVox is licensed under the **PolyForm Noncommercial License 1.0.0**.

You may use, study, modify, and redistribute LocalVox for permitted noncommercial purposes under that license. Commercial use is not permitted.

See `LICENSE` and <https://polyformproject.org/licenses/noncommercial/1.0.0/>.

## Voice consent

Only create voice profiles from your own voice or from a speaker who has explicitly authorized you to clone their voice. Do not use LocalVox for fraud, impersonation, harassment, or deceptive representation.
