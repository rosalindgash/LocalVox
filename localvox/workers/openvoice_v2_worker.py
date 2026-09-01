from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalVox OpenVoice V2 worker")
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--text")
    parser.add_argument("--output")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--language", default="EN_NEWEST")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if not args.health_check:
        missing = [
            option
            for option, value in (
                ("--reference", args.reference),
                ("--text", args.text),
                ("--output", args.output),
            )
            if not value
        ]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(missing))
    return args


def checkpoint_root() -> Path:
    configured = os.getenv("LOCALVOX_OPENVOICE_CHECKPOINTS", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    raise RuntimeError("LocalVox did not provide an OpenVoice checkpoint directory")


def configure_runtime_environment() -> None:
    configured = os.getenv("LOCALVOX_RUNTIME_ROOT", "").strip()
    runtime_root = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parent
    )
    cache_root = runtime_root / "cache"
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["NLTK_DATA"] = str(cache_root / "nltk")
    os.environ["TORCH_HOME"] = str(cache_root / "torch")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    # Windows GUI processes commonly inherit a legacy console encoding. Keep
    # dependency progress output from turning a successful run into an encoding
    # exception when it contains non-ASCII text.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def create_converter(config: Path, device: str):
    from openvoice.api import OpenVoiceBaseClass, ToneColorConverter

    class WatermarkFreeToneColorConverter(ToneColorConverter):
        def __init__(self, config_path: str, *, device: str) -> None:
            # The pinned OpenVoice V2 implementation forwards its
            # ``enable_watermark`` keyword to OpenVoiceBaseClass, whose
            # constructor rejects it. Initialize the documented base model
            # directly and explicitly disable the optional wavmark component.
            OpenVoiceBaseClass.__init__(self, config_path, device=device)
            self.watermark_model = None
            self.version = getattr(self.hps, "_version_", "v1")

    return WatermarkFreeToneColorConverter(str(config), device=device)


def main() -> int:
    args = parse_args()
    configure_runtime_environment()

    import torch
    from melo.api import TTS

    checkpoints = checkpoint_root()
    converter_dir = checkpoints / "converter"
    speaker_embeddings = checkpoints / "base_speakers" / "ses"
    config = converter_dir / "config.json"
    model = converter_dir / "checkpoint.pth"

    missing = [path for path in (config, model, speaker_embeddings) if not path.exists()]
    if missing:
        rendered = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "OpenVoice V2 checkpoints are incomplete. Missing:\n" + rendered
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    converter = create_converter(config, device)
    converter.load_ckpt(str(model))
    tts = TTS(language=args.language, device=device)
    speaker_ids = tts.hps.data.spk2id
    if not speaker_ids:
        raise RuntimeError(f"MeloTTS returned no speakers for {args.language}")

    speaker_name, speaker_id = next(iter(speaker_ids.items()))
    embedding_name = speaker_name.lower().replace("_", "-") + ".pth"
    source_embedding_path = speaker_embeddings / embedding_name
    if not source_embedding_path.exists():
        available = ", ".join(path.name for path in speaker_embeddings.glob("*.pth"))
        raise FileNotFoundError(
            f"No source embedding for speaker {speaker_name!r}. "
            f"Expected {source_embedding_path.name}. Available: {available}"
        )

    source_embedding = torch.load(str(source_embedding_path), map_location=device)

    if args.health_check:
        with tempfile.TemporaryDirectory(prefix="localvox-health-") as tmp:
            health_audio = Path(tmp) / "base.wav"
            tts.tts_to_file(
                "LocalVox health check.",
                speaker_id,
                str(health_audio),
                quiet=True,
            )
            if not health_audio.exists() or health_audio.stat().st_size == 0:
                raise RuntimeError("MeloTTS health check did not create audio")
        return 0

    reference = Path(args.reference).expanduser().resolve()
    if not reference.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference}")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target_embedding = converter.extract_se([str(reference)])

    with tempfile.TemporaryDirectory(prefix="localvox-") as tmp:
        base_audio = Path(tmp) / "base.wav"
        tts.tts_to_file(args.text, speaker_id, str(base_audio), speed=args.speed)
        converter.convert(
            audio_src_path=str(base_audio),
            src_se=source_embedding,
            tgt_se=target_embedding,
            output_path=str(output),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
