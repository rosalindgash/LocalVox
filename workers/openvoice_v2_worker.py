from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalVox OpenVoice V2 worker")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--language", default="EN_NEWEST")
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def checkpoint_root() -> Path:
    configured = os.getenv("LOCALVOX_OPENVOICE_CHECKPOINTS", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1] / "runtime" / "openvoice" / "checkpoints_v2"


def main() -> int:
    args = parse_args()

    import torch
    from melo.api import TTS
    from openvoice import se_extractor
    from openvoice.api import ToneColorConverter

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

    reference = Path(args.reference).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    converter = ToneColorConverter(str(config), device=device)
    converter.load_ckpt(str(model))
    target_embedding, _ = se_extractor.get_se(str(reference), converter, vad=True)

    tts = TTS(language=args.language, device=device)
    speaker_ids = tts.hps.data.spk2id
    if not speaker_ids:
        raise RuntimeError(f"MeloTTS returned no speakers for {args.language}")

    speaker_name = next(iter(speaker_ids))
    speaker_id = speaker_ids[speaker_name]
    embedding_name = speaker_name.lower().replace("_", "-") + ".pth"
    source_embedding_path = speaker_embeddings / embedding_name
    if not source_embedding_path.exists():
        available = ", ".join(path.name for path in speaker_embeddings.glob("*.pth"))
        raise FileNotFoundError(
            f"No source embedding for speaker {speaker_name!r}. "
            f"Expected {source_embedding_path.name}. Available: {available}"
        )

    source_embedding = torch.load(str(source_embedding_path), map_location=device)

    with tempfile.TemporaryDirectory(prefix="localvox-") as tmp:
        base_audio = Path(tmp) / "base.wav"
        tts.tts_to_file(args.text, speaker_id, str(base_audio), speed=args.speed)
        converter.convert(
            audio_src_path=str(base_audio),
            src_se=source_embedding,
            tgt_se=target_embedding,
            output_path=str(output),
            message="@LocalVox",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
