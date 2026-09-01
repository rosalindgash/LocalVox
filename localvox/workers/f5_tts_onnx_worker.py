from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import wave
from pathlib import Path

# The ONNX graph contract and text preprocessing in this worker are adapted
# from DakeQQ/F5-TTS-ONNX (Apache-2.0). See docs/third-party.md.

# These constants describe the exact CPU_F32.zip bundle pinned by LocalVox.
# That older bundle has no ONNX metadata carrier, so changing any of them requires
# pinning and validating a different archive in f5_runtime_installer.py.
MODEL_BUNDLE_ID = "7cfd12d27f0f7bad554c667fe2ebde9328106129"
MODEL_SAMPLE_RATE = 24_000
OUTPUT_SAMPLE_RATE = 24_000
ASR_SAMPLE_RATE = 16_000
HOP_LENGTH = 256
MEL_CHANNELS = 100
CONDITIONING_WIDTH = 612
MAX_REFERENCE_SECONDS = 10
NFE_STEPS = 32
REFERENCE_CACHE_VERSION = 1
ZH_PAUSE_PUNCTUATION = r"[。，、；：？！]"
VOCAB_SHA256 = "2a05f992e00af9b0bd3800a8d23e78d520dbd705284ed2eedb5f4bd29398fa3c"
MODEL_FILE_SHA256 = {
    "F5_Preprocess.onnx": (
        "e71d3ed14e90ba3fc1e83560512e86a771e25b6f0be7789b2cf53a5c9ba5617d"
    ),
    "F5_Transformer.onnx": (
        "c63aeb96953ccae551df6717d892f73a8408e59f1a6fe8edc2aed8063bd195b8"
    ),
    "F5_Decode.onnx": (
        "a16fea891beb4889b47e5987b80c841bb996beaccc1d1fe27a1f9323a089a6da"
    ),
    "vocab.txt": VOCAB_SHA256,
}
MODEL_FILE_SIZES = {
    "F5_Preprocess.onnx": 68_549_853,
    "F5_Transformer.onnx": 1_321_718_282,
    "F5_Decode.onnx": 62_550_703,
    "vocab.txt": 13_800,
}

EXPECTED_GRAPH_CONTRACT = {
    "F5_Preprocess.onnx": {
        "inputs": (
            ("audio", "tensor(int16)", (1, 1, "audio_len")),
            ("text_ids", "tensor(int32)", (1, "text_ids_len")),
            ("max_duration", "tensor(int64)", (1,)),
        ),
        "outputs": (
            (
                "noise",
                "tensor(float)",
                (
                    "RandomNormalLikenoise_dim_0",
                    "max_duration",
                    "RandomNormalLikenoise_dim_2",
                ),
            ),
            ("rope_cos_q", "tensor(float)", (2, 16, "max_duration", 64)),
            ("rope_sin_q", "tensor(float)", (2, 16, "max_duration", 64)),
            ("rope_cos_k", "tensor(float)", (2, 16, 64, "max_duration")),
            ("rope_sin_k", "tensor(float)", (2, 16, 64, "max_duration")),
            ("cat_mel_text", "tensor(float)", (1, "max_duration", CONDITIONING_WIDTH)),
            (
                "cat_mel_text_drop",
                "tensor(float)",
                (
                    "Concatcat_mel_text_drop_dim_0",
                    "max_duration",
                    "Concatcat_mel_text_drop_dim_2",
                ),
            ),
            ("ref_signal_len", "tensor(int64)", ()),
        ),
    },
    "F5_Transformer.onnx": {
        "inputs": (
            ("noise", "tensor(float)", (1, "max_duration", MEL_CHANNELS)),
            ("rope_cos_q", "tensor(float)", (2, 16, "max_duration", 64)),
            ("rope_sin_q", "tensor(float)", (2, 16, "max_duration", 64)),
            ("rope_cos_k", "tensor(float)", (2, 16, 64, "max_duration")),
            ("rope_sin_k", "tensor(float)", (2, 16, 64, "max_duration")),
            ("cat_mel_text", "tensor(float)", (1, "max_duration", CONDITIONING_WIDTH)),
            (
                "cat_mel_text_drop",
                "tensor(float)",
                (1, "max_duration", CONDITIONING_WIDTH),
            ),
            ("time_step.1", "tensor(int32)", (1,)),
        ),
        "outputs": (
            ("denoised", "tensor(float)", (1, "max_duration", MEL_CHANNELS)),
            ("time_step", "tensor(int32)", (1,)),
        ),
    },
    "F5_Decode.onnx": {
        "inputs": (
            ("denoised", "tensor(float)", (1, "max_duration", MEL_CHANNELS)),
            ("ref_signal_len", "tensor(int64)", ()),
        ),
        "outputs": (
            (
                "output_audio",
                "tensor(int16)",
                (1, 1, "ConvTranspose_143_o0__d2 - 1024"),
            ),
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LocalVox F5-TTS ONNX worker")
    parser.add_argument("--health-check", action="store_true")
    parser.add_argument("--reference")
    parser.add_argument("--text")
    parser.add_argument("--output")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--profile-cache")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if not args.health_check:
        missing = [
            option
            for option, value in (
                ("--reference", args.reference),
                ("--text", args.text),
                ("--output", args.output),
                ("--profile-cache", args.profile_cache),
            )
            if not value
        ]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(missing))
    if args.speed <= 0:
        parser.error("--speed must be greater than zero")
    return args


def configure_runtime_environment() -> Path:
    configured = os.getenv("LOCALVOX_F5_RUNTIME_ROOT", "").strip()
    runtime_root = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parent
    )
    cache_root = runtime_root / "cache"
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    return runtime_root


def model_directory() -> Path:
    configured = os.getenv("LOCALVOX_F5_MODEL_DIR", "").strip()
    if not configured:
        raise RuntimeError("LocalVox did not provide an F5-TTS model directory")
    model_dir = Path(configured).expanduser().resolve()
    missing = [
        model_dir / name
        for name in (
            "F5_Preprocess.onnx",
            "F5_Transformer.onnx",
            "F5_Decode.onnx",
            "vocab.txt",
        )
        if not (model_dir / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "F5-TTS ONNX model is incomplete. Missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    return model_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_bundle(model_dir: Path, *, verify_hashes: bool) -> None:
    for name, expected_size in MODEL_FILE_SIZES.items():
        path = model_dir / name
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise RuntimeError(
                f"Pinned F5-TTS model file has the wrong size: {name} "
                f"({actual_size} != {expected_size})"
            )
        if verify_hashes and sha256_file(path) != MODEL_FILE_SHA256[name]:
            raise RuntimeError(f"Pinned F5-TTS model checksum failed: {name}")

    vocab = (model_dir / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if len(vocab) != 2545 or not vocab or vocab[0] != " ":
        raise RuntimeError(
            "Pinned F5-TTS tokenizer contract changed: expected 2,545 tokens "
            "with the space token at index zero"
        )
    if not verify_hashes and sha256_file(model_dir / "vocab.txt") != VOCAB_SHA256:
        raise RuntimeError("Pinned F5-TTS vocabulary checksum failed")


def node_signatures(nodes) -> tuple[tuple[str, str, tuple], ...]:
    return tuple((node.name, node.type, tuple(node.shape)) for node in nodes)


def validate_graph_contract(sessions) -> None:
    for name, session in zip(EXPECTED_GRAPH_CONTRACT, sessions, strict=True):
        expected = EXPECTED_GRAPH_CONTRACT[name]
        actual_inputs = node_signatures(session.get_inputs())
        actual_outputs = node_signatures(session.get_outputs())
        if actual_inputs != expected["inputs"]:
            raise RuntimeError(
                f"Pinned F5-TTS graph input contract changed for {name}: "
                f"{actual_inputs!r}"
            )
        if actual_outputs != expected["outputs"]:
            raise RuntimeError(
                f"Pinned F5-TTS graph output contract changed for {name}: "
                f"{actual_outputs!r}"
            )


def load_asr_model(runtime_root: Path):
    from faster_whisper import WhisperModel

    model_name = os.getenv(
        "LOCALVOX_F5_ASR_MODEL", "Systran/faster-whisper-tiny.en"
    ).strip()
    return WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
        download_root=str(runtime_root / "models" / "asr"),
    )


def load_sessions(model_dir: Path):
    import onnxruntime

    options = onnxruntime.SessionOptions()
    threads = min(max(os.cpu_count() or 1, 1), 8)
    options.inter_op_num_threads = threads
    options.intra_op_num_threads = threads
    options.enable_cpu_mem_arena = True
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    sessions = tuple(
        onnxruntime.InferenceSession(
            str(model_dir / name),
            sess_options=options,
            providers=providers,
        )
        for name in (
            "F5_Preprocess.onnx",
            "F5_Transformer.onnx",
            "F5_Decode.onnx",
        )
    )
    validate_graph_contract(sessions)
    return sessions


def decode_reference(path: Path):
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if not streams:
            raise RuntimeError(f"Reference file contains no audio stream: {path}")
        stream = streams[0]
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=MODEL_SAMPLE_RATE,
        )
        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                chunks.append(converted.to_ndarray().reshape(-1))
        for converted in resampler.resample(None):
            chunks.append(converted.to_ndarray().reshape(-1))
    if not chunks:
        raise RuntimeError(f"Reference file decoded to no audio: {path}")
    return np.concatenate(chunks).astype(np.int16, copy=False)


def select_reference_window(audio):
    import numpy as np

    if audio.size == 0:
        raise RuntimeError("Reference audio is empty")
    absolute = np.abs(audio.astype(np.int32))
    peak = int(absolute.max(initial=0))
    if peak == 0:
        raise RuntimeError("Reference audio is silent")
    threshold = max(256, int(peak * 0.02))
    voiced = np.flatnonzero(absolute >= threshold)
    if not voiced.size:
        raise RuntimeError("Reference audio contains no detectable speech")
    padding = int(MODEL_SAMPLE_RATE * 0.2)
    start = max(0, int(voiced[0]) - padding)
    voiced_end = min(audio.size, int(voiced[-1]) + padding)
    maximum = MODEL_SAMPLE_RATE * MAX_REFERENCE_SECONDS
    end = min(voiced_end, start + maximum)
    selected = audio[start:end]
    if selected.size < MODEL_SAMPLE_RATE:
        raise RuntimeError("Reference speech must be at least one second long")
    return selected


def resample_for_asr(audio):
    import numpy as np

    source = audio.astype(np.float32) / 32768.0
    output_length = max(1, round(source.size * ASR_SAMPLE_RATE / MODEL_SAMPLE_RATE))
    source_positions = np.arange(source.size, dtype=np.float64)
    target_positions = np.linspace(0, source.size - 1, output_length)
    return np.interp(target_positions, source_positions, source).astype(np.float32)


def transcribe_reference(audio, runtime_root: Path) -> str:
    model = load_asr_model(runtime_root)
    segments, _info = model.transcribe(
        resample_for_asr(audio),
        language="en",
        beam_size=5,
        vad_filter=True,
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    del model
    gc.collect()
    if not transcript:
        raise RuntimeError(
            "Local transcription could not detect speech in the reference audio"
        )
    return transcript


def reference_fingerprint(path: Path, supplied_transcript: str) -> str:
    digest = hashlib.sha256()
    digest.update(f"localvox-f5-reference-v{REFERENCE_CACHE_VERSION}\0".encode())
    digest.update(MODEL_BUNDLE_ID.encode())
    digest.update(b"\0")
    digest.update(supplied_transcript.strip().encode("utf-8"))
    digest.update(b"\0")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cached_reference(cache_dir: Path, fingerprint: str):
    import numpy as np

    manifest_path = cache_dir / "prepared-reference.json"
    audio_path = cache_dir / "prepared-reference.npy"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("cache_version") != REFERENCE_CACHE_VERSION
            or manifest.get("model_bundle") != MODEL_BUNDLE_ID
            or manifest.get("fingerprint") != fingerprint
        ):
            return None
        transcript = manifest["transcript"].strip()
        audio = np.load(audio_path, allow_pickle=False)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        not transcript
        or audio.dtype != np.int16
        or audio.ndim != 1
        or not MODEL_SAMPLE_RATE
        <= audio.size
        <= MODEL_SAMPLE_RATE * MAX_REFERENCE_SECONDS
    ):
        return None
    return audio, transcript


def save_cached_reference(
    cache_dir: Path,
    fingerprint: str,
    audio,
    transcript: str,
) -> None:
    import numpy as np

    cache_dir.mkdir(parents=True, exist_ok=True)
    audio_path = cache_dir / "prepared-reference.npy"
    manifest_path = cache_dir / "prepared-reference.json"
    temporary_audio = cache_dir / f"prepared-reference.{os.getpid()}.npy.tmp"
    temporary_manifest = cache_dir / f"prepared-reference.{os.getpid()}.json.tmp"
    try:
        with temporary_audio.open("wb") as handle:
            np.save(handle, np.ascontiguousarray(audio, dtype=np.int16))
        temporary_audio.replace(audio_path)
        temporary_manifest.write_text(
            json.dumps(
                {
                    "cache_version": REFERENCE_CACHE_VERSION,
                    "model_bundle": MODEL_BUNDLE_ID,
                    "fingerprint": fingerprint,
                    "sample_rate": MODEL_SAMPLE_RATE,
                    "transcript": transcript,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_audio.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)


def prepare_reference(
    reference: Path,
    supplied_transcript: str,
    runtime_root: Path,
    cache_dir: Path,
):
    fingerprint = reference_fingerprint(reference, supplied_transcript)
    cached = load_cached_reference(cache_dir, fingerprint)
    if cached is not None:
        return cached

    audio = select_reference_window(decode_reference(reference))
    transcript = supplied_transcript.strip() or transcribe_reference(
        audio,
        runtime_root,
    )
    save_cached_reference(cache_dir, fingerprint, audio, transcript)
    return audio, transcript


def convert_char_to_pinyin(text_list: list[str]) -> list[list[str]]:
    import jieba
    from pypinyin import Style, lazy_pinyin

    if jieba.dt.initialized is False:
        jieba.default_logger.setLevel(50)
        jieba.initialize()

    final_text_list = []
    translation = str.maketrans({";": ",", "“": '"', "”": '"', "‘": "'", "’": "'"})

    def is_chinese(character: str) -> bool:
        return "\u3100" <= character <= "\u9fff"

    for text in text_list:
        characters = []
        for segment in jieba.cut(text.translate(translation)):
            byte_length = len(segment.encode("utf-8"))
            if byte_length == len(segment):
                if characters and byte_length > 1 and characters[-1] not in " :'\"":
                    characters.append(" ")
                characters.extend(segment)
            elif byte_length == 3 * len(segment):
                pinyin = lazy_pinyin(segment, style=Style.TONE3, tone_sandhi=True)
                for index, character in enumerate(segment):
                    if is_chinese(character):
                        characters.append(" ")
                    characters.append(pinyin[index])
            else:
                for character in segment:
                    if ord(character) < 256:
                        characters.append(character)
                    elif is_chinese(character):
                        characters.append(" ")
                        characters.extend(
                            lazy_pinyin(
                                character,
                                style=Style.TONE3,
                                tone_sandhi=True,
                            )
                        )
                    else:
                        characters.append(character)
        final_text_list.append(characters)
    return final_text_list


def numpy_dtype(onnx_type: str):
    import numpy as np

    types = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int16)": np.int16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
    }
    try:
        return types[onnx_type]
    except KeyError as exc:
        raise TypeError(f"Unsupported ONNX tensor type: {onnx_type}") from exc


def text_ids(text: str, vocab_path: Path, dtype):
    import numpy as np

    with vocab_path.open(encoding="utf-8") as handle:
        vocabulary = {line.rstrip("\n"): index for index, line in enumerate(handle)}
    characters = convert_char_to_pinyin([text])[0]
    return np.asarray(
        [[vocabulary.get(character, 0) for character in characters]],
        dtype=dtype,
    )


def duration_text_length(text: str) -> int:
    return len(text.encode("utf-8")) + 3 * len(re.findall(ZH_PAUSE_PUNCTUATION, text))


def run_inference(
    sessions,
    model_dir: Path,
    reference_audio,
    reference_text: str,
    generated_text: str,
    speed: float,
):
    import numpy as np

    preprocess, transformer, decoder = sessions
    preprocess_inputs = preprocess.get_inputs()
    audio_argument, text_argument, duration_argument = preprocess_inputs

    audio_dtype = numpy_dtype(audio_argument.type)
    if np.issubdtype(audio_dtype, np.floating):
        audio = reference_audio.astype(np.float32) / 32768.0
    else:
        audio = reference_audio.astype(audio_dtype, copy=False)
    audio = audio.reshape(1, 1, -1)

    reference_text = reference_text.strip()
    generated_text = generated_text.strip()
    if not reference_text or not generated_text:
        raise RuntimeError("Reference and generated text must not be empty")
    if reference_text[-1].isascii():
        reference_text += " "
    combined_text = reference_text + generated_text
    encoded = text_ids(
        combined_text,
        model_dir / "vocab.txt",
        numpy_dtype(text_argument.type),
    )

    reference_frames = reference_audio.size // HOP_LENGTH + 1
    reference_length = max(1, duration_text_length(reference_text))
    generated_length = duration_text_length(generated_text)
    maximum_duration = reference_frames + int(
        reference_frames * generated_length / reference_length / speed
    )
    duration = np.asarray([maximum_duration], dtype=numpy_dtype(duration_argument.type))

    preprocess_values = preprocess.run(
        None,
        {
            audio_argument.name: audio,
            text_argument.name: encoded,
            duration_argument.name: duration,
        },
    )
    preprocess_outputs = {
        argument.name: value
        for argument, value in zip(preprocess.get_outputs(), preprocess_values)
    }
    transformer_inputs = {
        argument.name: argument for argument in transformer.get_inputs()
    }
    state = preprocess_outputs["noise"]
    step = np.asarray(
        [0],
        dtype=numpy_dtype(transformer_inputs["time_step.1"].type),
    )
    for _index in range(NFE_STEPS - 1):
        state, step = transformer.run(
            None,
            {
                "noise": state,
                "rope_cos_q": preprocess_outputs["rope_cos_q"],
                "rope_sin_q": preprocess_outputs["rope_sin_q"],
                "rope_cos_k": preprocess_outputs["rope_cos_k"],
                "rope_sin_k": preprocess_outputs["rope_sin_k"],
                "cat_mel_text": preprocess_outputs["cat_mel_text"],
                "cat_mel_text_drop": preprocess_outputs["cat_mel_text_drop"],
                "time_step.1": step,
            },
        )

    generated = decoder.run(
        None,
        {
            "denoised": state,
            "ref_signal_len": preprocess_outputs["ref_signal_len"],
        },
    )[0]
    if generated.dtype != np.int16:
        raise RuntimeError(
            "Pinned F5-TTS decoder must return PCM int16; refusing implicit scaling"
        )
    return np.ascontiguousarray(generated.reshape(-1), dtype=np.int16)


def main() -> int:
    args = parse_args()
    runtime_root = configure_runtime_environment()
    model_dir = model_directory()
    validate_model_bundle(model_dir, verify_hashes=args.health_check)

    if args.health_check:
        asr = load_asr_model(runtime_root)
        del asr
        gc.collect()
        load_sessions(model_dir)
        return 0

    reference = Path(args.reference).expanduser().resolve()
    if not reference.exists():
        raise FileNotFoundError(f"Reference audio not found: {reference}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.profile_cache).expanduser().resolve()
    reference_audio, reference_text = prepare_reference(
        reference,
        args.transcript,
        runtime_root,
        cache_dir,
    )
    sessions = load_sessions(model_dir)
    generated = run_inference(
        sessions,
        model_dir,
        reference_audio,
        reference_text,
        args.text,
        args.speed,
    )

    # The pinned decoder returns fully scaled PCM int16. Write those samples
    # byte-for-byte; applying float normalization here would corrupt the output.
    with wave.open(str(output), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(OUTPUT_SAMPLE_RATE)
        wav_file.writeframes(generated.astype("<i2", copy=False).tobytes())
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("F5-TTS completed without creating a WAV file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
