from localvox.engines.f5_tts import F5TTSEngine
from localvox.engines.openvoice import OpenVoiceEngine


def engines():
    return {
        "f5-tts-onnx": F5TTSEngine(),
        "openvoice-v2": OpenVoiceEngine(),
    }
