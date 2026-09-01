from pathlib import Path
from types import SimpleNamespace

import pytest

from localvox.workers import f5_tts_onnx_worker as worker


class FakeSession:
    def __init__(self, contract):
        self.inputs = [
            SimpleNamespace(name=name, type=tensor_type, shape=list(shape))
            for name, tensor_type, shape in contract["inputs"]
        ]
        self.outputs = [
            SimpleNamespace(name=name, type=tensor_type, shape=list(shape))
            for name, tensor_type, shape in contract["outputs"]
        ]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


def fake_sessions():
    return tuple(
        FakeSession(contract) for contract in worker.EXPECTED_GRAPH_CONTRACT.values()
    )


def test_pinned_graph_contract_accepts_exact_signatures():
    worker.validate_graph_contract(fake_sessions())


def test_pinned_graph_contract_rejects_tensor_shape_change():
    sessions = fake_sessions()
    sessions[1].inputs[0].shape[-1] = 80

    with pytest.raises(RuntimeError, match="input contract changed"):
        worker.validate_graph_contract(sessions)


def test_reference_fingerprint_invalidates_audio_or_transcript(tmp_path: Path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"first recording")
    original = worker.reference_fingerprint(reference, "spoken words")

    assert worker.reference_fingerprint(reference, "spoken words") == original
    assert worker.reference_fingerprint(reference, "corrected words") != original

    reference.write_bytes(b"replacement recording")
    assert worker.reference_fingerprint(reference, "spoken words") != original


def test_duration_length_matches_pinned_multilingual_preprocessing():
    assert worker.duration_text_length("hello") == 5
    assert worker.duration_text_length("你好。") == len("你好。".encode()) + 3
