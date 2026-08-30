from pathlib import Path

from localvox import storage


def test_slugify():
    assert storage.slugify("Rosalind Conversational") == "rosalind-conversational"
    assert storage.slugify("  My Voice!! ") == "my-voice"


def test_voice_profile_persists_reference(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(storage, "app_data_root", lambda: tmp_path)
    source = tmp_path / "sample.wav"
    source.write_bytes(b"RIFF-test")

    profile = storage.create_voice_profile("Test Voice", source, "hello world")

    assert profile.name == "Test Voice"
    assert Path(profile.reference_audio).exists()
    assert Path(profile.reference_audio).parent == tmp_path / "voices" / "test-voice"
    loaded = storage.list_voice_profiles()
    assert len(loaded) == 1
    assert loaded[0].transcript == "hello world"


def test_duplicate_voice_names_get_unique_slugs(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(storage, "app_data_root", lambda: tmp_path)
    source = tmp_path / "sample.wav"
    source.write_bytes(b"RIFF-test")

    first = storage.create_voice_profile("Same Voice", source)
    second = storage.create_voice_profile("Same Voice", source)

    assert first.slug == "same-voice"
    assert second.slug == "same-voice-2"
