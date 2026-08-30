from __future__ import annotations

from datetime import UTC, datetime

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from localvox.engines.registry import engines
from localvox.storage import create_voice_profile, list_voice_profiles, outputs_root
from localvox.ui.add_voice_dialog import AddVoiceDialog


class GenerationThread(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, engine, profile, text: str, output):
        super().__init__()
        self.engine = engine
        self.profile = profile
        self.text = text
        self.output = output

    def run(self):
        try:
            result = self.engine.generate(
                voice=self.profile,
                text=self.text,
                output_path=self.output,
            )
        except Exception as exc:  # noqa: BLE001 - worker boundary must surface engine failures
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(str(result))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LocalVox")
        self.resize(900, 650)
        self.engine_map = engines()
        self.voice_profiles = []
        self.generation_thread = None

        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setSpacing(12)

        title = QLabel("LocalVox")
        title.setStyleSheet("font-size: 26px; font-weight: 600;")
        subtitle = QLabel("Save your voice once. Type what you want it to say. Everything stays on your PC.")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        voice_row = QHBoxLayout()
        voice_row.addWidget(QLabel("Voice"))
        self.voice_combo = QComboBox()
        self.voice_combo.currentIndexChanged.connect(self.refresh_engine_status)
        voice_row.addWidget(self.voice_combo, 1)
        add_voice = QPushButton("+ Add Voice")
        add_voice.clicked.connect(self.add_voice)
        voice_row.addWidget(add_voice)
        layout.addLayout(voice_row)

        self.engine_label = QLabel()
        layout.addWidget(self.engine_label)

        layout.addWidget(QLabel("Script"))
        self.script = QPlainTextEdit()
        self.script.setPlaceholderText("Paste or type your narration here…")
        layout.addWidget(self.script, 1)

        actions = QHBoxLayout()
        self.generate_button = QPushButton("Generate Narration")
        self.generate_button.clicked.connect(self.generate)
        actions.addWidget(self.generate_button)
        open_outputs = QPushButton("Open Output Folder")
        open_outputs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(outputs_root()))))
        actions.addWidget(open_outputs)
        actions.addStretch()
        layout.addLayout(actions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

        self.setCentralWidget(root)
        self.refresh_voices()
        self.refresh_engine_status()

    def refresh_voices(self):
        previous_slug = self.voice_combo.currentData()
        self.voice_profiles = list_voice_profiles()
        self.voice_combo.blockSignals(True)
        self.voice_combo.clear()
        for profile in self.voice_profiles:
            self.voice_combo.addItem(profile.name, profile.slug)
        if not self.voice_profiles:
            self.voice_combo.addItem("No saved voices yet", None)
        elif previous_slug:
            index = self.voice_combo.findData(previous_slug)
            if index >= 0:
                self.voice_combo.setCurrentIndex(index)
        self.voice_combo.blockSignals(False)

    def selected_profile(self):
        slug = self.voice_combo.currentData()
        return next((profile for profile in self.voice_profiles if profile.slug == slug), None)

    def refresh_engine_status(self, *_):
        profile = self.selected_profile()
        if profile is None:
            self.engine_label.setText("Engine: waiting for a saved voice")
            self.generate_button.setEnabled(False)
            return
        engine = self.engine_map.get(profile.engine)
        if engine is None:
            self.engine_label.setText(f"Engine unavailable: {profile.engine}")
            self.generate_button.setEnabled(False)
            return
        status = engine.status()
        self.engine_label.setText(f"Engine: {engine.display_name} — {status.message}")
        self.generate_button.setEnabled(status.available)

    def add_voice(self):
        dialog = AddVoiceDialog(self)
        if dialog.exec() != dialog.Accepted:
            return
        name, audio_path, transcript = dialog.values()
        if not name:
            QMessageBox.warning(self, "Voice name required", "Give this voice a name.")
            return
        if not audio_path.exists():
            QMessageBox.warning(self, "Reference audio required", "Choose an existing reference audio file.")
            return
        try:
            profile = create_voice_profile(name, audio_path, transcript)
        except OSError as exc:
            QMessageBox.critical(self, "Could not save voice", str(exc))
            return
        self.refresh_voices()
        index = self.voice_combo.findData(profile.slug)
        if index >= 0:
            self.voice_combo.setCurrentIndex(index)
        self.refresh_engine_status()
        self.status.setText(f"Saved voice profile: {name}")

    def generate(self):
        profile = self.selected_profile()
        if profile is None:
            return
        text = self.script.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Nothing to generate", "Enter a narration script first.")
            return
        engine = self.engine_map[profile.engine]
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        filename = f"{profile.slug}-{timestamp}.wav"
        output = outputs_root() / filename

        self.status.setText("Generating…")
        self.progress.setRange(0, 0)
        self.generate_button.setEnabled(False)
        self.generation_thread = GenerationThread(engine, profile, text, output)
        self.generation_thread.succeeded.connect(self._generation_succeeded)
        self.generation_thread.failed.connect(self._generation_failed)
        self.generation_thread.finished.connect(self._generation_finished)
        self.generation_thread.start()

    def _generation_succeeded(self, path: str):
        self.status.setText(f"Saved: {path}")
        QMessageBox.information(self, "Narration generated", f"Saved to:\n{path}")

    def _generation_failed(self, error: str):
        QMessageBox.critical(self, "Generation failed", error)
        self.status.setText("Generation failed")

    def _generation_finished(self):
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.generation_thread = None
        self.refresh_engine_status()
