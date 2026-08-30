from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QWidget,
)


class AddVoiceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Voice")
        self.setMinimumWidth(560)

        self.name_edit = QLineEdit()
        self.audio_edit = QLineEdit()
        self.audio_edit.setReadOnly(True)
        self.transcript_edit = QTextEdit()
        self.transcript_edit.setPlaceholderText("Optional reference transcript")
        self.transcript_edit.setMaximumHeight(120)

        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        audio_row = QWidget()
        audio_layout = QHBoxLayout(audio_row)
        audio_layout.setContentsMargins(0, 0, 0, 0)
        audio_layout.addWidget(self.audio_edit)
        audio_layout.addWidget(browse)

        self.consent = QCheckBox(
            "I confirm this is my voice or I have explicit permission from the speaker to clone it."
        )

        form = QFormLayout(self)
        form.addRow("Voice name", self.name_edit)
        form.addRow("Reference audio", audio_row)
        form.addRow("Transcript", self.transcript_edit)
        form.addRow(QLabel("Consent"), self.consent)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Save).setEnabled(False)
        self.consent.toggled.connect(self.buttons.button(QDialogButtonBox.Save).setEnabled)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)

    def _browse(self):
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Choose reference audio",
            "",
            "Audio (*.wav *.mp3 *.flac *.m4a);;All Files (*)",
        )
        if filename:
            self.audio_edit.setText(filename)

    def values(self) -> tuple[str, Path, str]:
        return (
            self.name_edit.text().strip(),
            Path(self.audio_edit.text().strip()),
            self.transcript_edit.toPlainText().strip(),
        )
