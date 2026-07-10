from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DistancesTab(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.addStretch(1)

        title = QLabel("Distances")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: 700;")
        layout.addWidget(title)

        message = QLabel("This feature is coming soon.")
        message.setAlignment(Qt.AlignCenter)
        message.setStyleSheet("font-size: 15px; color: #6b7280;")
        layout.addWidget(message)

        layout.addStretch(1)
