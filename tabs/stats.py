

import requests
from datetime import datetime, timezone, timedelta

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

STATS_URL = "https://lrphoton-stats.lrphoton-stats.workers.dev/stats"
INSTALLS_URL = "https://lrphoton-stats.lrphoton-stats.workers.dev/installs"


class LRPhotonStatsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("LRPhoton statistics")
        self.resize(1120, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("LRPhoton usage statistics")
        title.setStyleSheet("font-size: 20px; font-weight: 700; color: #222222;")
        layout.addWidget(title)

        self.summary_label = QLabel("Loading statistics…")
        self.summary_label.setStyleSheet("font-size: 13px; color: #333333;")
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "Install ID",
            "First seen",
            "Last seen",
            "Last update",
            "LRPhoton version",
            "Platform",
            "System version",
            "Country",
            "Language",
            "Channel",
        ])
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        button_row = QHBoxLayout()
        button_row.addStretch()

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.load_statistics)
        button_row.addWidget(refresh_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_row.addWidget(close_button)

        layout.addLayout(button_row)
        self.load_statistics()

    def load_statistics(self):
        try:
            summary = requests.get(STATS_URL, timeout=8).json()
            installs_response = requests.get(INSTALLS_URL, timeout=8)
            installs_response.raise_for_status()
            installs = installs_response.json()
            if isinstance(installs, dict) and "results" in installs:
                installs = installs["results"]
            if not isinstance(installs, list):
                installs = []
        except Exception as exc:
            QMessageBox.warning(self, "Statistics error", f"Could not load LRPhoton statistics:\n{exc}")
            return

        self.update_summary(summary, installs)
        self.update_table(installs)

    def update_summary(self, summary, installs):
        now = datetime.now(timezone.utc)

        def parse_date(value):
            try:
                return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except Exception:
                return None

        active_today = 0
        active_7_days = 0
        active_30_days = 0
        stable_count = 0
        development_count = 0

        for row in installs:
            last_seen = parse_date(row.get("last_seen"))
            if last_seen is not None:
                if last_seen.date() == now.date():
                    active_today += 1
                if last_seen >= now - timedelta(days=7):
                    active_7_days += 1
                if last_seen >= now - timedelta(days=30):
                    active_30_days += 1

            channel = str(row.get("channel", "")).lower()
            if channel == "stable":
                stable_count += 1
            elif channel == "development":
                development_count += 1

        total_installs = summary.get("total_installs", len(installs)) if isinstance(summary, dict) else len(installs)

        self.summary_label.setText(
            f"Installations: {total_installs}   |   "
            f"Active today: {active_today}   |   "
            f"Active 7 days: {active_7_days}   |   "
            f"Active 30 days: {active_30_days}   |   "
            f"Stable: {stable_count}   |   "
            f"Development: {development_count}"
        )

    def update_table(self, installs):
        columns = [
            "install_id",
            "first_seen",
            "last_seen",
            "last_update",
            "current_version",
            "platform",
            "platform_version",
            "country",
            "language",
            "channel",
        ]

        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(installs))

        for row_index, row in enumerate(installs):
            for column_index, key in enumerate(columns):
                value = row.get(key, "")
                item = QTableWidgetItem("" if value is None else str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(row_index, column_index, item)

        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()