import sys
import json
import platform
import webbrowser
from pathlib import Path

import requests

from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtCore import Qt, QTimer

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMessageBox,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QTabBar,
    QVBoxLayout,
    QWidget
)

from tabs.view_tab import ViewTab
from tabs.centre_tab import CentreTab
from tabs.cave_tab import CaveTab
from tabs.radial_tab import RadialTab
from tabs.azimuthal_tab import AzimuthalTab
from tabs.hermans_tab import HermansTab
from tabs.datplot_tab import DatPlotTab


APP_NAME = "LRPhoton 1"
APP_VERSION = "1.0.1"
UPDATE_INFO_URL = "https://raw.githubusercontent.com/nathan38p/LRPhoton-releases/main/version.json"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(APP_NAME)
        self.resize(1300, 700)

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(24, 18, 24, 24)
        main_layout.setSpacing(10)

        # ============================================================
        # HEADER
        # ============================================================

        header = QFrame()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 0, 8, 0)
        header_layout.setSpacing(20)

        logo_path = Path(__file__).parent / "assets" / "LRP.svg"

        logo = QSvgWidget(str(logo_path))
        logo.setFixedSize(42, 42)

        title = QLabel(APP_NAME)
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 700;
                color: #333333;
            }
        """)

        subtitle = QLabel("SAXS / WAXS data processing")
        subtitle.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #777777;
            }
        """)

        title_box = QVBoxLayout()
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(0)
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        header_layout.addWidget(logo)
        header_layout.addLayout(title_box)

        # ============================================================
        # TAB BAR IN HEADER
        # ============================================================

        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setMovable(False)
        self.tab_bar.setUsesScrollButtons(True)

        self.tab_bar.addTab("View")
        self.tab_bar.addTab("Plot")
        self.tab_bar.addTab("Centre")
        self.tab_bar.addTab("Cave")
        self.radial_tab_index = self.tab_bar.addTab("Radial")
        self.tab_bar.setTabEnabled(self.radial_tab_index, False)
        self.tab_bar.setTabToolTip(self.radial_tab_index, "Right-click to unlock radial integration.")
        self.tab_bar.addTab("Azimuthal")
        self.tab_bar.addTab("Anisotropy")
        self.tab_bar.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tab_bar.customContextMenuRequested.connect(self.unlock_radial_from_right_click)

        header_layout.addStretch()
        header_layout.addWidget(self.tab_bar)
        header_layout.addStretch()

        self.version_label = QLabel(f"Version {APP_VERSION} · Checking for updates…")
        self.version_label.setStyleSheet("""
            QLabel {
                font-size: 11px;
                color: #777777;
                padding: 4px 8px;
                border-radius: 8px;
                background: #f2f2f2;
            }
        """)
        header_layout.addWidget(self.version_label)

        main_layout.addWidget(header)

        # ============================================================
        # PAGE CONTENT FULL WIDTH
        # ============================================================

        self.pages = QStackedWidget()

        self.view_tab = ViewTab()
        self.datplot_tab = DatPlotTab()
        self.centre_tab = CentreTab()
        self.cave_tab = CaveTab()
        self.radial_tab = RadialTab()
        self.azimuthal_tab = AzimuthalTab()
        self.hermans_tab = HermansTab()
        self.view_tab.set_q_geometry_source_tab(self.azimuthal_tab)

        self.pages.addWidget(self.view_tab)
        self.pages.addWidget(self.datplot_tab)
        self.pages.addWidget(self.centre_tab)
        self.pages.addWidget(self.cave_tab)
        self.pages.addWidget(self.radial_tab)
        self.pages.addWidget(self.azimuthal_tab)
        self.pages.addWidget(self.hermans_tab)

        # Folder synchronisation between tabs using a folder browser.
        # When one tab changes folder, all the others are updated too.
        self.folder_synced_tabs = [
            self.view_tab,
            self.datplot_tab,
            self.radial_tab,
            self.azimuthal_tab,
            self.hermans_tab,
        ]

        default_folder = str(Path.home())
        for tab in self.folder_synced_tabs:
            if hasattr(tab, "set_folder_from_external_tab"):
                tab.set_folder_from_external_tab(default_folder)

        for source_tab in self.folder_synced_tabs:
            for target_tab in self.folder_synced_tabs:
                if source_tab is target_tab:
                    continue
                source_tab.folder_changed.connect(target_tab.set_folder_from_external_tab)

        self.tab_bar.currentChanged.connect(self.pages.setCurrentIndex)

        main_layout.addWidget(self.pages)

        self.setCentralWidget(container)

    def unlock_radial_from_right_click(self, position):
        if self.tab_bar.tabAt(position) != self.radial_tab_index:
            return

        self.tab_bar.setTabEnabled(self.radial_tab_index, True)
        self.tab_bar.setTabText(self.radial_tab_index, "Radial")
        self.tab_bar.setTabToolTip(self.radial_tab_index, "")
        self.tab_bar.setCurrentIndex(self.radial_tab_index)

    def _version_tuple(self, version: str) -> tuple:
        parts = []
        for item in version.strip().replace("v", "").split("."):
            try:
                parts.append(int(item))
            except ValueError:
                parts.append(0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts[:3])

    def check_for_updates(self):
        try:
            response = requests.get(UPDATE_INFO_URL, timeout=3)
            data = response.json()

            latest_version = str(data.get("version", "")).strip()
            message = str(data.get("message", "A new version of LRPhoton is available.")).strip()

            system_name = platform.system()
            if system_name == "Darwin":
                download_url = str(data.get("macos_download", "")).strip()
            elif system_name == "Windows":
                download_url = str(data.get("windows_download", "")).strip()
            else:
                download_url = str(data.get("download_url", "")).strip()

            if not download_url:
                download_url = str(data.get("download_url", "https://github.com/nathan38p/LRPhoton-releases/releases/latest")).strip()

            if not latest_version:
                self.version_label.setText(f"Version {APP_VERSION} · Update status unavailable")
                return

            if self._version_tuple(latest_version) <= self._version_tuple(APP_VERSION):
                self.version_label.setText(f"Version {APP_VERSION} · Up to date")
                return

            self.version_label.setText(f"Version {APP_VERSION} · Update available: {latest_version}")

            box = QMessageBox(self)
            box.setWindowTitle("Update available")
            box.setIcon(QMessageBox.Information)
            box.setText("A new version of LRPhoton is available.")
            box.setInformativeText(
                f"Installed version: {APP_VERSION}\n"
                f"Available version: {latest_version}\n\n"
                f"{message}"
            )
            box.setStandardButtons(QMessageBox.Open | QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Open)

            if box.exec() == QMessageBox.Open:
                webbrowser.open(download_url)

        except Exception:
            # Update checking must never prevent the application from starting.
            self.version_label.setText(f"Version {APP_VERSION} · Update status unavailable")


def main():
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        QFrame {
            background: transparent;
        }

        QStackedWidget {
            background: transparent;
        }

        QTabBar {
            background: transparent;
        }

        QTabWidget::pane {
            border: none;
            background: transparent;
        }

        QTabBar::tab {
            padding: 7px 18px;
            margin-right: 2px;
            border-radius: 8px;
            border: none;
            background: #eeeeee;
            color: #222222;
            font-size: 13px;
        }

        QTabBar::tab:disabled {
            background: #f5f5f5;
            color: #9a9a9a;
        }

        QTabBar::tab:selected {
            background: #007aff;
            color: white;
        }

        QTabBar::tab:hover:!selected {
            background: #dddddd;
        }
    """)

    window = MainWindow()
    window.show()
    QTimer.singleShot(1200, window.check_for_updates)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
