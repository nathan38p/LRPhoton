from pathlib import Path
import fnmatch
import shutil

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QFrame,
    QGridLayout,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QBrush, QColor, QKeySequence, QShortcut

from tabs.ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    PAGE_MARGINS,
)
from tabs.file_ratings import install_file_rating_menu, set_item_file_path, should_hide_file_in_browser

try:
    import fabio
    from fabio.edfimage import EdfImage
except Exception:
    fabio = None
    EdfImage = None

try:
    import h5py
except Exception:
    h5py = None


GEOMETRY_HEADER_FIELDS = [
    ("Center X", "px", ["Center_1", "center_1", "Center1", "CenterX", "center_x", "BeamCenterX", "beam_center_x", "beam_x", "Beam_x", "direct_beam_x", "x_beam", "poni1", "PONI1"]),
    ("Center Y", "px", ["Center_2", "center_2", "Center2", "CenterY", "center_y", "BeamCenterY", "beam_center_y", "beam_y", "Beam_y", "direct_beam_y", "y_beam", "poni2", "PONI2"]),
    ("Pixel size X", "m", ["PSize_1", "psize_1", "PSize_X", "PSizeX", "PSize1", "PixelSizeX", "PixelSize1", "PixelSize_1", "pixel_size_x", "pixel_size_1", "x_pixel_size", "pixelsize_x", "detector_pixel_size_x", "x_pixel_size_m"]),
    ("Pixel size Y", "m", ["PSize_2", "psize_2", "PSize_Y", "PSizeY", "PSize2", "PixelSizeY", "PixelSize2", "PixelSize_2", "pixel_size_y", "pixel_size_2", "y_pixel_size", "pixelsize_y", "detector_pixel_size_y", "y_pixel_size_m"]),
    ("Sample distance", "m", ["SampleDistance", "sample_distance", "sampledistance", "Distance", "distance", "DetectorDistance", "detector_distance", "detectorDistance", "SDD", "sdd", "LDet", "Ldet"]),
    ("Wavelength", "m", ["WaveLength", "Wavelength", "wave_length", "wavelength", "Lambda", "lambda", "incident_wavelength", "beam_wavelength", "energy_wavelength"]),
    ("Sample thickness", "m", ["SampleThickness", "sample_thickness", "sample_thick", "Thickness", "thickness", "Sample_Thickness", "sample_length", "SampleLength", "sample_thickness_m"]),
    ("Exposure time", "s", ["ExposureTime", "Exposure", "exposure", "exposure_time", "count_time", "CountTime", "counttime", "acquisition_time", "AcquisitionTime", "integration_time", "IntegrationTime"]),
    ("Transmitted flux", "counts/s", ["TransmittedFlux", "transmitted_flux", "Monitor", "monitor", "Flux", "flux", "IncidentFlux", "incident_flux", "Transmission", "transmission", "I0", "i0"]),
]


class HeaderEditorTab(QWidget):
    folder_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_folder = Path.home()
        self._syncing_folder = False
        self.current_file = None
        self.current_file_kind = None
        self.current_h5_dataset_path = None
        self.current_data = None
        self.current_header = {}
        self.saved_header_snapshot = {}
        self.saved_table_snapshot = []
        self.last_table_snapshot = []
        self.undo_stack = []
        self.user_has_unsaved_edits = False
        self._reverting_file_selection = False
        self._updating_table_colors = False
        self._restoring_table = False
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(content_layout, 1)

        left_panel = QWidget()
        left_panel.setFixedWidth(FILE_BROWSER_WIDTH)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(BLOCK_SPACING)
        content_layout.addWidget(left_panel, 0)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(BLOCK_SPACING)
        content_layout.addWidget(center_panel, 1)

        file_box = QGroupBox("File browser")
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_layout.setSpacing(6)
        left_layout.addWidget(file_box, 1)

        self.folder_edit = QLineEdit(str(self.current_folder))
        self.folder_edit.setPlaceholderText("Folder")
        self.folder_edit.returnPressed.connect(self.folder_from_text)
        file_layout.addWidget(self.folder_edit)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        file_layout.addWidget(browse_button)

        filters_layout = QGridLayout()
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setHorizontalSpacing(4)
        filters_layout.setVerticalSpacing(6)
        self.name_filter = QLineEdit("*")
        self.name_filter.returnPressed.connect(self.refresh_files)
        self.extension_filter = QLineEdit("*.edf *.h5 *.hdf5")
        self.extension_filter.returnPressed.connect(self.refresh_files)
        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extension_filter, 1, 1)
        file_layout.addLayout(filters_layout)

        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(4)
        self.subfolders_checkbox = QCheckBox("Show subfolders")
        self.subfolders_checkbox.toggled.connect(self.refresh_files)
        options_layout.addWidget(self.subfolders_checkbox)
        options_layout.addStretch(1)
        file_layout.addLayout(options_layout)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentItemChanged.connect(self.on_file_selection_changed)
        file_layout.addWidget(self.file_list, 1)

        tools_box = QGroupBox("Header tools")
        tools_box.setStyleSheet(GROUP_BOX_STYLE)
        tools_box_layout = QVBoxLayout(tools_box)
        tools_box_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        tools_box_layout.setSpacing(6)
        left_layout.addWidget(tools_box, 0)

        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.reload_current_file)
        tools_box_layout.addWidget(self.reload_button)

        self.add_row_button = QPushButton("Add header key")
        self.add_row_button.clicked.connect(self.add_header_row)
        tools_box_layout.addWidget(self.add_row_button)

        self.remove_row_button = QPushButton("Remove selected key")
        self.remove_row_button.clicked.connect(self.remove_selected_rows)
        tools_box_layout.addWidget(self.remove_row_button)

        self.save_copy_button = QPushButton("Save as copy")
        self.save_copy_button.clicked.connect(self.save_as_copy)
        tools_box_layout.addWidget(self.save_copy_button)

        geometry_box = QGroupBox("Main parameters")
        geometry_box.setStyleSheet(GROUP_BOX_STYLE)
        geometry_layout = QVBoxLayout(geometry_box)
        geometry_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        geometry_layout.setSpacing(6)
        center_layout.addWidget(geometry_box, 0)

        self.geometry_table = QTableWidget(0, 4)
        self.geometry_table.setHorizontalHeaderLabels(["Field", "Header key", "Value", "Expected unit"])
        self.geometry_table.verticalHeader().setVisible(False)
        self.geometry_table.verticalHeader().setDefaultSectionSize(22)
        self.geometry_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.geometry_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.geometry_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.geometry_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.geometry_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.geometry_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.configure_table_header(self.geometry_table)
        geometry_table_height = (
            self.geometry_table.horizontalHeader().height()
            + self.geometry_table.verticalHeader().defaultSectionSize() * len(GEOMETRY_HEADER_FIELDS)
            + 6
        )
        self.geometry_table.setFixedHeight(geometry_table_height)
        geometry_layout.addWidget(self.geometry_table)

        editor_box = QGroupBox("Full header")
        editor_box.setStyleSheet(GROUP_BOX_STYLE)
        editor_layout = QVBoxLayout(editor_box)
        editor_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        editor_layout.setSpacing(6)
        center_layout.addWidget(editor_box, 1)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Key", "Value"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.configure_table_header(self.table)
        editor_layout.addWidget(self.table, 1)

        self.status_label = QLabel("")
        self.table.itemChanged.connect(self.on_full_header_item_changed)
        self.undo_shortcut = QShortcut(QKeySequence.Undo, self.table)
        self.undo_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        self.undo_shortcut.activated.connect(self.undo_last_header_change)
        self.update_geometry_table()

        self.update_buttons()

    def configure_table_header(self, table):
        header = table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignCenter)
        header.setHighlightSections(False)
        header.setStyleSheet("""
            QHeaderView::section {
                background-color: #f4f4f4;
                color: #222222;
                font-weight: 700;
                border: 0px;
                border-right: 1px solid #c7c7c7;
                border-bottom: 1px solid #d8d8d8;
                padding: 4px 8px;
            }
        """)

    def folder_from_text(self):
        self.set_folder(self.folder_edit.text().strip())

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", self.folder_edit.text() or str(Path.home()))
        if folder:
            self.set_folder(folder)

    def set_folder(self, folder):
        if folder is None:
            return
        path = Path(folder).expanduser()
        if not path.exists():
            return
        self.current_folder = path
        self.folder_edit.setText(str(path))
        self.refresh_files()

    def refresh_files(self):
        self.file_list.clear()
        folder = Path(self.folder_edit.text()).expanduser()
        if not folder.exists():
            return

        self.current_folder = folder
        if not self._syncing_folder:
            self.folder_changed.emit(str(folder))

        name_pattern = self.name_filter.text().strip() or "*"
        extension_patterns = self.extension_filter.text().split() or ["*.edf"]
        iterator = folder.rglob("*") if self.subfolders_checkbox.isChecked() else folder.glob("*")

        files = []
        for path in iterator:
            if not path.is_file():
                continue
            if should_hide_file_in_browser(path):
                continue
            lower_name = path.name.lower()
            if not any(fnmatch.fnmatch(lower_name, pattern.lower()) for pattern in extension_patterns):
                continue
            if not fnmatch.fnmatch(path.name.lower(), name_pattern.lower()):
                continue
            files.append(path)

        for path in sorted(files):
            display_name = str(path.relative_to(folder)) if self.subfolders_checkbox.isChecked() else path.name
            item = QListWidgetItem(display_name)
            set_item_file_path(item, path)
            self.file_list.addItem(item)

    def on_file_selection_changed(self, current, previous):
        if self._reverting_file_selection:
            return
        path = self.path_from_file_item(current)
        if path is None:
            return
        if self.current_file is not None and Path(path) != self.current_file and self.has_unsaved_changes() and not self.confirm_discard_unsaved_changes():
            self._reverting_file_selection = True
            if previous is not None:
                self.file_list.setCurrentItem(previous)
            else:
                self.select_current_file_in_list()
            self._reverting_file_selection = False
            return
        self.load_file(path)

    def path_from_file_item(self, item):
        if item is None:
            return None
        path = item.data(Qt.UserRole)
        if path is None:
            path = Path(self.current_folder) / item.text()
        return Path(path)

    def selected_file(self):
        item = self.file_list.currentItem()
        if item is None:
            selected = self.file_list.selectedItems()
            item = selected[0] if selected else None
        return self.path_from_file_item(item)

    def reload_current_file(self):
        if self.current_file is not None:
            if self.has_unsaved_changes() and not self.confirm_discard_unsaved_changes():
                return
            self.load_file(self.current_file)

    def load_file(self, filename):
        path = Path(filename)
        suffix = path.suffix.lower()
        if suffix in (".h5", ".hdf5"):
            self.load_h5_file(path)
            return

        if suffix != ".edf":
            QMessageBox.warning(self, "Unsupported file", f"Unsupported file type: {path.suffix}")
            return

        if fabio is None:
            self.status_label.setText("fabio is required to read EDF files.")
            return

        try:
            edf = fabio.open(str(path))
            try:
                data = np.asarray(edf.data)
                header = {str(key): str(value) for key, value in dict(edf.header).items()}
            finally:
                try:
                    edf.close()
                except Exception:
                    pass
        except Exception as exc:
            QMessageBox.warning(self, "EDF read error", str(exc))
            return

        self.current_file = path
        self.current_file_kind = "edf"
        self.current_h5_dataset_path = None
        self.current_folder = path.parent
        self.current_data = data
        self.current_header = header
        self.saved_header_snapshot = dict(header)
        self.populate_table(header)
        self.status_label.clear()
        self.folder_changed.emit(str(path.parent))
        self.update_buttons()

    def load_h5_file(self, path):
        if h5py is None:
            self.status_label.setText("h5py is required to read H5 files.")
            return

        try:
            with h5py.File(path, "r") as h5:
                dataset_path = self.find_first_h5_image_dataset_path(h5)
                dataset = h5[dataset_path] if dataset_path else None
                header = self.extract_h5_header(h5, dataset)
        except Exception as exc:
            QMessageBox.warning(self, "H5 read error", str(exc))
            return

        self.current_file = path
        self.current_file_kind = "h5"
        self.current_h5_dataset_path = dataset_path
        self.current_folder = path.parent
        self.current_data = None
        self.current_header = header
        self.saved_header_snapshot = dict(header)
        self.populate_table(header)
        self.status_label.clear()
        self.folder_changed.emit(str(path.parent))
        self.update_buttons()

    def find_first_h5_image_dataset_path(self, h5):
        dataset_path = None

        def visitor(name, obj):
            nonlocal dataset_path
            if dataset_path is not None:
                return
            if isinstance(obj, h5py.Dataset) and len(obj.shape) >= 2:
                dataset_path = name

        h5.visititems(visitor)
        return dataset_path

    def header_value_to_text(self, value):
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return self.header_value_to_text(value.item())
            if value.size == 1:
                return self.header_value_to_text(value.reshape(-1)[0])
            return np.array2string(value, separator=", ")
        return str(value)

    def extract_h5_header(self, h5, dataset):
        header = {}
        for key, value in h5.attrs.items():
            header[f"file/{key}"] = self.header_value_to_text(value)
        if dataset is not None:
            for key, value in dataset.attrs.items():
                header[f"dataset/{key}"] = self.header_value_to_text(value)
        return header

    def populate_table(self, header):
        self._restoring_table = True
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for key in sorted(header):
            self.add_header_row(key, header[key])
        self.table.blockSignals(False)
        self._restoring_table = False
        self.last_table_snapshot = self.table_snapshot()
        self.saved_table_snapshot = list(self.last_table_snapshot)
        self.undo_stack.clear()
        self.user_has_unsaved_edits = False
        self.update_geometry_table()
        self.update_full_header_dirty_rows()

    def add_header_row(self, key="", value=""):
        if not self._restoring_table:
            self.push_current_table_snapshot_for_undo()
        row = self.table.rowCount()
        signals_were_blocked = self.table.blockSignals(True)
        self.table.insertRow(row)
        key_item = QTableWidgetItem(str(key))
        value_item = QTableWidgetItem(str(value))
        self.table.setItem(row, 0, key_item)
        self.table.setItem(row, 1, value_item)
        self.table.blockSignals(signals_were_blocked)
        self.table.setCurrentCell(row, 0 if not key else 1)
        if not self._restoring_table:
            self.last_table_snapshot = self.table_snapshot()
            self.user_has_unsaved_edits = self.last_table_snapshot != self.saved_table_snapshot
        self.update_buttons()
        if hasattr(self, "geometry_table"):
            self.update_geometry_table()
            self.update_full_header_dirty_rows()

    def remove_selected_rows(self):
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        if rows:
            self.push_current_table_snapshot_for_undo()
        for row in rows:
            self.table.removeRow(row)
        if rows:
            self.last_table_snapshot = self.table_snapshot()
            self.user_has_unsaved_edits = self.last_table_snapshot != self.saved_table_snapshot
        self.update_buttons()
        self.update_geometry_table()
        self.update_full_header_dirty_rows()

    def on_full_header_item_changed(self, item):
        if self._updating_table_colors or self._restoring_table:
            return
        previous_snapshot = self.last_table_snapshot
        current_snapshot = self.table_snapshot()
        if previous_snapshot and current_snapshot != previous_snapshot:
            self.undo_stack.append(previous_snapshot)
        self.last_table_snapshot = current_snapshot
        focus_widget = QApplication.focusWidget()
        edit_came_from_table = (
            focus_widget is self.table
            or focus_widget is self.table.viewport()
            or (focus_widget is not None and self.table.isAncestorOf(focus_widget))
        )
        if edit_came_from_table:
            self.user_has_unsaved_edits = current_snapshot != self.saved_table_snapshot
        self.update_geometry_table()
        self.update_full_header_dirty_rows()

    def table_snapshot(self):
        snapshot = []
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            key = "" if key_item is None else key_item.text()
            value = "" if value_item is None else value_item.text()
            snapshot.append((key, value))
        return snapshot

    def push_current_table_snapshot_for_undo(self):
        snapshot = self.table_snapshot()
        if snapshot != self.last_table_snapshot:
            self.last_table_snapshot = snapshot
        self.undo_stack.append(snapshot)

    def restore_table_snapshot(self, snapshot):
        self._restoring_table = True
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(0)
            for row, (key, value) in enumerate(snapshot):
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(str(key)))
                self.table.setItem(row, 1, QTableWidgetItem(str(value)))
        finally:
            self.table.blockSignals(False)
            self._restoring_table = False
        self.last_table_snapshot = self.table_snapshot()
        self.user_has_unsaved_edits = self.last_table_snapshot != self.saved_table_snapshot
        self.update_geometry_table()
        self.update_full_header_dirty_rows()
        self.update_buttons()

    def undo_last_header_change(self):
        if not self.undo_stack:
            return
        snapshot = self.undo_stack.pop()
        self.restore_table_snapshot(snapshot)

    def has_unsaved_changes(self):
        return self.user_has_unsaved_edits and self.table_snapshot() != self.saved_table_snapshot

    def confirm_discard_unsaved_changes(self):
        answer = QMessageBox.warning(
            self,
            "Unsaved header changes",
            "Some header changes have not been saved as a copy.\nOpen another file and discard these changes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def select_current_file_in_list(self):
        if self.current_file is None:
            return
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            item_path = item.data(Qt.UserRole)
            if item_path is not None and Path(item_path) == self.current_file:
                self.file_list.setCurrentItem(item)
                return

    def is_full_header_row_dirty(self, row):
        snapshot = self.table_snapshot()
        if row < 0 or row >= len(snapshot):
            return False
        if row >= len(self.saved_table_snapshot):
            return True
        return snapshot[row] != self.saved_table_snapshot[row]

    def update_full_header_dirty_rows(self):
        if not hasattr(self, "table"):
            return
        dirty_color = QColor("#ffdca8")
        self._updating_table_colors = True
        self.table.blockSignals(True)
        try:
            for row in range(self.table.rowCount()):
                dirty = self.is_full_header_row_dirty(row)
                brush = QBrush(dirty_color) if dirty else QBrush()
                for column in range(self.table.columnCount()):
                    item = self.table.item(row, column)
                    if item is not None:
                        item.setBackground(brush)
        finally:
            self.table.blockSignals(False)
            self._updating_table_colors = False

    def header_map_from_table(self):
        header = {}
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            key = "" if key_item is None else key_item.text().strip()
            if not key:
                continue
            value = "" if value_item is None else value_item.text().strip()
            header[key] = value
        return header

    def find_header_value(self, header, aliases):
        lower_lookup = {str(key).lower(): key for key in header}
        for alias in aliases:
            if alias in header:
                return alias, header[alias]
            match = lower_lookup.get(str(alias).lower())
            if match is not None:
                return match, header[match]
            for key in header:
                short_key = str(key).split("/")[-1]
                if short_key.lower() == str(alias).lower():
                    return key, header[key]
                if short_key.lower() == f"edf_header_{str(alias).lower()}":
                    return key, header[key]
        return None, None

    def update_geometry_table(self):
        if not hasattr(self, "geometry_table"):
            return

        header = self.header_map_from_table()
        present_color = QColor("#d8f5dc")

        self.geometry_table.setRowCount(len(GEOMETRY_HEADER_FIELDS))
        for row, (field, unit, aliases) in enumerate(GEOMETRY_HEADER_FIELDS):
            found_key, value = self.find_header_value(header, aliases)
            present = found_key is not None and str(value).strip() != ""

            values = [
                field,
                found_key if present else "",
                str(value) if present else "",
                unit,
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsEnabled)
                if present:
                    item.setBackground(present_color)
                self.geometry_table.setItem(row, column, item)

    def header_from_table(self):
        header = {}
        duplicates = []
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            value_item = self.table.item(row, 1)
            key = "" if key_item is None else key_item.text().strip()
            value = "" if value_item is None else value_item.text().strip()
            if not key:
                continue
            if key in header:
                duplicates.append(key)
            header[key] = value
        if duplicates:
            duplicate_text = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate header key(s): {duplicate_text}")
        return header

    def save_as_copy(self):
        if self.current_file is None:
            return
        suffix = self.current_file.suffix.lower()
        default_path = self.current_file.with_name(f"{self.current_file.stem}_editedheader{suffix}")
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save header copy",
            str(default_path),
            "Image files (*.edf *.h5 *.hdf5);;EDF files (*.edf);;HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if filename:
            output_path = Path(filename)
            if self.current_file_kind == "h5":
                self.write_h5(output_path, copy_source=True)
            else:
                self.write_edf(output_path)

    def write_edf(self, output_path):
        if EdfImage is None:
            self.status_label.setText("fabio EDF support is not available.")
            return
        if self.current_data is None:
            return
        try:
            header = self.header_from_table()
            edf = EdfImage(data=np.asarray(self.current_data), header=header)
            edf.write(str(output_path))
        except Exception as exc:
            QMessageBox.warning(self, "EDF save error", str(exc))
            return

        self.saved_header_snapshot = self.header_from_table()
        self.last_table_snapshot = self.table_snapshot()
        self.saved_table_snapshot = list(self.last_table_snapshot)
        self.undo_stack.clear()
        self.user_has_unsaved_edits = False
        self.update_full_header_dirty_rows()
        self.status_label.setText(f"Saved EDF: {output_path}")

    def write_h5(self, output_path, copy_source):
        if h5py is None:
            self.status_label.setText("h5py support is not available.")
            return
        if self.current_file is None:
            return

        output_path = Path(output_path)
        try:
            if copy_source:
                shutil.copy2(self.current_file, output_path)
            self.apply_h5_header(output_path)
        except Exception as exc:
            QMessageBox.warning(self, "H5 save error", str(exc))
            return

        self.saved_header_snapshot = self.header_from_table()
        self.last_table_snapshot = self.table_snapshot()
        self.saved_table_snapshot = list(self.last_table_snapshot)
        self.undo_stack.clear()
        self.user_has_unsaved_edits = False
        self.update_full_header_dirty_rows()
        self.status_label.setText(f"Saved H5: {output_path}")

    def apply_h5_header(self, output_path):
        header = self.header_from_table()
        with h5py.File(output_path, "r+") as h5:
            dataset = None
            if self.current_h5_dataset_path and self.current_h5_dataset_path in h5:
                dataset = h5[self.current_h5_dataset_path]
            for key, value in header.items():
                if key.startswith("file/"):
                    h5.attrs[key.split("/", 1)[1]] = value
                elif key.startswith("dataset/"):
                    if dataset is None:
                        continue
                    dataset.attrs[key.split("/", 1)[1]] = value
                else:
                    h5.attrs[key] = value
                    if dataset is not None:
                        dataset.attrs[key] = value

    def set_folder_from_external_tab(self, folder):
        if folder:
            path = Path(folder)
            if path.exists():
                self._syncing_folder = True
                self.current_folder = path
                self.folder_edit.setText(str(path))
                self.refresh_files()
                self._syncing_folder = False

    def update_buttons(self):
        has_file = self.current_file is not None and (self.current_file_kind == "h5" or self.current_data is not None)
        for button in [
            self.reload_button,
            self.add_row_button,
            self.remove_row_button,
            self.save_copy_button,
        ]:
            button.setEnabled(has_file)

    def show_project_selector(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "setCurrentIndex"):
                parent.setCurrentIndex(0)
                return
            parent = parent.parentWidget()
