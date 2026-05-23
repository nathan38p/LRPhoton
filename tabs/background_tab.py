

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
)

try:
    from tabs.ui_style import (
        PAGE_MARGINS,
        PANEL_MARGINS,
        BLOCK_SPACING,
        TOOL_GROUP_BOX_STYLE,
    )
except Exception:
    PAGE_MARGINS = (4, 4, 4, 4)
    PANEL_MARGINS = (0, 0, 0, 0)
    BLOCK_SPACING = 8
    TOOL_GROUP_BOX_STYLE = ""


class BackgroundTab(QWidget):
    folder_changed = Signal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample_file_path = ""
        self.background_file_path = ""
        self.output_folder_path = ""
        self.current_folder = ""
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(*PANEL_MARGINS)
        top_layout.setSpacing(BLOCK_SPACING)

        files_box = QGroupBox("Files")
        files_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        files_layout = QVBoxLayout(files_box)
        files_layout.setContentsMargins(8, 20, 8, 8)
        files_layout.setSpacing(6)

        self.sample_file_edit = QLineEdit()
        self.sample_file_edit.setPlaceholderText("Sample file")
        self.sample_file_edit.setReadOnly(True)
        self.sample_file_button = QPushButton("Open sample")
        self.sample_file_button.clicked.connect(self.select_sample_file)

        sample_row = QHBoxLayout()
        sample_row.addWidget(QLabel("Sample"))
        sample_row.addWidget(self.sample_file_edit, 1)
        sample_row.addWidget(self.sample_file_button)
        files_layout.addLayout(sample_row)

        self.background_file_edit = QLineEdit()
        self.background_file_edit.setPlaceholderText("Background file")
        self.background_file_edit.setReadOnly(True)
        self.background_file_button = QPushButton("Open background")
        self.background_file_button.clicked.connect(self.select_background_file)

        background_row = QHBoxLayout()
        background_row.addWidget(QLabel("Background"))
        background_row.addWidget(self.background_file_edit, 1)
        background_row.addWidget(self.background_file_button)
        files_layout.addLayout(background_row)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Output folder")
        self.output_folder_edit.setReadOnly(True)
        self.output_folder_button = QPushButton("Output folder")
        self.output_folder_button.clicked.connect(self.select_output_folder)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output"))
        output_row.addWidget(self.output_folder_edit, 1)
        output_row.addWidget(self.output_folder_button)
        files_layout.addLayout(output_row)

        parameters_box = QGroupBox("Parameters")
        parameters_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        parameters_layout = QVBoxLayout(parameters_box)
        parameters_layout.setContentsMargins(8, 20, 8, 8)
        parameters_layout.setSpacing(6)

        scale_row = QHBoxLayout()
        self.background_scale_spin = QDoubleSpinBox()
        self.background_scale_spin.setDecimals(4)
        self.background_scale_spin.setRange(-999999.0, 999999.0)
        self.background_scale_spin.setSingleStep(0.01)
        self.background_scale_spin.setValue(1.0)
        scale_row.addWidget(QLabel("Background factor"))
        scale_row.addWidget(self.background_scale_spin)
        parameters_layout.addLayout(scale_row)

        offset_row = QHBoxLayout()
        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setDecimals(4)
        self.offset_spin.setRange(-999999.0, 999999.0)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setValue(0.0)
        offset_row.addWidget(QLabel("Offset"))
        offset_row.addWidget(self.offset_spin)
        parameters_layout.addLayout(offset_row)

        frame_row = QHBoxLayout()
        self.frame_spin = QSpinBox()
        self.frame_spin.setRange(0, 999999)
        self.frame_spin.setValue(0)
        frame_row.addWidget(QLabel("Frame"))
        frame_row.addWidget(self.frame_spin)
        parameters_layout.addLayout(frame_row)

        self.keep_negative_checkbox = QCheckBox("Keep negative values")
        self.keep_negative_checkbox.setChecked(True)
        parameters_layout.addWidget(self.keep_negative_checkbox)

        self.save_preview_checkbox = QCheckBox("Save preview image")
        self.save_preview_checkbox.setChecked(False)
        parameters_layout.addWidget(self.save_preview_checkbox)

        actions_box = QGroupBox("Actions")
        actions_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        actions_layout = QVBoxLayout(actions_box)
        actions_layout.setContentsMargins(8, 20, 8, 8)
        actions_layout.setSpacing(6)

        self.run_button = QPushButton("Subtract background")
        self.run_button.clicked.connect(self.run_background_subtraction)
        actions_layout.addWidget(self.run_button)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        actions_layout.addWidget(self.status_label)

        left_column = QVBoxLayout()
        left_column.setContentsMargins(*PANEL_MARGINS)
        left_column.setSpacing(BLOCK_SPACING)
        left_column.addWidget(files_box)
        left_column.addWidget(parameters_box)
        left_column.addWidget(actions_box)
        left_column.addStretch(1)

        log_box = QGroupBox("Log")
        log_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        log_layout = QVBoxLayout(log_box)
        log_layout.setContentsMargins(8, 20, 8, 8)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("Background processing messages will appear here.")
        log_layout.addWidget(self.log_text)

        top_layout.addLayout(left_column, 1)
        top_layout.addWidget(log_box, 2)
        main_layout.addLayout(top_layout, 1)

    def set_working_folder(self, folder_path):
        self.current_folder = folder_path or ""
        if folder_path and not self.output_folder_path:
            self.output_folder_path = folder_path
            self.output_folder_edit.setText(folder_path)

    def set_folder_from_external_tab(self, folder_path):
        self.set_working_folder(folder_path)

    def select_sample_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select sample file",
            self.current_folder,
            "Data files (*.edf *.h5 *.hdf5 *.dat *.txt);;All files (*)",
        )
        if file_path:
            self.sample_file_path = file_path
            self.sample_file_edit.setText(file_path)
            self.set_working_folder(file_path.rsplit("/", 1)[0])
            self.folder_changed.emit(self.current_folder)

    def select_background_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select background file",
            self.current_folder,
            "Data files (*.edf *.h5 *.hdf5 *.dat *.txt);;All files (*)",
        )
        if file_path:
            self.background_file_path = file_path
            self.background_file_edit.setText(file_path)
            self.set_working_folder(file_path.rsplit("/", 1)[0])
            self.folder_changed.emit(self.current_folder)

    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            self.current_folder,
        )
        if folder_path:
            self.output_folder_path = folder_path
            self.output_folder_edit.setText(folder_path)
            self.set_working_folder(folder_path)
            self.folder_changed.emit(self.current_folder)

    def run_background_subtraction(self):
        if not self.sample_file_path:
            self.status_label.setText("Select a sample file first.")
            return

        if not self.background_file_path:
            self.status_label.setText("Select a background file first.")
            return

        if not self.output_folder_path:
            self.status_label.setText("Select an output folder first.")
            return

        self.log_text.append("Background subtraction is not implemented yet.")
        self.log_text.append(f"Sample: {self.sample_file_path}")
        self.log_text.append(f"Background: {self.background_file_path}")
        self.log_text.append(f"Factor: {self.background_scale_spin.value()}")
        self.log_text.append(f"Offset: {self.offset_spin.value()}")
        self.log_text.append(f"Frame: {self.frame_spin.value()}")
        self.status_label.setText("Interface ready. Processing code can now be added.")