from PySide6.QtCore import Qt, Signal, QDir, QSignalBlocker
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
    QSpinBox,
    QTextEdit,
    QSlider,
    QScrollArea,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QGridLayout,
)

import os
import re
import fnmatch
import numpy as np
from scipy.ndimage import binary_dilation, binary_opening, label

from .cave_tab import ImageCanvas, read_cave_mask_image
from .file_ratings import is_file_rated_up, install_file_rating_menu, set_item_file_path, should_hide_file_in_browser

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

try:
    import hdf5plugin  # noqa: F401
except Exception:
    pass

try:
    from tabs.ui_style import (
        PAGE_MARGINS,
        PANEL_MARGINS,
        BLOCK_SPACING,
        FILE_BROWSER_WIDTH,
        FlexibleDoubleSpinBox as QDoubleSpinBox,
        FRAME_BUTTON_WIDTH,
        FRAME_COUNTER_WIDTH,
        FRAME_NAV_SPACING,
        FRAME_SPIN_WIDTH,
        GROUP_BOX_MARGINS,
        TOOL_GROUP_BOX_STYLE,
    )
except Exception:
    from PySide6.QtWidgets import QDoubleSpinBox

    PAGE_MARGINS = (4, 4, 4, 4)
    PANEL_MARGINS = (0, 0, 0, 0)
    BLOCK_SPACING = 8
    FILE_BROWSER_WIDTH = 320
    FRAME_BUTTON_WIDTH = 44
    FRAME_COUNTER_WIDTH = 72
    FRAME_NAV_SPACING = 8
    FRAME_SPIN_WIDTH = 80
    GROUP_BOX_MARGINS = (8, 20, 8, 8)
    TOOL_GROUP_BOX_STYLE = ""


class LazyImageStack:
    def __init__(self, file_path, kind, data=None, dataset_path=None, frame_count=1, shape=None, frame_axis=None, header=None):
        self.file_path = file_path
        self.kind = kind
        self.data = data
        self.dataset_path = dataset_path
        self.frame_count = int(frame_count)
        self.shape = shape
        self.frame_axis = frame_axis
        self.header = dict(header or {})

    def get_frame(self, frame_index):
        frame_index = max(0, min(int(frame_index), self.frame_count - 1))

        if self.kind in ("edf", "text"):
            if self.data.ndim == 2:
                return self.data.astype(np.float64)
            return self.data[frame_index].astype(np.float64)

        if self.kind == "hdf5":
            if h5py is None:
                raise ImportError("h5py is required to read HDF5 files.")
            with h5py.File(self.file_path, "r") as handle:
                dataset = handle[self.dataset_path]
                if dataset.ndim == 2:
                    return np.asarray(dataset[:, :], dtype=np.float64)
                frame_axis = 0 if self.frame_axis is None else self.frame_axis
                if frame_axis == 0:
                    return np.asarray(dataset[frame_index, :, :], dtype=np.float64)
                if frame_axis == 1:
                    return np.asarray(dataset[:, frame_index, :], dtype=np.float64)
                return np.asarray(dataset[:, :, frame_index], dtype=np.float64)

        return None


class BackgroundTab(QWidget):
    folder_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sample_file_path = ""
        self.sample_file_paths = []
        self.background_file_path = ""
        self.mask_file_path = ""
        self.mask_array = None
        self.output_folder_path = ""
        self.current_folder = ""
        self.sample_stack = None
        self.background_stack = None
        self.result_data = None
        self.contrast_vmin = None
        self.contrast_vmax = None
        self.contrast_range_min = None
        self.contrast_range_max = None
        self.contrast_min_slider_low = None
        self.contrast_min_slider_high = None
        self.contrast_max_slider_low = None
        self.contrast_max_slider_high = None
        self.contrast_auto_initialized = False
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(*PANEL_MARGINS)
        content_layout.setSpacing(BLOCK_SPACING)

        original_box = QGroupBox("Original pattern")
        original_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        original_box.setMinimumWidth(0)
        original_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        original_layout = QVBoxLayout(original_box)
        original_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        original_layout.setSpacing(6)

        self.original_canvas = ImageCanvas()
        self.original_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_canvas.setMinimumWidth(0)
        self.original_canvas.setMinimumHeight(0)
        self.original_ax = self.original_canvas.ax
        self.original_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.original_coordinate_label.setMinimumHeight(28)
        self.original_coordinate_label.setAlignment(Qt.AlignCenter)
        self.original_coordinate_label.setStyleSheet(self.coordinate_label_style())
        self.original_canvas.set_coordinate_label(self.original_coordinate_label, "")
        original_layout.addWidget(self.original_canvas, 1)
        original_layout.addWidget(self.original_coordinate_label, 0)

        parameters_box = QGroupBox("Background")
        parameters_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        parameters_layout = QVBoxLayout(parameters_box)
        parameters_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        parameters_layout.setSpacing(6)

        file_browser_box = QGroupBox("File browser")
        file_browser_layout = QVBoxLayout(file_browser_box)
        file_browser_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_browser_layout.setSpacing(6)
        self.folder_path_edit = QLineEdit()
        self.folder_path_edit.setPlaceholderText("Folder")
        self.folder_path_edit.returnPressed.connect(self.refresh_file_browser)
        file_browser_layout.addWidget(self.folder_path_edit)
        browse_folder_button = QPushButton("Browse")
        browse_folder_button.clicked.connect(self.select_working_folder)
        file_browser_layout.addWidget(browse_folder_button)
        filters_layout = QGridLayout()
        self.name_filter = QLineEdit("*")
        self.extensions_filter = QLineEdit("*.edf *.h5 *.hdf5")
        self.name_filter.textChanged.connect(self.refresh_file_browser)
        self.extensions_filter.textChanged.connect(self.refresh_file_browser)
        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extensions_filter, 1, 1)
        file_browser_layout.addLayout(filters_layout)
        self.show_subfolders_checkbox = QCheckBox("Show subfolders")
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.show_subfolders_checkbox.stateChanged.connect(self.refresh_file_browser)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_file_browser)
        options_layout = QHBoxLayout()
        options_layout.addWidget(self.show_subfolders_checkbox)
        options_layout.addWidget(self.only_thumbs_up_checkbox)
        options_layout.addStretch(1)
        file_browser_layout.addLayout(options_layout)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_file_browser)
        file_browser_layout.addWidget(refresh_button)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        install_file_rating_menu(self.file_list)
        self.file_list.itemSelectionChanged.connect(self.select_sample_files_from_browser)
        file_browser_layout.addWidget(self.file_list, 1)

        mask_box = QGroupBox("Mask")
        mask_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        mask_layout = QVBoxLayout(mask_box)
        mask_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        mask_layout.setSpacing(6)
        self.mask_file_edit = QLineEdit()
        self.mask_file_edit.setPlaceholderText("Mask file")
        self.mask_file_edit.setReadOnly(True)
        mask_layout.addWidget(self.mask_file_edit)
        mask_button = QPushButton("Open mask")
        mask_button.clicked.connect(self.select_mask_file)
        mask_layout.addWidget(mask_button)
        expand_mask_layout = QHBoxLayout()
        expand_mask_layout.addWidget(QLabel("Expand NaN by"))
        self.mask_expand_spin = QSpinBox()
        self.mask_expand_spin.setRange(0, 10)
        self.mask_expand_spin.setValue(0)
        self.mask_expand_spin.setSuffix(" px")
        self.mask_expand_spin.valueChanged.connect(self.update_result_preview)
        expand_mask_layout.addWidget(self.mask_expand_spin)
        mask_layout.addLayout(expand_mask_layout)
        mask_layout.addStretch(1)

        self.sample_file_edit = QLineEdit()
        self.sample_file_edit.setPlaceholderText("Sample file")
        self.sample_file_edit.setReadOnly(True)
        self.sample_file_button = QPushButton("Open sample")
        self.sample_file_button.clicked.connect(self.select_sample_file)

        self.background_file_edit = QLineEdit()
        self.background_file_edit.setPlaceholderText("Background file")
        self.background_file_edit.setReadOnly(True)
        self.background_file_button = QPushButton("Open background")
        self.background_file_button.clicked.connect(self.select_background_file)
        parameters_layout.addWidget(self.background_file_edit)
        parameters_layout.addWidget(self.background_file_button)

        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Output folder")
        self.output_folder_edit.setReadOnly(True)
        self.output_folder_button = QPushButton("Output folder")
        self.output_folder_button.clicked.connect(self.select_output_folder)
        parameters_layout.addWidget(self.output_folder_edit)
        parameters_layout.addWidget(self.output_folder_button)
        self.output_folder_edit.hide()
        self.output_folder_button.hide()

        self.background_scale_spin = QDoubleSpinBox()
        self.background_scale_spin.setDecimals(4)
        self.background_scale_spin.setRange(-999999.0, 999999.0)
        self.background_scale_spin.setSingleStep(0.01)
        self.background_scale_spin.setValue(1.0)
        self.background_scale_spin.valueChanged.connect(self.update_result_preview)
        parameters_layout.addWidget(QLabel("Background factor"))
        parameters_layout.addWidget(self.background_scale_spin)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setDecimals(4)
        self.offset_spin.setRange(-999999.0, 999999.0)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setValue(0.0)
        self.offset_spin.valueChanged.connect(self.update_result_preview)
        parameters_layout.addWidget(QLabel("Offset"))
        parameters_layout.addWidget(self.offset_spin)

        self.frame_spin = QSpinBox(self)
        self.frame_spin.setRange(1, 1)
        self.frame_spin.setValue(1)
        self.frame_spin.hide()
        self.frame_spin.valueChanged.connect(self.sync_frame_slider_from_spin)

        self.keep_negative_checkbox = QCheckBox("Keep negative values")
        self.keep_negative_checkbox.setChecked(True)
        self.keep_negative_checkbox.stateChanged.connect(self.update_result_preview)
        parameters_layout.addWidget(self.keep_negative_checkbox)

        contrast_box = QGroupBox("Contrast")
        contrast_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        contrast_layout = QVBoxLayout(contrast_box)
        contrast_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        contrast_layout.setSpacing(6)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min"))
        self.intensity_min_spin = QDoubleSpinBox()
        self.intensity_min_spin.setDecimals(2)
        self.intensity_min_spin.setRange(-1e12, 1e12)
        self.intensity_min_spin.setSingleStep(100.0)
        self.intensity_min_spin.valueChanged.connect(self.update_contrast_from_spins)
        min_row.addWidget(self.intensity_min_spin)
        contrast_layout.addLayout(min_row)

        self.intensity_min_slider = QSlider(Qt.Horizontal)
        self.intensity_min_slider.setRange(0, 1000)
        self.intensity_min_slider.setValue(0)
        self.intensity_min_slider.valueChanged.connect(self.update_contrast_from_sliders)
        contrast_layout.addWidget(self.intensity_min_slider)

        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Max"))
        self.intensity_max_spin = QDoubleSpinBox()
        self.intensity_max_spin.setDecimals(2)
        self.intensity_max_spin.setRange(-1e12, 1e12)
        self.intensity_max_spin.setSingleStep(100.0)
        self.intensity_max_spin.valueChanged.connect(self.update_contrast_from_spins)
        max_row.addWidget(self.intensity_max_spin)
        contrast_layout.addLayout(max_row)

        self.intensity_max_slider = QSlider(Qt.Horizontal)
        self.intensity_max_slider.setRange(0, 1000)
        self.intensity_max_slider.setValue(1000)
        self.intensity_max_slider.valueChanged.connect(self.update_contrast_from_sliders)
        contrast_layout.addWidget(self.intensity_max_slider)

        self.auto_contrast_button = QPushButton("Auto contrast")
        self.auto_contrast_button.clicked.connect(self.auto_contrast)
        contrast_layout.addWidget(self.auto_contrast_button)
        contrast_box.hide()

        self.save_preview_checkbox = QCheckBox("Save preview image")
        self.save_preview_checkbox.setChecked(False)
        parameters_layout.addWidget(self.save_preview_checkbox)
        self.save_preview_checkbox.hide()

        self.save_current_button = QPushButton("💾 Save current frame")
        self.save_current_button.clicked.connect(self.save_current_frame)
        parameters_layout.addWidget(self.save_current_button)
        self.save_current_button.hide()

        parameters_layout.addStretch(1)
        self.run_button = QPushButton("▶️ Run and Save")
        self.run_button.clicked.connect(self.run_background_subtraction)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(80)
        self.log_text.setPlaceholderText("Background processing messages will appear here.")
        self.log_text.hide()

        normalization_box = QGroupBox("🚧 Normalization")
        normalization_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        normalization_layout = QVBoxLayout(normalization_box)
        normalization_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        self.normalization_enabled = QCheckBox("Enable normalization")
        self.normalization_enabled.stateChanged.connect(self.update_result_preview)
        normalization_layout.addWidget(self.normalization_enabled)
        normalization_form = QGridLayout()
        normalization_form.setHorizontalSpacing(6)
        normalization_form.setVerticalSpacing(4)
        normalization_form.addWidget(QLabel("Sample"), 0, 0, 1, 2)
        normalization_form.addWidget(QLabel("Exposure sample"), 1, 0)
        self.sample_exposure_spin = QDoubleSpinBox()
        self.sample_exposure_spin.setRange(1e-12, 1e12)
        self.sample_exposure_spin.setDecimals(6)
        self.sample_exposure_spin.setValue(1.0)
        self.sample_exposure_spin.setSuffix(" s")
        self.sample_exposure_spin.valueChanged.connect(self.update_result_preview)
        normalization_form.addWidget(self.sample_exposure_spin, 1, 1)
        normalization_form.addWidget(QLabel("Sample thickness"), 2, 0)
        self.sample_thickness_spin = QDoubleSpinBox()
        self.sample_thickness_spin.setRange(1e-12, 1e12)
        self.sample_thickness_spin.setDecimals(6)
        self.sample_thickness_spin.setValue(1.0)
        self.sample_thickness_spin.setSuffix(" mm")
        self.sample_thickness_spin.valueChanged.connect(self.update_result_preview)
        normalization_form.addWidget(self.sample_thickness_spin, 2, 1)
        normalization_form.addWidget(QLabel("Flux sample"), 3, 0)
        self.sample_flux_spin = QDoubleSpinBox()
        self.sample_flux_spin.setRange(1e-12, 1e12)
        self.sample_flux_spin.setDecimals(6)
        self.sample_flux_spin.setValue(1.0)
        self.sample_flux_spin.valueChanged.connect(self.update_result_preview)
        normalization_form.addWidget(self.sample_flux_spin, 3, 1)
        normalization_form.addWidget(QLabel("Background"), 4, 0, 1, 2)
        normalization_form.addWidget(QLabel("Exposure background"), 5, 0)
        self.background_exposure_spin = QDoubleSpinBox()
        self.background_exposure_spin.setRange(1e-12, 1e12)
        self.background_exposure_spin.setDecimals(6)
        self.background_exposure_spin.setValue(1.0)
        self.background_exposure_spin.setSuffix(" s")
        self.background_exposure_spin.valueChanged.connect(self.update_result_preview)
        normalization_form.addWidget(self.background_exposure_spin, 5, 1)
        normalization_form.addWidget(QLabel("Flux background"), 6, 0)
        self.background_flux_spin = QDoubleSpinBox()
        self.background_flux_spin.setRange(1e-12, 1e12)
        self.background_flux_spin.setDecimals(6)
        self.background_flux_spin.setValue(1.0)
        self.background_flux_spin.valueChanged.connect(self.update_result_preview)
        normalization_form.addWidget(self.background_flux_spin, 6, 1)
        normalization_layout.addLayout(normalization_form)
        normalization_layout.addStretch(1)

        result_box = QGroupBox("Background-subtracted pattern")
        result_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        result_box.setMinimumWidth(0)
        result_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_layout = QVBoxLayout(result_box)
        result_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        result_layout.setSpacing(6)

        self.result_canvas = ImageCanvas()
        self.result_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_canvas.setMinimumWidth(0)
        self.result_canvas.setMinimumHeight(0)
        self.result_ax = self.result_canvas.ax
        self.result_coordinate_label = QLabel("x = - | y = - | q = - | I = -")
        self.result_coordinate_label.setMinimumHeight(28)
        self.result_coordinate_label.setAlignment(Qt.AlignCenter)
        self.result_coordinate_label.setStyleSheet(self.coordinate_label_style())
        self.result_canvas.set_coordinate_label(self.result_coordinate_label, "")
        result_layout.addWidget(self.result_canvas, 1)
        result_layout.addWidget(self.result_coordinate_label, 0)

        content_layout.addWidget(original_box, 2)

        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(BLOCK_SPACING)
        columns_layout = QHBoxLayout()
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(BLOCK_SPACING)
        left_controls = QVBoxLayout()
        left_controls.setContentsMargins(0, 0, 0, 0)
        left_controls.setSpacing(BLOCK_SPACING)
        left_controls.addWidget(file_browser_box, 1)
        right_controls = QVBoxLayout()
        right_controls.setContentsMargins(0, 0, 0, 0)
        right_controls.setSpacing(BLOCK_SPACING)
        right_controls.addWidget(mask_box, 1)
        right_controls.addWidget(parameters_box, 1)
        right_controls.addWidget(normalization_box, 1)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 1)
        self.batch_progress.setValue(0)
        self.batch_progress.setVisible(False)
        right_controls.addWidget(self.batch_progress)
        right_controls.addWidget(self.run_button)
        columns_layout.addLayout(left_controls, 1)
        columns_layout.addLayout(right_controls, 1)
        controls_layout.addLayout(columns_layout, 1)
        controls_panel.setFixedWidth(FILE_BROWSER_WIDTH * 2 + BLOCK_SPACING)
        controls_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        content_layout.addWidget(controls_panel, 0)

        content_layout.addWidget(result_box, 2)
        main_layout.addLayout(content_layout, 1)

        contrast_bar_box = QGroupBox("Contrast")
        contrast_bar_box.setStyleSheet(TOOL_GROUP_BOX_STYLE)
        contrast_bar_layout = QHBoxLayout(contrast_bar_box)
        contrast_bar_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        contrast_bar_layout.setSpacing(6)
        contrast_bar_layout.addWidget(QLabel("Min"))
        contrast_bar_layout.addWidget(self.intensity_min_spin)
        contrast_bar_layout.addWidget(self.intensity_min_slider, 1)
        contrast_bar_layout.addWidget(self.auto_contrast_button)
        contrast_bar_layout.addWidget(self.intensity_max_slider, 1)
        contrast_bar_layout.addWidget(QLabel("Max"))
        contrast_bar_layout.addWidget(self.intensity_max_spin)
        main_layout.addWidget(contrast_bar_box)

        frame_slider_layout = QHBoxLayout()
        frame_slider_layout.setContentsMargins(0, 0, 0, 0)
        frame_slider_layout.setSpacing(FRAME_NAV_SPACING)

        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(1, 1)
        self.frame_start_spin.setValue(1)
        self.frame_start_spin.setFixedWidth(FRAME_SPIN_WIDTH)

        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(1, 1)
        self.frame_end_spin.setValue(1)
        self.frame_end_spin.setFixedWidth(FRAME_SPIN_WIDTH)

        self.prev_frame_button = QPushButton("<")
        self.next_frame_button = QPushButton(">")
        self.prev_frame_button.setFixedWidth(FRAME_BUTTON_WIDTH)
        self.next_frame_button.setFixedWidth(FRAME_BUTTON_WIDTH)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(1, 1)
        self.frame_slider.setValue(1)
        self.frame_slider.valueChanged.connect(self.sync_frame_spin_from_slider)

        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        frame_slider_layout.addWidget(QLabel("Start:"))
        frame_slider_layout.addWidget(self.frame_start_spin)
        frame_slider_layout.addWidget(self.prev_frame_button)
        frame_slider_layout.addWidget(self.frame_slider, 1)
        frame_slider_layout.addWidget(self.next_frame_button)
        frame_slider_layout.addWidget(QLabel("End:"))
        frame_slider_layout.addWidget(self.frame_end_spin)
        frame_slider_layout.addWidget(self.frame_counter_label)
        main_layout.addLayout(frame_slider_layout)

        self.frame_start_spin.valueChanged.connect(self.update_frame_bounds)
        self.frame_end_spin.valueChanged.connect(self.update_frame_bounds)
        self.prev_frame_button.clicked.connect(self.previous_frame)
        self.next_frame_button.clicked.connect(self.next_frame)
        self.update_frame_bounds()

    def coordinate_label_style(self):
        return """
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """

    def open_image_stack(self, file_path):
        lower_path = file_path.lower()

        if lower_path.endswith(".edf"):
            if fabio is None:
                raise ImportError("fabio is required to read EDF files.")
            edf = fabio.open(file_path)
            data = np.asarray(edf.data)
            header = dict(getattr(edf, "header", {}) or {})
            if data.ndim == 2:
                return LazyImageStack(file_path, "edf", data=data, frame_count=1, shape=data.shape, header=header)
            if data.ndim == 3:
                return LazyImageStack(file_path, "edf", data=data, frame_count=data.shape[0], shape=data.shape[-2:], header=header)
            raise ValueError("Unsupported EDF data dimensions.")

        if lower_path.endswith((".h5", ".hdf5")):
            if h5py is None:
                raise ImportError("h5py is required to read HDF5 files.")
            with h5py.File(file_path, "r") as handle:
                dataset = self.find_first_image_dataset(handle)
                if dataset is None:
                    raise ValueError("No 2D or 3D image dataset found in HDF5 file.")
                dataset_path = dataset.name
                if dataset.ndim == 2:
                    frame_count = 1
                    shape = dataset.shape
                    frame_axis = None
                elif dataset.ndim == 3:
                    frame_axis, frame_count, shape = self.h5_dataset_image_info(dataset.shape)
                else:
                    raise ValueError("Unsupported HDF5 data dimensions.")
                header = {}
                def collect_exposure(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.ndim == 0 and "exposure_time" in name.lower():
                        try:
                            header["ExposureTime"] = float(obj[()])
                        except (TypeError, ValueError):
                            pass
                handle.visititems(collect_exposure)
            return LazyImageStack(
                file_path,
                "hdf5",
                dataset_path=dataset_path,
                frame_count=frame_count,
                shape=shape,
                frame_axis=frame_axis,
                header=header,
            )

        data = np.loadtxt(file_path)
        if data.ndim == 2:
            return LazyImageStack(file_path, "text", data=data, frame_count=1, shape=data.shape)
        raise ValueError("Only 2D text data can be displayed as an image.")

    def find_first_image_dataset(self, h5_group):
        best_dataset = None
        best_score = None

        def visitor(name, obj):
            nonlocal best_dataset, best_score
            if not isinstance(obj, h5py.Dataset) or obj.ndim not in (2, 3):
                return
            if not np.issubdtype(obj.dtype, np.number):
                return

            try:
                frame_axis, frame_count, image_shape = self.h5_dataset_image_info(obj.shape)
            except ValueError:
                return
            if min(image_shape) <= 16:
                return

            score = self.h5_dataset_image_score(name, obj, frame_axis, frame_count, image_shape)
            if best_score is None or score > best_score:
                best_dataset = obj
                best_score = score

        h5_group.visititems(visitor)
        return best_dataset

    def h5_dataset_image_info(self, shape):
        shape = tuple(int(size) for size in shape)
        if len(shape) == 2:
            return None, 1, shape
        if len(shape) == 3:
            frame_axis = int(np.argmin(shape))
            frame_count = int(shape[frame_axis])
            image_shape = tuple(size for axis, size in enumerate(shape) if axis != frame_axis)
            return frame_axis, frame_count, image_shape
        raise ValueError("Dataset must be 2D or 3D.")

    def h5_dataset_image_score(self, name, dataset, frame_axis, frame_count, image_shape):
        lower_name = str(name).lower()
        score = float(image_shape[0]) * float(image_shape[1])

        if len(tuple(dataset.shape)) == 3:
            score *= 10.0

        if min(image_shape) >= 128:
            score *= 4.0
        elif min(image_shape) < 32:
            score *= 0.05

        if any(token in lower_name for token in ["data", "image", "eiger", "detector", "pilatus"]):
            score *= 3.0
        if any(token in lower_name for token in ["mask", "flat", "dark", "background", "metadata"]):
            score *= 0.1

        if frame_axis is not None and frame_count <= 1:
            score *= 0.1

        return score

    def update_frame_controls(self):
        frame_count = 1 if self.sample_stack is None else max(1, self.sample_stack.frame_count)

        self.frame_slider.blockSignals(True)
        self.frame_spin.blockSignals(True)
        self.frame_start_spin.blockSignals(True)
        self.frame_end_spin.blockSignals(True)

        self.frame_spin.setRange(1, frame_count)
        self.frame_start_spin.setRange(1, frame_count)
        self.frame_start_spin.setValue(1)
        self.frame_end_spin.setRange(1, frame_count)
        self.frame_end_spin.setValue(frame_count)
        self.frame_slider.setRange(1, frame_count)
        if self.frame_spin.value() > frame_count:
            self.frame_spin.setValue(frame_count)
        self.frame_slider.setValue(self.frame_spin.value())

        self.frame_end_spin.blockSignals(False)
        self.frame_start_spin.blockSignals(False)
        self.frame_slider.blockSignals(False)
        self.frame_spin.blockSignals(False)
        self.update_frame_navigation_state()

    def current_frame_index(self):
        return max(0, self.frame_spin.value() - 1)

    def update_frame_bounds(self):
        start = self.frame_start_spin.value()
        end = self.frame_end_spin.value()

        if start > end:
            sender = self.sender()
            if sender is self.frame_start_spin:
                self.frame_end_spin.setValue(start)
                end = start
            else:
                self.frame_start_spin.setValue(end)
                start = end

        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(start, end)
        self.frame_slider.blockSignals(False)

        current = self.frame_spin.value()
        if current < start:
            self.frame_spin.setValue(start)
        elif current > end:
            self.frame_spin.setValue(end)
        self.update_frame_navigation_state()

    def update_frame_navigation_state(self):
        frame_count = 1 if self.sample_stack is None else max(1, self.sample_stack.frame_count)
        current = self.frame_spin.value()
        can_navigate = frame_count > 1

        self.frame_counter_label.setText(f"{current} / {frame_count}")
        self.frame_start_spin.setEnabled(can_navigate)
        self.frame_end_spin.setEnabled(can_navigate)
        self.frame_slider.setEnabled(can_navigate)
        self.prev_frame_button.setEnabled(can_navigate and current > self.frame_slider.minimum())
        self.next_frame_button.setEnabled(can_navigate and current < self.frame_slider.maximum())

    def previous_frame(self):
        self.frame_spin.setValue(max(self.frame_slider.minimum(), self.frame_spin.value() - 1))

    def next_frame(self):
        self.frame_spin.setValue(min(self.frame_slider.maximum(), self.frame_spin.value() + 1))

    def display_image(self, ax, canvas, image, title):
        if image is None:
            ax.clear()
            ax.set_axis_off()
            if hasattr(canvas, "raw_image"):
                canvas.raw_image = None
                canvas.image_artist = None
            canvas.draw_idle()
            return

        image = np.asarray(image, dtype=np.float64)
        image = np.where(np.isfinite(image), image, np.nan)

        if hasattr(canvas, "show_image"):
            canvas.show_image(
                image,
                title="",
                vmin=self.contrast_vmin,
                vmax=self.contrast_vmax,
            )
            return

        ax.clear()
        ax.set_axis_off()
        ax.imshow(
            image,
            cmap="jet",
            origin="upper",
            vmin=self.contrast_vmin,
            vmax=self.contrast_vmax,
        )
        canvas.draw_idle()

    def display_values_for_contrast(self, image):
        display = np.asarray(image, dtype=np.float64).copy()
        display[~np.isfinite(display)] = np.nan
        display[display < 0] = np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1)
        return display

    def block_contrast_signals(self, blocked):
        self.intensity_min_spin.blockSignals(blocked)
        self.intensity_max_spin.blockSignals(blocked)
        self.intensity_min_slider.blockSignals(blocked)
        self.intensity_max_slider.blockSignals(blocked)

    def set_contrast_values(self, vmin, vmax):
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return
        if vmax <= vmin:
            vmax = vmin + 1.0

        self.contrast_vmin = float(vmin)
        self.contrast_vmax = float(vmax)
        span = max(self.contrast_vmax - self.contrast_vmin, 1.0)
        self.contrast_min_slider_low = self.contrast_vmin - abs(self.contrast_vmin) * 0.2
        self.contrast_min_slider_high = self.contrast_vmin + abs(self.contrast_vmin) * 0.2
        self.contrast_max_slider_low = self.contrast_vmax - abs(self.contrast_vmax) * 0.2
        self.contrast_max_slider_high = self.contrast_vmax + abs(self.contrast_vmax) * 0.2
        if self.contrast_min_slider_low == self.contrast_min_slider_high:
            self.contrast_min_slider_low = self.contrast_vmin - span * 0.2
            self.contrast_min_slider_high = self.contrast_vmin + span * 0.2
        if self.contrast_max_slider_low == self.contrast_max_slider_high:
            self.contrast_max_slider_low = self.contrast_vmax - span * 0.2
            self.contrast_max_slider_high = self.contrast_vmax + span * 0.2
        self.contrast_auto_initialized = True

        self.block_contrast_signals(True)
        self.intensity_min_spin.setValue(self.contrast_vmin)
        self.intensity_max_spin.setValue(self.contrast_vmax)
        self.intensity_min_slider.setValue(0)
        self.intensity_max_slider.setValue(1000)
        self.block_contrast_signals(False)

        self.refresh_displayed_images()

    def update_contrast_from_spins(self):
        vmin = self.intensity_min_spin.value()
        vmax = self.intensity_max_spin.value()
        if vmax <= vmin:
            vmax = vmin + 1.0
            self.intensity_max_spin.blockSignals(True)
            self.intensity_max_spin.setValue(vmax)
            self.intensity_max_spin.blockSignals(False)

        self.contrast_vmin = vmin
        self.contrast_vmax = vmax
        span = max(vmax - vmin, 1.0)
        self.contrast_min_slider_low = vmin - abs(vmin) * 0.2
        self.contrast_min_slider_high = vmin + abs(vmin) * 0.2
        self.contrast_max_slider_low = vmax - abs(vmax) * 0.2
        self.contrast_max_slider_high = vmax + abs(vmax) * 0.2
        if self.contrast_min_slider_low == self.contrast_min_slider_high:
            self.contrast_min_slider_low = vmin - span * 0.2
            self.contrast_min_slider_high = vmin + span * 0.2
        if self.contrast_max_slider_low == self.contrast_max_slider_high:
            self.contrast_max_slider_low = vmax - span * 0.2
            self.contrast_max_slider_high = vmax + span * 0.2
        self.contrast_auto_initialized = True
        self.refresh_displayed_images()

    def update_contrast_from_sliders(self):
        if self.contrast_vmin is None or self.contrast_vmax is None:
            self.auto_contrast()
            return

        slider_min = self.intensity_min_slider.value()
        slider_max = self.intensity_max_slider.value()
        if slider_max <= slider_min:
            slider_max = slider_min + 1
            self.intensity_max_slider.blockSignals(True)
            self.intensity_max_slider.setValue(slider_max)
            self.intensity_max_slider.blockSignals(False)

        if self.contrast_min_slider_low is None or self.contrast_max_slider_high is None:
            return

        vmin = self.contrast_min_slider_low + (
            self.contrast_min_slider_high - self.contrast_min_slider_low
        ) * (slider_min / 1000.0)
        vmax = self.contrast_max_slider_low + (
            self.contrast_max_slider_high - self.contrast_max_slider_low
        ) * (slider_max / 1000.0)

        self.block_contrast_signals(True)
        self.intensity_min_spin.setValue(vmin)
        self.intensity_max_spin.setValue(vmax)
        self.block_contrast_signals(False)

        self.contrast_vmin = vmin
        self.contrast_vmax = vmax
        self.contrast_auto_initialized = True
        self.refresh_displayed_images()

    def auto_contrast(self):
        frame = self.result_data
        if frame is None and self.sample_stack is not None:
            frame = self.sample_stack.get_frame(self.current_frame_index())
        if frame is None:
            return

        finite_values = self.display_values_for_contrast(frame)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return

        vmin, vmax = np.nanpercentile(finite_values, [1, 99])
        self.set_contrast_values(vmin, vmax)

    def refresh_displayed_images(self):
        sample_frame = None
        if self.sample_stack is not None:
            sample_frame = self.sample_stack.get_frame(self.current_frame_index())
        self.display_image(self.original_ax, self.original_canvas, sample_frame, "Original file")

        if self.result_data is not None:
            self.display_image(self.result_ax, self.result_canvas, self.result_data, "Result")
        else:
            self.display_image(self.result_ax, self.result_canvas, None, "Result")

    def update_sample_preview(self):
        frame = None
        if self.sample_stack is not None:
            frame = self.sample_stack.get_frame(self.current_frame_index())
            if not self.contrast_auto_initialized:
                finite_values = self.display_values_for_contrast(frame)
                finite_values = finite_values[np.isfinite(finite_values)]
                if finite_values.size:
                    vmin, vmax = np.nanpercentile(finite_values, [1, 99])
                    self.set_contrast_values(vmin, vmax)
        self.display_image(self.original_ax, self.original_canvas, frame, "Original file")
        self.update_result_preview()

    def update_result_preview(self):
        sample_frame = None
        background_frame = None
        frame_index = self.current_frame_index()

        if self.sample_stack is not None:
            sample_frame = self.sample_stack.get_frame(frame_index)
        if self.background_stack is not None:
            background_frame = self.background_stack.get_frame(frame_index)

        if sample_frame is None:
            self.result_data = None
            self.display_image(self.result_ax, self.result_canvas, None, "Result")
            return

        if background_frame is None:
            if self.mask_array is not None or self.normalization_enabled.isChecked():
                try:
                    self.result_data = self.compute_result_frame(frame_index)
                except Exception:
                    self.result_data = None
                    self.display_image(self.result_ax, self.result_canvas, None, "Preview unavailable")
                    return
                self.display_image(self.result_ax, self.result_canvas, self.result_data, "Pre-treatment preview")
                return
            self.result_data = None
            self.display_image(self.result_ax, self.result_canvas, None, "Result")
            return

        if sample_frame.shape != background_frame.shape:
            self.result_data = None
            self.status_label.setText("Sample and background frames do not have the same shape.")
            self.display_image(self.result_ax, self.result_canvas, None, "Shape mismatch")
            return

        result = self.compute_result_frame(frame_index)
        self.result_data = result
        self.display_image(self.result_ax, self.result_canvas, result, "Result")
        self.status_label.setText("")

    def sync_frame_slider_from_spin(self, value):
        if self.frame_slider.value() != value:
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(value)
            self.frame_slider.blockSignals(False)
        self.update_sample_preview()
        self.update_frame_navigation_state()

    def sync_frame_spin_from_slider(self, value):
        if self.frame_spin.value() != value:
            self.frame_spin.blockSignals(True)
            self.frame_spin.setValue(value)
            self.frame_spin.blockSignals(False)
        self.update_sample_preview()
        self.update_frame_navigation_state()

    def set_working_folder(self, folder_path):
        self.current_folder = folder_path or ""
        if hasattr(self, "folder_path_edit"):
            self.folder_path_edit.setText(self.current_folder)
            if self.current_folder and os.path.isdir(self.current_folder) and hasattr(self, "file_list"):
                self.refresh_file_browser()
        if folder_path and hasattr(self, "output_folder_edit") and not self.output_folder_path:
            self.output_folder_path = folder_path
            self.output_folder_edit.setText(folder_path)

    def set_folder_from_external_tab(self, folder_path):
        self.set_working_folder(folder_path)

    def open_data_file_dialog(self, title):
        start_folder = self.current_folder if self.current_folder and os.path.isdir(self.current_folder) else QDir.homePath()
        dialog = QFileDialog(self, title, start_folder)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setNameFilter("Data files (*.edf *.h5 *.hdf5 *.dat *.txt);;All files (*)")
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.raise_()
        dialog.activateWindow()
        if dialog.exec() == QFileDialog.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                return selected_files[0]
        return ""

    def stack_description(self, stack):
        if stack is None or stack.kind != "hdf5":
            return ""
        details = [f"dataset: {stack.dataset_path}"]
        if stack.frame_axis is not None:
            details.append(f"frame axis: {stack.frame_axis}")
        return " (" + ", ".join(details) + ")"

    def apply_stack_exposure(self, stack, spin):
        try:
            exposure = float(stack.header.get("ExposureTime"))
        except (AttributeError, TypeError, ValueError):
            return
        if exposure > 0:
            spin.blockSignals(True)
            spin.setValue(exposure)
            spin.blockSignals(False)

    def select_sample_file(self):
        file_path = self.open_data_file_dialog("Select sample file")
        if file_path:
            self.sample_file_path = file_path
            self.sample_file_edit.setText(file_path)
            self.set_working_folder(file_path.rsplit("/", 1)[0])
            self.folder_changed.emit(self.current_folder)
            try:
                self.sample_stack = self.open_image_stack(file_path)
                self.apply_stack_exposure(self.sample_stack, self.sample_exposure_spin)
                self.auto_select_detector_files(file_path)
                if self.mask_file_path:
                    self.mask_array = read_cave_mask_image(self.mask_file_path, expected_shape=self.sample_stack.shape)
                self.contrast_auto_initialized = False
                self.contrast_vmin = None
                self.contrast_vmax = None
                self.update_frame_controls()
                self.update_sample_preview()
                self.status_label.setText(
                    f"Sample loaded: {self.sample_stack.frame_count} frame(s)"
                    f"{self.stack_description(self.sample_stack)}."
                )
            except Exception as exc:
                self.sample_stack = None
                self.update_frame_controls()
                self.display_image(self.original_ax, self.original_canvas, None, "Original file")
                self.status_label.setText(f"Sample loading error: {exc}")

    def select_working_folder(self):
        start_folder = self.current_folder if self.current_folder and os.path.isdir(self.current_folder) else QDir.homePath()
        folder = QFileDialog.getExistingDirectory(self, "Select data folder", start_folder)
        if folder:
            self.set_working_folder(folder)
            self.folder_path_edit.setText(folder)
            self.refresh_file_browser()

    def refresh_file_browser(self):
        folder = self.folder_path_edit.text().strip()
        if not os.path.isdir(folder):
            return
        self.current_folder = folder
        blocker = QSignalBlocker(self.file_list)
        try:
            self.file_list.clear()
            name_pattern = self.name_filter.text().strip() or "*"
            extension_patterns = self.extensions_filter.text().split() or ["*.edf", "*.h5", "*.hdf5"]
            if self.show_subfolders_checkbox.isChecked():
                candidates = (
                    os.path.join(root, name)
                    for root, _dirs, names in os.walk(folder)
                    for name in names
                )
            else:
                candidates = (os.path.join(folder, name) for name in os.listdir(folder))
            paths = []
            for path in candidates:
                if not os.path.isfile(path) or should_hide_file_in_browser(path):
                    continue
                name = os.path.basename(path)
                if not fnmatch.fnmatch(name.lower(), name_pattern.lower()):
                    continue
                if not any(fnmatch.fnmatch(name.lower(), pattern.lower()) for pattern in extension_patterns):
                    continue
                if self.only_thumbs_up_checkbox.isChecked() and not is_file_rated_up(path):
                    continue
                paths.append(path)
            for path in sorted(paths):
                item = QListWidgetItem(os.path.relpath(path, folder))
                set_item_file_path(item, path)
                self.file_list.addItem(item)
        finally:
            del blocker

    def select_sample_files_from_browser(self):
        paths = [item.data(Qt.UserRole) for item in self.file_list.selectedItems()]
        if not paths:
            return
        self.sample_file_paths = paths
        self.sample_file_path = paths[0]
        self.sample_file_edit.setText("; ".join(paths))
        try:
            self.sample_stack = self.open_image_stack(paths[0])
            self.apply_stack_exposure(self.sample_stack, self.sample_exposure_spin)
            self.auto_select_detector_files(paths[0])
            if self.mask_file_path:
                self.mask_array = read_cave_mask_image(self.mask_file_path, expected_shape=self.sample_stack.shape)
            self.contrast_auto_initialized = False
            self.contrast_vmin = None
            self.contrast_vmax = None
            self.update_frame_controls()
            self.update_sample_preview()
            self.status_label.setText(f"Preview: {os.path.basename(paths[0])} ({len(paths)} selected)")
        except Exception as exc:
            self.sample_stack = None
            self.update_frame_controls()
            self.status_label.setText(f"Sample loading error: {exc}")

    def select_background_file(self):
        file_path = self.open_data_file_dialog("Select background file")
        if file_path:
            self.background_file_path = file_path
            self.background_file_edit.setText(file_path)
            self.set_working_folder(file_path.rsplit("/", 1)[0])
            self.folder_changed.emit(self.current_folder)
            try:
                self.background_stack = self.open_image_stack(file_path)
                self.apply_stack_exposure(self.background_stack, self.background_exposure_spin)
                self.update_result_preview()
                self.status_label.setText(
                    f"Background loaded: {self.background_stack.frame_count} frame(s)"
                    f"{self.stack_description(self.background_stack)}."
                )
            except Exception as exc:
                self.background_stack = None
                self.update_result_preview()
                self.status_label.setText(f"Background loading error: {exc}")

    def auto_select_detector_files(self, sample_path):
        """Load matching detector mask and capillary/background from the sample folder."""
        folder = os.path.dirname(sample_path)
        sample_name = os.path.basename(sample_path)
        detector_aliases = (
            "eiger4m", "eiger", "pilatus", "si4m", "udetx", "id13", "id02",
            "xenocs", "mar300", "mar165", "lambda", "saxs", "waxs",
        )
        sample_lower = sample_name.lower()
        detectors = [alias for alias in detector_aliases if alias in sample_lower]
        if not detectors:
            return

        try:
            candidates = [
                os.path.join(folder, name)
                for name in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, name))
                and os.path.splitext(name)[1].lower() in {".edf", ".h5", ".hdf5"}
                and os.path.abspath(os.path.join(folder, name)) != os.path.abspath(sample_path)
            ]
        except OSError:
            return

        def has_word(name, words):
            lower = os.path.splitext(os.path.basename(name))[0].lower()
            return any(re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", lower) for word in words)

        def rank(path, words):
            lower = os.path.basename(path).lower()
            detector_score = max((len(alias) for alias in detectors if alias in lower), default=0)
            keyword_score = 10 if has_word(path, words) else 0
            return keyword_score + detector_score

        mask_candidates = [path for path in candidates if has_word(path, ("mask",)) and any(alias in os.path.basename(path).lower() for alias in detectors)]
        background_candidates = [
            path for path in candidates
            if has_word(path, ("capillaire", "background", "bck", "bg"))
            and any(alias in os.path.basename(path).lower() for alias in detectors)
        ]

        if mask_candidates:
            mask_path = max(sorted(mask_candidates), key=lambda path: rank(path, ("mask",)))
            try:
                expected_shape = self.sample_stack.shape if self.sample_stack is not None else None
                self.mask_array = read_cave_mask_image(mask_path, expected_shape=expected_shape)
                self.mask_file_path = mask_path
                self.mask_file_edit.setText(mask_path)
            except Exception:
                pass

        if background_candidates:
            background_path = max(sorted(background_candidates), key=lambda path: rank(path, ("capillaire", "background", "bck", "bg")))
            try:
                self.background_stack = self.open_image_stack(background_path)
                self.apply_stack_exposure(self.background_stack, self.background_exposure_spin)
                self.background_file_path = background_path
                self.background_file_edit.setText(background_path)
            except Exception:
                pass

        if mask_candidates or background_candidates:
            self.update_result_preview()

    def select_mask_file(self):
        file_path = self.open_data_file_dialog("Select mask file")
        if not file_path:
            return
        try:
            expected_shape = self.sample_stack.shape if self.sample_stack is not None else None
            self.mask_array = read_cave_mask_image(file_path, expected_shape=expected_shape)
            self.mask_file_path = file_path
            self.mask_file_edit.setText(file_path)
            self.update_result_preview()
        except Exception as exc:
            self.mask_array = None
            self.mask_file_path = ""
            self.mask_file_edit.clear()
            self.status_label.setText(f"Mask loading error: {exc}")

    def select_output_folder(self):
        start_folder = self.current_folder if self.current_folder and os.path.isdir(self.current_folder) else QDir.homePath()
        dialog = QFileDialog(self, "Select output folder", start_folder)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.raise_()
        dialog.activateWindow()
        folder_path = ""
        if dialog.exec() == QFileDialog.Accepted:
            selected_folders = dialog.selectedFiles()
            if selected_folders:
                folder_path = selected_folders[0]
        if folder_path:
            self.output_folder_path = folder_path
            self.output_folder_edit.setText(folder_path)
            self.set_working_folder(folder_path)
            self.folder_changed.emit(self.current_folder)

    def get_output_base_name(self):
        if self.sample_file_path:
            base_name = os.path.splitext(os.path.basename(self.sample_file_path))[0]
        else:
            base_name = "background_subtracted"
        return f"{base_name}_nrm"

    def compute_result_frame(self, frame_index):
        if self.sample_stack is None:
            raise ValueError("No sample file loaded.")

        sample_frame = self.sample_stack.get_frame(frame_index)
        background_frame = None
        if self.background_stack is not None:
            background_frame = self.background_stack.get_frame(frame_index)

        if background_frame is not None and sample_frame.shape != background_frame.shape:
            raise ValueError("Sample and background frames do not have the same shape.")

        if self.mask_array is not None:
            if self.mask_array.shape != sample_frame.shape:
                raise ValueError("Mask shape does not match the current image frame.")
            mask = self.expanded_mask(self.mask_array, self.mask_expand_spin.value())
            sample_frame = np.asarray(sample_frame, dtype=np.float64).copy()
            sample_frame[mask] = np.nan
            if background_frame is not None:
                background_frame = np.asarray(background_frame, dtype=np.float64).copy()
                background_frame[mask] = np.nan

        if self.normalization_enabled.isChecked():
            sample_factor = self.sample_exposure_spin.value() * self.sample_flux_spin.value()
            sample_frame = np.asarray(sample_frame, dtype=np.float64) / sample_factor
            if background_frame is not None:
                background_factor = self.background_exposure_spin.value() * self.background_flux_spin.value()
                background_frame = np.asarray(background_frame, dtype=np.float64) / background_factor

        result = np.asarray(sample_frame, dtype=np.float64).copy()
        if background_frame is not None:
            result = result - self.background_scale_spin.value() * background_frame + self.offset_spin.value()
        if self.normalization_enabled.isChecked():
            result /= self.sample_thickness_spin.value()
        if not self.keep_negative_checkbox.isChecked():
            result = np.maximum(result, 0)
        return result

    @staticmethod
    def expanded_mask(mask, radius):
        source = np.asarray(mask, dtype=bool)
        expanded = source.copy()
        radius = max(0, int(radius))
        if radius == 0:
            return expanded
        # Remove small one-pixel appendages before identifying the large
        # detector-gap rectangles. The original mask is still preserved.
        rectangle_core = binary_opening(
            source,
            structure=np.ones((3, 3), dtype=bool),
        )
        components, count = label(rectangle_core, structure=np.ones((3, 3), dtype=bool))
        large_components = np.zeros_like(source, dtype=bool)
        for component_index in range(1, count + 1):
            component = components == component_index
            touches_border = bool(
                component[0, :].any()
                or component[-1, :].any()
                or component[:, 0].any()
                or component[:, -1].any()
            )
            if int(component.sum()) > 20 and touches_border:
                large_components |= component
        if not large_components.any():
            return expanded
        structure_size = 2 * radius + 1
        expanded |= binary_dilation(
            large_components,
            structure=np.ones((structure_size, structure_size), dtype=bool),
        )
        return expanded

    def output_metadata(self):
        return {
            "background_subtracted": "True",
            "background_source": str(self.background_file_path),
            "background_scale": str(self.background_scale_spin.value()),
            "background_offset": str(self.offset_spin.value()),
            "normalization_enabled": str(self.normalization_enabled.isChecked()),
            "sample_exposure": str(self.sample_exposure_spin.value()),
            "background_exposure": str(self.background_exposure_spin.value()),
            "sample_thickness_mm": str(self.sample_thickness_spin.value()),
            "sample_flux": str(self.sample_flux_spin.value()),
            "background_flux": str(self.background_flux_spin.value()),
        }

    def save_current_frame(self):
        if not self.output_folder_path:
            self.status_label.setText("Select an output folder first.")
            return

        if self.sample_stack is None:
            self.status_label.setText("Load a sample first.")
            return
        if self.sample_stack.kind != "hdf5" and EdfImage is None:
            self.status_label.setText("fabio EDF support is not available.")
            return

        try:
            frame_index = self.current_frame_index()
            result = self.compute_result_frame(frame_index)
            base_name = self.get_output_base_name()

            if self.sample_stack.kind == "hdf5":
                h5_path = os.path.join(
                    self.output_folder_path,
                    f"{base_name}_frame_{frame_index:04d}.h5",
                )
                self.write_background_h5(h5_path, result, frame_index)
                saved_path = h5_path
            else:
                edf_path = os.path.join(
                    self.output_folder_path,
                    f"{base_name}_frame_{frame_index:04d}.edf",
                )
                edf_image = EdfImage(data=np.asarray(result, dtype=np.float32))
                edf_image.header.update(self.sample_stack.header)
                edf_image.header.update(self.output_metadata())
                edf_image.write(edf_path)
                saved_path = edf_path

            if self.save_preview_checkbox.isChecked():
                png_path = os.path.join(
                    self.output_folder_path,
                    f"{base_name}_frame_{frame_index:04d}.png",
                )
                self.result_canvas.figure.savefig(
                    png_path,
                    dpi=300,
                    bbox_inches="tight",
                )
                self.log_text.append(f"Saved preview: {png_path}")

            self.log_text.append(f"Saved: {saved_path}")
            self.status_label.setText("Current frame saved.")

        except Exception as exc:
            self.status_label.setText(f"Save error: {exc}")

    def save_all_frames_as_npy(self):
        if not self.output_folder_path:
            self.status_label.setText("Select an output folder first.")
            return

        if self.sample_stack is None:
            self.status_label.setText("Load a sample first.")
            return

        if self.sample_stack.kind != "hdf5" and EdfImage is None:
            self.status_label.setText("fabio EDF support is not available.")
            return

        try:
            frame_count = self.sample_stack.frame_count
            if self.background_stack is not None:
                frame_count = min(frame_count, self.background_stack.frame_count)
            start_frame = max(0, self.frame_start_spin.value() - 1)
            end_frame = min(frame_count, self.frame_end_spin.value())

            base_name = self.get_output_base_name()

            for frame_index in range(start_frame, end_frame):
                result = self.compute_result_frame(frame_index)

                if self.sample_stack.kind == "hdf5":
                    h5_path = os.path.join(
                        self.output_folder_path,
                        f"{base_name}_frame_{frame_index:04d}.h5",
                    )
                    self.write_background_h5(h5_path, result, frame_index)
                    saved_path = h5_path
                else:
                    edf_path = os.path.join(
                        self.output_folder_path,
                        f"{base_name}_frame_{frame_index:04d}.edf",
                    )
                    edf_image = EdfImage(data=np.asarray(result, dtype=np.float32))
                    edf_image.header.update(self.sample_stack.header)
                    edf_image.header.update(self.output_metadata())
                    edf_image.write(edf_path)
                    saved_path = edf_path

                self.log_text.append(f"Saved: {saved_path}")

            self.status_label.setText(f"Saved {max(0, end_frame - start_frame)} frame(s).")

        except Exception as exc:
            self.status_label.setText(f"Save all error: {exc}")

    def write_background_h5(self, output_path, result, frame_index):
        """Copy the complete source H5/header and replace only the selected frame."""
        import h5py

        source_path = self.sample_file_path
        dataset_path = self.sample_stack.dataset_path
        with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as out:
            for key, value in source.attrs.items():
                out.attrs[key] = value
            for key in source.keys():
                source.copy(key, out, name=key)
            if dataset_path not in out:
                raise ValueError(f"Dataset {dataset_path} is missing from copied H5.")
            dataset = out[dataset_path]
            dataset_attrs = dict(dataset.attrs.items())
            source_data = np.asarray(dataset[()], dtype=np.float32)
            if self.sample_stack.frame_axis is None:
                source_data = np.asarray(result, dtype=np.float32)
            else:
                index = [slice(None)] * dataset.ndim
                index[self.sample_stack.frame_axis] = int(frame_index)
                source_data[tuple(index)] = np.asarray(result, dtype=np.float32)
            parent = out[dataset_path.rsplit("/", 1)[0] or "/"]
            name = dataset_path.rsplit("/", 1)[-1]
            del parent[name]
            dataset = parent.create_dataset(name, data=source_data, compression="gzip")
            for key, value in dataset_attrs.items():
                dataset.attrs[key] = value
            dataset.attrs["background_subtracted"] = True
            dataset.attrs["background_source"] = str(self.background_file_path)
            dataset.attrs["background_scale"] = float(self.background_scale_spin.value())
            dataset.attrs["background_offset"] = float(self.offset_spin.value())
            out.attrs["background_subtracted"] = True
            out.attrs["background_source"] = str(self.background_file_path)
            out.attrs["background_scale"] = float(self.background_scale_spin.value())
            out.attrs["background_offset"] = float(self.offset_spin.value())
            out.attrs["normalization_enabled"] = bool(self.normalization_enabled.isChecked())
            out.attrs["sample_exposure"] = float(self.sample_exposure_spin.value())
            out.attrs["background_exposure"] = float(self.background_exposure_spin.value())
            out.attrs["sample_thickness_mm"] = float(self.sample_thickness_spin.value())
            out.attrs["sample_flux"] = float(self.sample_flux_spin.value())
            out.attrs["background_flux"] = float(self.background_flux_spin.value())

    def run_background_subtraction(self):
        paths = self.sample_file_paths or ([self.sample_file_path] if self.sample_file_path else [])
        if not paths:
            self.status_label.setText("Select a sample file first.")
            return

        if not self.background_file_path and self.mask_array is None and not self.normalization_enabled.isChecked():
            self.status_label.setText("Select a mask, background or normalization first.")
            return

        saved_files = 0
        try:
            self.batch_progress.setVisible(len(paths) > 1)
            self.batch_progress.setRange(0, len(paths))
            self.batch_progress.setValue(0)
            self.batch_progress.setFormat("%v / %m files")
            self.run_button.setEnabled(False)
            for path in paths:
                self.sample_file_path = path
                self.sample_stack = self.open_image_stack(path)
                self.output_folder_path = os.path.dirname(path)
                self.update_frame_controls()
                self.save_all_frames_as_npy()
                saved_files += 1
                self.batch_progress.setValue(saved_files)
            self.status_label.setText(f"Run and Save complete: {saved_files} file(s).")
        except Exception as exc:
            self.status_label.setText(f"Run and Save error: {exc}")
        finally:
            self.run_button.setEnabled(True)
            if saved_files == len(paths):
                self.batch_progress.setVisible(False)
