from pathlib import Path
import fnmatch

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QCheckBox,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import cm, colors

from tabs.file_ratings import install_file_rating_menu, set_item_file_path
from tabs.ui_style import GROUP_BOX_STYLE, PAGE_MARGINS, PANEL_MARGINS


class SandboxTab(QWidget):
    folder_changed = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.folder_path = None
        self.current_folder = None
        self.current_image = None
        self.current_path = None
        self.current_stack = None
        self.current_frame_index = 0
        self.current_frame_count = 1
        self.max_3d_points = 90
        self.geometry_presets = {
            "XENOCS": {"center_x": 612.0, "center_y": 649.0, "pixel_size": 75.0, "distance": 900.0},
            "ID02": {"center_x": 0.0, "center_y": 0.0, "pixel_size": 75.0, "distance": 900.0},
            "ID13": {"center_x": 1294.689, "center_y": 1310.29, "pixel_size": 75.0, "distance": 900.0},
            "Custom": {"center_x": None, "center_y": None, "pixel_size": None, "distance": None},
            "+": {"center_x": None, "center_y": None, "pixel_size": None, "distance": None},
        }
        self.current_geometry = self.geometry_presets["XENOCS"].copy()
        self.build_ui()

    def build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        file_box = QGroupBox("File browser")
        file_box.setStyleSheet(GROUP_BOX_STYLE)
        file_box.setMinimumWidth(280)
        file_box.setMaximumWidth(420)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*PANEL_MARGINS)
        file_layout.setSpacing(6)

        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Folder")
        self.folder_edit.returnPressed.connect(self.folder_from_text)
        file_layout.addWidget(self.folder_edit)

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        file_layout.addWidget(browse_button)

        filters_layout = QHBoxLayout()
        filters_layout.setContentsMargins(0, 0, 0, 0)
        filters_layout.setSpacing(4)
        filters_layout.addWidget(QLabel("Name:"))
        self.name_filter = QLineEdit("*")
        filters_layout.addWidget(self.name_filter, 1)
        file_layout.addLayout(filters_layout)

        ext_layout = QHBoxLayout()
        ext_layout.setContentsMargins(0, 0, 0, 0)
        ext_layout.setSpacing(4)
        ext_layout.addWidget(QLabel("Extensions:"))
        self.extension_filter = QLineEdit("*.edf *.h5 *.hdf5")
        ext_layout.addWidget(self.extension_filter, 1)
        file_layout.addLayout(ext_layout)

        options_layout = QHBoxLayout()
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(4)
        self.subfolders_checkbox = QCheckBox("Show subfolders")
        options_layout.addWidget(self.subfolders_checkbox)
        options_layout.addStretch(1)
        file_layout.addLayout(options_layout)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.itemClicked.connect(self.open_selected_file)
        file_layout.addWidget(self.file_list, 1)

        plot_box = QGroupBox("3D SAXS pattern")
        plot_box.setStyleSheet(GROUP_BOX_STYLE)
        plot_layout = QVBoxLayout(plot_box)
        plot_layout.setContentsMargins(*PANEL_MARGINS)
        plot_layout.setSpacing(6)

        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar, 0)

        plot_area_layout = QHBoxLayout()
        plot_area_layout.setContentsMargins(0, 0, 0, 0)
        plot_area_layout.setSpacing(6)
        plot_area_layout.addWidget(self.canvas, 1)

        side_box = QGroupBox("Line geometry")
        side_box.setStyleSheet(GROUP_BOX_STYLE)
        side_box.setFixedWidth(180)
        side_layout = QVBoxLayout(side_box)
        side_layout.setContentsMargins(8, 20, 8, 8)
        side_layout.setSpacing(6)

        side_layout.addWidget(QLabel("Geometry"))
        self.geometry_combo = QComboBox()
        self.geometry_combo.addItems(["XENOCS", "ID02", "ID13", "Custom", "+"])
        self.geometry_combo.currentTextChanged.connect(self.apply_geometry_preset)
        side_layout.addWidget(self.geometry_combo)

        side_layout.addWidget(QLabel("Center X"))
        self.center_x_edit = QLineEdit()
        self.center_x_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addWidget(self.center_x_edit)

        side_layout.addWidget(QLabel("Center Y"))
        self.center_y_edit = QLineEdit()
        self.center_y_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addWidget(self.center_y_edit)

        side_layout.addWidget(QLabel("Pixel size (µm)"))
        self.pixel_size_edit = QLineEdit()
        self.pixel_size_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addWidget(self.pixel_size_edit)

        side_layout.addWidget(QLabel("Distance (mm)"))
        self.distance_edit = QLineEdit()
        self.distance_edit.returnPressed.connect(self.apply_custom_geometry_from_fields)
        side_layout.addWidget(self.distance_edit)

        self.apply_geometry_button = QPushButton("Apply")
        self.apply_geometry_button.clicked.connect(self.apply_custom_geometry_from_fields)
        side_layout.addWidget(self.apply_geometry_button)

        self.wireframe_checkbox = QCheckBox("Wireframe")
        self.wireframe_checkbox.setChecked(False)
        self.wireframe_checkbox.toggled.connect(self.replot_current_image)
        side_layout.addWidget(self.wireframe_checkbox)

        side_layout.addWidget(QLabel("Displayed pixels"))
        self.displayed_pixels_spinbox = QSpinBox()
        self.displayed_pixels_spinbox.setRange(20, 600)
        self.displayed_pixels_spinbox.setSingleStep(10)
        self.displayed_pixels_spinbox.setValue(self.max_3d_points)
        self.displayed_pixels_spinbox.setToolTip("Maximum number of displayed points along the largest image dimension. Higher values are more precise but slower.")
        self.displayed_pixels_spinbox.valueChanged.connect(self.update_display_precision)
        side_layout.addWidget(self.displayed_pixels_spinbox)

        side_layout.addWidget(QLabel("Precision step"))
        self.precision_step_spinbox = QSpinBox()
        self.precision_step_spinbox.setRange(1, 100)
        self.precision_step_spinbox.setValue(1)
        self.precision_step_spinbox.setToolTip("Additional pixel skipping. 1 keeps the automatic display density; higher values show fewer pixels.")
        self.precision_step_spinbox.valueChanged.connect(self.replot_current_image)
        side_layout.addWidget(self.precision_step_spinbox)

        side_layout.addWidget(QLabel("Intensity min"))
        self.z_min_edit = QLineEdit()
        self.z_min_edit.setPlaceholderText("auto")
        self.z_min_edit.returnPressed.connect(self.replot_current_image)
        side_layout.addWidget(self.z_min_edit)

        side_layout.addWidget(QLabel("Intensity max"))
        self.z_max_edit = QLineEdit()
        self.z_max_edit.setPlaceholderText("auto")
        self.z_max_edit.returnPressed.connect(self.replot_current_image)
        side_layout.addWidget(self.z_max_edit)

        self.reset_z_scale_button = QPushButton("Auto intensity scale")
        self.reset_z_scale_button.clicked.connect(self.reset_intensity_scale)
        side_layout.addWidget(self.reset_z_scale_button)

        side_layout.addStretch(1)
        self.update_geometry_fields()

        plot_area_layout.addWidget(side_box, 0)
        plot_layout.addLayout(plot_area_layout, 1)

        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(0, 0, 0, 0)
        bottom_bar.setSpacing(6)

        self.previous_frame_button = QPushButton("◀")
        self.previous_frame_button.clicked.connect(self.previous_frame)
        bottom_bar.addWidget(self.previous_frame_button)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.valueChanged.connect(self.set_frame_from_slider)
        bottom_bar.addWidget(self.frame_slider, 1)

        self.next_frame_button = QPushButton("▶")
        self.next_frame_button.clicked.connect(self.next_frame)
        bottom_bar.addWidget(self.next_frame_button)

        self.frame_label = QLabel("Frame 1 / 1")
        self.frame_label.setMinimumWidth(90)
        bottom_bar.addWidget(self.frame_label)

        self.auto_contrast_button = QPushButton("Auto contrast")
        self.auto_contrast_button.clicked.connect(self.auto_contrast)
        bottom_bar.addWidget(self.auto_contrast_button)

        bottom_bar.addWidget(QLabel("Contrast min"))
        self.contrast_min_slider = QSlider(Qt.Horizontal)
        self.contrast_min_slider.setRange(0, 999)
        self.contrast_min_slider.setValue(10)
        self.contrast_min_slider.valueChanged.connect(self.replot_current_image)
        bottom_bar.addWidget(self.contrast_min_slider, 1)

        bottom_bar.addWidget(QLabel("max"))
        self.contrast_max_slider = QSlider(Qt.Horizontal)
        self.contrast_max_slider.setRange(1, 1000)
        self.contrast_max_slider.setValue(990)
        self.contrast_max_slider.valueChanged.connect(self.replot_current_image)
        bottom_bar.addWidget(self.contrast_max_slider, 1)


        plot_layout.addLayout(bottom_bar, 0)

        self.status_label = QLabel("Open an EDF/H5 SAXS image to display it as a 3D surface.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(22)
        plot_layout.addWidget(self.status_label, 0)

        splitter.addWidget(file_box)
        splitter.addWidget(plot_box)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def apply_geometry_preset(self, name):
        preset = self.geometry_presets.get(name)
        if preset is None:
            return
        self.current_geometry = preset.copy()
        self.update_geometry_fields()
        self.replot_current_image()

    def update_geometry_fields(self):
        geometry = self.current_geometry
        self.center_x_edit.setText("" if geometry.get("center_x") is None else f"{geometry['center_x']:.3f}")
        self.center_y_edit.setText("" if geometry.get("center_y") is None else f"{geometry['center_y']:.3f}")
        self.pixel_size_edit.setText("" if geometry.get("pixel_size") is None else f"{geometry['pixel_size']:.3f}")
        self.distance_edit.setText("" if geometry.get("distance") is None else f"{geometry['distance']:.3f}")

    def apply_custom_geometry_from_fields(self):
        self.current_geometry = {
            "center_x": self.optional_float_from_text(self.center_x_edit.text()),
            "center_y": self.optional_float_from_text(self.center_y_edit.text()),
            "pixel_size": self.optional_float_from_text(self.pixel_size_edit.text()),
            "distance": self.optional_float_from_text(self.distance_edit.text()),
        }
        if self.geometry_combo.currentText() not in {"Custom", "+"}:
            self.geometry_combo.blockSignals(True)
            self.geometry_combo.setCurrentText("Custom")
            self.geometry_combo.blockSignals(False)
        self.replot_current_image()

    def optional_float_from_text(self, text):
        text = text.strip().replace(",", ".")
        if not text:
            return None
        return float(text)

    def set_folder(self, folder):
        if folder is None:
            return
        folder = Path(folder)
        if not folder.exists():
            return
        self.folder_path = folder
        self.current_folder = folder
        self.folder_edit.setText(str(folder))
        self.refresh_files()

    def set_folder_from_external_tab(self, folder):
        self.set_folder(folder)

    def folder_from_text(self):
        self.set_folder(self.folder_edit.text().strip())

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", self.folder_edit.text() or str(Path.home()))
        if folder:
            self.set_folder(folder)
            self.folder_changed.emit(Path(folder), self)

    def refresh_files(self):
        self.file_list.clear()
        if self.folder_path is None:
            return

        folder = Path(self.folder_path)
        if not folder.exists():
            return

        name_pattern = self.name_filter.text().strip() or "*"
        extension_patterns = self.extension_filter.text().split() or ["*.edf", "*.h5", "*.hdf5"]
        iterator = folder.rglob("*") if self.subfolders_checkbox.isChecked() else folder.glob("*")

        files = []
        for path in iterator:
            if not path.is_file():
                continue
            lower_name = path.name.lower()
            if not any(fnmatch.fnmatch(lower_name, pattern.lower()) for pattern in extension_patterns):
                continue
            if not fnmatch.fnmatch(path.name, name_pattern):
                continue
            files.append(path)

        for path in sorted(files):
            item = QListWidgetItem(str(path.relative_to(folder)))
            set_item_file_path(item, path)
            self.file_list.addItem(item)

    def selected_file(self):
        items = self.file_list.selectedItems()
        if not items:
            return None
        path = items[0].data(Qt.UserRole)
        if path is None and self.current_folder is not None:
            path = Path(self.current_folder) / items[0].text()
        return Path(path) if path is not None else None

    def open_selected_file(self):
        path = self.selected_file()
        if path is None:
            return
        try:
            stack = self.load_stack(path)
        except Exception as exc:
            self.status_label.setText(f"Could not open file: {exc}")
            return

        self.current_path = path
        self.current_stack = stack
        self.current_frame_count = stack.shape[0]
        self.current_frame_index = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setRange(0, max(0, self.current_frame_count - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.update_frame_controls()
        self.set_current_image_from_stack()

    def load_stack(self, path):
        suffix = path.suffix.lower()
        if suffix == ".edf":
            return self.load_edf_stack(path)
        if suffix in {".h5", ".hdf5"}:
            import h5py

            return self.load_h5_stack(path, h5py)
        raise ValueError(f"Unsupported file type: {suffix}")

    def load_edf_stack(self, path):
        import fabio

        edf = fabio.open(str(path))
        frames = []
        index = 0
        while edf is not None:
            data = np.asarray(edf.data, dtype=float)
            frames.append(self.clean_image(data))
            index += 1
            try:
                edf = edf.next()
            except Exception:
                break
        if not frames:
            raise ValueError("No EDF frame found")
        return np.stack(frames, axis=0)

    def clean_image(self, data):
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D image, got shape {data.shape}")
        data[data > 4e9] = np.nan
        return data

    def load_h5_stack(self, path, h5py):
        candidates = []

        with h5py.File(path, "r") as h5:
            def visitor(name, obj):
                if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                    shape = tuple(obj.shape)
                    if len(shape) in {2, 3}:
                        candidates.append((name, shape))

            h5.visititems(visitor)
            if not candidates:
                raise ValueError("No 2D or 3D dataset found in H5 file")

            name, shape = sorted(candidates, key=lambda item: (len(item[1]) != 3, item[1]))[0]
            dataset = h5[name]
            data = np.asarray(dataset[...], dtype=float)

        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            pass
        else:
            raise ValueError(f"Expected a 2D or 3D dataset, got shape {data.shape}")

        frames = [self.clean_image(frame) for frame in data]
        return np.stack(frames, axis=0)

    def set_current_image_from_stack(self):
        if self.current_stack is None:
            return
        self.current_frame_index = max(0, min(self.current_frame_index, self.current_frame_count - 1))
        self.current_image = self.current_stack[self.current_frame_index]
        self.update_frame_controls()
        self.plot_3d(self.current_image)

    def update_frame_controls(self):
        self.frame_label.setText(f"Frame {self.current_frame_index + 1} / {self.current_frame_count}")
        enabled = self.current_frame_count > 1
        self.previous_frame_button.setEnabled(enabled and self.current_frame_index > 0)
        self.next_frame_button.setEnabled(enabled and self.current_frame_index < self.current_frame_count - 1)
        self.frame_slider.setEnabled(enabled)

    def set_frame_from_slider(self, value):
        if self.current_stack is None:
            return
        self.current_frame_index = int(value)
        self.set_current_image_from_stack()

    def previous_frame(self):
        if self.current_stack is None:
            return
        self.current_frame_index = max(0, self.current_frame_index - 1)
        self.frame_slider.setValue(self.current_frame_index)

    def next_frame(self):
        if self.current_stack is None:
            return
        self.current_frame_index = min(self.current_frame_count - 1, self.current_frame_index + 1)
        self.frame_slider.setValue(self.current_frame_index)

    def replot_current_image(self):
        if self.current_image is not None:
            self.plot_3d(self.current_image)

    def update_display_precision(self, value):
        self.max_3d_points = int(value)
        self.replot_current_image()

    def auto_contrast(self):
        self.contrast_min_slider.blockSignals(True)
        self.contrast_max_slider.blockSignals(True)
        self.contrast_min_slider.setValue(10)
        self.contrast_max_slider.setValue(990)
        self.contrast_min_slider.blockSignals(False)
        self.contrast_max_slider.blockSignals(False)
        self.z_max_edit.clear()
        self.replot_current_image()

    def reset_intensity_scale(self):
        self.z_min_edit.clear()
        self.z_max_edit.clear()
        self.replot_current_image()

    def plot_colored_wireframe(self, ax, xx, yy, z, vmin, vmax):
        z_data = np.asarray(z.filled(np.nan), dtype=float)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("jet")

        for row_index in range(z_data.shape[0]):
            row_z = z_data[row_index, :]
            finite = np.isfinite(row_z)
            if np.count_nonzero(finite) < 2:
                continue
            row_color = cmap(norm(np.nanmean(row_z[finite])))
            ax.plot(
                xx[row_index, finite],
                yy[row_index, finite],
                row_z[finite],
                color=row_color,
                linewidth=0.45,
            )

        for column_index in range(z_data.shape[1]):
            column_z = z_data[:, column_index]
            finite = np.isfinite(column_z)
            if np.count_nonzero(finite) < 2:
                continue
            column_color = cmap(norm(np.nanmean(column_z[finite])))
            ax.plot(
                xx[finite, column_index],
                yy[finite, column_index],
                column_z[finite],
                color=column_color,
                linewidth=0.45,
            )

    def plot_3d(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        self.ax = ax

        display = np.array(image, dtype=float)
        finite = np.isfinite(display)
        if not np.any(finite):
            self.status_label.setText("Image contains no finite intensity values.")
            self.canvas.draw_idle()
            return

        finite_display = np.isfinite(display)
        low_percentile = self.contrast_min_slider.value() / 10.0
        high_percentile = self.contrast_max_slider.value() / 10.0
        if high_percentile <= low_percentile:
            high_percentile = min(100.0, low_percentile + 0.1)
        vmin = np.nanpercentile(display[finite_display], low_percentile)
        vmax = np.nanpercentile(display[finite_display], high_percentile)
        if vmax <= vmin:
            vmax = vmin + 1e-12

        auto_z_min = float(np.nanmin(display[finite_display]))
        auto_z_max = float(np.nanmax(display[finite_display]))
        z_min = self.optional_float_from_text(self.z_min_edit.text())
        z_max = self.optional_float_from_text(self.z_max_edit.text())
        if z_min is None:
            z_min = auto_z_min
        if z_max is None:
            z_max = auto_z_max
        if z_max <= z_min:
            z_max = z_min + 1e-12

        max_points = self.max_3d_points
        precision_step = self.precision_step_spinbox.value()
        step_y = max(1, int(np.ceil(display.shape[0] / max_points))) * precision_step
        step_x = max(1, int(np.ceil(display.shape[1] / max_points))) * precision_step

        z = display[::step_y, ::step_x]
        y = np.arange(0, display.shape[0], step_y)
        x = np.arange(0, display.shape[1], step_x)
        xx, yy = np.meshgrid(x, y)

        z = np.asarray(z, dtype=float)
        z[(z > z_max) | (z < z_min)] = np.nan
        z = np.ma.masked_invalid(z)
        z_plot = z.filled(np.nan)
        if z.count() == 0:
            self.status_label.setText("3D downsample contains no finite intensity values.")
            self.canvas.draw_idle()
            return

        geometry = self.current_geometry
        center_x = geometry.get("center_x")
        center_y = geometry.get("center_y")
        if center_x is not None and center_y is not None:
            ax.set_xlabel("x - center (px)")
            ax.set_ylabel("y - center (px)")
            ax.set_xlim(float(np.nanmin(xx - center_x)), float(np.nanmax(xx - center_x)))
            ax.set_ylim(float(np.nanmin(yy - center_y)), float(np.nanmax(yy - center_y)))
            if self.wireframe_checkbox.isChecked():
                self.plot_colored_wireframe(ax, xx - center_x, yy - center_y, z, max(vmin, z_min), min(vmax, z_max))
            else:
                surface = ax.plot_surface(
                    xx - center_x,
                    yy - center_y,
                    z_plot,
                    cmap="jet",
                    vmin=max(vmin, z_min),
                    vmax=min(vmax, z_max),
                    linewidth=0,
                    antialiased=False,
                    rstride=1,
                    cstride=1,
                )
                surface.set_clim(max(vmin, z_min), min(vmax, z_max))
        else:
            ax.set_xlabel("x (px)")
            ax.set_ylabel("y (px)")
            if self.wireframe_checkbox.isChecked():
                self.plot_colored_wireframe(ax, xx, yy, z, max(vmin, z_min), min(vmax, z_max))
            else:
                surface = ax.plot_surface(
                    xx,
                    yy,
                    z_plot,
                    cmap="jet",
                    vmin=max(vmin, z_min),
                    vmax=min(vmax, z_max),
                    linewidth=0,
                    antialiased=False,
                    rstride=1,
                    cstride=1,
                )
                surface.set_clim(max(vmin, z_min), min(vmax, z_max))
        ax.set_zlim(z_min, z_max)
        ax.set_zlabel("Intensity")
        ax.view_init(elev=35, azim=-60)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status_label.clear()
