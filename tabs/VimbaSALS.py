from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from tabs.line_geometry import LineGeometrySelector, default_center_text
from tabs.ui_style import GROUP_BOX_STYLE, PAGE_MARGINS, PANEL_MARGINS


class VimbaSALSWidget(QWidget):
    back_requested = Signal()
    CAMERA_ID = "DEV_000F315BB2BF"
    CAMERA_MODEL = "Mako G-419B"
    DEFAULT_ROI_SIZE = 796
    DEFAULT_PIXEL_FORMAT = "Mono12Packed"
    DEFAULT_OFFSET_X = "626"
    DEFAULT_OFFSET_Y = "626"
    DEFAULT_DISTANCE_M = "0,0035"
    DEFAULT_PIXEL_SIZE_M = "5,5e-6"
    DEFAULT_WAVELENGTH_M = "632,8e-9"
    TEST_IMAGE_PATH = Path("/Users/nathanpiaget/Documents/Thèse LRP/Expériences/SALSGels/cnc6/10.6relax_nrm.edf")

    def __init__(self):
        super().__init__()
        self.vmb = None
        self.camera = None
        self.current_frame = None
        self.frame_index = 0
        self.output_folder = Path.home() / "LRPhoton_SALS"
        self.current_geometry_name = "SALS default"

        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self.grab_live_frame)

        self.build_ui()
        self.update_connection_state(False)

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(6)

        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(6)
        main_layout.addLayout(body_layout, 1)

        controls_box = QGroupBox("SALS acquisition")
        controls_box.setStyleSheet(GROUP_BOX_STYLE)
        controls_box.setFixedWidth(360)
        controls_layout = QVBoxLayout(controls_box)
        controls_layout.setContentsMargins(*PANEL_MARGINS)
        controls_layout.setSpacing(6)
        body_layout.addWidget(controls_box, 0)

        self.connect_button = QPushButton("Connect camera")
        self.connect_button.clicked.connect(self.connect_camera)
        controls_layout.addWidget(self.connect_button)

        live_buttons_layout = QHBoxLayout()
        live_buttons_layout.setContentsMargins(0, 0, 0, 0)
        live_buttons_layout.setSpacing(4)
        self.start_button = QPushButton("Start live")
        self.start_button.clicked.connect(self.start_live)
        live_buttons_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop live")
        self.stop_button.clicked.connect(self.stop_live)
        live_buttons_layout.addWidget(self.stop_button)
        controls_layout.addLayout(live_buttons_layout)

        self.test_image_button = QPushButton("Test image")
        self.test_image_button.setToolTip("Load the SALS EDF test image instead of grabbing the camera.")
        self.test_image_button.clicked.connect(self.load_test_image)
        controls_layout.addWidget(self.test_image_button)

        self.exposure_edit = QLineEdit("10000")
        self.exposure_edit.setToolTip("Vimba ExposureTime in microseconds when available.")
        self.gain_edit = QLineEdit("")
        self.gain_edit.setPlaceholderText("auto/unchanged")
        self.width_spinbox = QSpinBox()
        self.width_spinbox.setRange(1, 10000)
        self.width_spinbox.setValue(self.DEFAULT_ROI_SIZE)
        self.height_spinbox = QSpinBox()
        self.height_spinbox.setRange(1, 10000)
        self.height_spinbox.setValue(self.DEFAULT_ROI_SIZE)
        self.offset_x_spinbox = QSpinBox()
        self.offset_x_spinbox.setRange(0, 10000)
        self.offset_x_spinbox.setValue(0)
        self.offset_y_spinbox = QSpinBox()
        self.offset_y_spinbox.setRange(0, 10000)
        self.offset_y_spinbox.setValue(0)
        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.setEditable(True)
        self.pixel_format_combo.addItems([
            self.DEFAULT_PIXEL_FORMAT,
            "Mono12",
            "Mono12p",
            "Mono16",
            "Mono8",
        ])
        self.pixel_format_combo.setCurrentText(self.DEFAULT_PIXEL_FORMAT)
        self.pixel_format_combo.setToolTip("Camera PixelFormat. Packed formats are converted internally for display.")
        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(1, 60)
        self.fps_spinbox.setValue(2)

        camera_form = QFormLayout()
        camera_form.setContentsMargins(0, 0, 0, 0)
        camera_form.setSpacing(6)
        camera_form.addRow("Exposure (µs)", self.exposure_edit)
        camera_form.addRow("Gain", self.gain_edit)
        roi_x_layout = QHBoxLayout()
        roi_x_layout.setContentsMargins(0, 0, 0, 0)
        roi_x_layout.setSpacing(4)
        roi_x_layout.addWidget(self.width_spinbox, 1)
        roi_x_layout.addWidget(QLabel("Offset X"))
        roi_x_layout.addWidget(self.offset_x_spinbox, 1)
        camera_form.addRow("Width", roi_x_layout)
        roi_y_layout = QHBoxLayout()
        roi_y_layout.setContentsMargins(0, 0, 0, 0)
        roi_y_layout.setSpacing(4)
        roi_y_layout.addWidget(self.height_spinbox, 1)
        roi_y_layout.addWidget(QLabel("Offset Y"))
        roi_y_layout.addWidget(self.offset_y_spinbox, 1)
        camera_form.addRow("Height", roi_y_layout)
        camera_form.addRow("Pixel format", self.pixel_format_combo)
        camera_form.addRow("Preview fps", self.fps_spinbox)
        controls_layout.addLayout(camera_form)

        self.apply_camera_button = QPushButton("Apply camera settings")
        self.apply_camera_button.clicked.connect(self.apply_camera_settings)
        controls_layout.addWidget(self.apply_camera_button)

        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(4)
        self.output_edit = QLineEdit(str(self.output_folder))
        output_layout.addWidget(self.output_edit, 1)
        self.output_button = QPushButton("Browse")
        self.output_button.clicked.connect(self.choose_output_folder)
        output_layout.addWidget(self.output_button)
        controls_layout.addWidget(QLabel("Output folder"))
        controls_layout.addLayout(output_layout)

        self.prefix_edit = QLineEdit("sals")
        self.save_button = QPushButton("Save current EDF")
        self.save_button.clicked.connect(self.save_current_edf)
        controls_layout.addWidget(QLabel("File prefix"))
        controls_layout.addWidget(self.prefix_edit)
        controls_layout.addWidget(self.save_button)

        sals_box = QGroupBox("EDF SALS parameters")
        sals_box.setStyleSheet(GROUP_BOX_STYLE)
        sals_layout = QFormLayout(sals_box)
        sals_layout.setContentsMargins(8, 14, 8, 6)
        sals_layout.setSpacing(5)
        controls_layout.addWidget(sals_box)

        self.geometry_selector = LineGeometrySelector(self, "SALS default")
        self.geometry_selector.geometry_selected.connect(self.apply_line_geometry)

        self.distance_edit = QLineEdit("")
        self.pixel_x_edit = QLineEdit("")
        self.pixel_y_edit = QLineEdit("")
        self.wavelength_edit = QLineEdit("")
        self.center_x_edit = QLineEdit(self.default_center_text(self.DEFAULT_ROI_SIZE))
        self.center_y_edit = QLineEdit(self.default_center_text(self.DEFAULT_ROI_SIZE))
        self.width_spinbox.valueChanged.connect(self.update_default_center_x)
        self.height_spinbox.valueChanged.connect(self.update_default_center_y)

        self.distance_edit.setPlaceholderText("m")
        self.pixel_x_edit.setPlaceholderText("m")
        self.pixel_y_edit.setPlaceholderText("m")
        self.wavelength_edit.setPlaceholderText("m")
        sals_layout.addRow("Ligne", self.geometry_selector)
        sals_layout.addRow("Distance (m)", self.distance_edit)
        sals_layout.addRow("Pixel X (m)", self.pixel_x_edit)
        sals_layout.addRow("Pixel Y (m)", self.pixel_y_edit)
        sals_layout.addRow("Wavelength (m)", self.wavelength_edit)
        sals_layout.addRow("Center X", self.center_x_edit)
        sals_layout.addRow("Center Y", self.center_y_edit)
        self.apply_line_geometry(self.geometry_selector.current_name, self.geometry_selector.current_geometry())

        controls_layout.addStretch(1)

        preview_box = QGroupBox("Live EDF preview")
        preview_box.setStyleSheet(GROUP_BOX_STYLE)
        preview_layout = QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(*PANEL_MARGINS)
        preview_layout.setSpacing(6)
        body_layout.addWidget(preview_box, 1)

        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.image_artist = None
        preview_layout.addWidget(self.canvas, 1)

        self.status_label = QLabel("Vimba is not connected.")
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.status_label.setMinimumHeight(22)
        self.status_label.setWordWrap(True)
        preview_layout.addWidget(self.status_label, 0)

    def update_connection_state(self, connected):
        self.connect_button.setEnabled(not connected)
        self.start_button.setEnabled(connected and not self.live_timer.isActive())
        self.stop_button.setEnabled(self.live_timer.isActive())
        self.apply_camera_button.setEnabled(connected)
        self.save_button.setEnabled(self.current_frame is not None)

    def request_back(self):
        if self.live_timer.isActive():
            self.stop_live()
        self.back_requested.emit()

    def connect_camera(self):
        try:
            from vmbpy import VmbSystem
        except ImportError:
            self.status_label.setText("VmbPy is missing. Reinstall or update LRPhoton with the bundled camera support.")
            return

        try:
            self.vmb = VmbSystem.get_instance()
            self.vmb.__enter__()
            cameras = list(self.vmb.get_all_cameras())
            if not cameras:
                raise RuntimeError("No Vimba camera detected. Close Vimba X Viewer if it has full access.")

            self.camera = self.select_mako_camera(cameras)
            self.camera.__enter__()
            self.apply_camera_settings()
            self.sync_fields_from_camera()
            self.status_label.setText(f"Connected: {self.camera.get_id()}")
            self.update_connection_state(True)
        except Exception as exc:
            self.disconnect_camera()
            self.status_label.setText(f"Camera connection failed: {exc}")

    def select_mako_camera(self, cameras):
        for camera in cameras:
            if camera.get_id() == self.CAMERA_ID:
                return camera
            text = f"{camera.get_id()} {getattr(camera, 'get_name', lambda: '')()}".lower()
            if "g419" in text or "mako" in text:
                return camera
        return cameras[0]

    def apply_camera_settings(self):
        if self.camera is None:
            return
        self.set_camera_feature("OffsetX", self.offset_x_spinbox.value())
        self.set_camera_feature("OffsetY", self.offset_y_spinbox.value())
        self.set_camera_feature("Width", self.width_spinbox.value())
        self.set_camera_feature("Height", self.height_spinbox.value())
        requested_pixel_format = self.pixel_format_combo.currentText().strip()
        self.set_camera_feature("PixelFormat", requested_pixel_format)
        self.set_camera_feature("ExposureTime", self.optional_float(self.exposure_edit.text()))
        self.set_camera_feature("Gain", self.optional_float(self.gain_edit.text()))
        self.set_camera_feature("AcquisitionFrameRate", float(self.fps_spinbox.value()))

    def set_camera_feature(self, name, value):
        if value is None:
            return
        try:
            feature = self.camera.get_feature_by_name(name)
            feature.set(value)
        except Exception:
            pass

    def sync_fields_from_camera(self):
        width = self.camera_feature_value("Width")
        height = self.camera_feature_value("Height")
        offset_x = self.camera_feature_value("OffsetX")
        offset_y = self.camera_feature_value("OffsetY")
        pixel_format = self.camera_feature_value("PixelFormat")
        exposure = self.camera_feature_value("ExposureTime")
        gain = self.camera_feature_value("Gain")
        fps = self.camera_feature_value("AcquisitionFrameRate")
        self.refresh_pixel_format_choices()

        if width is not None:
            self.width_spinbox.setValue(int(width))
        if height is not None:
            self.height_spinbox.setValue(int(height))
        if offset_x is not None:
            self.offset_x_spinbox.setValue(int(offset_x))
        if offset_y is not None:
            self.offset_y_spinbox.setValue(int(offset_y))
        if pixel_format is not None:
            self.set_pixel_format_text(str(pixel_format))
        if exposure is not None:
            self.exposure_edit.setText(f"{float(exposure):.10g}")
        if gain is not None:
            self.gain_edit.setText(f"{float(gain):.10g}")
        if fps is not None:
            self.fps_spinbox.setValue(max(1, min(60, int(round(float(fps))))))

    def camera_feature_value(self, name):
        if self.camera is None:
            return None
        try:
            feature = self.camera.get_feature_by_name(name)
            return feature.get()
        except Exception:
            return None

    def refresh_pixel_format_choices(self):
        if self.camera is None:
            return
        current_text = self.pixel_format_combo.currentText().strip() or self.DEFAULT_PIXEL_FORMAT
        try:
            formats = [str(pixel_format) for pixel_format in self.camera.get_pixel_formats()]
        except Exception:
            return
        if not formats:
            return
        self.pixel_format_combo.blockSignals(True)
        self.pixel_format_combo.clear()
        for pixel_format in formats:
            self.pixel_format_combo.addItem(pixel_format)
        if current_text in formats:
            self.pixel_format_combo.setCurrentText(current_text)
        elif self.DEFAULT_PIXEL_FORMAT in formats:
            self.pixel_format_combo.setCurrentText(self.DEFAULT_PIXEL_FORMAT)
        else:
            self.pixel_format_combo.setCurrentIndex(0)
        self.pixel_format_combo.blockSignals(False)

    def set_pixel_format_text(self, text):
        if self.pixel_format_combo.findText(text) < 0:
            self.pixel_format_combo.addItem(text)
        self.pixel_format_combo.setCurrentText(text)

    def apply_line_geometry(self, name, geometry):
        self.current_geometry_name = name
        self.center_x_edit.setText(str(geometry.get("center_x", "")))
        self.center_y_edit.setText(str(geometry.get("center_y", "")))
        self.pixel_x_edit.setText(str(geometry.get("pixel_x_m", "")))
        self.pixel_y_edit.setText(str(geometry.get("pixel_y_m", "")))
        self.distance_edit.setText(str(geometry.get("distance_m", "")))
        self.wavelength_edit.setText(str(geometry.get("wavelength_m", "")))
        if self.current_frame is not None:
            self.update_preview()

    def update_default_center_x(self, value):
        if self.current_geometry_name == "SALS default":
            self.center_x_edit.setText(self.default_center_text(value))

    def update_default_center_y(self, value):
        if self.current_geometry_name == "SALS default":
            self.center_y_edit.setText(self.default_center_text(value))

    def start_live(self):
        if self.camera is None:
            self.connect_camera()
        if self.camera is None:
            return
        self.apply_camera_settings()
        interval_ms = max(1, int(1000 / max(1, self.fps_spinbox.value())))
        self.live_timer.start(interval_ms)
        self.status_label.setText("Live acquisition running.")
        self.update_connection_state(True)

    def stop_live(self):
        self.live_timer.stop()
        self.status_label.setText("Live acquisition stopped.")
        self.update_connection_state(self.camera is not None)

    def load_test_image(self):
        if self.live_timer.isActive():
            self.stop_live()
        try:
            import fabio

            edf = fabio.open(str(self.TEST_IMAGE_PATH))
            image = np.asarray(edf.data)
            if image.ndim != 2:
                raise ValueError(f"Expected a 2D EDF image, got shape {image.shape}.")
            self.current_frame = np.flipud(image)
            self.frame_index += 1
            self.width_spinbox.setValue(int(self.current_frame.shape[1]))
            self.height_spinbox.setValue(int(self.current_frame.shape[0]))
            self.set_pixel_format_text("Test EDF")
            self.update_preview()
            self.save_button.setEnabled(True)
            self.status_label.setText(f"Test image loaded: {self.TEST_IMAGE_PATH.name}")
            self.update_connection_state(self.camera is not None)
        except Exception as exc:
            self.status_label.setText(f"Could not load test image: {exc}")

    def grab_live_frame(self):
        if self.camera is None:
            self.stop_live()
            return
        try:
            frame = self.camera.get_frame(timeout_ms=1000)
            image = np.asarray(self.frame_to_numpy(frame))
            if image.ndim == 3 and image.shape[-1] == 1:
                image = image[:, :, 0]
            elif image.ndim == 3:
                image = image.mean(axis=2)
            image = np.flipud(image)
            self.current_frame = np.asarray(image)
            self.frame_index += 1
            self.update_preview()
            self.save_button.setEnabled(True)
        except Exception as exc:
            self.status_label.setText(f"Frame grab failed: {exc}")

    def frame_to_numpy(self, frame):
        source_format = frame.get_pixel_format()
        source_name = str(source_format)
        if "Packed" in source_name or source_name.endswith("p"):
            return self.convert_frame_to_numpy(frame, source_format)

        try:
            return frame.as_numpy_ndarray()
        except Exception as error:
            if "PixelFormat" not in str(error):
                raise

        return self.convert_frame_to_numpy(frame, source_format)

    def convert_frame_to_numpy(self, frame, source_format):
        from vmbpy import PixelFormat

        convertible = source_format.get_convertible_formats()
        preferred_formats = (PixelFormat.Mono16, PixelFormat.Mono12, PixelFormat.Mono8)
        for target_format in preferred_formats:
            if target_format not in convertible:
                continue
            try:
                converted_frame = frame.convert_pixel_format(target_format)
                return converted_frame.as_numpy_ndarray()
            except Exception:
                continue

        raise ValueError(f"PixelFormat {source_format} cannot be converted to Mono16, Mono12 or Mono8.")

    def update_preview(self):
        if self.current_frame is None:
            return
        image = np.asarray(self.current_frame, dtype=float)
        self.ax.clear()
        finite = np.isfinite(image)
        if np.any(finite):
            vmin, vmax = np.nanpercentile(image[finite], [1, 99])
            if vmax <= vmin:
                vmax = vmin + 1
        else:
            vmin, vmax = 0, 1
        self.ax.imshow(image, cmap="jet", origin="upper", vmin=vmin, vmax=vmax)
        center_x = self.optional_float(self.center_x_edit.text())
        center_y = self.optional_float(self.center_y_edit.text())
        if center_x is None:
            center_x = (image.shape[1] - 1.0) / 2.0
        if center_y is None:
            center_y = (image.shape[0] - 1.0) / 2.0
        self.ax.axvline(center_x, color="white", linewidth=0.8, alpha=0.9)
        self.ax.axhline(center_y, color="white", linewidth=0.8, alpha=0.9)
        self.ax.set_title(f"Frame {self.frame_index} - {image.shape[1]} x {image.shape[0]}")
        self.ax.set_axis_off()
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def choose_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose EDF output folder", self.output_edit.text())
        if folder:
            self.output_folder = Path(folder)
            self.output_edit.setText(str(self.output_folder))

    def save_current_edf(self):
        if self.current_frame is None:
            self.status_label.setText("No live frame to save.")
            return
        try:
            from fabio.edfimage import EdfImage
        except ImportError:
            self.status_label.setText("fabio EDF support is not available.")
            return

        folder = Path(self.output_edit.text()).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        prefix = self.prefix_edit.text().strip() or "sals"
        output_path = folder / f"{prefix}_{timestamp}.edf"

        image = np.asarray(self.current_frame)
        header = self.edf_header(image, timestamp)
        try:
            edf = EdfImage(data=image, header=header)
            edf.write(str(output_path))
        except TypeError:
            edf = EdfImage(data=image)
            edf.header.update(header)
            edf.write(str(output_path))
        self.status_label.setText(f"Saved EDF: {output_path.name}")

    def edf_header(self, image, timestamp):
        ny, nx = image.shape[:2]
        center_x = self.center_x_edit.text().strip()
        center_y = self.center_y_edit.text().strip()
        distance_m = self.optional_float(self.distance_edit.text())
        pixel_x_m = self.optional_float(self.pixel_x_edit.text())
        pixel_y_m = self.optional_float(self.pixel_y_edit.text())
        wavelength_m = self.optional_float(self.wavelength_edit.text())
        header = {
            "HeaderID": "EH:000001:000000:000000",
            "Image": str(self.frame_index),
            "ByteOrder": "LowByteFirst",
            "DataType": str(image.dtype),
            "Dim_1": str(nx),
            "Dim_2": str(ny),
            "Camera": f"Allied Vision {self.CAMERA_MODEL}",
            "CameraID": self.CAMERA_ID,
            "PixelFormat": self.pixel_format_combo.currentText().strip(),
            "ROIWidth": str(nx),
            "ROIHeight": str(ny),
            "OffsetX": str(self.offset_x_spinbox.value()),
            "OffsetY": str(self.offset_y_spinbox.value()),
            "AcquisitionDate": timestamp,
            "ExposureTime": self.exposure_edit.text().strip(),
            "Gain": self.gain_edit.text().strip(),
            "LRPhotonModule": "VimbaSALS",
            "LineGeometry": self.current_geometry_name,
            "SampleDistance": "" if distance_m is None else f"{distance_m:.10g}",
            "PSize_1": "" if pixel_x_m is None else f"{pixel_x_m:.10g}",
            "PSize_2": "" if pixel_y_m is None else f"{pixel_y_m:.10g}",
            "WaveLength": "" if wavelength_m is None else f"{wavelength_m:.10g}",
            "SALS_Distance_m": self.distance_edit.text().strip(),
            "SALS_PixelX_m": self.pixel_x_edit.text().strip(),
            "SALS_PixelY_m": self.pixel_y_edit.text().strip(),
            "SALS_Wavelength_m": self.wavelength_edit.text().strip(),
            "Center_1": center_x,
            "Center_2": center_y,
            "Offset_1": self.DEFAULT_OFFSET_X,
            "Offset_2": self.DEFAULT_OFFSET_Y,
            "YReversed": "1",
        }
        return {key: value for key, value in header.items() if value not in {None, ""}}

    def optional_float(self, text):
        text = str(text).strip().replace(",", ".")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    def scaled_optional_float(self, text, scale):
        value = self.optional_float(text)
        if value is None:
            return None
        return value * scale

    def default_center_text(self, size):
        return default_center_text(size)

    def disconnect_camera(self):
        self.live_timer.stop()
        if self.camera is not None:
            try:
                self.camera.__exit__(None, None, None)
            except Exception:
                pass
            self.camera = None
        if self.vmb is not None:
            try:
                self.vmb.__exit__(None, None, None)
            except Exception:
                pass
            self.vmb = None
        self.update_connection_state(False)

    def closeEvent(self, event):
        self.disconnect_camera()
        super().closeEvent(event)
