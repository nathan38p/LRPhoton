from pathlib import Path
import copy
import json
import tempfile

import numpy as np

from PySide6.QtCore import QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFileDialog,
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QMenu,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from tabs.cave_tab import (
    collect_h5_attrs_json,
    copy_h5_attrs,
    inspect_h5_image_dataset,
    read_edf_frame,
    read_h5_frame,
    sanitize_cave_output_image,
    set_h5_attr,
)
from tabs.ui_style import FILE_BROWSER_WIDTH, GROUP_BOX_STYLE, PANEL_MARGINS


DOUBLE_DETECTOR_DEFAULT_FOLDER = Path("/Users/nathanpiaget/data_ESRF_BD_5CB_27°C")
DOUBLE_DETECTOR_DOWNLOADS_FOLDER = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Downloads"


DOUBLE_DETECTOR_FILES = {
    "Si4M": {
        "poni": "Si4M_scan0010.poni",
        "mask": "Si4M_scan0020_mask.edf",
        "5CB": "Si4M_0000 5CB.h5",
        "pixel_size_m": 75e-6,
        "geometry_note": "Eiger2_4M PONI geometry",
    },
    "WOS": {
        "poni": "WOS_scan0014.poni",
        "mask": "WOS_scan0021_mask.edf",
        "5CB": "WOS_0000 5CB.h5",
        "pixel_size_m": 130e-6,
        "geometry_note": "Pixel photon counting detector IMXPAD WOS-S700 on BM02-D2AM.",
    },
}


def read_text_file(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def parse_simple_poni(path: Path):
    values = {}
    for line in read_text_file(path).splitlines():
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            values[key] = float(value)
        except ValueError:
            values[key] = value
    return values


def resolve_detector_geometry_file(original_path, folder: Path):
    if not original_path:
        return None, None

    requested = Path(original_path)
    candidates = [
        requested,
        folder / requested.name,
        DOUBLE_DETECTOR_DOWNLOADS_FOLDER / requested.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return requested, candidate
    return requested, None


def pyfai_ready_poni_path(poni_path: Path, folder: Path):
    values = parse_simple_poni(poni_path)
    detector_config = values.get("Detector_config")
    if not detector_config:
        return poni_path, None

    try:
        detector_config = json.loads(detector_config)
    except (TypeError, ValueError):
        return poni_path, None

    geometry_file = detector_config.get("filename")
    requested_geometry, resolved_geometry = resolve_detector_geometry_file(geometry_file, folder)
    if resolved_geometry is None or resolved_geometry == requested_geometry:
        return poni_path, resolved_geometry

    patched_lines = []
    for line in read_text_file(poni_path).splitlines():
        if line.startswith("Detector_config:"):
            patched_lines.append(f"Detector_config: {json.dumps({'filename': str(resolved_geometry)})}")
        else:
            patched_lines.append(line)

    temp_file = tempfile.NamedTemporaryFile("w", suffix=".poni", delete=False, encoding="utf-8")
    temp_file.write("\n".join(patched_lines) + "\n")
    temp_file.close()
    return Path(temp_file.name), resolved_geometry


def detector_summary(folder: Path, detector: str):
    config = DOUBLE_DETECTOR_FILES[detector]
    poni_path = folder / config["poni"]
    mask_path = folder / config["mask"]
    sample_paths = {name: folder / filename for name, filename in config.items() if name in {"5CB"}}

    lines = [detector, config["geometry_note"]]
    pixel_size_m = config.get("pixel_size_m")
    if pixel_size_m is not None:
        lines.append(f"Pixel size: {pixel_size_m * 1e6:.0f} x {pixel_size_m * 1e6:.0f} µm")
        if detector == "WOS":
            lines.append("Note: IMXPAD module-border pixels are wider; pyFAI needs the WOS NeXus geometry for full correction.")
    for label, path in [("PONI", poni_path), ("Mask", mask_path), *sample_paths.items()]:
        lines.append(f"{label}: {'OK' if path.exists() else 'missing'} - {path.name}")

    if poni_path.exists():
        values = parse_simple_poni(poni_path)
        detector_config = values.get("Detector_config")
        if detector_config:
            try:
                detector_config = json.loads(detector_config)
            except (TypeError, ValueError):
                detector_config = {}
            geometry_file = detector_config.get("filename")
            if geometry_file:
                requested_geometry, resolved_geometry = resolve_detector_geometry_file(geometry_file, folder)
                if resolved_geometry is not None:
                    lines.append(f"pyFAI NeXus geometry: OK - {resolved_geometry}")
                    if resolved_geometry != requested_geometry:
                        lines.append(f"Original PONI path: {requested_geometry}")
                else:
                    lines.append(f"pyFAI NeXus geometry: missing - {requested_geometry}")
        distance = values.get("Distance")
        poni1 = values.get("Poni1")
        poni2 = values.get("Poni2")
        wavelength = values.get("Wavelength")
        rot1 = values.get("Rot1", 0.0)
        rot2 = values.get("Rot2", 0.0)
        rot3 = values.get("Rot3", 0.0)
        if distance is not None:
            lines.append(f"Distance: {distance:.6g} m")
        if poni1 is not None and poni2 is not None:
            lines.append(f"PONI point: y={poni1:.6g} m, x={poni2:.6g} m")
        lines.append(f"Rotations: rot1={rot1:.6g}, rot2={rot2:.6g}, rot3={rot3:.6g} rad")
        if wavelength is not None:
            lines.append(f"Wavelength: {wavelength:.6g} m")
        ready_path, _resolved_geometry = pyfai_ready_poni_path(poni_path, folder)
        try:
            import pyFAI

            integrator = pyFAI.load(str(ready_path))
            fit2d = integrator.getFit2D()
            lines.append(
                "Direct beam center: "
                f"x={fit2d['centerX']:.3f} px, y={fit2d['centerY']:.3f} px"
            )
            lines.append(
                f"Detector tilt: {fit2d['tilt']:.6g} deg; "
                f"tilt plane: {fit2d['tiltPlanRotation']:.6g} deg"
            )
        except Exception as exc:
            lines.append(f"Direct beam center: unavailable ({exc})")
        finally:
            if ready_path != poni_path:
                try:
                    ready_path.unlink()
                except OSError:
                    pass

    for label, path in sample_paths.items():
        if path.exists():
            lines.extend(["", f"{label} H5 header", *h5_scalar_header_lines(path)])

    return "\n".join(lines)


def h5_scalar_header_lines(path: Path):
    """Return a short, readable NeXus/LIMA header without internal HDF5 paths."""
    import h5py

    fields = (
        ("/start_time", "Start time"),
        ("/end_time", "End time"),
        ("/title", "Title"),
        ("/acquisition/exposure_time", "Exposure time"),
        ("/acquisition/latency_time", "Latency time"),
        ("/acquisition/mode", "Acquisition mode"),
        ("/acquisition/nb_frames", "Frames"),
        ("/acquisition/trigger_mode", "Trigger mode"),
        ("/detector_information/name", "Detector name"),
        ("/detector_information/model", "Detector model"),
        ("/detector_information/type", "Detector type"),
        ("/detector_information/max_image_size/xsize", "Maximum width"),
        ("/detector_information/max_image_size/ysize", "Maximum height"),
        ("/detector_information/pixel_size/xsize", "Pixel width"),
        ("/detector_information/pixel_size/ysize", "Pixel height"),
        ("/image_operation/dimension/xsize", "Image width"),
        ("/image_operation/dimension/ysize", "Image height"),
        ("/image_operation/binning/x", "Binning X"),
        ("/image_operation/binning/y", "Binning Y"),
        ("/image_operation/rotation", "Image rotation"),
    )
    values_by_name = {}
    with h5py.File(path, "r") as h5:
        def collect(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.shape != ():
                return
            value = obj[()]
            if isinstance(value, bytes):
                value = value.decode(errors="replace")
            elif isinstance(value, np.generic):
                value = value.item()
            values_by_name[f"/{name}"] = value

        h5.visititems(collect)

    lines = []
    for suffix, label in fields:
        match = next((value for name, value in values_by_name.items() if name.endswith(suffix)), None)
        if match is not None:
            lines.append(f"{label}: {match}")
    return lines or ["No readable header values found"]


def cave_regions_sidecar_path(h5_path: Path):
    return Path(h5_path).with_suffix(".lrphoton.json")


def normalize_cave_region(region):
    if isinstance(region, dict):
        bounds = region.get("bounds_mm", region.get("bounds", []))
        mode = str(region.get("mode", "central"))
        rotate_with_reference = bool(region.get("rotate_with_reference", True))
        visible = bool(region.get("visible", True))
    else:
        bounds = region
        mode = "central"
        rotate_with_reference = True
        visible = True
    if mode not in {"central", "vertical", "horizontal", "extend_nan"} or len(bounds) != 4:
        mode = "central"
    return {
        "bounds_mm": list(map(float, bounds)),
        "mode": mode,
        "rotate_with_reference": rotate_with_reference,
        "visible": visible,
        "expand_nan_px": max(0, int(region.get("expand_nan_px", 0))) if isinstance(region, dict) else 0,
        "pixel_indices": region.get("pixel_indices", []) if isinstance(region, dict) else [],
    }


def read_cave_regions(h5_path: Path):
    sidecar = cave_regions_sidecar_path(h5_path)
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            raw_regions = payload.get("cave_regions", payload.get("cave_nan_regions_mm", []))
            return [normalize_cave_region(region) for region in raw_regions]
        except Exception:
            pass
    try:
        import h5py

        with h5py.File(h5_path, "r") as h5:
            raw = h5.attrs.get("cave_nan_regions_json", "")
            if isinstance(raw, bytes):
                raw = raw.decode(errors="replace")
            if raw:
                return [normalize_cave_region(region) for region in json.loads(str(raw))]
    except Exception:
        pass
    return []


def read_cave_reference_angle(h5_path: Path):
    sidecar = cave_regions_sidecar_path(h5_path)
    if sidecar.exists():
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
            return float(payload.get("reference_angle_deg", 0.0))
        except Exception:
            pass
    try:
        import h5py

        with h5py.File(h5_path, "r") as h5:
            return float(h5.attrs.get("cave_reference_angle_deg", 0.0))
    except Exception:
        return 0.0


def write_cave_regions(h5_path: Path, detector: str, regions, reference_angle_deg=0.0):
    sidecar = cave_regions_sidecar_path(h5_path)
    payload = {
        "format": "LRPhoton cave NaN regions",
        "version": 1,
        "source_h5": Path(h5_path).name,
        "detector": detector,
        "coordinate_system": "reference-angle-aligned detector coordinates in mm",
        "reference_angle_deg": float(reference_angle_deg),
        "cave_regions": [normalize_cave_region(region) for region in regions],
    }
    sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return sidecar


class ReorderableZoneList(QListWidget):
    reorder_requested = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_source_row = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.position().toPoint())
            self._drag_source_row = self.row(item) if item is not None else None
        super().mousePressEvent(event)

    def dropEvent(self, event):
        source_row = self._drag_source_row
        self._drag_source_row = None
        if source_row is None or not (0 <= source_row < self.count()):
            event.ignore()
            return

        position = event.position().toPoint()
        target_item = self.itemAt(position)
        if target_item is None:
            insertion_row = self.count()
        else:
            target_row = self.row(target_item)
            target_rect = self.visualItemRect(target_item)
            insertion_row = target_row + (1 if position.y() >= target_rect.center().y() else 0)

        if insertion_row > source_row:
            insertion_row -= 1
        insertion_row = max(0, min(insertion_row, self.count() - 1))
        if insertion_row != source_row:
            QTimer.singleShot(
                0,
                lambda source=source_row, target=insertion_row: self.reorder_requested.emit(source, target),
            )
        # The backing list is reordered by reorder_requested. Ignoring the
        # native drop prevents Qt's InternalMove from deleting the source item
        # a second time after the UI has been rebuilt.
        event.ignore()


class SelectingComboBox(QComboBox):
    pressed = Signal()

    def mousePressEvent(self, event):
        self.pressed.emit()
        super().mousePressEvent(event)


class DoubleDetectorCanvas(FigureCanvas):
    save_requested = Signal(str, str, str)
    regions_changed = Signal(str)
    regions_about_to_change = Signal(str)

    def __init__(self):
        self.figure = Figure(figsize=(9, 7))
        self.axes_grid = self.figure.subplots(2, 3)
        self.axes = np.ravel(self.axes_grid)
        self._panel_positions = (
            (0.03, 0.51, 0.30, 0.40), (0.35, 0.51, 0.30, 0.40),
            (0.67, 0.51, 0.30, 0.40), (0.03, 0.05, 0.30, 0.40),
            (0.35, 0.05, 0.30, 0.40), (0.67, 0.05, 0.30, 0.40),
        )
        self.expanded_panels = False
        self.extend_nan_enabled = False
        self.extend_nan_pixels = 1
        self.lock_panel_positions()
        self.loaded = {}
        self.mask_disabled = False
        self.selected_detector = "WOS"
        self.coordinate_labels = {}
        self.save_icons = {}
        self.expand_icons = {}
        self.figure_buttons = []
        self._syncing_limits = False
        self._pan_detector = None
        self._pan_last_pos = None
        self._last_pointer_axis = None
        self._selection_detector = None
        self._selection_start = None
        self._selection_last = None
        self._selection_patch = None
        self._symmetry_source_marker = None
        self._cave_modifications_overlay = None
        self.selected_region_index = None
        self.colormap_name = "turbo"
        self.intensity_min_percentile = 1.0
        self.intensity_max_percentile = 99.5
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("button_press_event", self.on_button_press)
        self.mpl_connect("button_release_event", self.on_button_release)
        self.mpl_connect("motion_notify_event", self.on_pan_motion)
        self.mpl_connect("figure_leave_event", self.clear_coordinate_labels)
        self.draw_empty()

    def lock_panel_positions(self):
        if self.expanded_panels:
            self.axes[2].set_position((0.03, 0.10, 0.46, 0.82), which="both")
            self.axes[3].set_position((0.51, 0.10, 0.46, 0.82), which="both")
            return
        for axis, position in zip(self.axes, self._panel_positions):
            axis.set_position(position, which="both")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_raw_controls()
        self.position_figure_buttons()

    def position_raw_controls(self):
        controls = getattr(self, "raw_controls_widget", None)
        if controls is None or not self.axes:
            return
        self.draw()
        bbox = self.axes[0].get_window_extent(self.renderer)
        x = int(round(bbox.x0))
        y = int(round(self.height() - bbox.y0 + 6))
        width = max(260, int(round(bbox.width)))
        controls.setGeometry(x, y, width, controls.sizeHint().height())

    def draw_empty(self):
        self.lock_panel_positions()
        self.save_icons = {}
        self.expand_icons = {}
        for button in getattr(self, "figure_buttons", []):
            button.deleteLater()
        self.figure_buttons = []
        for ax in np.ravel(self.axes):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_visible(False)
        self.draw_idle()

    def position_figure_buttons(self):
        for button, axis, offset in getattr(self, "figure_buttons", []):
            bbox = axis.get_window_extent(self.renderer)
            x = int(bbox.x0 + bbox.width - offset - button.width())
            y = int(self.height() - bbox.y1 - button.height() - 4)
            button.move(max(0, x), max(0, y))
            button.show()
            button.raise_()

    def draw_detectors(self, loaded, selected_detector=None):
        self.lock_panel_positions()
        self.loaded = loaded
        if selected_detector is not None:
            self.selected_detector = selected_detector
        self.save_icons = {}
        self.expand_icons = {}
        for index, ax in enumerate(self.axes):
            ax.set_visible(not self.expanded_panels or index in (2, 3))
        detector = self.selected_detector
        if detector == "Combo (test)" and not self.expanded_panels:
            # Les boutons des six tuiles ne doivent pas rester flottants lorsque
            # l'affichage Combo est réduit à trois panneaux.
            for button, *_ in getattr(self, "figure_buttons", []):
                button.hide()
            for index, ax in enumerate(self.axes):
                ax.set_visible(index in (2, 4, 5))
            combo = self.loaded.get(detector, {})
            self._draw_combo_layout(combo)
            self.draw()
            self.position_figure_buttons()
            QTimer.singleShot(0, self.position_figure_buttons)
            self.draw_idle()
            return
        if detector == "Si4M" and not self.expanded_panels:
            si4m_positions = {
                0: (0.03, 0.12, 0.30, 0.76),
                2: (0.35, 0.12, 0.30, 0.76),
                4: (0.67, 0.12, 0.30, 0.76),
            }
            for index, ax in enumerate(self.axes):
                ax.set_visible(index in si4m_positions)
                if index in si4m_positions:
                    ax.set_position(si4m_positions[index], which="both")
        sample = "5CB"
        self.draw_raw_panel(detector, sample)
        self.draw_real_panel(detector, sample, "zones")
        self.draw_zones_panel(detector, sample)
        self.draw_cave_panel(detector, sample)
        self.draw_extended_panel(detector, sample)
        self.draw_resized_panel(detector, sample)
        self.position_raw_controls()
        self.draw()
        self.position_figure_buttons()
        QTimer.singleShot(0, self.position_figure_buttons)
        self.draw_idle()

    def _draw_combo_layout(self, data):
        positions = {
            2: (0.04, 0.08, 0.62, 0.84),
            4: (0.72, 0.53, 0.25, 0.38),
            5: (0.72, 0.08, 0.25, 0.38),
        }
        for index, position in positions.items():
            self.axes[index].set_position(position, which="both")
        combo_axis = self.axes[2]
        combo_axis.clear()
        combo_axis.set_title("COMBO (TEST)", loc="left", pad=10, fontsize=12)
        self._draw_combo_image_or_empty(combo_axis, data.get("real_image"))
        combo_axis.set_visible(True)
        # La colonne de droite montre toujours les matrices natives : le
        # recalage q utilisé pour le panneau central ne doit jamais dégrader
        # les pixels Si4M/WOS affichés séparément.
        layers = data.get("combo_native_layers", data.get("combo_layers", {}))
        for axis, name in ((self.axes[4], "Si4M"), (self.axes[5], "WOS")):
            axis.clear()
            axis.set_title(name, loc="left", pad=10, fontsize=11)
            image = layers.get(name)
            if name == "Si4M" and image is not None:
                angle = float(getattr(self.parent(), "si4m_display_angle_spin", None).value()) if getattr(self.parent(), "si4m_display_angle_spin", None) is not None else 0.0
                if abs(angle) > 1e-9:
                    from scipy import ndimage
                    finite = np.isfinite(image)
                    values = ndimage.rotate(np.where(finite, image, 0.0), angle, reshape=True, order=0, mode="constant", cval=0.0, prefilter=False)
                    weights = ndimage.rotate(finite.astype(float), angle, reshape=True, order=0, mode="constant", cval=0.0, prefilter=False)
                    image = np.where(weights > 0.5, values, np.nan)
            self._draw_combo_image_or_empty(axis, image)
            axis.set_visible(True)

    def _draw_combo_image_or_empty(self, axis, image):
        """Render a Combo panel without passing an invalid scalar/empty array to imshow."""
        if image is None:
            axis.set_facecolor("white")
            axis.set_xticks([])
            axis.set_yticks([])
            return
        array = np.asarray(image)
        if array.ndim != 2 or array.size == 0:
            axis.set_facecolor("white")
            axis.set_xticks([])
            axis.set_yticks([])
            return
        self.draw_image(axis, array)

    def toggle_expanded_panels(self):
        self.expanded_panels = not self.expanded_panels
        self.draw_detectors(self.loaded, self.selected_detector)

    def draw_raw_panel(self, detector, sample):
        ax = self.detector_axis(detector, "raw")
        ax.clear()
        data = self.loaded.get(detector, {})
        image = data.get(sample)
        self.set_axis_title_with_save(ax, "RAW FILE", detector, "raw")
        if image is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            return
        self.draw_image(ax, image)

    def draw_resized_panel(self, detector, sample):
        ax = self.detector_axis(detector, "resized")
        ax.clear()
        data = self.loaded.get(detector, {})
        source = data.get("extended_image")
        self.set_axis_title_with_save(ax, "RESIZED PATTERN", detector, "resized")
        if source is None:
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 0.5, "missing image", ha="center", va="center", transform=ax.transAxes)
            return
        rows, cols = map(int, data.get("resized_shape", np.shape(source)))
        source = np.asarray(source, dtype=float)
        resized = np.full((rows, cols), np.nan, dtype=float)
        sr, sc = source.shape
        source_center = data.get("extended_center_pixel")
        if source_center is None:
            source_center = (0.5 * (sc - 1), 0.5 * (sr - 1))
        source_cx, source_cy = map(float, source_center)
        h, w = min(rows, sr), min(cols, sc)
        source_x0 = int(round(source_cx - w / 2.0))
        source_y0 = int(round(source_cy - h / 2.0))
        source_x0 = max(0, min(source_x0, sc - w))
        source_y0 = max(0, min(source_y0, sr - h))
        resized[(rows-h)//2:(rows-h)//2+h, (cols-w)//2:(cols-w)//2+w] = source[source_y0:source_y0+h, source_x0:source_x0+w]
        self.draw_image(ax, resized)
        self.draw_center_axes(ax, ((cols - 1) * 0.5, (rows - 1) * 0.5), data.get("reference_angle_deg", 0.0))

    def draw_image(self, ax, image):
        display = self.masked_log_display(np.asarray(image, dtype=float), None)
        finite = display[np.isfinite(display)]
        vmin, vmax = (
            np.nanpercentile(finite, [self.intensity_min_percentile, self.intensity_max_percentile])
            if finite.size else (None, None)
        )
        cmap = self.display_cmap()
        # Les pixels NaN doivent rester visibles comme des pixels blancs,
        # jamais transparents (sinon le fond de l'axe apparaît en sombre).
        cmap.set_bad((1.0, 1.0, 1.0, 1.0))
        ax.imshow(display, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", zorder=2)
        ax.set_facecolor("white")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xticks([]); ax.set_yticks([])

    def draw_real_panel(self, detector, sample, panel_kind="real"):
        ax = self.detector_axis(detector, panel_kind)
        ax.clear()
        self._cave_modifications_overlay = None
        data = self.loaded.get(detector, {})
        self.ensure_real_grid(detector, data)
        image = self.real_image_with_extended_nan(detector, data)
        center_mm = data.get("center_mm")
        title = "REAL COORDINATES" if panel_kind == "real" else (
            "ZONES TO EXTEND" if detector == "Si4M" else "ZONES TO CAVE"
        )
        self.set_axis_title_with_save(ax, title, detector, panel_kind)
        if image is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "missing geometry", ha="center", va="center", transform=ax.transAxes)
            return
        display_reference_angle = float(data.get("reference_angle_deg", 0.0))
        self.draw_regular_grid_image(
            ax,
            image,
            data.get("real_x_centers_mm"),
            data.get("real_y_centers_mm"),
            center_mm,
            display_reference_angle,
        )
        if panel_kind == "zones":
            self.draw_cave_nan_regions(ax, data)

    def real_image_with_extended_nan(self, detector, data):
        image = data.get("real_image")
        if image is None:
            return image
        self.ensure_real_grid(detector, data)
        result = np.asarray(image, dtype=float).copy()
        x_centers = np.asarray(data.get("real_x_centers_mm"), dtype=float)
        y_centers = np.asarray(data.get("real_y_centers_mm"), dtype=float)
        center_mm = data.get("center_mm")
        angle = float(data.get("reference_angle_deg", 0.0))
        for region in data.get("cave_nan_regions", []):
            normalized = normalize_cave_region(region)
            if not normalized.get("visible", True):
                continue
            if normalized["mode"] != "extend_nan":
                continue
            result[self.region_grid_mask(normalized, x_centers, y_centers, center_mm, angle)] = np.nan
        return result

    def draw_zones_panel(self, detector, sample):
        self.draw_real_panel(detector, sample, "real")

    def draw_cave_panel(self, detector, sample):
        ax = self.detector_axis(detector, "cave")
        ax.clear()
        data = self.loaded.get(detector, {})
        self.ensure_real_grid(detector, data)
        image = self.real_image_with_extended_nan(detector, data)
        center_mm = data.get("center_mm")
        center_pixel = data.get("real_center_pixel")
        self.set_axis_title_with_save(ax, "CAVED PATTERN", detector, "cave")
        if image is None or center_pixel is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "missing geometry", ha="center", va="center", transform=ax.transAxes)
            return
        original_image = np.asarray(image, dtype=float)
        x_centers = np.asarray(data.get("real_x_centers_mm"), dtype=float)
        y_centers = np.asarray(data.get("real_y_centers_mm"), dtype=float)
        normalized_regions = [normalize_cave_region(region) for region in data.get("cave_nan_regions", [])]
        reference_angle = data.get("reference_angle_deg", 0.0)

        # "NaN for caving" is a preprocessing mask: all of these zones must
        # become NaN before the first, global central-symmetry Cave pass.
        cave_input = original_image.copy()
        for region in normalized_regions:
            if not region.get("visible", True):
                continue
            preprocessing_modes = {"extend_nan"} if detector == "Si4M" else {"central", "extend_nan"}
            if region["mode"] not in preprocessing_modes:
                continue
            region_mask = self.region_grid_mask(
                region, x_centers, y_centers, center_mm, reference_angle
            )
            cave_input[region_mask] = np.nan

        # Si4M workflow: first reconstruct the four tiles with the selected
        # vertical/horizontal operations; central Cave symmetry comes later.
        filled = cave_input.copy() if detector in {"Si4M", "Combo (test)"} else self.axial_symmetry_cave(cave_input, None, center_pixel)
        row_grid, col_grid = np.indices(original_image.shape)
        source_rows = np.where(np.isfinite(cave_input), row_grid, -1)
        source_cols = np.where(np.isfinite(cave_input), col_grid, -1)
        missing_rows, missing_cols = np.where(~np.isfinite(cave_input))
        center_col, center_row = map(float, center_pixel)
        central_cols = np.rint(2.0 * center_col - missing_cols).astype(int)
        central_rows = np.rint(2.0 * center_row - missing_rows).astype(int)
        valid_central = (
            (central_rows >= 0) & (central_rows < original_image.shape[0])
            & (central_cols >= 0) & (central_cols < original_image.shape[1])
        )
        can_trace = np.zeros(missing_rows.shape, dtype=bool)
        can_trace[valid_central] = np.isfinite(
            cave_input[central_rows[valid_central], central_cols[valid_central]]
        )
        source_rows[missing_rows[can_trace], missing_cols[can_trace]] = central_rows[can_trace]
        source_cols[missing_rows[can_trace], missing_cols[can_trace]] = central_cols[can_trace]
        remaining_layer_nans = ~np.isfinite(cave_input)
        for region in normalized_regions:
            non_layer_modes = {"extend_nan"} if detector == "Si4M" else {"central", "extend_nan"}
            if region["mode"] in non_layer_modes:
                continue
            region_mask = self.region_grid_mask(
                region, x_centers, y_centers, center_mm, reference_angle
            )
            was_nan_for_caving = remaining_layer_nans & region_mask
            filled[was_nan_for_caving] = np.nan
            source_rows[was_nan_for_caving] = -1
            source_cols[was_nan_for_caving] = -1
            # Symmetry layers fill current NaNs, but their intensities always
            # come from the untouched REAL CORDINATES image.
            layer_source = original_image
            replacement_info = self.apply_region_symmetry(
                filled,
                layer_source,
                region["bounds_mm"],
                center_pixel,
                region["mode"],
                x_centers,
                y_centers,
                data.get("reference_angle_deg", 0.0),
                target_mask=region_mask,
            )
            if replacement_info is not None:
                target_rows, target_cols, reflected_rows, reflected_cols, replace = replacement_info
                source_rows[target_rows[replace], target_cols[replace]] = reflected_rows[replace]
                source_cols[target_rows[replace], target_cols[replace]] = reflected_cols[replace]
            remaining_layer_nans[was_nan_for_caving] = ~np.isfinite(
                filled[was_nan_for_caving]
            )

        # Finish Cave with the same central-symmetry pass that made EXTENDED
        # complete. Existing vertical/horizontal reconstructions stay intact;
        # only pixels that are still NaN can be filled here.
        before_final_cave = filled.copy()
        data["zones_preview_image"] = before_final_cave.copy()
        data["zones_preview_modified_mask"] = (
            np.isfinite(before_final_cave)
            & ((source_rows != row_grid) | (source_cols != col_grid))
        )
        final_missing_rows, final_missing_cols = np.where(~np.isfinite(before_final_cave))
        filled = before_final_cave.copy() if detector in {"Si4M", "Combo (test)"} else self.axial_symmetry_cave(before_final_cave, None, center_pixel)
        if final_missing_rows.size:
            final_source_cols = np.rint(2.0 * center_col - final_missing_cols).astype(int)
            final_source_rows = np.rint(2.0 * center_row - final_missing_rows).astype(int)
            final_valid = (
                (final_source_rows >= 0) & (final_source_rows < before_final_cave.shape[0])
                & (final_source_cols >= 0) & (final_source_cols < before_final_cave.shape[1])
            )
            final_filled = np.zeros(final_missing_rows.shape, dtype=bool)
            final_filled[final_valid] = (
                np.isfinite(before_final_cave[final_source_rows[final_valid], final_source_cols[final_valid]])
                & np.isfinite(filled[final_missing_rows[final_valid], final_missing_cols[final_valid]])
            )
            inherited_rows = source_rows[
                final_source_rows[final_filled], final_source_cols[final_filled]
            ]
            inherited_cols = source_cols[
                final_source_rows[final_filled], final_source_cols[final_filled]
            ]
            missing_provenance = (inherited_rows < 0) | (inherited_cols < 0)
            inherited_rows[missing_provenance] = final_source_rows[final_filled][missing_provenance]
            inherited_cols[missing_provenance] = final_source_cols[final_filled][missing_provenance]
            source_rows[final_missing_rows[final_filled], final_missing_cols[final_filled]] = inherited_rows
            source_cols[final_missing_rows[final_filled], final_missing_cols[final_filled]] = inherited_cols
        data["cave_image"] = filled
        data["cave_source_rows"] = source_rows
        data["cave_source_cols"] = source_cols
        data["cave_modified_mask"] = (
            np.isfinite(filled)
            & ((source_rows != row_grid) | (source_cols != col_grid))
        )
        real_q = data.get("real_q_nm")
        if real_q is not None and np.shape(real_q) == filled.shape:
            cave_q = np.asarray(real_q, dtype=float).copy()
            valid_sources = (
                np.isfinite(filled)
                & (source_rows >= 0) & (source_rows < filled.shape[0])
                & (source_cols >= 0) & (source_cols < filled.shape[1])
            )
            cave_q[valid_sources] = np.asarray(real_q, dtype=float)[
                source_rows[valid_sources], source_cols[valid_sources]
            ]
            data["cave_q_nm"] = cave_q
        self.draw_regular_grid_image(
            ax,
            filled,
            data.get("real_x_centers_mm"),
            data.get("real_y_centers_mm"),
            center_mm,
            data.get("reference_angle_deg", 0.0),
        )
        self._symmetry_source_marker, = ax.plot(
            [], [], marker="o", markersize=9, markerfacecolor="none",
            markeredgecolor="#ff00a8", markeredgewidth=1.8,
            linestyle="none", zorder=50,
        )

    def draw_extended_panel(self, detector, sample):
        ax = self.detector_axis(detector, "extended")
        ax.clear()
        data = self.loaded.get(detector, {})
        cave = data.get("cave_image")
        x_centers = np.asarray(data.get("real_x_centers_mm"), dtype=float)
        y_centers = np.asarray(data.get("real_y_centers_mm"), dtype=float)
        center_mm = data.get("center_mm")
        extended_title = "EXTENDED PATTERN" if detector == "Si4M" else "EXTENDED CAVE PATTERN"
        self.set_axis_title_with_save(ax, extended_title, detector, "extended")
        if cave is None or center_mm is None or x_centers.size < 2 or y_centers.size < 2:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "missing Cave geometry", ha="center", va="center", transform=ax.transAxes)
            return

        step = max(
            abs(float(np.nanmedian(np.diff(x_centers)))),
            abs(float(np.nanmedian(np.diff(y_centers)))),
        )
        center_x, center_y = map(float, center_mm)
        x_edges = (x_centers[0] - step / 2.0, x_centers[-1] + step / 2.0)
        y_edges = (y_centers[0] - step / 2.0, y_centers[-1] + step / 2.0)
        radius = max(
            abs(float(x_edges[0]) - center_x), abs(float(x_edges[1]) - center_x),
            abs(float(y_edges[0]) - center_y), abs(float(y_edges[1]) - center_y),
        )
        half_pixels = max(1, int(np.ceil(radius / step)))
        offsets = np.arange(-half_pixels, half_pixels + 1, dtype=float)
        extended_x = center_x + offsets * step
        extended_y = center_y + offsets * step
        extended = np.full((offsets.size, offsets.size), np.nan, dtype=float)
        extended_q = np.full_like(extended, np.nan)

        target_cols = np.rint((x_centers - extended_x[0]) / step).astype(int)
        target_rows = np.rint((y_centers - extended_y[0]) / step).astype(int)
        valid_cols = (target_cols >= 0) & (target_cols < extended.shape[1])
        valid_rows = (target_rows >= 0) & (target_rows < extended.shape[0])
        cave_array = np.asarray(cave, dtype=float)
        cave_q = data.get("cave_q_nm")
        for source_row, target_row in enumerate(target_rows):
            if not valid_rows[source_row]:
                continue
            source_columns = np.where(valid_cols)[0]
            extended[target_row, target_cols[source_columns]] = cave_array[source_row, source_columns]
            if cave_q is not None and np.shape(cave_q) == cave_array.shape:
                extended_q[target_row, target_cols[source_columns]] = np.asarray(cave_q)[source_row, source_columns]

        extended_center = (float(half_pixels), float(half_pixels))
        if detector in {"Si4M", "Combo (test)"}:
            extended_source = extended.copy()
            for raw_region in data.get("cave_nan_regions", []):
                region = normalize_cave_region(raw_region)
                if not region.get("visible", True) or region["mode"] == "extend_nan":
                    continue
                target_mask = self.region_grid_mask(
                    region, extended_x, extended_y, center_mm,
                    data.get("reference_angle_deg", 0.0),
                )
                self.apply_region_symmetry(
                    extended, extended_source, region["bounds_mm"],
                    extended_center, region["mode"], extended_x, extended_y,
                    data.get("reference_angle_deg", 0.0), target_mask=target_mask,
                )
            valid_columns = np.where(np.any(np.isfinite(extended), axis=0))[0]
            if valid_columns.size:
                first_col = int(valid_columns[0])
                last_col = int(valid_columns[-1]) + 1
                extended = extended[:, first_col:last_col]
                extended_q = extended_q[:, first_col:last_col]
                extended_x = extended_x[first_col:last_col]
                extended_center = (extended_center[0] - first_col, extended_center[1])
        else:
            extended = self.axial_symmetry_cave(extended, None, extended_center)
            extended_q = self.axial_symmetry_cave(extended_q, None, extended_center)
        geometric_extended_q = self.geometric_q_grid(
            extended_x, extended_y, center_mm,
            data.get("distance_m"), data.get("wavelength_m"),
        )
        extended_q[~np.isfinite(extended_q)] = geometric_extended_q[~np.isfinite(extended_q)]
        data["extended_image"] = extended
        data["extended_q_nm"] = extended_q
        data["extended_x_centers_mm"] = extended_x
        data["extended_y_centers_mm"] = extended_y
        data["extended_center_pixel"] = extended_center
        self.draw_regular_grid_image(
            ax, extended, extended_x, extended_y, center_mm,
            data.get("reference_angle_deg", 0.0),
        )

    def point_to_reference(self, x_value, y_value, center_mm, angle_deg):
        center_x, center_y = map(float, center_mm)
        angle = np.deg2rad(float(angle_deg))
        dx = float(x_value) - center_x
        dy = float(y_value) - center_y
        return (
            center_x + dx * np.cos(angle) + dy * np.sin(angle),
            center_y - dx * np.sin(angle) + dy * np.cos(angle),
        )

    def point_from_reference(self, u_value, v_value, center_mm, angle_deg):
        center_x, center_y = map(float, center_mm)
        angle = np.deg2rad(float(angle_deg))
        du = float(u_value) - center_x
        dv = float(v_value) - center_y
        return (
            center_x + du * np.cos(angle) - dv * np.sin(angle),
            center_y + du * np.sin(angle) + dv * np.cos(angle),
        )

    def region_polygon(self, region, center_mm, angle_deg):
        normalized = normalize_cave_region(region)
        x0, x1, y0, y1 = normalized["bounds_mm"]
        zone_angle = float(angle_deg) if normalized["rotate_with_reference"] else 0.0
        return np.asarray([
            self.point_from_reference(x0, y0, center_mm, zone_angle),
            self.point_from_reference(x1, y0, center_mm, zone_angle),
            self.point_from_reference(x1, y1, center_mm, zone_angle),
            self.point_from_reference(x0, y1, center_mm, zone_angle),
        ], dtype=float)

    def region_grid_mask(self, region, x_centers, y_centers, center_mm, angle_deg):
        normalized = normalize_cave_region(region)
        if isinstance(region, dict) and region.get("pixel_indices"):
            mask = np.zeros((len(y_centers), len(x_centers)), dtype=bool)
            for row, col in region["pixel_indices"]:
                if 0 <= int(row) < mask.shape[0] and 0 <= int(col) < mask.shape[1]:
                    mask[int(row), int(col)] = True
            return mask
        x0, x1, y0, y1 = normalized["bounds_mm"]
        expansion = self.extend_nan_pixels if self.extend_nan_enabled else 0
        if expansion:
            dx = abs(float(np.nanmedian(np.diff(x_centers)))) if len(x_centers) > 1 else 0.0
            dy = abs(float(np.nanmedian(np.diff(y_centers)))) if len(y_centers) > 1 else 0.0
            x0 -= expansion * dx; x1 += expansion * dx
            y0 -= expansion * dy; y1 += expansion * dy
        zone_angle = float(angle_deg) if normalized["rotate_with_reference"] else 0.0
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)
        center_x, center_y = map(float, center_mm)
        angle = np.deg2rad(zone_angle)
        dx = x_grid - center_x
        dy = y_grid - center_y
        u_grid = center_x + dx * np.cos(angle) + dy * np.sin(angle)
        v_grid = center_y - dx * np.sin(angle) + dy * np.cos(angle)
        return (u_grid >= x0) & (u_grid <= x1) & (v_grid >= y0) & (v_grid <= y1)

    def draw_cave_nan_regions(self, ax, data):
        fixed_xlim = ax.get_xlim()
        fixed_ylim = ax.get_ylim()
        colors = {
            "central": ("#ef4444", "#b91c1c"),
            "extend_nan": ("#f59e0b", "#b45309"),
            "vertical": ("#3b82f6", "#1d4ed8"),
            "horizontal": ("#22c55e", "#15803d"),
        }
        for region_index, raw_region in enumerate(data.get("cave_nan_regions", [])):
            region = normalize_cave_region(raw_region)
            if not region["visible"]:
                continue
            face_color, edge_color = colors[region["mode"]]
            selected = region_index == self.selected_region_index
            ax.add_patch(Polygon(
                self.region_polygon(
                    region,
                    data.get("center_mm"),
                    data.get("reference_angle_deg", 0.0),
                ),
                closed=True,
                facecolor=face_color if selected else "none",
                edgecolor=edge_color,
                linewidth=1.2,
                alpha=0.25 if selected else 0.9,
                zorder=30,
            ))
        ax.set_xlim(*fixed_xlim)
        ax.set_ylim(*fixed_ylim)

    def draw_cave_modifications_overlay(self, ax, data):
        real_image = data.get("real_image")
        # The overlay must use the same final Cave image as CAVED PATTERN;
        # using the pre-final-symmetry preview made the two layers disagree.
        cave_image = data.get("cave_image")
        x_centers = np.asarray(data.get("real_x_centers_mm"), dtype=float)
        y_centers = np.asarray(data.get("real_y_centers_mm"), dtype=float)
        if real_image is None or cave_image is None or real_image.shape != cave_image.shape:
            return
        if x_centers.size < 2 or y_centers.size < 2:
            return

        real = np.asarray(real_image, dtype=float)
        cave = np.asarray(cave_image, dtype=float)
        finite_real = np.isfinite(real)
        finite_cave = np.isfinite(cave)
        # This preview is based on the final Cave provenance, including pixels
        # deliberately made NaN before the initial central-symmetry pass.
        reconstructed = data.get("zones_preview_modified_mask")
        if reconstructed is None or np.shape(reconstructed) != cave.shape:
            reconstructed = ~finite_real & finite_cave
        else:
            reconstructed = np.asarray(reconstructed, dtype=bool) & finite_cave
        if not np.any(reconstructed):
            return

        with np.errstate(invalid="ignore", divide="ignore"):
            log_cave = np.log10(cave + 1.0)
        finite_values = log_cave[np.isfinite(log_cave)]
        if finite_values.size:
            vmin, vmax = np.nanpercentile(finite_values, [1, 99.5])
            scale = max(float(vmax - vmin), 1e-12)
            normalized = np.clip((log_cave - vmin) / scale, 0.0, 1.0)
        else:
            normalized = np.zeros(cave.shape, dtype=float)

        rgba = self.display_cmap()(np.nan_to_num(normalized, nan=0.0))
        rgba[..., 3] = 0.0
        rgba[reconstructed, 3] = 0.34

        dx = float(np.nanmedian(np.diff(x_centers)))
        dy = float(np.nanmedian(np.diff(y_centers)))
        extent = (
            float(x_centers[0] - dx / 2.0),
            float(x_centers[-1] + dx / 2.0),
            float(y_centers[-1] + dy / 2.0),
            float(y_centers[0] - dy / 2.0),
        )
        self._cave_modifications_overlay = ax.imshow(
            rgba, origin="upper", extent=extent,
            interpolation="nearest", aspect="equal", zorder=1,
        )
        # Keep the transparent reconstruction layer in the same reference
        # frame as the axes.  Previously it stayed horizontal when switching
        # between 0° and Tilt plane.
        from matplotlib.transforms import Affine2D
        center_x, center_y = map(float, data.get("center_mm"))
        angle = float(data.get("reference_angle_deg", 0.0))
        self._cave_modifications_overlay.set_transform(
            Affine2D().rotate_deg_around(center_x, center_y, angle) + ax.transData
        )

    def apply_region_symmetry(
        self,
        result,
        source,
        bounds_mm,
        center_pixel,
        mode,
        x_centers,
        y_centers,
        reference_angle_deg=0.0,
        replace_entire_region=False,
        target_mask=None,
    ):
        if target_mask is not None:
            target_mask = np.asarray(target_mask, dtype=bool)
            rows = np.where(np.any(target_mask, axis=1))[0]
            cols = np.where(np.any(target_mask, axis=0))[0]
        else:
            x0, x1, y0, y1 = bounds_mm
            rows = np.where((y_centers >= y0) & (y_centers <= y1))[0]
            cols = np.where((x_centers >= x0) & (x_centers <= x1))[0]
        if rows.size == 0 or cols.size == 0:
            return None
        target_rows, target_cols = np.meshgrid(rows, cols, indexing="ij")
        center_col, center_row = map(float, center_pixel)
        center_x = float(np.interp(center_col, np.arange(x_centers.size), x_centers))
        center_y = float(np.interp(center_row, np.arange(y_centers.size), y_centers))
        target_x = x_centers[target_cols] - center_x
        target_y = y_centers[target_rows] - center_y
        angle = np.deg2rad(float(reference_angle_deg))
        if mode == "central":
            reflected_x = -target_x + center_x
            reflected_y = -target_y + center_y
        elif mode == "horizontal":
            axis_x, axis_y = np.cos(angle), np.sin(angle)
        else:
            axis_x, axis_y = -np.sin(angle), np.cos(angle)
        if mode != "central":
            projection = target_x * axis_x + target_y * axis_y
            reflected_x = 2.0 * projection * axis_x - target_x + center_x
            reflected_y = 2.0 * projection * axis_y - target_y + center_y
        x_step = float(np.nanmedian(np.diff(x_centers)))
        y_step = float(np.nanmedian(np.diff(y_centers)))
        source_cols = np.rint((reflected_x - x_centers[0]) / x_step).astype(int)
        source_rows = np.rint((reflected_y - y_centers[0]) / y_step).astype(int)
        valid = (
            (source_rows >= 0) & (source_rows < source.shape[0])
            & (source_cols >= 0) & (source_cols < source.shape[1])
        )
        replacements = np.full(target_rows.shape, np.nan, dtype=float)
        replacements[valid] = source[source_rows[valid], source_cols[valid]]
        if replace_entire_region:
            replace = np.isfinite(replacements)
            if target_mask is not None:
                replace &= target_mask[target_rows, target_cols]
            result[target_rows[replace], target_cols[replace]] = replacements[replace]
        else:
            target_was_nan = ~np.isfinite(result[target_rows, target_cols])
            replace = target_was_nan & np.isfinite(replacements)
            if target_mask is not None:
                replace &= target_mask[target_rows, target_cols]
            result[target_rows[replace], target_cols[replace]] = replacements[replace]
        return target_rows, target_cols, source_rows, source_cols, replace

    def ensure_real_grid(self, detector, data):
        """Build the rectified H5-equivalent grid if a loaded session predates it."""
        if data.get("real_image") is not None:
            return
        image = data.get("5CB")
        corners = data.get("pixel_corners")
        if image is None or corners is None:
            return
        config = DOUBLE_DETECTOR_FILES.get(detector, {})
        pixel_mm = float(config.get("pixel_size_m") or 0.0) * 1000.0
        if pixel_mm <= 0:
            return

        source = np.asarray(image, dtype=float).copy()
        mask = data.get("mask")
        if mask is not None and mask.shape == source.shape:
            source[np.asarray(mask, dtype=bool)] = np.nan
        source[~np.isfinite(source) | (source < 0)] = np.nan
        corners = np.asarray(corners, dtype=float)
        x_corners_mm = corners[:, :, :, 2] * 1000.0
        y_corners_mm = corners[:, :, :, 1] * 1000.0

        def rectify(values):
            values = np.asarray(values, dtype=float)
            x_min, x_max = float(np.nanmin(x_corners_mm)), float(np.nanmax(x_corners_mm))
            y_min, y_max = float(np.nanmin(y_corners_mm)), float(np.nanmax(y_corners_mm))
            nx = max(1, int(np.ceil((x_max - x_min) / pixel_mm)))
            ny = max(1, int(np.ceil((y_max - y_min) / pixel_mm)))
            output = np.full((ny, nx), np.nan, dtype=float)
            x_center = np.nanmean(x_corners_mm, axis=2)
            y_center = np.nanmean(y_corners_mm, axis=2)
            cols = np.rint((x_center - x_min) / pixel_mm).astype(int)
            rows = np.rint((y_center - y_min) / pixel_mm).astype(int)
            valid = (
                np.isfinite(values) & np.isfinite(x_center) & np.isfinite(y_center)
                & (rows >= 0) & (rows < ny) & (cols >= 0) & (cols < nx)
            )
            output[rows[valid], cols[valid]] = values[valid]
            return output, x_min, y_min

        real_image, x_min, y_min = rectify(source)
        real_q, _, _ = rectify(data.get("q_nm"))
        real_psi, _, _ = rectify(data.get("psi_deg"))
        real_x = x_min + (np.arange(real_image.shape[1], dtype=float) + 0.5) * pixel_mm
        real_y = y_min + (np.arange(real_image.shape[0], dtype=float) + 0.5) * pixel_mm
        data["real_image"] = real_image
        data["real_mask"] = ~np.isfinite(real_image)
        data["real_q_nm"] = real_q
        data["real_psi_deg"] = real_psi
        data["real_x_centers_mm"] = real_x
        data["real_y_centers_mm"] = real_y
        center_mm = data.get("center_mm")
        if center_mm is not None:
            data["real_center_pixel"] = (
                float(self.nearest_sorted_index(real_x, center_mm[0])),
                float(self.nearest_sorted_index(real_y, center_mm[1])),
            )

    def geometric_q_grid(self, x_centers_mm, y_centers_mm, center_mm, distance_m, wavelength_m):
        x_grid, y_grid = np.meshgrid(
            np.asarray(x_centers_mm, dtype=float),
            np.asarray(y_centers_mm, dtype=float),
        )
        center_x, center_y = map(float, center_mm)
        radius_m = np.hypot(x_grid - center_x, y_grid - center_y) * 1e-3
        two_theta = np.arctan2(radius_m, float(distance_m))
        return (
            4.0 * np.pi / float(wavelength_m)
            * np.sin(0.5 * two_theta)
            / 1e9
        )

    def draw_regular_grid_image(
        self, ax, image, x_centers_mm, y_centers_mm, center_mm=None, reference_angle_deg=0.0
    ):
        display = self.masked_log_display(image, None)
        finite = display[np.isfinite(display)]
        vmin, vmax = (
            np.nanpercentile(finite, [self.intensity_min_percentile, self.intensity_max_percentile])
            if finite.size else (None, None)
        )
        x_centers = np.asarray(x_centers_mm, dtype=float)
        y_centers = np.asarray(y_centers_mm, dtype=float)
        if x_centers.size < 2 or y_centers.size < 2:
            return
        dx = float(np.nanmedian(np.diff(x_centers)))
        dy = float(np.nanmedian(np.diff(y_centers)))
        extent = (
            float(x_centers[0] - dx / 2.0),
            float(x_centers[-1] + dx / 2.0),
            float(y_centers[-1] + dy / 2.0),
            float(y_centers[0] - dy / 2.0),
        )
        cmap = self.display_cmap()
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        ax.imshow(
            display,
            cmap=cmap,
            origin="upper",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="auto",
            zorder=2,
        )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        self.draw_center_axes(ax, center_mm, reference_angle_deg)

    def set_axis_title_with_save(self, ax, title, detector, kind):
        ax.set_title(title, loc="left", pad=10, fontsize=11)
        if kind == "resized":
            h5_button = QPushButton("💾", self)
            h5_button.setFixedSize(32, 28)
            h5_button.setToolTip("Enregistrer le H5")
            h5_button.clicked.connect(lambda: self.save_requested.emit(detector, kind, "h5"))
            self.figure_buttons.append((h5_button, ax, 4))
        if kind == "zones" and self.expanded_panels:
            return_button = QPushButton("❌", self)
            return_button.setFixedSize(34, 28)
            return_button.setToolTip("Revenir aux 6 images")
            return_button.clicked.connect(self.toggle_expanded_panels)
            self.figure_buttons.append((return_button, ax, 4))

    def detector_axis(self, detector, kind):
        _detector = detector
        col = {"raw": 0, "real": 1, "zones": 2, "cave": 3, "extended": 4, "resized": 5}[kind]
        return self.axes[col]

    def masked_log_display(self, image, mask):
        display = np.asarray(image, dtype=float)
        if mask is not None and mask.shape == image.shape:
            display = display.copy()
            display[mask] = np.nan
        display = np.where(np.isfinite(display) & (display >= 0), display, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log10(display + 1.0)

    def display_cmap(self):
        from matplotlib import colormaps
        cmap = colormaps.get(self.colormap_name, colormaps["turbo"]).copy()
        cmap.set_bad("white")
        return cmap

    def draw_true_coordinate_image(self, ax, image, mask, corners, center_mm=None):
        display = self.masked_log_display(image, mask)
        finite = display[np.isfinite(display)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [1, 99.5])
        else:
            vmin, vmax = None, None

        corners = np.asarray(corners, dtype=float)
        for row_slice, col_slice in self.contiguous_detector_block_slices(corners):
            block_display = display[row_slice, col_slice]
            if not np.any(np.isfinite(block_display)):
                continue
            block_corners = corners[row_slice, col_slice]
            x_edges, y_edges = self.full_pixel_corner_grid_mm(block_corners)
            ax.pcolormesh(
                x_edges,
                y_edges,
                block_display,
                cmap=self.display_cmap(),
                shading="flat",
                vmin=vmin,
                vmax=vmax,
                linewidth=0,
                antialiased=False,
            )
        x_all = corners[:, :, :, 2] * 1000.0
        y_all = corners[:, :, :, 1] * 1000.0
        ax.set_xlim(float(np.nanmin(x_all)), float(np.nanmax(x_all)))
        ax.set_ylim(float(np.nanmax(y_all)), float(np.nanmin(y_all)))
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        self.draw_center_axes(ax, center_mm)

    def contiguous_detector_block_slices(self, corners):
        corners = np.asarray(corners, dtype=float)
        if corners.ndim < 4:
            return [(slice(None), slice(None))]

        y_centers = np.nanmedian(np.mean(corners[:, :, :, 1], axis=2), axis=1) * 1000.0
        x_centers = np.nanmedian(np.mean(corners[:, :, :, 2], axis=2), axis=0) * 1000.0
        row_breaks = self.geometry_gap_breaks(y_centers)
        col_breaks = self.geometry_gap_breaks(x_centers)

        blocks = []
        row_starts = [0] + row_breaks
        row_stops = row_breaks + [corners.shape[0]]
        col_starts = [0] + col_breaks
        col_stops = col_breaks + [corners.shape[1]]
        for row_start, row_stop in zip(row_starts, row_stops):
            for col_start, col_stop in zip(col_starts, col_stops):
                if row_stop > row_start and col_stop > col_start:
                    blocks.append((slice(row_start, row_stop), slice(col_start, col_stop)))
        return blocks or [(slice(None), slice(None))]

    def geometry_gap_breaks(self, centers):
        centers = np.asarray(centers, dtype=float)
        if centers.size < 2:
            return []
        diffs = np.abs(np.diff(centers))
        finite = diffs[np.isfinite(diffs) & (diffs > 0)]
        if finite.size == 0:
            return []
        pixel_step = float(np.nanmedian(finite))
        if not np.isfinite(pixel_step) or pixel_step <= 0:
            return []
        return [int(index + 1) for index, diff in enumerate(diffs) if np.isfinite(diff) and diff > pixel_step * 2.5]

    def draw_center_axes(self, ax, center_mm, reference_angle_deg=0.0):
        if center_mm is None:
            return
        center_x_mm, center_y_mm = center_mm
        if center_x_mm is None or center_y_mm is None:
            return
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        angle = np.deg2rad(float(reference_angle_deg))
        length = 2.0 * np.hypot(x_max - x_min, y_max - y_min)
        for direction_x, direction_y in (
            (np.cos(angle), np.sin(angle)),
            (-np.sin(angle), np.cos(angle)),
        ):
            ax.plot(
                [center_x_mm - length * direction_x, center_x_mm + length * direction_x],
                [center_y_mm - length * direction_y, center_y_mm + length * direction_y],
                color="red",
                linewidth=1.0,
                alpha=0.9,
                zorder=10,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    def axial_symmetry_cave(self, image, mask, center_pixel):
        raw = np.asarray(image, dtype=float)
        source = raw.copy()
        invalid_pixels = ~np.isfinite(raw) | (raw < 0)
        if mask is not None and mask.shape == source.shape:
            invalid_pixels = invalid_pixels | np.asarray(mask, dtype=bool)
        source[invalid_pixels] = np.nan
        filled = source.copy()

        missing_rows, missing_cols = np.where(invalid_pixels)
        ny, nx = source.shape
        if missing_rows.size == 0:
            return filled

        center_col, center_row = map(float, center_pixel)
        central_cols = np.rint(2.0 * center_col - missing_cols).astype(int)
        central_rows = np.rint(2.0 * center_row - missing_rows).astype(int)

        valid_central = (
            (central_cols >= 0) & (central_cols < nx)
            & (central_rows >= 0) & (central_rows < ny)
        )
        replacement = np.full(missing_rows.shape, np.nan, dtype=float)
        replacement[valid_central] = source[central_rows[valid_central], central_cols[valid_central]]
        can_fill = np.isfinite(replacement)
        filled[missing_rows[can_fill], missing_cols[can_fill]] = replacement[can_fill]
        return filled

    def local_nan_estimate(self, source):
        try:
            from scipy import ndimage
        except Exception:
            return None

        valid = np.isfinite(source)
        if not np.any(valid):
            return None
        values = np.where(valid, source, 0.0)
        weights = valid.astype(float)
        sigma = 2.0
        smoothed_values = ndimage.gaussian_filter(values, sigma=sigma, mode="nearest")
        smoothed_weights = ndimage.gaussian_filter(weights, sigma=sigma, mode="nearest")
        estimate = np.full(source.shape, np.nan, dtype=float)
        good = smoothed_weights > 0.05
        estimate[good] = smoothed_values[good] / smoothed_weights[good]
        return estimate

    def thin_mask_pixels(self, mask, max_run_length=32):
        mask = np.asarray(mask, dtype=bool)
        horizontal_lengths = np.zeros(mask.shape, dtype=np.uint16)
        vertical_lengths = np.zeros(mask.shape, dtype=np.uint16)

        for row_index in range(mask.shape[0]):
            row = mask[row_index]
            starts, stops = self.true_runs(row)
            for start, stop in zip(starts, stops):
                horizontal_lengths[row_index, start:stop] = stop - start

        for col_index in range(mask.shape[1]):
            col = mask[:, col_index]
            starts, stops = self.true_runs(col)
            for start, stop in zip(starts, stops):
                vertical_lengths[start:stop, col_index] = stop - start

        return mask & (
            (horizontal_lengths <= max_run_length)
            & (vertical_lengths <= max_run_length)
        )

    def true_runs(self, values):
        values = np.asarray(values, dtype=bool)
        padded = np.r_[False, values, False]
        changes = np.diff(padded.astype(np.int8))
        starts = np.where(changes == 1)[0]
        stops = np.where(changes == -1)[0]
        return starts, stops

    def downsample_display_blocks(self, display, mask, row_edges, col_edges):
        sampled = np.full((len(row_edges) - 1, len(col_edges) - 1), np.nan, dtype=float)
        for row_index, (row_start, row_stop) in enumerate(zip(row_edges[:-1], row_edges[1:])):
            for col_index, (col_start, col_stop) in enumerate(zip(col_edges[:-1], col_edges[1:])):
                block = display[row_start:row_stop, col_start:col_stop]
                if block.size == 0:
                    continue
                if mask is not None and mask.shape == display.shape:
                    mask_block = mask[row_start:row_stop, col_start:col_stop]
                    if float(np.mean(mask_block)) >= 0.25:
                        continue
                finite = block[np.isfinite(block)]
                if finite.size:
                    sampled[row_index, col_index] = float(np.mean(finite))
        return sampled

    def pixel_edge_vectors_mm(self, corners):
        corners = np.asarray(corners, dtype=float)
        ny, nx = corners.shape[:2]
        x_edges = np.empty(nx + 1, dtype=float)
        y_edges = np.empty(ny + 1, dtype=float)

        x_edges[:-1] = np.nanmedian(corners[:, :, 0, 2], axis=0) * 1000.0
        x_edges[-1] = np.nanmedian(corners[:, -1, 3, 2]) * 1000.0
        y_edges[:-1] = np.nanmedian(corners[:, :, 0, 1], axis=1) * 1000.0
        y_edges[-1] = np.nanmedian(corners[-1, :, 1, 1]) * 1000.0

        if not self.is_strictly_increasing(x_edges):
            x_edges = self.monotonic_edges_from_centers(np.nanmedian(np.mean(corners[:, :, :, 2], axis=2), axis=0) * 1000.0)
        if not self.is_strictly_increasing(y_edges):
            y_edges = self.monotonic_edges_from_centers(np.nanmedian(np.mean(corners[:, :, :, 1], axis=2), axis=1) * 1000.0)
        return x_edges, y_edges

    def is_strictly_increasing(self, values):
        values = np.asarray(values, dtype=float)
        return np.all(np.isfinite(values)) and np.all(np.diff(values) > 0)

    def monotonic_edges_from_centers(self, centers):
        centers = np.asarray(centers, dtype=float)
        if centers.size == 0:
            return np.array([], dtype=float)
        edges = np.empty(centers.size + 1, dtype=float)
        if centers.size == 1:
            edges[0] = centers[0] - 0.5
            edges[1] = centers[0] + 0.5
            return edges
        edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
        return edges

    def full_pixel_corner_grid_mm(self, corners):
        corners = np.asarray(corners, dtype=float)
        ny, nx = corners.shape[:2]
        x_grid = np.empty((ny + 1, nx + 1), dtype=float)
        y_grid = np.empty_like(x_grid)

        x_grid[:-1, :-1] = corners[:, :, 0, 2] * 1000.0
        y_grid[:-1, :-1] = corners[:, :, 0, 1] * 1000.0
        x_grid[1:, :-1] = corners[:, :, 1, 2] * 1000.0
        y_grid[1:, :-1] = corners[:, :, 1, 1] * 1000.0
        x_grid[1:, 1:] = corners[:, :, 2, 2] * 1000.0
        y_grid[1:, 1:] = corners[:, :, 2, 1] * 1000.0
        x_grid[:-1, 1:] = corners[:, :, 3, 2] * 1000.0
        y_grid[:-1, 1:] = corners[:, :, 3, 1] * 1000.0
        return x_grid, y_grid

    def pixel_corner_grid_mm(self, corners, row_edges, col_edges):
        ny, nx = corners.shape[:2]
        y_grid = np.empty((len(row_edges), len(col_edges)), dtype=float)
        x_grid = np.empty_like(y_grid)
        for i, row in enumerate(row_edges):
            for j, col in enumerate(col_edges):
                if row < ny and col < nx:
                    corner = corners[row, col, 0]
                elif row >= ny and col < nx:
                    corner = corners[ny - 1, col, 1]
                elif row < ny and col >= nx:
                    corner = corners[row, nx - 1, 3]
                else:
                    corner = corners[ny - 1, nx - 1, 2]
                y_grid[i, j] = float(corner[1]) * 1000.0
                x_grid[i, j] = float(corner[2]) * 1000.0
        return x_grid, y_grid

    def set_coordinate_labels(self, labels):
        self.coordinate_labels = labels

    def clear_coordinate_labels(self, _event=None):
        for detector, label in self.coordinate_labels.items():
            label.setText(f"{detector}: x = - | y = - | q = - | I = - | psi = -")
        self.hide_symmetry_source_marker()

    def hide_symmetry_source_marker(self):
        marker = self._symmetry_source_marker
        if marker is not None and marker.get_visible():
            marker.set_visible(False)
            self.draw_idle()

    def update_symmetry_source_marker(self, data, row, col):
        marker = self._symmetry_source_marker
        source_rows = data.get("cave_source_rows")
        source_cols = data.get("cave_source_cols")
        x_centers = data.get("real_x_centers_mm")
        y_centers = data.get("real_y_centers_mm")
        if (
            marker is None or source_rows is None or source_cols is None
            or x_centers is None or y_centers is None
            or source_rows.shape != source_cols.shape
        ):
            self.hide_symmetry_source_marker()
            return None
        source_row = int(source_rows[row, col])
        source_col = int(source_cols[row, col])
        if (
            source_row < 0 or source_col < 0
            or (source_row == row and source_col == col)
            or source_row >= len(y_centers) or source_col >= len(x_centers)
        ):
            self.hide_symmetry_source_marker()
            return None
        marker.set_data([float(x_centers[source_col])], [float(y_centers[source_row])])
        marker.set_visible(True)
        self.draw_idle()
        return source_row, source_col

    def on_motion(self, event):
        if self.detector_for_axis(event.inaxes) is not None:
            self._last_pointer_axis = event.inaxes
        detector = self.detector_for_real_axis(event.inaxes)
        if detector is None or event.xdata is None or event.ydata is None:
            self.hide_symmetry_source_marker()
            return

        data = self.loaded.get(detector, {})
        label = self.coordinate_labels.get(detector)
        if label is None:
            return

        kind = self.axis_kind(event.inaxes)
        if kind == "extended":
            row_centers = data.get("extended_y_centers_mm")
            col_centers = data.get("extended_x_centers_mm")
            image = data.get("extended_image")
            q_map = data.get("extended_q_nm")
            psi_map = None
        elif kind == "resized":
            image = self.resized_image_for_data(data)
            rows, cols = image.shape
            extended_x = np.asarray(data.get("extended_x_centers_mm"), dtype=float)
            side_mm = abs(float(np.nanmedian(np.diff(extended_x)))) if extended_x.size > 1 else 1.0
            row_centers = (np.arange(rows) - (rows - 1) / 2.0) * side_mm
            col_centers = (np.arange(cols) - (cols - 1) / 2.0) * side_mm
            q_map = None
            psi_map = None
        else:
            row_centers = data.get("real_y_centers_mm")
            col_centers = data.get("real_x_centers_mm")
            image = data.get("cave_image") if kind == "cave" else data.get("real_image")
            q_map = data.get("cave_q_nm") if kind == "cave" else data.get("real_q_nm")
            psi_map = data.get("real_psi_deg")
        if row_centers is None or col_centers is None or image is None:
            return

        row = self.nearest_sorted_index(row_centers, event.ydata)
        col = self.nearest_sorted_index(col_centers, event.xdata)
        if row is None or col is None or not (0 <= row < image.shape[0] and 0 <= col < image.shape[1]):
            label.setText(f"{detector}: x = - | y = - | q = - | I = - | psi = -")
            self.hide_symmetry_source_marker()
            return

        source_pixel = None
        if self.axis_kind(event.inaxes) == "cave":
            source_pixel = self.update_symmetry_source_marker(data, row, col)
        else:
            self.hide_symmetry_source_marker()

        value = float(image[row, col])
        if np.isfinite(value):
            intensity_text = f"I = {value:.6g}"
        else:
            intensity_text = "I = NaN"

        q_text = "q = -"
        if kind == "resized" and q_map is None:
            distance_m = float(data.get("distance_m", 1.0))
            wavelength_m = float(data.get("wavelength_m", 1e-10))
            radius_m = float(np.hypot(col_centers[col], row_centers[row]) * 1e-3)
            q_value = 4.0 * np.pi / wavelength_m * np.sin(0.5 * np.arctan2(radius_m, distance_m)) / 1e9
            q_text = f"q = {q_value:.6g} nm⁻¹"
        elif (
            q_map is not None
            and q_map.shape == image.shape
            and np.isfinite(q_map[row, col])
        ):
            q_text = f"q = {float(q_map[row, col]):.6g} nm⁻¹"

        psi_text = "psi = -"
        if psi_map is not None and psi_map.shape == image.shape and np.isfinite(psi_map[row, col]):
            psi_text = f"psi = {float(psi_map[row, col]):.3f}°"

        source_text = ""
        if source_pixel is not None:
            source_row, source_col = source_pixel
            source_text = f" | source sym. x = {source_col + 1}, y = {source_row + 1}"
        label.setText(
            f"{detector}: x = {col + 1} | y = {row + 1} | {q_text} | "
            f"{intensity_text} | {psi_text}{source_text}"
        )

    def point_inside_detector_pixel(self, pixel_corners, x_mm, y_mm):
        """True only when a real-coordinate cursor is inside a measured detector pixel."""
        corners = np.asarray(pixel_corners, dtype=float)
        if corners.shape != (4, 3) or not np.all(np.isfinite(corners)):
            return False
        x_values = corners[:, 2] * 1000.0
        y_values = corners[:, 1] * 1000.0
        tolerance = 1e-9
        return (
            float(np.nanmin(x_values)) - tolerance <= float(x_mm) <= float(np.nanmax(x_values)) + tolerance
            and float(np.nanmin(y_values)) - tolerance <= float(y_mm) <= float(np.nanmax(y_values)) + tolerance
        )

    def detector_for_real_axis(self, axis):
        if any(axis is candidate for candidate in self.axes):
            return self.selected_detector
        return None

    def nearest_sorted_index(self, values, target):
        if values is None or len(values) == 0 or not np.isfinite(target):
            return None
        index = int(np.searchsorted(values, target))
        if index <= 0:
            return 0
        if index >= len(values):
            return len(values) - 1
        before = values[index - 1]
        after = values[index]
        return index - 1 if abs(target - before) <= abs(after - target) else index

    def event(self, event):
        if event.type() == QEvent.Type.NativeGesture and self.handle_native_gesture(event):
            return True
        return super().event(event)

    def handle_native_gesture(self, event):
        gesture_type = event.gestureType()
        if gesture_type not in {
            Qt.NativeGestureType.ZoomNativeGesture,
            Qt.NativeGestureType.PanNativeGesture,
            Qt.NativeGestureType.SmartZoomNativeGesture,
        }:
            return False

        display_x, display_y = self.display_coords_from_qt_position(event.position())
        axis = self.axis_at_display_point(display_x, display_y)
        used_axis_fallback = False
        if axis is None or self.detector_for_axis(axis) is None:
            axis = self._last_pointer_axis
            used_axis_fallback = True
        detector = self.detector_for_axis(axis)
        if detector is None:
            return False
        if self.axis_kind(axis) == "raw":
            return False

        if self.axis_kind(axis) == "extended":
            if gesture_type == Qt.NativeGestureType.SmartZoomNativeGesture:
                axis.autoscale()
            elif gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
                value = float(event.value())
                if abs(value) < 1e-9:
                    value = float(event.delta().y()) * 0.01
                if abs(value) < 1e-9:
                    return False
                x_data, y_data = axis.transData.inverted().transform((display_x, display_y))
                self.zoom_extended_axis(axis, x_data, y_data, float(np.exp(-value)))
            else:
                delta = event.delta()
                self.pan_extended_axis(axis, float(delta.x()), -float(delta.y()))
            self.draw_idle()
            event.accept()
            return True

        if (
            self.axis_kind(axis) in {"real_base", "real", "zones", "cave"}
            and gesture_type == Qt.NativeGestureType.ZoomNativeGesture
        ):
            value = float(event.value())
            if abs(value) < 1e-9:
                value = float(event.delta().y()) * 0.01
            if abs(value) < 1e-9:
                return False
            x_data, y_data = axis.transData.inverted().transform((display_x, display_y))
            self.zoom_extended_axis(axis, x_data, y_data, float(np.exp(-value)))
            event.accept()
            return True

        if gesture_type == Qt.NativeGestureType.SmartZoomNativeGesture:
            bounds = self.default_detector_pixel_bounds(detector)
            if bounds is not None:
                self.apply_detector_pixel_view(detector, bounds)
                event.accept()
                return True
            return False

        if gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
            center = None if used_axis_fallback else self.pixel_point_from_display(detector, axis, display_x, display_y)
            if center is None:
                bounds = self.current_detector_pixel_bounds(detector, axis)
                if bounds is None:
                    return False
                center = (
                    0.5 * (bounds[0] + bounds[1]),
                    0.5 * (bounds[2] + bounds[3]),
                )
            value = float(event.value())
            if not np.isfinite(value):
                return False
            if abs(value) < 1e-9:
                delta = event.delta()
                value = float(delta.y()) * 0.01
            if abs(value) < 1e-9:
                return False
            factor = float(np.exp(-value))
            self.zoom_detector(detector, center, factor)
            event.accept()
            return True

        delta = event.delta()
        self.pan_detector_by_display_delta(detector, axis, float(delta.x()), -float(delta.y()))
        event.accept()
        return True

    def wheelEvent(self, event):
        display_x, display_y = self.display_coords_from_qt_position(event.position())
        axis = self.axis_at_display_point(display_x, display_y)
        detector = self.detector_for_axis(axis)
        if detector is None:
            super().wheelEvent(event)
            return
        if self.axis_kind(axis) == "raw":
            event.accept()
            return

        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        dx = float(pixel_delta.x())
        dy = float(pixel_delta.y())
        if dx == 0.0 and dy == 0.0:
            dx = float(angle_delta.x()) / 8.0
            dy = float(angle_delta.y()) / 8.0

        if self.axis_kind(axis) == "extended":
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier and dy != 0.0:
                x_data, y_data = axis.transData.inverted().transform((display_x, display_y))
                self.zoom_extended_axis(axis, x_data, y_data, 1.0 / 1.15 if dy > 0 else 1.15)
            else:
                self.pan_extended_axis(axis, dx, -dy)
            event.accept()
            return


        if (
            self.axis_kind(axis) in {"real_base", "real", "zones", "cave"}
            and event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and dy != 0.0
        ):
            x_data, y_data = axis.transData.inverted().transform((display_x, display_y))
            self.zoom_extended_axis(axis, x_data, y_data, 1.0 / 1.15 if dy > 0 else 1.15)
            event.accept()
            return

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and dy != 0.0:
            x_data, y_data = axis.transData.inverted().transform((display_x, display_y))
            factor = 1.0 / 1.15 if dy > 0 else 1.15
            self.zoom_extended_axis(axis, x_data, y_data, factor)
            event.accept()
            return

        self.pan_extended_axis(axis, dx, -dy)
        event.accept()

    def on_scroll(self, event):
        detector = self.detector_for_axis(event.inaxes)
        if detector is None or event.x is None or event.y is None:
            return
        if self.axis_kind(event.inaxes) == "raw":
            return
        if self.axis_kind(event.inaxes) in {"real_base", "real", "zones", "cave", "extended"}:
            if event.xdata is not None and event.ydata is not None:
                factor = 1.0 / 1.25 if event.step > 0 else 1.25
                self.zoom_extended_axis(event.inaxes, event.xdata, event.ydata, factor)
            return
        factor = 1.0 / 1.25 if event.step > 0 else 1.25
        if event.xdata is not None and event.ydata is not None:
            self.zoom_extended_axis(event.inaxes, event.xdata, event.ydata, factor)

    def on_button_press(self, event):
        if self.handle_expand_icon_click(event):
            return
        if self.handle_save_icon_click(event):
            return
        detector = self.detector_for_axis(event.inaxes)
        if detector is None:
            return
        if self.axis_kind(event.inaxes) == "raw":
            return
        if self.axis_kind(event.inaxes) == "zones":
            if event.xdata is None or event.ydata is None:
                return
            if event.button == 3:
                self.show_cave_region_menu(detector, float(event.xdata), float(event.ydata))
                return
            if event.button == 1:
                if self._cave_modifications_overlay is not None:
                    self._cave_modifications_overlay.set_visible(False)
                self._selection_detector = detector
                data = self.loaded.get(detector, {})
                self._selection_start = self.point_to_reference(
                    float(event.xdata), float(event.ydata),
                    data.get("center_mm"), data.get("reference_angle_deg", 0.0),
                )
                self._selection_last = self._selection_start
                self._selection_patch = Polygon(
                    [(float(event.xdata), float(event.ydata))] * 4,
                    closed=True,
                    facecolor="#ef4444",
                    edgecolor="#b91c1c",
                    linewidth=1.2,
                    alpha=0.28,
                    zorder=40,
                )
                fixed_xlim = event.inaxes.get_xlim()
                fixed_ylim = event.inaxes.get_ylim()
                event.inaxes.add_patch(self._selection_patch)
                event.inaxes.set_xlim(*fixed_xlim)
                event.inaxes.set_ylim(*fixed_ylim)
                self.draw_idle()
            return
        if self.axis_kind(event.inaxes) == "real" and event.button == 3:
            if event.xdata is not None and event.ydata is not None:
                self.show_nan_extension_menu(detector, float(event.xdata), float(event.ydata))
            return
        if event.button != 1 or event.x is None or event.y is None:
            return
        self._pan_detector = detector
        self._pan_last_pos = (float(event.x), float(event.y), event.inaxes)

    def handle_save_icon_click(self, event):
        if event.x is None or event.y is None:
            return False
        for icon, target in list(self.save_icons.items()):
            contains, _details = icon.contains(event)
            if contains:
                detector, kind, file_format = target
                self.save_requested.emit(detector, kind, file_format)
                return True
        return False

    def handle_expand_icon_click(self, event):
        if event.x is None or event.y is None:
            return False
        for icon in list(self.expand_icons):
            contains, _details = icon.contains(event)
            if contains:
                self.toggle_expanded_panels()
                return True
        return False

    def on_button_release(self, event):
        if self._selection_detector is not None and self._selection_start is not None:
            detector = self._selection_detector
            start_x, start_y = self._selection_start
            endpoint = self.selection_endpoint(event)
            if endpoint is None:
                endpoint = self._selection_last
            else:
                data = self.loaded.get(detector, {})
                endpoint = self.point_to_reference(
                    endpoint[0], endpoint[1], data.get("center_mm"),
                    data.get("reference_angle_deg", 0.0),
                )
            if endpoint is not None:
                end_x, end_y = endpoint
                x0, x1 = sorted((start_x, end_x))
                y0, y1 = sorted((start_y, end_y))
                data = self.loaded.get(detector, {})
                x_centers = np.asarray(data.get("real_x_centers_mm"), dtype=float)
                y_centers = np.asarray(data.get("real_y_centers_mm"), dtype=float)
                min_width = abs(float(np.nanmedian(np.diff(x_centers)))) if x_centers.size > 1 else 0.0
                min_height = abs(float(np.nanmedian(np.diff(y_centers)))) if y_centers.size > 1 else 0.0
                if x1 - x0 >= min_width and y1 - y0 >= min_height:
                    self.regions_about_to_change.emit(detector)
                    data.setdefault("cave_nan_regions", []).append({
                        "bounds_mm": [x0, x1, y0, y1],
                        "mode": "central",
                    })
                    self.regions_changed.emit(detector)
            self._selection_detector = None
            self._selection_start = None
            self._selection_last = None
            self._selection_patch = None
            self.refresh_detector_preserve_view(detector)
        self._pan_detector = None
        self._pan_last_pos = None

    def on_pan_motion(self, event):
        if self._selection_detector is not None and self._selection_start is not None:
            endpoint = self.selection_endpoint(event)
            if endpoint is not None:
                start_x, start_y = self._selection_start
                data = self.loaded.get(self._selection_detector, {})
                end_x, end_y = self.point_to_reference(
                    endpoint[0], endpoint[1], data.get("center_mm"),
                    data.get("reference_angle_deg", 0.0),
                )
                self._selection_last = (end_x, end_y)
                x0, x1 = sorted((start_x, end_x))
                y0, y1 = sorted((start_y, end_y))
                self._selection_patch.set_xy(self.region_polygon(
                    {"bounds_mm": [x0, x1, y0, y1], "mode": "central"},
                    data.get("center_mm"), data.get("reference_angle_deg", 0.0),
                ))
                self.draw_idle()
            return
        if self._pan_detector is None or self._pan_last_pos is None:
            return
        if event.x is None or event.y is None:
            return
        last_x, last_y, axis = self._pan_last_pos
        dx = float(event.x) - last_x
        dy = float(event.y) - last_y
        self.pan_extended_axis(axis, dx, dy)
        self._pan_last_pos = (float(event.x), float(event.y), axis)

    def selection_endpoint(self, event):
        axis = self.axes[2]
        if event.inaxes is axis and event.xdata is not None and event.ydata is not None:
            x_value, y_value = float(event.xdata), float(event.ydata)
        elif event.x is not None and event.y is not None:
            x_value, y_value = axis.transData.inverted().transform((float(event.x), float(event.y)))
        else:
            return None
        x_min, x_max = sorted(axis.get_xlim())
        y_min, y_max = sorted(axis.get_ylim())
        return (
            float(np.clip(x_value, x_min, x_max)),
            float(np.clip(y_value, y_min, y_max)),
        )

    def cave_region_at(self, detector, x_mm, y_mm):
        data = self.loaded.get(detector, {})
        regions = data.get("cave_nan_regions", [])
        for index in range(len(regions) - 1, -1, -1):
            region = normalize_cave_region(regions[index])
            zone_angle = (
                data.get("reference_angle_deg", 0.0)
                if region["rotate_with_reference"] else 0.0
            )
            reference_x, reference_y = self.point_to_reference(
                x_mm, y_mm, data.get("center_mm"), zone_angle
            )
            x0, x1, y0, y1 = region["bounds_mm"]
            if x0 <= reference_x <= x1 and y0 <= reference_y <= y1:
                return index
        return None

    def show_cave_region_menu(self, detector, x_mm, y_mm):
        index = self.cave_region_at(detector, x_mm, y_mm)
        data = self.loaded.get(detector, {})
        if detector == "Si4M" and index is None and len(data.get("cave_nan_regions", [])) >= 4:
            center_mm = data.get("center_mm")
            reference_x, reference_y = self.point_to_reference(
                x_mm, y_mm, center_mm, data.get("reference_angle_deg", 0.0)
            )
            center_x, center_y = map(float, center_mm)
            if reference_x < center_x:
                index = 0 if reference_y < center_y else 2
            else:
                index = 1 if reference_y < center_y else 3
        menu = QMenu(self)
        expand_action = menu.addAction("❌ Retour" if self.expanded_panels else "✏️ Agrandir")
        if index is None:
            selected = menu.exec(QCursor.pos())
            if selected is expand_action:
                self.toggle_expanded_panels()
            return
        current = normalize_cave_region(data["cave_nan_regions"][index])
        actions = {}
        for label, mode in (
            ("Central symmetry" if detector == "Si4M" else "NaN for caving", "central"),
            ("Vertical symmetry", "vertical"),
            ("Horizontal symmetry", "horizontal"),
        ):
            action = menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(current["mode"] == mode)
            actions[action] = mode
        menu.addSeparator()
        menu.addSeparator()
        delete_action = menu.addAction("Delete zone")
        selected_action = menu.exec(QCursor.pos())
        if selected_action is None:
            return
        if selected_action is expand_action:
            self.toggle_expanded_panels()
            return
        if selected_action is delete_action:
            self.regions_about_to_change.emit(detector)
            data["cave_nan_regions"].pop(index)
        elif selected_action in actions:
            self.regions_about_to_change.emit(detector)
            current["mode"] = actions[selected_action]
            current["expand_nan_px"] = 1 if current["mode"] == "extend_nan" else 0
            data["cave_nan_regions"][index] = current
        else:
            return
        self.regions_changed.emit(detector)
        self.refresh_detector_preserve_view(detector)

    def show_nan_extension_menu(self, detector, x_mm, y_mm):
        data = self.loaded.get(detector, {})
        image = np.asarray(data.get("real_image"))
        x_centers = np.asarray(data.get("real_x_centers_mm"), dtype=float)
        y_centers = np.asarray(data.get("real_y_centers_mm"), dtype=float)
        if image.ndim != 2 or x_centers.size != image.shape[1] or y_centers.size != image.shape[0]:
            return
        center_mm = data.get("center_mm")
        reference_x, reference_y = self.point_to_reference(
            x_mm, y_mm, center_mm, data.get("reference_angle_deg", 0.0)
        )
        col = int(np.argmin(np.abs(x_centers - reference_x)))
        row = int(np.argmin(np.abs(y_centers - reference_y)))
        if np.isfinite(image[row, col]):
            return
        menu = QMenu(self)
        actions = {menu.addAction(f"Extend NaN by {n} px"): n for n in range(1, 6)}
        selected = menu.exec(QCursor.pos())
        if selected not in actions:
            return
        n = actions[selected]
        nan_mask = ~np.isfinite(image)
        component = {(row, col)}
        pending = [(row, col)]
        while pending:
            rr, cc = pending.pop()
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if 0 <= nr < nan_mask.shape[0] and 0 <= nc < nan_mask.shape[1] and nan_mask[nr, nc] and (nr, nc) not in component:
                    component.add((nr, nc)); pending.append((nr, nc))
        for _ in range(n):
            component |= {(rr + dr, cc + dc) for rr, cc in component for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                          if 0 <= rr + dr < nan_mask.shape[0] and 0 <= cc + dc < nan_mask.shape[1]}
        zone = {
            "bounds_mm": [float(np.min(x_centers)), float(np.max(x_centers)), float(np.min(y_centers)), float(np.max(y_centers))],
            "pixel_indices": [[int(rr), int(cc)] for rr, cc in component],
            "mode": "extend_nan", "expand_nan_px": n,
            "rotate_with_reference": True, "visible": True,
        }
        self.regions_about_to_change.emit(detector)
        data.setdefault("cave_nan_regions", []).insert(0, zone)
        self.regions_changed.emit(detector)
        self.refresh_detector_preserve_view(detector)

    def refresh_detector_preserve_view(self, detector):
        limits = [(axis.get_xlim(), axis.get_ylim()) for axis in self.axes]
        self.draw_detectors(self.loaded, detector)
        for axis, (x_limits, y_limits) in zip(self.axes, limits):
            axis.set_xlim(*x_limits)
            axis.set_ylim(*y_limits)
        self.draw_idle()

    def zoom_detector(self, detector, center, factor):
        axis = self.detector_axis(detector, "raw")
        bounds = self.current_detector_pixel_bounds(detector, axis)
        if bounds is None:
            return
        col_min, col_max, row_min, row_max = bounds
        center_col, center_row = center
        if not all(np.isfinite(value) for value in (center_col, center_row, factor)):
            return
        factor = float(np.clip(factor, 0.05, 20.0))
        new_bounds = (
            center_col + (col_min - center_col) * factor,
            center_col + (col_max - center_col) * factor,
            center_row + (row_min - center_row) * factor,
            center_row + (row_max - center_row) * factor,
        )
        self.apply_detector_pixel_view(detector, new_bounds)

    def pan_detector_by_display_delta(self, detector, axis, dx, dy):
        if axis is None or axis.bbox.width <= 0 or axis.bbox.height <= 0:
            return
        if self.axis_kind(axis) in {"real_base", "real", "zones", "cave", "extended"}:
            self.pan_extended_axis(axis, dx, dy)
            return
        bounds = self.current_detector_pixel_bounds(detector, axis)
        if bounds is None:
            return
        col_min, col_max, row_min, row_max = bounds
        col_span = col_max - col_min
        row_span = row_max - row_min
        col_shift = -float(dx) / float(axis.bbox.width) * col_span
        row_shift = float(dy) / float(axis.bbox.height) * row_span
        self.apply_detector_pixel_view(
            detector,
            (
                col_min + col_shift,
                col_max + col_shift,
                row_min + row_shift,
                row_max + row_shift,
            ),
        )

    def pan_extended_axis(self, axis, dx, dy):
        origin = axis.transData.inverted().transform((0.0, 0.0))
        shifted = axis.transData.inverted().transform((float(dx), float(dy)))
        data_dx = float(shifted[0] - origin[0])
        data_dy = float(shifted[1] - origin[1])
        for target_axis in self.axes:
            x0, x1 = target_axis.get_xlim()
            y0, y1 = target_axis.get_ylim()
            target_axis.set_xlim(x0 - data_dx, x1 - data_dx)
            target_axis.set_ylim(y0 - data_dy, y1 - data_dy)
        self.draw_idle()

    def zoom_extended_axis(self, axis, center_x, center_y, factor):
        if not all(np.isfinite(value) for value in (center_x, center_y, factor)):
            return
        factor = float(np.clip(factor, 0.05, 20.0))
        for target_axis in self.axes:
            x0, x1 = target_axis.get_xlim()
            y0, y1 = target_axis.get_ylim()
            target_axis.set_xlim(center_x + (x0 - center_x) * factor, center_x + (x1 - center_x) * factor)
            target_axis.set_ylim(center_y + (y0 - center_y) * factor, center_y + (y1 - center_y) * factor)
        self.draw_idle()

    def current_detector_pixel_bounds(self, detector, axis):
        data = self.loaded.get(detector, {})
        image = data.get("5CB")
        if image is None:
            return None
        kind = self.axis_kind(axis)
        if kind == "raw":
            x0, x1 = axis.get_xlim()
            y0, y1 = axis.get_ylim()
            return (*sorted((x0, x1)), *sorted((y0, y1)))
        if kind in {"real_base", "real", "zones", "cave"}:
            col_centers = data.get("col_centers_mm")
            row_centers = data.get("row_centers_mm")
            if col_centers is None or row_centers is None:
                return None
            x0, x1 = sorted(axis.get_xlim())
            y0, y1 = sorted(axis.get_ylim())
            return (
                self.fractional_index_from_sorted(col_centers, x0),
                self.fractional_index_from_sorted(col_centers, x1),
                self.fractional_index_from_sorted(row_centers, y0),
                self.fractional_index_from_sorted(row_centers, y1),
            )
        return self.default_detector_pixel_bounds(detector)

    def default_detector_pixel_bounds(self, detector):
        image = self.loaded.get(detector, {}).get("5CB")
        if image is None:
            return None
        ny, nx = image.shape
        return -0.5, nx - 0.5, -0.5, ny - 0.5

    def apply_detector_pixel_view(self, detector, bounds):
        bounds = self.clamp_detector_pixel_bounds(detector, bounds)
        if bounds is None:
            return
        col_min, col_max, row_min, row_max = bounds
        self._syncing_limits = True
        try:
            for kind in ("raw", "real", "cave"):
                axis = self.detector_axis(detector, kind)
                x0, x1, y0, y1 = self.pixel_bounds_to_real_bounds(detector, bounds)
                if None in (x0, x1, y0, y1):
                    continue
                axis.set_xlim(x0, x1)
                axis.set_ylim(y1, y0)
        finally:
            self._syncing_limits = False
        self.draw_idle()

    def clamp_detector_pixel_bounds(self, detector, bounds):
        image = self.loaded.get(detector, {}).get("5CB")
        if image is None:
            return None
        ny, nx = image.shape
        col_min, col_max, row_min, row_max = bounds
        col_min, col_max = sorted((float(col_min), float(col_max)))
        row_min, row_max = sorted((float(row_min), float(row_max)))
        min_span = 8.0
        col_span = max(col_max - col_min, min_span)
        row_span = max(row_max - row_min, min_span)
        max_col_span = float(nx)
        max_row_span = float(ny)
        if col_span >= max_col_span:
            col_min, col_max = -0.5, nx - 0.5
        else:
            center = (col_min + col_max) * 0.5
            col_min = center - col_span * 0.5
            col_max = center + col_span * 0.5
            if col_min < -0.5:
                col_max += -0.5 - col_min
                col_min = -0.5
            if col_max > nx - 0.5:
                col_min -= col_max - (nx - 0.5)
                col_max = nx - 0.5
        if row_span >= max_row_span:
            row_min, row_max = -0.5, ny - 0.5
        else:
            center = (row_min + row_max) * 0.5
            row_min = center - row_span * 0.5
            row_max = center + row_span * 0.5
            if row_min < -0.5:
                row_max += -0.5 - row_min
                row_min = -0.5
            if row_max > ny - 0.5:
                row_min -= row_max - (ny - 0.5)
                row_max = ny - 0.5
        return col_min, col_max, row_min, row_max

    def pixel_bounds_to_real_bounds(self, detector, bounds):
        data = self.loaded.get(detector, {})
        col_centers = data.get("col_centers_mm")
        row_centers = data.get("row_centers_mm")
        if col_centers is None or row_centers is None:
            return None, None, None, None
        col_min, col_max, row_min, row_max = bounds
        return (
            self.value_at_fractional_index(col_centers, col_min),
            self.value_at_fractional_index(col_centers, col_max),
            self.value_at_fractional_index(row_centers, row_min),
            self.value_at_fractional_index(row_centers, row_max),
        )

    def pixel_point_from_display(self, detector, axis, display_x, display_y):
        if axis is None:
            return None
        x_data, y_data = axis.transData.inverted().transform((display_x, display_y))
        kind = self.axis_kind(axis)
        if kind == "raw":
            return float(x_data), float(y_data)
        if kind in {"real_base", "real", "zones", "cave"}:
            data = self.loaded.get(detector, {})
            col_centers = data.get("col_centers_mm")
            row_centers = data.get("row_centers_mm")
            if col_centers is None or row_centers is None:
                return None
            return (
                self.fractional_index_from_sorted(col_centers, x_data),
                self.fractional_index_from_sorted(row_centers, y_data),
            )
        return None

    def fractional_index_from_sorted(self, values, target):
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(target):
            return np.nan
        indexes = np.arange(values.size, dtype=float)
        if values[0] <= values[-1]:
            return float(np.interp(target, values, indexes, left=0.0, right=float(values.size - 1)))
        return float(np.interp(target, values[::-1], indexes[::-1], left=float(values.size - 1), right=0.0))

    def value_at_fractional_index(self, values, index):
        values = np.asarray(values, dtype=float)
        if values.size == 0 or not np.isfinite(index):
            return None
        indexes = np.arange(values.size, dtype=float)
        index = float(np.clip(index, 0.0, float(values.size - 1)))
        return float(np.interp(index, indexes, values))

    def display_coords_from_qt_position(self, position):
        return float(position.x()), float(self.height()) - float(position.y())

    def axis_at_display_point(self, display_x, display_y):
        for axis in np.ravel(self.axes):
            if axis.bbox.contains(display_x, display_y):
                return axis
        return None

    def detector_for_axis(self, axis):
        if any(axis is candidate for candidate in self.axes):
            return self.selected_detector
        return None

    def axis_kind(self, axis):
        if axis is None:
            return None
        if axis is self.axes[0]:
            return "raw"
        if axis is self.axes[1]:
            return "real"
        if axis is self.axes[2]:
            return "zones"
        if axis is self.axes[3]:
            return "cave"
        if axis is self.axes[4]:
            return "extended"
        if axis is self.axes[5]:
            return "resized"
        return None


class DoubleDetectorProject(QWidget):
    folder_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.folder = DOUBLE_DETECTOR_DEFAULT_FOLDER
        self.loaded = {}
        self.mask_disabled = False
        self.zone_undo_stacks = {detector: [] for detector in DOUBLE_DETECTOR_FILES}
        self.build_ui()
        self.undo_zones_shortcut = QShortcut(QKeySequence.Undo, self)
        self.undo_zones_shortcut.activated.connect(self.undo_zone_change)
        self.set_folder(self.folder)

    def build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        side = QGroupBox("BM02 D2AM")
        side.setStyleSheet(GROUP_BOX_STYLE)
        side.setFixedWidth(FILE_BROWSER_WIDTH)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(*PANEL_MARGINS)
        side_layout.setSpacing(6)

        self.folder_edit = QLineEdit()
        self.folder_edit.returnPressed.connect(lambda: self.set_folder(self.folder_edit.text().strip()))
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)
        self.detector_combo = QComboBox()
        self.detector_combo.addItems(["Si4M", "WOS", "Combo (test)"])
        self.detector_combo.setCurrentText("WOS")
        self.detector_combo.currentTextChanged.connect(self.set_selected_detector)
        load_button = QPushButton("Load test files")
        load_button.clicked.connect(self.load_test_files)

        side_layout.addWidget(QLabel("Test folder"))
        side_layout.addWidget(self.folder_edit)
        side_layout.addWidget(browse_button)
        side_layout.addWidget(QLabel("Detector"))
        side_layout.addWidget(self.detector_combo)
        self.combo_top_selector = QComboBox()
        self.combo_top_selector.addItems(["WOS au-dessus", "Si4M au-dessus"])
        self.combo_top_selector.setCurrentIndex(1)
        self.combo_top_selector.setVisible(False)
        self.combo_top_selector.currentIndexChanged.connect(self.rebuild_combo)
        side_layout.addWidget(self.combo_top_selector)
        self.si4m_display_angle_spin = QDoubleSpinBox()
        self.si4m_display_angle_spin.setRange(-180.0, 180.0)
        self.si4m_display_angle_spin.setDecimals(2)
        self.si4m_display_angle_spin.setSingleStep(1.0)
        self.si4m_display_angle_spin.setSuffix(" °")
        self.si4m_display_angle_spin.setValue(0.0)
        self.si4m_display_angle_spin.setToolTip("Rotation d'affichage de l'image Si4M")
        self.si4m_display_angle_spin.valueChanged.connect(self._si4m_display_angle_changed)
        self.si4m_display_angle_spin.setVisible(False)
        side_layout.addWidget(QLabel("Angle Si4M"))
        self.si4m_display_angle_label = side_layout.itemAt(side_layout.count() - 1).widget()
        self.si4m_display_angle_label.setVisible(False)
        side_layout.addWidget(self.si4m_display_angle_spin)
        side_layout.addWidget(load_button)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(0)
        side_layout.addWidget(self.summary, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #4b5563;")
        self.status.setVisible(False)

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_content_layout = QHBoxLayout()
        preview_content_layout.setContentsMargins(0, 0, 0, 0)
        preview_content_layout.setSpacing(8)
        self.canvas = DoubleDetectorCanvas()
        self.canvas.save_requested.connect(self.save_panel)
        self.canvas.regions_changed.connect(self.auto_save_cave_regions)
        self.canvas.regions_changed.connect(lambda _detector: self.refresh_custom_zones_panel())
        self.canvas.regions_about_to_change.connect(self.push_zone_undo_state)
        raw_preview_layout = QVBoxLayout()
        raw_preview_layout.setContentsMargins(0, 0, 0, 0)
        raw_preview_layout.setSpacing(4)
        raw_preview_layout.addWidget(self.canvas, 1)
        raw_file_layout = QHBoxLayout()
        raw_file_layout.setContentsMargins(0, 0, 0, 0)
        raw_file_layout.setSpacing(4)
        raw_file_layout.setAlignment(Qt.AlignLeft)
        self.raw_file_edit = QLineEdit()
        self.raw_file_edit.setFixedWidth(280)
        self.raw_file_edit.setPlaceholderText("Raw file (.h5)")
        self.raw_file_edit.setToolTip("Fichier H5 correspondant au détecteur sélectionné")
        raw_file_layout.addWidget(self.raw_file_edit, 1)
        raw_browse_button = QPushButton("…")
        raw_browse_button.setFixedWidth(30)
        raw_browse_button.clicked.connect(self.browse_raw_file)
        raw_file_layout.addWidget(raw_browse_button)
        raw_file_layout.setStretch(0, 1)
        raw_file_layout.setStretch(1, 0)
        raw_file_layout.setStretch(2, 0)
        mask_file_layout = QHBoxLayout()
        mask_file_layout.setContentsMargins(0, 0, 0, 0)
        mask_file_layout.setSpacing(4)
        self.mask_file_edit = QLineEdit()
        self.mask_file_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.mask_file_edit.setPlaceholderText("Mask (.edf)")
        mask_file_layout.addWidget(self.mask_file_edit, 1)
        mask_browse_button = QPushButton("…")
        mask_browse_button.setFixedWidth(30)
        mask_browse_button.clicked.connect(self.browse_mask_file)
        mask_file_layout.addWidget(mask_browse_button)
        self.apply_mask_button = QPushButton("Apply mask")
        self.apply_mask_button.clicked.connect(self.apply_selected_mask)
        mask_file_layout.addWidget(self.apply_mask_button)
        mask_cancel_button = QPushButton("×")
        mask_cancel_button.setFixedWidth(32)
        mask_cancel_button.setToolTip("Cancel mask")
        mask_cancel_button.clicked.connect(self.cancel_selected_mask)
        mask_file_layout.addWidget(mask_cancel_button)
        background_file_layout = QHBoxLayout()
        background_file_layout.setContentsMargins(0, 0, 0, 0)
        background_file_layout.setSpacing(4)
        self.background_file_edit = QLineEdit()
        self.background_file_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.background_file_edit.setPlaceholderText("Background (.h5)")
        background_file_layout.addWidget(self.background_file_edit, 1)
        background_browse_button = QPushButton("…")
        background_browse_button.setFixedWidth(30)
        background_browse_button.clicked.connect(self.choose_background_file)
        background_file_layout.addWidget(background_browse_button)
        self.apply_background_button = QPushButton("Apply background")
        self.apply_background_button.clicked.connect(self.apply_selected_background)
        background_file_layout.addWidget(self.apply_background_button)
        background_cancel_button = QPushButton("×")
        background_cancel_button.setFixedWidth(32)
        background_cancel_button.setToolTip("Cancel background")
        background_cancel_button.clicked.connect(self.cancel_selected_background)
        background_file_layout.addWidget(background_cancel_button)
        raw_controls_layout = QVBoxLayout()
        raw_controls_layout.setContentsMargins(0, 0, 0, 0)
        raw_controls_layout.setSpacing(2)
        raw_controls_layout.addLayout(mask_file_layout)
        raw_controls_layout.addLayout(background_file_layout)
        self.raw_controls_widget = QWidget()
        self.raw_controls_widget.setLayout(raw_controls_layout)
        side_layout.addWidget(self.raw_controls_widget, 0)
        preview_content_layout.addLayout(raw_preview_layout, 1)

        self.custom_zones_box = QGroupBox("Custom zones")
        self.custom_zones_box.setStyleSheet(GROUP_BOX_STYLE)
        self.custom_zones_box.setFixedWidth(FILE_BROWSER_WIDTH)
        custom_zones_box_layout = QVBoxLayout(self.custom_zones_box)
        custom_zones_box_layout.setContentsMargins(8, 20, 8, 8)
        custom_zones_box_layout.setSpacing(4)
        custom_zones_box_layout.addWidget(QLabel("Reference angle"))
        self.reference_angle_spin = QDoubleSpinBox()
        self.reference_angle_spin.setRange(-360.0, 360.0)
        self.reference_angle_spin.setDecimals(6)
        self.reference_angle_spin.setSingleStep(1.0)
        self.reference_angle_spin.setSuffix(" °")
        self.reference_angle_spin.valueChanged.connect(self.set_reference_angle)
        reference_angle_layout = QHBoxLayout()
        reference_angle_layout.setContentsMargins(0, 0, 0, 0)
        reference_angle_layout.setSpacing(4)
        reference_angle_layout.addWidget(self.reference_angle_spin, 1)
        self.reference_angle_toggle = QPushButton("Tilt plane")
        self.reference_angle_toggle.setToolTip("Toggle between 0° and −tilt plane")
        self.reference_angle_toggle.clicked.connect(self.toggle_reference_angle)
        reference_angle_layout.addWidget(self.reference_angle_toggle)
        custom_zones_box_layout.addLayout(reference_angle_layout)
        table_header = QHBoxLayout()
        table_header.setContentsMargins(0, 0, 0, 0)
        table_header.setSpacing(4)
        zone_header = QLabel("Zone")
        zone_header.setFixedWidth(52)
        mode_header = QLabel("Cave operation")
        mode_header.setAlignment(Qt.AlignCenter)
        rotation_header = QLabel("Ref.")
        rotation_header.setFixedWidth(28)
        delete_header = QLabel(" ")
        delete_header.setFixedWidth(24)
        table_header.addWidget(zone_header)
        table_header.addWidget(mode_header, 1)
        table_header.addWidget(rotation_header)
        table_header.addWidget(delete_header)
        custom_zones_box_layout.addLayout(table_header)
        self._refreshing_custom_zones = False
        self.custom_zones_list = ReorderableZoneList()
        self.custom_zones_list.setFrameShape(QListWidget.NoFrame)
        self.custom_zones_list.setDragEnabled(True)
        self.custom_zones_list.setAcceptDrops(True)
        self.custom_zones_list.setDropIndicatorShown(True)
        self.custom_zones_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.custom_zones_list.setDefaultDropAction(Qt.MoveAction)
        self.custom_zones_list.setDragDropOverwriteMode(False)
        self.custom_zones_list.setSpacing(1)
        self.custom_zones_list.setUniformItemSizes(True)
        self.custom_zones_list.reorder_requested.connect(self.reorder_custom_zone)
        self.custom_zones_list.currentRowChanged.connect(self.select_custom_zone)
        custom_zones_box_layout.addWidget(self.custom_zones_list, 1)
        self.auto_tile_zones_button = QPushButton("Auto")
        self.auto_tile_zones_button.setToolTip(
            "Create horizontal and vertical symmetry zones for all four reference-axis quadrants"
        )
        self.auto_tile_zones_button.clicked.connect(self.add_auto_tile_zones)
        custom_zones_box_layout.addWidget(self.auto_tile_zones_button)
        preview_layout.addLayout(preview_content_layout, 1)

        coordinate_layout = QHBoxLayout()
        coordinate_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_layout.setSpacing(8)
        self.si4m_coordinate_label = self.make_coordinate_label("Si4M")
        self.wos_coordinate_label = self.make_coordinate_label("WOS")
        self.combo_coordinate_label = self.make_coordinate_label("Combo")
        coordinate_layout.addWidget(self.si4m_coordinate_label)
        coordinate_layout.addWidget(self.wos_coordinate_label)
        coordinate_layout.addWidget(self.combo_coordinate_label)
        preview_layout.addLayout(coordinate_layout)
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        controls_layout.addWidget(QLabel("Intensity min"))
        self.intensity_min_slider = QSlider(Qt.Horizontal)
        self.intensity_min_slider.setRange(0, 1000)
        self.intensity_min_slider.setValue(0)
        self.intensity_min_slider.valueChanged.connect(self.change_intensity_limits)
        controls_layout.addWidget(self.intensity_min_slider, 1)
        controls_layout.addWidget(QLabel("Intensity max"))
        self.intensity_max_slider = QSlider(Qt.Horizontal)
        self.intensity_max_slider.setRange(0, 1000)
        self.intensity_max_slider.setValue(1000)
        self.intensity_max_slider.valueChanged.connect(self.change_intensity_limits)
        controls_layout.addWidget(self.intensity_max_slider, 1)
        controls_layout.addWidget(QLabel("Colormap"))
        self.colormap_combo = QComboBox()
        for name in ("turbo", "viridis", "inferno", "magma", "plasma", "gray", "jet"):
            self.colormap_combo.addItem(name, name)
        self.colormap_combo.currentIndexChanged.connect(self.change_colormap)
        controls_layout.addWidget(self.colormap_combo)
        controls_layout.addWidget(QLabel("Resize px"))
        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(1, 10000)
        self.resize_width_spin.setValue(1000)
        self.resize_width_spin.setToolTip("Largeur de RESIZED PATTERN en pixels")
        self.resize_width_spin.valueChanged.connect(self.change_resize_shape)
        controls_layout.addWidget(self.resize_width_spin)
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(1, 10000)
        self.resize_height_spin.setValue(1000)
        self.resize_height_spin.setToolTip("Hauteur de RESIZED PATTERN en pixels")
        self.resize_height_spin.valueChanged.connect(self.change_resize_shape)
        controls_layout.addWidget(self.resize_height_spin)
        self.final_save_button = QPushButton("💾")
        self.final_save_button.setToolTip("Enregistrer le motif final en H5")
        self.final_save_button.setFixedSize(34, 28)
        self.final_save_button.clicked.connect(
            lambda: self.save_panel(
                self.selected_detector(),
                "extended" if self.selected_detector() == "Si4M" else "resized",
                "h5",
            )
        )
        controls_layout.addWidget(self.final_save_button)
        preview_layout.addLayout(controls_layout)
        self.canvas.set_coordinate_labels({
            "Si4M": self.si4m_coordinate_label,
            "WOS": self.wos_coordinate_label,
            "Combo (test)": self.combo_coordinate_label,
        })
        self.update_coordinate_label_visibility()
        self.refresh_custom_zones_panel()

        side_splitter = QSplitter(Qt.Vertical)
        side_splitter.setChildrenCollapsible(False)
        side_splitter.addWidget(side)
        side_splitter.addWidget(self.custom_zones_box)
        side_splitter.setStretchFactor(0, 1)
        side_splitter.setStretchFactor(1, 1)
        side_splitter.setSizes([420, 420])
        layout.addWidget(side_splitter, 0)
        layout.addLayout(preview_layout, 1)

    def change_colormap(self, _index):
        self.canvas.colormap_name = self.colormap_combo.currentData() or "turbo"
        if self.loaded:
            self.canvas.draw_detectors(self.loaded, self.selected_detector())

    def change_extend_nan(self, _value=None):
        self.canvas.extend_nan_enabled = self.extend_nan_checkbox.isChecked()
        self.canvas.extend_nan_pixels = self.extend_nan_spin.value()
        if self.loaded:
            self.canvas.draw_detectors(self.loaded, self.selected_detector())

    def update_raw_file_field(self, detector=None):
        detector = detector or self.selected_detector()
        config = DOUBLE_DETECTOR_FILES.get(detector, {})
        self.raw_file_edit.setText(str(self.folder / config.get("5CB", "")))
        self.mask_file_edit.setText(str(self.folder / config.get("mask", "")))
        default_background = self.folder / f"{detector}_0000 capillaire.h5"
        self.background_file = default_background if default_background.exists() else None
        self.background_file_edit.setText(str(default_background) if default_background.exists() else "")

    def browse_raw_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose raw H5 file", str(self.folder), "H5 files (*.h5 *.hdf5)")
        if path:
            self.raw_file_edit.setText(path)

    def browse_mask_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose mask file", str(self.folder), "Mask files (*.edf *.h5 *.hdf5)")
        if path:
            self.mask_file_edit.setText(path)

    def choose_background_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose background file", str(self.folder), "H5 files (*.h5 *.hdf5)")
        if path:
            self.background_file = Path(path)
            self.background_file_edit.setText(path)
            self.status.setText(f"Background selected: {self.background_file.name}")

    def apply_selected_mask(self):
        self.mask_disabled = False
        self.load_test_files()

    def cancel_selected_mask(self):
        self.mask_disabled = True
        self.mask_file_edit.clear()
        self.load_test_files()

    def apply_selected_background(self):
        path = Path(self.background_file_edit.text().strip()).expanduser()
        if not path.exists():
            QMessageBox.warning(self, "Background", "Sélectionne un fichier background valide.")
            return
        self.background_file = path
        self.load_test_files()

    def cancel_selected_background(self):
        self.background_file = None
        self.background_file_edit.clear()
        self.load_test_files()

    def change_intensity_limits(self, _value):
        minimum = float(self.intensity_min_slider.value()) / 10.0
        maximum = float(self.intensity_max_slider.value()) / 10.0
        if minimum >= maximum:
            if self.sender() is self.intensity_min_slider:
                maximum = min(100.0, minimum + 0.1)
                self.intensity_max_slider.blockSignals(True)
                self.intensity_max_slider.setValue(round(maximum * 10))
                self.intensity_max_slider.blockSignals(False)
            else:
                minimum = max(0.0, maximum - 0.1)
                self.intensity_min_slider.blockSignals(True)
                self.intensity_min_slider.setValue(round(minimum * 10))
                self.intensity_min_slider.blockSignals(False)
        self.canvas.intensity_min_percentile = minimum
        self.canvas.intensity_max_percentile = maximum
        if self.loaded:
            self.canvas.draw_detectors(self.loaded, self.selected_detector())

    def change_resize_shape(self, _value):
        shape = (self.resize_height_spin.value(), self.resize_width_spin.value())
        for data in self.loaded.values():
            data["resized_shape"] = shape
        if self.loaded:
            self.canvas.draw_detectors(self.loaded, self.selected_detector())

    def make_coordinate_label(self, detector):
        label = QLabel(f"{detector}: x = - | y = - | q = - | I = - | psi = -")
        label.setMinimumHeight(28)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(15, 23, 42, 0.08);
                border-radius: 8px;
                color: #111827;
                font-family: Menlo, Monaco, Consolas, monospace;
                font-size: 11px;
                padding: 4px 6px;
            }
        """)
        return label

    def set_folder_from_external_tab(self, folder):
        self.set_folder(folder)

    def selected_detector(self):
        return self.detector_combo.currentText() if hasattr(self, "detector_combo") else "WOS"

    def set_selected_detector(self, detector):
        if detector not in DOUBLE_DETECTOR_FILES and detector != "Combo (test)":
            return
        self.combo_top_selector.setVisible(detector == "Combo (test)")
        self.si4m_display_angle_spin.setVisible(detector == "Combo (test)")
        self.si4m_display_angle_label.setVisible(detector == "Combo (test)")
        if detector in DOUBLE_DETECTOR_FILES:
            self.update_raw_file_field(detector)
        elif self.loaded:
            self.rebuild_combo()
        self.canvas.selected_detector = detector
        self.canvas.selected_region_index = None
        self.update_coordinate_label_visibility()
        self.canvas.clear_coordinate_labels()
        if self.loaded:
            self.canvas.draw_detectors(self.loaded, detector)
            self.status.setText(
                f"Showing {detector}. Layout: real coordinates, zones, caved pattern, extended cave pattern."
            )

    def _si4m_display_angle_changed(self, _value):
        if self.selected_detector() == "Combo (test)" and self.loaded:
            self.rebuild_combo()
        self.refresh_custom_zones_panel()
        resized_shape = self.loaded.get(detector, {}).get("resized_shape")
        if resized_shape:
            rows, cols = resized_shape
            self.resize_height_spin.blockSignals(True)
            self.resize_width_spin.blockSignals(True)
            self.resize_height_spin.setValue(int(rows))
            self.resize_width_spin.setValue(int(cols))
            self.resize_height_spin.blockSignals(False)
            self.resize_width_spin.blockSignals(False)

    def default_resized_shape(self, data):
        image = data.get("real_image")
        image = data.get("extended_image")
        center = data.get("extended_center_pixel")
        if image is None or center is None:
            image = data.get("real_image")
            center = data.get("real_center_pixel")
        if image is None or center is None:
            return None
        rows, cols = np.shape(image)
        finite = np.isfinite(np.asarray(image, dtype=float))
        valid_rows = np.where(np.any(finite, axis=1))[0]
        valid_cols = np.where(np.any(finite, axis=0))[0]
        if valid_rows.size and valid_cols.size:
            row_min, row_max = int(valid_rows[0]), int(valid_rows[-1])
            col_min, col_max = int(valid_cols[0]), int(valid_cols[-1])
        else:
            row_min, row_max, col_min, col_max = 0, rows - 1, 0, cols - 1
        center_col, center_row = map(float, center)
        max_square_width = 2 * int(np.floor(min(center_col - col_min, float(col_max) - center_col))) + 1
        max_square_height = 2 * int(np.floor(min(center_row - row_min, float(row_max) - center_row))) + 1
        side = max(1, min(max_square_width, max_square_height))
        return side, side

    def update_coordinate_label_visibility(self):
        detector = self.selected_detector()
        if hasattr(self, "si4m_coordinate_label"):
            self.si4m_coordinate_label.setVisible(detector == "Si4M")
        if hasattr(self, "wos_coordinate_label"):
            self.wos_coordinate_label.setVisible(detector == "WOS")
        if hasattr(self, "combo_coordinate_label"):
            self.combo_coordinate_label.setVisible(detector == "Combo (test)")

    def refresh_custom_zones_panel(self):
        if not hasattr(self, "custom_zones_list"):
            return
        scroll_value = self.custom_zones_list.verticalScrollBar().value()
        self._refreshing_custom_zones = True
        self.custom_zones_list.clear()
        detector = self.selected_detector()
        data = self.loaded.get(detector, {})
        if hasattr(self, "reference_angle_spin"):
            self.reference_angle_spin.blockSignals(True)
            current_angle = float(data.get("reference_angle_deg", 0.0))
            self.reference_angle_spin.setValue(current_angle)
            self.reference_angle_spin.blockSignals(False)
            tilt_angle = -float(data.get("tilt_plane_deg", 0.0))
            self.reference_angle_toggle.setText(
                "0" if abs(current_angle - tilt_angle) < 5e-6 else "Tilt plane"
            )
        regions = data.get("cave_nan_regions", [])
        if not regions:
            empty_item = QListWidgetItem("No custom zone")
            empty_item.setFlags(Qt.NoItemFlags)
            empty_item.setTextAlignment(Qt.AlignCenter)
            self.custom_zones_list.addItem(empty_item)
        else:
            for index, raw_region in enumerate(regions):
                region = normalize_cave_region(raw_region)
                data["cave_nan_regions"][index] = region
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(4)
                number_label = QPushButton(f"#{index + 1}")
                number_label.setFixedWidth(26)
                number_label.setFixedHeight(24)
                mode_colors = {
                    "central": "#fee2e2",
                    "extend_nan": "#fef3c7",
                    "vertical": "#dbeafe",
                    "horizontal": "#dcfce7",
                }
                number_label.setStyleSheet(
                    f"background: {mode_colors[region['mode']] if region['visible'] else '#e5e7eb'}; "
                    "border: none; border-radius: 4px; padding: 2px;"
                )
                number_label.setToolTip(
                    "Hide zone overlay" if region["visible"] else "Show zone overlay"
                )
                number_label.clicked.connect(
                    lambda _checked=False, d=detector, i=index: self.toggle_zone_visibility(d, i)
                )
                mode_combo = SelectingComboBox()
                mode_combo.addItem(
                    "Central symmetry" if detector == "Si4M" else "NaN for caving",
                    "central",
                )
                mode_combo.addItem(
                    f"Extend NaN by {region.get('expand_nan_px', 1)} px", "extend_nan"
                )
                mode_combo.addItem("Vertical symmetry", "vertical")
                mode_combo.addItem("Horizontal symmetry", "horizontal")
                mode_combo.setCurrentIndex(mode_combo.findData(region["mode"]))
                mode_combo.pressed.connect(
                    lambda i=index: self.custom_zones_list.setCurrentRow(i)
                )
                mode_combo.currentIndexChanged.connect(
                    lambda _value, d=detector, i=index, combo=mode_combo: self.set_zone_mode_from_panel(
                        d, i, combo.currentData()
                    )
                )
                orientation_button = QPushButton(
                    "↻" if region["rotate_with_reference"] else "0"
                )
                orientation_button.setFixedSize(28, 24)
                orientation_button.setToolTip(
                    "Zone follows Reference angle"
                    if region["rotate_with_reference"]
                    else "Zone fixed without Reference angle tilt"
                )
                orientation_button.clicked.connect(
                    lambda _checked=False, d=detector, i=index: self.toggle_zone_reference_rotation(d, i)
                )
                delete_button = QPushButton("−")
                delete_button.setFixedSize(24, 24)
                delete_button.setToolTip("Delete zone")
                delete_button.clicked.connect(
                    lambda _checked=False, d=detector, i=index: self.delete_zone_from_panel(d, i)
                )
                row_layout.addWidget(number_label)
                row_layout.addWidget(mode_combo, 1)
                row_layout.addWidget(orientation_button)
                row_layout.addWidget(delete_button)
                list_item = QListWidgetItem()
                list_item.setData(Qt.UserRole, index)
                list_item.setSizeHint(row_widget.sizeHint())
                list_item.setFlags(
                    list_item.flags()
                    | Qt.ItemIsDragEnabled
                    | Qt.ItemIsDropEnabled
                    | Qt.ItemIsEnabled
                    | Qt.ItemIsSelectable
                )
                self.custom_zones_list.addItem(list_item)
                self.custom_zones_list.setItemWidget(list_item, row_widget)
        selected_index = self.canvas.selected_region_index
        if selected_index is not None and 0 <= selected_index < len(regions):
            self.custom_zones_list.setCurrentRow(selected_index)
        self.custom_zones_list.verticalScrollBar().setValue(scroll_value)
        self._refreshing_custom_zones = False

    def select_custom_zone(self, index):
        if self._refreshing_custom_zones:
            return
        regions = self.loaded.get(self.selected_detector(), {}).get("cave_nan_regions", [])
        self.canvas.selected_region_index = index if 0 <= index < len(regions) else None
        if self.loaded:
            self.canvas.refresh_detector_preserve_view(self.selected_detector())

    def toggle_zone_visibility(self, detector, index):
        data = self.loaded.get(detector, {})
        regions = data.get("cave_nan_regions", [])
        if not (0 <= index < len(regions)):
            return
        self.push_zone_undo_state(detector)
        region = normalize_cave_region(regions[index])
        region["visible"] = not region["visible"]
        regions[index] = region
        if not region["visible"] and self.canvas.selected_region_index == index:
            self.canvas.selected_region_index = None
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)

    def reorder_custom_zone(self, source_row, insertion_row):
        detector = self.selected_detector()
        data = self.loaded.get(detector, {})
        regions = data.get("cave_nan_regions", [])
        if not (
            0 <= source_row < len(regions)
            and 0 <= insertion_row < len(regions)
            and source_row != insertion_row
        ):
            return
        self.push_zone_undo_state(detector)
        moved_region = regions.pop(source_row)
        regions.insert(insertion_row, moved_region)
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)
        self.refresh_custom_zones_panel()

    def set_reference_angle(self, angle_deg):
        detector = self.selected_detector()
        data = self.loaded.get(detector)
        if data is None:
            return
        if abs(float(data.get("reference_angle_deg", 0.0)) - float(angle_deg)) < 1e-12:
            return
        self.push_zone_undo_state(detector)
        data["reference_angle_deg"] = float(angle_deg)
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)

    def toggle_reference_angle(self):
        detector = self.selected_detector()
        data = self.loaded.get(detector)
        if data is None:
            return
        tilt_angle = -float(data.get("tilt_plane_deg", 0.0))
        self.reference_angle_spin.setValue(
            0.0 if self.reference_angle_toggle.text() == "0" else tilt_angle
        )

    def add_auto_tile_zones(self):
        detector = self.selected_detector()
        data = self.loaded.get(detector)
        if data is None or data.get("real_image") is None:
            self.status.setText("Load the detector image before creating automatic tile zones.")
            return
        x_centers = np.asarray(data.get("real_x_centers_mm"), dtype=float)
        y_centers = np.asarray(data.get("real_y_centers_mm"), dtype=float)
        center_mm = data.get("center_mm")
        angle_deg = float(data.get("reference_angle_deg", 0.0))
        x_step = abs(float(np.nanmedian(np.diff(x_centers))))
        y_step = abs(float(np.nanmedian(np.diff(y_centers))))
        center_x, center_y = map(float, center_mm)
        x_edges = (float(x_centers[0] - x_step / 2.0), float(x_centers[-1] + x_step / 2.0))
        y_edges = (float(y_centers[0] - y_step / 2.0), float(y_centers[-1] + y_step / 2.0))
        reference_corners = [
            self.canvas.point_to_reference(x_value, y_value, center_mm, angle_deg)
            for x_value in x_edges for y_value in y_edges
        ]
        u_min = min(point[0] for point in reference_corners)
        u_max = max(point[0] for point in reference_corners)
        v_min = min(point[1] for point in reference_corners)
        v_max = max(point[1] for point in reference_corners)

        if detector == "Si4M":
            radius = max(
                abs(u_min - center_x), abs(u_max - center_x),
                abs(v_min - center_y), abs(v_max - center_y),
            )
            quadrant_bounds = [
                [center_x - radius, center_x, center_y - radius, center_y],
                [center_x, center_x + radius, center_y - radius, center_y],
                [center_x - radius, center_x, center_y, center_y + radius],
                [center_x, center_x + radius, center_y, center_y + radius],
            ]
            self.push_zone_undo_state(detector)
            data["cave_nan_regions"] = [
                {"bounds_mm": list(bounds), "mode": "central", "visible": True,
                 "rotate_with_reference": True}
                for bounds in quadrant_bounds
            ]
            max_pixels = 2 * int(np.ceil(radius / max(x_step, y_step))) + 1
            data["si4m_max_extended_shape"] = (max_pixels, max_pixels)
            self.canvas.regions_changed.emit(detector)
            self.canvas.refresh_detector_preserve_view(detector)
            self.status.setText(
                f"Si4M: 4 zones created; maximum reconstructed image {max_pixels} × {max_pixels} px."
            )
            return

        quadrant_bounds = [
            [u_min, center_x, v_min, center_y],
            [center_x, u_max, v_min, center_y],
            [u_min, center_x, center_y, v_max],
            [center_x, u_max, center_y, v_max],
        ]
        quadrant_zones = [
            {"bounds_mm": list(bounds), "mode": mode}
            for bounds in quadrant_bounds
            for mode in ("horizontal", "vertical")
        ]

        self.push_zone_undo_state(detector)

        manual_nan_regions = [
            normalize_cave_region(region)
            for region in data.get("cave_nan_regions", [])
            if normalize_cave_region(region)["mode"] == "central"
        ]
        data["cave_nan_regions"] = manual_nan_regions + quadrant_zones
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)
        self.status.setText("Auto zones: 4 axis quadrants, 8 symmetry zones created and saved.")

    def set_zone_mode_from_panel(self, detector, index, mode):
        data = self.loaded.get(detector, {})
        regions = data.get("cave_nan_regions", [])
        if not (0 <= index < len(regions)) or mode not in {"central", "extend_nan", "vertical", "horizontal"}:
            return
        region = normalize_cave_region(regions[index])
        if region["mode"] == mode:
            return
        self.push_zone_undo_state(detector)
        region["mode"] = mode
        region["expand_nan_px"] = 1 if mode == "extend_nan" else 0
        regions[index] = region
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)

    def toggle_zone_reference_rotation(self, detector, index):
        data = self.loaded.get(detector, {})
        regions = data.get("cave_nan_regions", [])
        if not (0 <= index < len(regions)):
            return
        self.push_zone_undo_state(detector)
        region = normalize_cave_region(regions[index])
        old_angle = (
            float(data.get("reference_angle_deg", 0.0))
            if region["rotate_with_reference"] else 0.0
        )
        new_rotate = not region["rotate_with_reference"]
        new_angle = float(data.get("reference_angle_deg", 0.0)) if new_rotate else 0.0
        x0, x1, y0, y1 = region["bounds_mm"]
        old_center = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
        lab_center = self.canvas.point_from_reference(
            old_center[0], old_center[1], data.get("center_mm"), old_angle
        )
        new_center = self.canvas.point_to_reference(
            lab_center[0], lab_center[1], data.get("center_mm"), new_angle
        )
        half_width = (x1 - x0) * 0.5
        half_height = (y1 - y0) * 0.5
        region["bounds_mm"] = [
            new_center[0] - half_width, new_center[0] + half_width,
            new_center[1] - half_height, new_center[1] + half_height,
        ]
        region["rotate_with_reference"] = new_rotate
        regions[index] = region
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)

    def delete_zone_from_panel(self, detector, index):
        data = self.loaded.get(detector, {})
        regions = data.get("cave_nan_regions", [])
        if not (0 <= index < len(regions)):
            return
        self.push_zone_undo_state(detector)
        regions.pop(index)
        selected = self.canvas.selected_region_index
        if selected == index:
            self.canvas.selected_region_index = None
        elif selected is not None and selected > index:
            self.canvas.selected_region_index = selected - 1
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)

    def set_folder(self, folder):
        folder = Path(folder).expanduser()
        self.folder = folder
        if hasattr(self, "raw_file_edit"):
            self.update_raw_file_field()
        self.folder_edit.setText(str(folder))
        self.refresh_summary()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose double-detector folder", str(self.folder))
        if folder:
            self.set_folder(folder)
            self.folder_changed.emit(Path(folder))

    def refresh_summary(self):
        lines = []
        for detector in ("Si4M", "WOS"):
            lines.append(detector_summary(self.folder, detector))
            lines.append("")
        self.summary.setPlainText("\n".join(lines).strip())

    def rebuild_combo(self, _index=None):
        if not hasattr(self, "loaded"):
            return
        try:
            combo = self.load_combo_data()
        except Exception as exc:
            self.status.setText(f"Combo unavailable: {exc}")
            return
        self.loaded["Combo (test)"] = combo
        if self.selected_detector() == "Combo (test)":
            self.canvas.draw_detectors(self.loaded, "Combo (test)")
            self.update_coordinate_label_visibility()

    def load_combo_data(self):
        import h5py
        from scipy import ndimage

        paths = {
            "Si4M": self.folder / "Si4M_5CB_extended_geometry.h5",
            "WOS": self.folder / "WOS_5CB_resized_geometry.h5",
        }
        sources = {}
        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(path.name)
            dataset_name, *_ = inspect_h5_image_dataset(path)
            image, _ = read_h5_frame(path, dataset_name, 0, add_matching_center=False)
            with h5py.File(path, "r") as h5:
                dataset = h5[dataset_name]
                attrs = dict(h5.attrs); attrs.update(dict(dataset.attrs))
                cx = float(attrs.get("Center_1", (image.shape[1] - 1) / 2.0))
                cy = float(attrs.get("Center_2", (image.shape[0] - 1) / 2.0))
                q_map = None
                for candidate in (
                    "/entry_0000/instrument/detector/q_nm_inverse",
                    "/entry_0000/instrument/detector/q_nm",
                ):
                    if candidate in h5:
                        q_map = np.asarray(h5[candidate], dtype=float)
                        break
            if q_map is None or q_map.shape != np.shape(image):
                raise ValueError(f"q map missing in {path.name}")
            offsets = (10, 20, 40)
            q_steps = []
            for offset in offsets:
                col = min(image.shape[1] - 1, int(round(cx)) + offset)
                row = int(np.clip(round(cy), 0, image.shape[0] - 1))
                if np.isfinite(q_map[row, col]):
                    q_steps.append(float(q_map[row, col]) / max(1, col - cx))
            if not q_steps:
                raise ValueError(f"invalid q geometry in {path.name}")
            sources[name] = {
                "image": np.asarray(image, dtype=float), "center": (cx, cy),
                "dq": abs(float(np.nanmedian(q_steps))), "q_map": q_map, "path": path,
            }

        # Le WOS est le fond WAXS, conservé à sa taille native. Le Si4M SAXS
        # est reprojeté sur la grille q du WOS puis placé sur son centre.
        wos = sources["WOS"]
        si4m = sources["Si4M"]
        shape = tuple(wos["image"].shape)
        left, top = map(float, wos["center"])
        target_dq = float(wos["dq"])
        layers = {
            "WOS": np.asarray(wos["image"], dtype=float).copy(),
            "Si4M": np.full(shape, np.nan, dtype=float),
        }
        native_layers = {name: source["image"] for name, source in sources.items()}

        si4m_image = np.asarray(si4m["image"], dtype=float)
        angle = float(self.si4m_display_angle_spin.value())
        if abs(angle) > 1e-9:
            finite = np.isfinite(si4m_image)
            rotated_values = ndimage.rotate(
                np.where(finite, si4m_image, 0.0), angle, reshape=False,
                order=0, mode="constant", cval=0.0, prefilter=False,
            )
            rotated_weights = ndimage.rotate(
                finite.astype(float), angle, reshape=False,
                order=0, mode="constant", cval=0.0, prefilter=False,
            )
            si4m_image = np.where(rotated_weights > 0.5, rotated_values, np.nan)

        q_scale = float(si4m["dq"]) / target_dq
        finite = np.isfinite(si4m_image)
        scaled_values = ndimage.zoom(
            np.where(finite, si4m_image, 0.0), q_scale,
            order=0, mode="constant", cval=0.0, prefilter=False,
        )
        scaled_weights = ndimage.zoom(
            finite.astype(float), q_scale,
            order=0, mode="constant", cval=0.0, prefilter=False,
        )
        scaled_si4m = np.where(scaled_weights > 0.5, scaled_values, np.nan)
        scaled_center = (si4m["center"][0] * q_scale, si4m["center"][1] * q_scale)

        x0 = int(round(left - scaled_center[0]))
        y0 = int(round(top - scaled_center[1]))
        src_x0 = max(0, -x0); src_y0 = max(0, -y0)
        dst_x0 = max(0, x0); dst_y0 = max(0, y0)
        width = min(scaled_si4m.shape[1] - src_x0, shape[1] - dst_x0)
        height = min(scaled_si4m.shape[0] - src_y0, shape[0] - dst_y0)
        if width > 0 and height > 0:
            layers["Si4M"][dst_y0:dst_y0 + height, dst_x0:dst_x0 + width] = \
                scaled_si4m[src_y0:src_y0 + height, src_x0:src_x0 + width]
        top_name = "WOS" if self.combo_top_selector.currentIndex() == 0 else "Si4M"
        bottom_name = "Si4M" if top_name == "WOS" else "WOS"
        combined = layers[bottom_name].copy()
        top_valid = np.isfinite(layers[top_name])
        combined[top_valid] = layers[top_name][top_valid]
        q_map = np.asarray(wos["q_map"], dtype=float)
        centers = (np.arange(shape[1]) - left) * target_dq
        rows = (np.arange(shape[0]) - top) * target_dq
        return {
            "5CB": combined, "real_image": combined, "cave_image": combined,
            "extended_image": combined, "real_q_nm": q_map, "cave_q_nm": q_map,
            "extended_q_nm": q_map, "q_nm": q_map,
            "real_x_centers_mm": centers, "real_y_centers_mm": rows,
            "extended_x_centers_mm": centers, "extended_y_centers_mm": rows,
            "center_mm": (0.0, 0.0), "center_pixel": (left, top),
            "real_center_pixel": (left, top), "extended_center_pixel": (left, top),
            "reference_angle_deg": 0.0, "cave_nan_regions": [],
            "resized_shape": shape, "combo_top": top_name,
            "combo_layers": layers,
            "combo_native_layers": native_layers,
            "source_path": paths[top_name], "distance_m": 1.0, "wavelength_m": 1e-10,
        }

    def load_test_files(self):
        try:
            self.loaded = {}
            self.zone_undo_stacks = {detector: [] for detector in DOUBLE_DETECTOR_FILES}
            for detector, config in DOUBLE_DETECTOR_FILES.items():
                detector_data = {}
                for sample in ("5CB",):
                    source_path = self.folder / config[sample]
                    if detector == self.selected_detector() and self.raw_file_edit.text().strip():
                        candidate = Path(self.raw_file_edit.text().strip()).expanduser()
                        if candidate.exists():
                            source_path = candidate
                    detector_data[sample] = self.read_first_h5_image(source_path)
                    detector_data["source_path"] = source_path
                    detector_data["source_dataset"] = inspect_h5_image_dataset(source_path)[0]
                mask_path = self.folder / config["mask"]
                if detector == self.selected_detector() and self.mask_disabled:
                    mask_path = None
                elif detector == self.selected_detector() and self.mask_file_edit.text().strip():
                    candidate = Path(self.mask_file_edit.text().strip()).expanduser()
                    if candidate.exists():
                        mask_path = candidate
                detector_data["mask"] = self.read_mask(mask_path) if mask_path is not None else None
                detector_data["mask_path"] = mask_path
                detector_data["poni_path"] = self.folder / config["poni"]
                (
                    detector_data["pixel_corners"],
                    detector_data["center_mm"],
                    detector_data["center_pixel"],
                    detector_data["q_nm"],
                    detector_data["psi_deg"],
                    detector_data["row_centers_mm"],
                    detector_data["col_centers_mm"],
                    detector_data["tilt_plane_deg"],
                    detector_data["distance_m"],
                    detector_data["wavelength_m"],
                ) = self.read_pyfai_geometry(
                    self.folder / config["poni"]
                )
                masked_image = np.asarray(detector_data["5CB"], dtype=float).copy()
                background_path = getattr(self, "background_file", None)
                if detector == self.selected_detector() and background_path is not None and Path(background_path).exists():
                    background_image = self.read_first_h5_image(Path(background_path))
                    if background_image.shape != masked_image.shape:
                        raise ValueError("Le background doit avoir la même taille que l'image brute.")
                    masked_image -= np.asarray(background_image, dtype=float)
                    detector_data["background_path"] = Path(background_path)
                    detector_data["background_subtracted"] = True
                else:
                    detector_data["background_subtracted"] = False
                detector_mask = detector_data.get("mask")
                if detector_mask is not None and detector_mask.shape == masked_image.shape:
                    masked_image[detector_mask] = np.nan
                masked_image[~np.isfinite(masked_image) | (masked_image < 0)] = np.nan
                real_package = self.prepare_panel_save_package(
                    detector,
                    "real",
                    masked_image,
                    detector_mask,
                    detector_data,
                    config,
                )
                detector_data["real_image"] = real_package["image"]
                detector_data["real_mask"] = real_package["mask"]
                detector_data["real_q_nm"] = real_package["q_nm"]
                detector_data["real_psi_deg"] = real_package["psi_deg"]
                detector_data["real_x_centers_mm"] = real_package["x_centers_mm"]
                detector_data["real_y_centers_mm"] = real_package["y_centers_mm"]
                detector_data["real_rectified_info"] = real_package["rectified_info"]
                geometric_q = self.canvas.geometric_q_grid(
                    real_package["x_centers_mm"], real_package["y_centers_mm"],
                    detector_data["center_mm"], detector_data["distance_m"],
                    detector_data["wavelength_m"],
                )
                if detector_data["real_q_nm"] is None:
                    detector_data["real_q_nm"] = geometric_q
                else:
                    real_q = np.asarray(detector_data["real_q_nm"], dtype=float).copy()
                    real_q[~np.isfinite(real_q)] = geometric_q[~np.isfinite(real_q)]
                    detector_data["real_q_nm"] = real_q
                center_x_mm, center_y_mm = detector_data["center_mm"]
                detector_data["real_center_pixel"] = (
                    float(self.canvas.nearest_sorted_index(real_package["x_centers_mm"], center_x_mm)),
                    float(self.canvas.nearest_sorted_index(real_package["y_centers_mm"], center_y_mm)),
                )
                detector_data["cave_nan_regions"] = read_cave_regions(detector_data["source_path"])
                # Start in the tilted reference frame; the panel button can
                # switch it back to the laboratory (0°) frame.
                detector_data["reference_angle_deg"] = -float(detector_data.get("tilt_plane_deg", 0.0))
                self.loaded[detector] = detector_data
            default_shape = self.loaded.get(self.selected_detector(), {}).get("resized_shape")
            if default_shape:
                self.resize_height_spin.blockSignals(True)
                self.resize_width_spin.blockSignals(True)
                self.resize_height_spin.setValue(int(default_shape[0]))
                self.resize_width_spin.setValue(int(default_shape[1]))
                self.resize_height_spin.blockSignals(False)
                self.resize_width_spin.blockSignals(False)
            self.canvas.draw_detectors(self.loaded, self.selected_detector())
            for detector_data in self.loaded.values():
                detector_data["resized_shape"] = (1000, 1000)
            self.canvas.draw_detectors(self.loaded, self.selected_detector())
            if self.selected_detector() == "Si4M":
                self.add_auto_tile_zones()
            elif self.selected_detector() == "Combo (test)":
                self.rebuild_combo()
            self.refresh_custom_zones_panel()
            self.status.setText(
                f"Loaded Si4M and WOS test files. Showing {self.selected_detector()}. Layout: real coordinates, zones, caved pattern, extended cave pattern."
            )
            self.refresh_summary()
        except Exception as exc:
            QMessageBox.critical(self, "BM02 D2AM", str(exc))
            self.status.setText(f"Load failed: {exc}")

    def read_first_h5_image(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        dataset_name, *_ = inspect_h5_image_dataset(path)
        image, _header = read_h5_frame(path, dataset_name, 0, add_matching_center=False)
        return np.asarray(image, dtype=float)

    def read_mask(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            image, *_ = read_edf_frame(path, 0)
        except ValueError as exc:
            # Some detector EDF masks use a non-standard header. Fabio knows
            # how to decode those variants while preserving the pixel array.
            try:
                import fabio
                image = fabio.open(str(path)).data
            except Exception:
                raise exc
        return np.asarray(image, dtype=float) > 0

    def push_zone_undo_state(self, detector):
        data = self.loaded.get(detector)
        if data is None:
            return
        snapshot = {
            "regions": copy.deepcopy(data.get("cave_nan_regions", [])),
            "reference_angle_deg": float(data.get("reference_angle_deg", 0.0)),
        }
        stack = self.zone_undo_stacks.setdefault(detector, [])
        if stack and stack[-1] == snapshot:
            return
        stack.append(snapshot)
        if len(stack) > 100:
            del stack[:-100]

    def undo_zone_change(self):
        detector = self.selected_detector()
        stack = self.zone_undo_stacks.get(detector, [])
        data = self.loaded.get(detector)
        if data is None or not stack:
            self.status.setText("No zone modification to undo.")
            return
        snapshot = stack.pop()
        data["cave_nan_regions"] = copy.deepcopy(snapshot["regions"])
        data["reference_angle_deg"] = float(snapshot["reference_angle_deg"])
        self.canvas.selected_region_index = None
        self.canvas.regions_changed.emit(detector)
        self.canvas.refresh_detector_preserve_view(detector)
        self.status.setText("Zone modification undone and JSON updated.")

    def auto_save_cave_regions(self, detector):
        data = self.loaded.get(detector, {})
        source_path = data.get("source_path")
        if source_path is None:
            return
        try:
            sidecar = write_cave_regions(
                source_path,
                detector,
                data.get("cave_nan_regions", []),
                data.get("reference_angle_deg", 0.0),
            )
            self.status.setText(f"Saved Cave regions: {sidecar.name}")
        except Exception as exc:
            self.status.setText(f"Unable to save Cave regions: {exc}")

    def save_panel_h5(self, detector, kind):
        if detector not in self.loaded:
            QMessageBox.warning(self, "Save H5", "Load the double-detector test files first.")
            return

        try:
            image = self.panel_image_for_save(detector, kind)
        except Exception as exc:
            QMessageBox.critical(self, "Save H5", str(exc))
            return

        default_name = f"{detector}_5CB_{kind}_geometry.h5"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {detector} {kind} H5",
            str(self.folder / default_name),
            "HDF5 files (*.h5 *.hdf5)",
        )
        if not output_path:
            return

        try:
            self.write_panel_h5(Path(output_path), detector, kind, image)
            self.status.setText(f"Saved {detector} {kind} H5: {Path(output_path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save H5 error", str(exc))
            self.status.setText(f"Save failed: {exc}")

    def save_panel(self, detector, kind, file_format):
        if file_format == "png":
            self.save_panel_png(detector, kind)
        else:
            self.save_panel_h5(detector, kind)

    def save_panel_png(self, detector, kind):
        if detector not in self.loaded:
            QMessageBox.warning(self, "Save PNG", "Load the double-detector test files first.")
            return
        data = self.loaded[detector]
        if kind == "raw":
            image = np.asarray(data.get("5CB"), dtype=float).copy()
            mask = data.get("mask")
            if mask is not None and mask.shape == image.shape:
                image[np.asarray(mask, dtype=bool)] = np.nan
        elif kind == "real":
            image = np.asarray(data.get("real_image"), dtype=float).copy()
        elif kind == "zones":
            image = np.asarray(data.get("zones_preview_image", data.get("real_image")), dtype=float).copy()
        elif kind == "extended":
            image = np.asarray(data.get("extended_image"), dtype=float).copy()
        elif kind == "resized":
            image = self.resized_image_for_data(data)
        else:
            image = np.asarray(data.get("cave_image"), dtype=float).copy()

        image[~np.isfinite(image) | (image < 0)] = np.nan
        default_name = f"{detector}_5CB_{kind}.png"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Save {detector} {kind} PNG",
            str(self.folder / default_name),
            "PNG images (*.png)",
        )
        if not output_path:
            return
        if not output_path.lower().endswith(".png"):
            output_path += ".png"

        try:
            from matplotlib import image as matplotlib_image

            with np.errstate(invalid="ignore", divide="ignore"):
                display = np.log10(image + 1.0)
            finite = display[np.isfinite(display)]
            vmin, vmax = np.nanpercentile(finite, [1, 99.5]) if finite.size else (None, None)
            matplotlib_image.imsave(
                output_path,
                display,
                cmap=self.canvas.display_cmap(),
                vmin=vmin,
                vmax=vmax,
                origin="upper",
            )
            self.status.setText(f"Saved {detector} {kind} PNG: {Path(output_path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save PNG error", str(exc))
            self.status.setText(f"Save failed: {exc}")

    def panel_image_for_save(self, detector, kind):
        data = self.loaded.get(detector, {})
        image = data.get("5CB")
        if image is None:
            raise ValueError(f"No {detector} 5CB image is loaded.")

        mask = data.get("mask")
        if kind == "real":
            real_image = data.get("real_image")
            if real_image is None:
                raise ValueError(f"No {detector} REAL image is loaded.")
            return np.asarray(real_image, dtype=float).copy()
        if kind == "zones":
            zones_image = data.get("zones_preview_image")
            if zones_image is None:
                zones_image = data.get("real_image")
            if zones_image is None:
                raise ValueError(f"No {detector} ZONES image is loaded.")
            return np.asarray(zones_image, dtype=float).copy()
        if kind == "cave":
            cave_image = data.get("cave_image")
            if cave_image is None:
                raise ValueError(f"No {detector} rectified Cave image is loaded.")
            return np.asarray(cave_image, dtype=float).copy()
        if kind == "extended":
            extended_image = data.get("extended_image")
            if extended_image is None:
                raise ValueError(f"No {detector} Extended image is loaded.")
            return np.asarray(extended_image, dtype=float).copy()
        if kind == "resized":
            return self.resized_image_for_data(data)

        output = np.asarray(image, dtype=float).copy()
        if mask is not None and mask.shape == output.shape:
            output[mask] = np.nan
        output[~np.isfinite(output)] = np.nan
        output[output < 0] = np.nan
        return output

    def resized_image_for_data(self, data):
        source = data.get("extended_image")
        center = data.get("extended_center_pixel")
        if source is None or center is None:
            raise ValueError("No EXTENDED image is available for RESIZED export.")
        rows, cols = map(int, data.get("resized_shape", (1000, 1000)))
        source = np.asarray(source, dtype=float)
        sr, sc = source.shape
        h, w = min(rows, sr), min(cols, sc)
        cx, cy = map(float, center)
        x0 = max(0, min(int(round(cx - w / 2.0)), sc - w))
        y0 = max(0, min(int(round(cy - h / 2.0)), sr - h))
        output = np.full((rows, cols), np.nan, dtype=float)
        output[(rows-h)//2:(rows-h)//2+h, (cols-w)//2:(cols-w)//2+w] = source[y0:y0+h, x0:x0+w]
        return output

    def write_panel_h5(self, output_path: Path, detector, kind, image):
        import h5py

        data = self.loaded[detector]
        config = DOUBLE_DETECTOR_FILES[detector]
        source_path = Path(data.get("source_path") or self.folder / config["5CB"])
        source_dataset = data.get("source_dataset") or inspect_h5_image_dataset(source_path)[0]
        mask = data.get("mask")
        poni_path = Path(data.get("poni_path") or self.folder / config["poni"])
        mask_path = Path(data.get("mask_path") or self.folder / config["mask"])
        poni_values = parse_simple_poni(poni_path)
        ready_poni_path, resolved_geometry = pyfai_ready_poni_path(poni_path, self.folder)
        if ready_poni_path != poni_path:
            try:
                ready_poni_path.unlink()
            except OSError:
                pass
        requested_geometry = None
        detector_config = poni_values.get("Detector_config")
        if detector_config:
            try:
                requested_geometry = json.loads(detector_config).get("filename")
            except (TypeError, ValueError):
                requested_geometry = None

        save_package = self.prepare_panel_save_package(detector, kind, image, mask, data, config)
        saved_image = save_package["image"]
        saved_mask = save_package["mask"]
        saved_q_nm = save_package["q_nm"]
        saved_psi_deg = save_package["psi_deg"]
        saved_x_centers_mm = save_package["x_centers_mm"]
        saved_y_centers_mm = save_package["y_centers_mm"]
        rectified_info = save_package["rectified_info"]

        output_path = Path(output_path)
        with h5py.File(output_path, "w") as out:
            dataset = out.create_dataset(
                "/entry_0000/instrument/detector/data",
                data=sanitize_cave_output_image(np.asarray(saved_image, dtype=float)),
                compression="gzip",
            )
            if saved_mask is not None:
                out.create_dataset(
                    "/entry_0000/instrument/detector/mask",
                    data=np.asarray(saved_mask, dtype=np.uint8),
                    compression="gzip",
                )
            self.write_optional_array(out, "/entry_0000/instrument/detector/q_nm_inverse", saved_q_nm)
            self.write_optional_array(out, "/entry_0000/instrument/detector/psi_deg", saved_psi_deg)
            self.write_optional_array(out, "/entry_0000/instrument/detector/x_center_mm", saved_x_centers_mm)
            self.write_optional_array(out, "/entry_0000/instrument/detector/y_center_mm", saved_y_centers_mm)

            self.add_panel_h5_metadata(
                out,
                dataset,
                detector,
                kind,
                source_path,
                source_dataset,
                mask_path,
                poni_path,
                poni_values,
                requested_geometry,
                resolved_geometry,
                data,
                config,
                rectified_info,
            )
            regions_json = json.dumps(data.get("cave_nan_regions", []), ensure_ascii=False)
            set_h5_attr(out.attrs, "cave_nan_regions_json", regions_json)
            set_h5_attr(dataset.attrs, "cave_nan_regions_json", regions_json)
            set_h5_attr(out.attrs, "cave_reference_angle_deg", float(data.get("reference_angle_deg", 0.0)))
            set_h5_attr(dataset.attrs, "cave_reference_angle_deg", float(data.get("reference_angle_deg", 0.0)))

        write_cave_regions(
            output_path,
            detector,
            data.get("cave_nan_regions", []),
            data.get("reference_angle_deg", 0.0),
        )

    def prepare_panel_save_package(self, detector, kind, image, mask, data, config):
        base_image = np.asarray(image, dtype=float)
        base_mask = ~np.isfinite(base_image)
        q_nm = data.get("q_nm")
        psi_deg = data.get("psi_deg")

        if kind == "extended" and data.get("extended_image") is not None:
            extended_x = np.asarray(data.get("extended_x_centers_mm"), dtype=float)
            extended_y = np.asarray(data.get("extended_y_centers_mm"), dtype=float)
            pixel_mm = float(np.nanmedian(np.abs(np.r_[np.diff(extended_x), np.diff(extended_y)])))
            rectified_info = {
                "rectified": True,
                "extended": True,
                "output_shape": tuple(base_image.shape),
                "source_shape": tuple(np.shape(data.get("real_image"))),
                "pixel_size_mm": pixel_mm,
                "x_min_mm": float(extended_x[0] - pixel_mm / 2.0),
                "x_max_mm": float(extended_x[-1] + pixel_mm / 2.0),
                "y_min_mm": float(extended_y[0] - pixel_mm / 2.0),
                "y_max_mm": float(extended_y[-1] + pixel_mm / 2.0),
            }
            return {
                "image": base_image,
                "mask": base_mask,
                "q_nm": data.get("extended_q_nm"),
                "psi_deg": None,
                "x_centers_mm": data.get("extended_x_centers_mm"),
                "y_centers_mm": data.get("extended_y_centers_mm"),
                "rectified_info": rectified_info,
            }

        if kind == "resized":
            resized = self.resized_image_for_data(data)
            # Resizing changes the array dimensions only; detector pixel size
            # remains the calibrated value from the selected geometry.
            side = float(config.get("pixel_size_m") or 0.0) * 1000.0
            if side <= 0:
                side = float(np.nanmedian(np.abs(np.diff(np.asarray(data.get("extended_x_centers_mm"), dtype=float)))) or 1.0)
            resized_x = (np.arange(resized.shape[1]) - (resized.shape[1] - 1) / 2.0) * side
            resized_y = (np.arange(resized.shape[0]) - (resized.shape[0] - 1) / 2.0) * side
            resized_q = self.canvas.geometric_q_grid(
                resized_x, resized_y, (0.0, 0.0),
                data.get("distance_m", 1.0), data.get("wavelength_m", 1e-10)
            )
            return {
                "image": resized,
                "mask": ~np.isfinite(resized),
                "q_nm": resized_q,
                "psi_deg": None,
                "x_centers_mm": resized_x,
                "y_centers_mm": resized_y,
                "rectified_info": {
                    "rectified": True,
                    "resized": True,
                    "extended": True,
                    "pixel_size_mm": side,
                    "x_min_mm": -resized.shape[1] * side / 2.0,
                    "x_max_mm": resized.shape[1] * side / 2.0,
                    "y_min_mm": -resized.shape[0] * side / 2.0,
                    "y_max_mm": resized.shape[0] * side / 2.0,
                    "output_shape": tuple(resized.shape),
                    "source_shape": tuple(np.shape(data.get("extended_image"))),
                },
            }

        if kind == "cave" and data.get("real_image") is not None and base_image.shape == data["real_image"].shape:
            return {
                "image": base_image,
                "mask": base_mask,
                "q_nm": data.get("real_q_nm"),
                "psi_deg": data.get("real_psi_deg"),
                "x_centers_mm": data.get("real_x_centers_mm"),
                "y_centers_mm": data.get("real_y_centers_mm"),
                "rectified_info": data.get("real_rectified_info"),
            }

        if kind == "zones" and data.get("real_image") is not None and base_image.shape == data["real_image"].shape:
            return {
                "image": base_image,
                "mask": base_mask,
                "q_nm": data.get("real_q_nm"),
                "psi_deg": data.get("real_psi_deg"),
                "x_centers_mm": data.get("real_x_centers_mm"),
                "y_centers_mm": data.get("real_y_centers_mm"),
                "rectified_info": data.get("real_rectified_info"),
            }

        if kind == "raw":
            return {
                "image": base_image,
                "mask": base_mask,
                "q_nm": q_nm,
                "psi_deg": psi_deg,
                "x_centers_mm": data.get("col_centers_mm"),
                "y_centers_mm": data.get("row_centers_mm"),
                "rectified_info": {
                    "rectified": False,
                    "output_shape": tuple(base_image.shape),
                    "source_shape": tuple(base_image.shape),
                },
            }

        corners = data.get("pixel_corners")
        if corners is None:
            raise ValueError(f"No {detector} pixel-corner geometry is loaded.")

        pixel_mm = float(config.get("pixel_size_m") or 0.0) * 1000.0
        if not np.isfinite(pixel_mm) or pixel_mm <= 0:
            x_centers = np.asarray(data.get("col_centers_mm"), dtype=float)
            y_centers = np.asarray(data.get("row_centers_mm"), dtype=float)
            pixel_mm = float(np.nanmedian(np.r_[np.diff(x_centers), np.diff(y_centers)]))

        x_corners_mm = np.asarray(corners[:, :, :, 2], dtype=float) * 1000.0
        y_corners_mm = np.asarray(corners[:, :, :, 1], dtype=float) * 1000.0
        rectified_image, rectified_info = self.rectify_to_regular_detector_grid(
            base_image,
            x_corners_mm,
            y_corners_mm,
            pixel_mm,
        )
        rectified_mask = ~np.isfinite(rectified_image)
        rectified_q = (
            self.rectify_to_regular_detector_grid(q_nm, x_corners_mm, y_corners_mm, pixel_mm)[0]
            if q_nm is not None else None
        )
        rectified_psi = (
            self.rectify_to_regular_detector_grid(psi_deg, x_corners_mm, y_corners_mm, pixel_mm)[0]
            if psi_deg is not None else None
        )
        rectified_info["rectified"] = True
        rectified_info["pixel_size_mm"] = pixel_mm
        rectified_info["source_shape"] = tuple(base_image.shape)
        rectified_info["output_shape"] = tuple(rectified_image.shape)
        return {
            "image": rectified_image,
            "mask": rectified_mask,
            "q_nm": rectified_q,
            "psi_deg": rectified_psi,
            "x_centers_mm": rectified_info["x_centers_mm"],
            "y_centers_mm": rectified_info["y_centers_mm"],
            "rectified_info": rectified_info,
        }

    def rectify_to_regular_detector_grid(self, values, x_corners_mm, y_corners_mm, pixel_mm):
        values = np.asarray(values, dtype=float)
        x_corners_mm = np.asarray(x_corners_mm, dtype=float)
        y_corners_mm = np.asarray(y_corners_mm, dtype=float)
        x_min = float(np.nanmin(x_corners_mm))
        x_max = float(np.nanmax(x_corners_mm))
        y_min = float(np.nanmin(y_corners_mm))
        y_max = float(np.nanmax(y_corners_mm))
        nx_out = self.pixel_count_from_span(x_max - x_min, pixel_mm)
        ny_out = self.pixel_count_from_span(y_max - y_min, pixel_mm)
        output = np.full((ny_out, nx_out), np.nan, dtype=float)

        x_centers = np.nanmean(x_corners_mm, axis=2)
        y_centers = np.nanmean(y_corners_mm, axis=2)
        rows = np.zeros(values.shape, dtype=int)
        cols = np.zeros(values.shape, dtype=int)
        finite_centers = np.isfinite(x_centers) & np.isfinite(y_centers)
        cols[finite_centers] = np.floor((x_centers[finite_centers] - x_min) / pixel_mm + 0.5).astype(int)
        rows[finite_centers] = np.floor((y_centers[finite_centers] - y_min) / pixel_mm + 0.5).astype(int)
        valid = (
            np.isfinite(values)
            & finite_centers
            & (rows >= 0)
            & (rows < ny_out)
            & (cols >= 0)
            & (cols < nx_out)
        )
        output[rows[valid], cols[valid]] = values[valid]

        x_centers_mm = x_min + (np.arange(nx_out, dtype=float) + 0.5) * pixel_mm
        y_centers_mm = y_min + (np.arange(ny_out, dtype=float) + 0.5) * pixel_mm
        info = {
            "x_min_mm": x_min,
            "x_max_mm": x_max,
            "y_min_mm": y_min,
            "y_max_mm": y_max,
            "x_centers_mm": x_centers_mm,
            "y_centers_mm": y_centers_mm,
        }
        return output, info

    def pixel_count_from_span(self, span_mm, pixel_mm):
        raw_count = float(span_mm) / float(pixel_mm)
        rounded = round(raw_count)
        if abs(raw_count - rounded) < 1e-3:
            return max(1, int(rounded))
        return max(1, int(np.ceil(raw_count)))

    def write_optional_array(self, h5_file, name, values):
        if values is None:
            return
        h5_file.create_dataset(name, data=np.asarray(values, dtype=np.float32), compression="gzip")

    def add_panel_h5_metadata(
        self,
        out,
        dataset,
        detector,
        kind,
        source_path,
        source_dataset,
        mask_path,
        poni_path,
        poni_values,
        requested_geometry,
        resolved_geometry,
        data,
        config,
        rectified_info,
    ):
        import h5py

        root_attrs = out.attrs
        dataset_attrs = dataset.attrs
        attrs_targets = (root_attrs, dataset_attrs)
        processing = {
            "raw": "masked raw detector image",
            "real": "masked detector image rectified to a regular detector grid from pyFAI real coordinates",
            "zones": "real-coordinate image with configured Cave zones applied",
            "cave": "axial-symmetry cave image rectified to a regular detector grid from pyFAI real coordinates",
            "extended": "Cave image extended by centered axial symmetry",
            "resized": "centered crop of the Extended Cave image to the selected pixel dimensions",
        }.get(kind, kind)

        for attrs in attrs_targets:
            set_h5_attr(attrs, "lrphoton_project", "double_detector")
            set_h5_attr(attrs, "detector", detector)
            set_h5_attr(attrs, "panel", kind)
            set_h5_attr(attrs, "sample", "5CB")
            set_h5_attr(attrs, "processing", processing)
            set_h5_attr(attrs, "source_file", source_path.name)
            set_h5_attr(attrs, "source_path", str(source_path))
            set_h5_attr(attrs, "source_dataset", source_dataset)
            set_h5_attr(attrs, "source_frame", 0)
            set_h5_attr(attrs, "mask_file", mask_path.name)
            set_h5_attr(attrs, "mask_path", str(mask_path))
            set_h5_attr(attrs, "mask_applied", True)
            set_h5_attr(attrs, "background_subtracted", bool(data.get("background_subtracted", False)))
            if data.get("background_path") is not None:
                set_h5_attr(attrs, "background_file", Path(data["background_path"]).name)
                set_h5_attr(attrs, "background_path", str(data["background_path"]))
            set_h5_attr(attrs, "poni_file", poni_path.name)
            set_h5_attr(attrs, "poni_path", str(poni_path))
            set_h5_attr(attrs, "poni_text", read_text_file(poni_path))
            set_h5_attr(attrs, "pyfai_geometry_requested", requested_geometry or "")
            set_h5_attr(attrs, "pyfai_geometry_resolved", str(resolved_geometry or ""))
            set_h5_attr(attrs, "geometry_note", config.get("geometry_note", ""))
            set_h5_attr(attrs, "pixel_size_m", config.get("pixel_size_m", ""))
            set_h5_attr(attrs, "rectified_regular_detector_grid", bool(rectified_info.get("rectified", False)))
            set_h5_attr(attrs, "source_image_shape", str(tuple(rectified_info.get("source_shape", ()))))
            set_h5_attr(attrs, "rectified_image_shape", str(tuple(rectified_info.get("output_shape", ()))))
            if rectified_info.get("rectified", False):
                pixel_size_mm = float(rectified_info["pixel_size_mm"])
                pixel_size_m = pixel_size_mm * 1e-3
                set_h5_attr(attrs, "rectified_pixel_size_mm", pixel_size_mm)
                set_h5_attr(attrs, "rectified_x_min_mm", float(rectified_info["x_min_mm"]))
                set_h5_attr(attrs, "rectified_y_min_mm", float(rectified_info["y_min_mm"]))
                set_h5_attr(attrs, "rectified_x_max_mm", float(rectified_info["x_max_mm"]))
                set_h5_attr(attrs, "rectified_y_max_mm", float(rectified_info["y_max_mm"]))
                set_h5_attr(attrs, "PSize_1", pixel_size_m)
                set_h5_attr(attrs, "PSize_2", pixel_size_m)
                set_h5_attr(attrs, "PixelSizeX_mm", pixel_size_mm)
                set_h5_attr(attrs, "PixelSizeY_mm", pixel_size_mm)
            set_h5_attr(attrs, "q_dataset", "/entry_0000/instrument/detector/q_nm_inverse")
            set_h5_attr(attrs, "q_unit", "nm^-1")
            set_h5_attr(attrs, "psi_dataset", "/entry_0000/instrument/detector/psi_deg")
            set_h5_attr(attrs, "psi_unit", "deg")
            set_h5_attr(attrs, "x_center_mm_dataset", "/entry_0000/instrument/detector/x_center_mm")
            set_h5_attr(attrs, "y_center_mm_dataset", "/entry_0000/instrument/detector/y_center_mm")

            center_mm = data.get("center_mm")
            if kind == "resized" and rectified_info.get("output_shape"):
                out_rows, out_cols = map(int, rectified_info["output_shape"])
                # The resized image is explicitly recentered: FIT2D-style
                # coordinates are the geometric midpoint of the new array.
                center_x_pixel = (out_cols - 1) / 2.0
                center_y_pixel = (out_rows - 1) / 2.0
                set_h5_attr(attrs, "Center_1", center_x_pixel)
                set_h5_attr(attrs, "Center_2", center_y_pixel)
                set_h5_attr(attrs, "rectified_center_x_pixel", center_x_pixel)
                set_h5_attr(attrs, "rectified_center_y_pixel", center_y_pixel)
                center_mm = (0.0, 0.0)
            if center_mm is not None:
                set_h5_attr(attrs, "center_x_mm", float(center_mm[0]))
                set_h5_attr(attrs, "center_y_mm", float(center_mm[1]))
                if rectified_info.get("rectified", False):
                    pixel_size_mm = float(rectified_info["pixel_size_mm"])
                    center_x_pixel = (float(center_mm[0]) - float(rectified_info["x_min_mm"])) / pixel_size_mm + 0.5
                    center_y_pixel = (float(center_mm[1]) - float(rectified_info["y_min_mm"])) / pixel_size_mm + 0.5
                    set_h5_attr(attrs, "rectified_center_x_pixel", center_x_pixel)
                    set_h5_attr(attrs, "rectified_center_y_pixel", center_y_pixel)
                    set_h5_attr(attrs, "Center_1", center_x_pixel)
                    set_h5_attr(attrs, "Center_2", center_y_pixel)
            if kind == "resized" and rectified_info.get("output_shape"):
                out_rows, out_cols = map(int, rectified_info["output_shape"])
                set_h5_attr(attrs, "Center_1", (out_cols - 1) / 2.0)
                set_h5_attr(attrs, "Center_2", (out_rows - 1) / 2.0)
            center_pixel = data.get("center_pixel")
            if center_pixel is not None:
                set_h5_attr(attrs, "center_x_pixel", float(center_pixel[0]))
                set_h5_attr(attrs, "center_y_pixel", float(center_pixel[1]))

            for key, value in poni_values.items():
                set_h5_attr(attrs, f"poni_{key}", value)

        try:
            with h5py.File(source_path, "r") as source:
                copy_h5_attrs(source.attrs, root_attrs, prefix="source_root_", overwrite=False)
                source_attrs_json = json.dumps(collect_h5_attrs_json(source), ensure_ascii=False)
                set_h5_attr(root_attrs, "source_h5_attrs_json", source_attrs_json)
                set_h5_attr(dataset_attrs, "source_h5_attrs_json", source_attrs_json)
                if source_dataset in source:
                    copy_h5_attrs(source[source_dataset].attrs, dataset_attrs, prefix="source_dataset_", overwrite=False)
        except Exception:
            pass

        if resolved_geometry is not None and Path(resolved_geometry).exists():
            try:
                with h5py.File(resolved_geometry, "r") as geometry_h5:
                    geometry_attrs_json = json.dumps(collect_h5_attrs_json(geometry_h5), ensure_ascii=False)
                    set_h5_attr(root_attrs, "pyfai_geometry_h5_attrs_json", geometry_attrs_json)
                    set_h5_attr(dataset_attrs, "pyfai_geometry_h5_attrs_json", geometry_attrs_json)
            except Exception:
                pass

    def read_pyfai_geometry(self, poni_path: Path):
        import pyFAI

        if not poni_path.exists():
            raise FileNotFoundError(poni_path)
        ready_path, _geometry_path = pyfai_ready_poni_path(poni_path, self.folder)
        try:
            integrator = pyFAI.load(str(ready_path))
            corners = np.asarray(integrator.detector.get_pixel_corners(), dtype=float)
            fit2d = integrator.getFit2D()
            center_pixel = (float(fit2d["centerX"]), float(fit2d["centerY"]))
            center_mm = (
                center_pixel[0] * float(integrator.detector.pixel2) * 1000.0,
                center_pixel[1] * float(integrator.detector.pixel1) * 1000.0,
            )
            shape = integrator.detector.shape
            q_nm = np.asarray(integrator.center_array(shape, unit="q_nm^-1"), dtype=float)
            psi_deg = np.rad2deg(np.asarray(integrator.center_array(shape, unit="chi_rad"), dtype=float))
            row_centers_mm = np.nanmedian(np.mean(corners[:, :, :, 1], axis=2), axis=1) * 1000.0
            col_centers_mm = np.nanmedian(np.mean(corners[:, :, :, 2], axis=2), axis=0) * 1000.0
            tilt_plane_deg = float(fit2d["tiltPlanRotation"])
            return (
                corners, center_mm, center_pixel, q_nm, psi_deg,
                row_centers_mm, col_centers_mm, tilt_plane_deg,
                float(integrator.dist), float(integrator.wavelength),
            )
        finally:
            if ready_path != poni_path:
                try:
                    ready_path.unlink()
                except OSError:
                    pass
