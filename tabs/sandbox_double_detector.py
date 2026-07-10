from pathlib import Path
import json
import tempfile

import numpy as np

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
        if distance is not None:
            lines.append(f"Distance: {distance:.6g} m")
        if poni1 is not None and poni2 is not None:
            lines.append(f"PONI: y={poni1:.6g} m, x={poni2:.6g} m")
        if wavelength is not None:
            lines.append(f"Wavelength: {wavelength:.6g} m")

    return "\n".join(lines)


class DoubleDetectorCanvas(FigureCanvas):
    save_requested = Signal(str, str)

    def __init__(self):
        self.figure = Figure(figsize=(9, 7), tight_layout=True)
        self.axes_grid = self.figure.subplots(2, 2)
        self.axes = np.ravel(self.axes_grid)
        self.loaded = {}
        self.selected_detector = "WOS"
        self.coordinate_labels = {}
        self.save_icons = {}
        self._syncing_limits = False
        self._pan_detector = None
        self._pan_last_pos = None
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.mpl_connect("motion_notify_event", self.on_motion)
        self.mpl_connect("scroll_event", self.on_scroll)
        self.mpl_connect("button_press_event", self.on_button_press)
        self.mpl_connect("button_release_event", self.on_button_release)
        self.mpl_connect("motion_notify_event", self.on_pan_motion)
        self.mpl_connect("figure_leave_event", self.clear_coordinate_labels)
        self.draw_empty()

    def draw_empty(self):
        self.save_icons = {}
        for ax in np.ravel(self.axes):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_visible(True)
        detector = self.selected_detector
        self.set_axis_title_with_save(self.axes[0], f"{detector} 5CB raw", detector, "raw")
        self.set_axis_title_with_save(self.axes[1], f"{detector} real coordinates", detector, "real")
        self.set_axis_title_with_save(self.axes[2], f"{detector} axial symmetry cave", detector, "cave")
        self.axes[3].set_visible(False)
        self.draw_idle()

    def draw_detectors(self, loaded, selected_detector=None):
        self.loaded = loaded
        if selected_detector is not None:
            self.selected_detector = selected_detector
        self.save_icons = {}
        for ax in self.axes:
            ax.set_visible(True)
        detector = self.selected_detector
        sample = "5CB"
        self.draw_raw_panel(detector, sample)
        self.draw_real_panel(detector, sample)
        self.draw_cave_panel(detector, sample)
        self.axes[3].clear()
        self.axes[3].set_visible(False)
        self.draw_idle()

    def draw_raw_panel(self, detector, sample):
        ax = self.detector_axis(detector, "raw")
        ax.clear()
        image = self.loaded.get(detector, {}).get(sample)
        mask = self.loaded.get(detector, {}).get("mask")
        self.set_axis_title_with_save(ax, f"{detector} {sample} raw", detector, "raw")
        ax.set_xticks([])
        ax.set_yticks([])
        if image is None:
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            return
        display = np.asarray(image, dtype=float)
        if mask is not None and mask.shape == image.shape:
            display = display.copy()
            display[mask] = np.nan
        display = np.where(np.isfinite(display) & (display >= 0), display, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            display = np.log10(display + 1.0)
        finite = display[np.isfinite(display)]
        if finite.size:
            vmin, vmax = np.nanpercentile(finite, [1, 99.5])
        else:
            vmin, vmax = None, None
        ax.imshow(display, cmap=self.display_cmap(), origin="upper", vmin=vmin, vmax=vmax)
        ax.set_facecolor("white")

    def draw_real_panel(self, detector, sample):
        ax = self.detector_axis(detector, "real")
        ax.clear()
        data = self.loaded.get(detector, {})
        image = data.get(sample)
        mask = data.get("mask")
        corners = data.get("pixel_corners")
        center_mm = data.get("center_mm")
        self.set_axis_title_with_save(ax, f"{detector} {sample} real coordinates", detector, "real")
        if image is None or corners is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "missing geometry", ha="center", va="center", transform=ax.transAxes)
            return
        self.draw_true_coordinate_image(ax, image, mask, corners, center_mm)

    def draw_cave_panel(self, detector, sample):
        ax = self.detector_axis(detector, "cave")
        ax.clear()
        data = self.loaded.get(detector, {})
        image = data.get(sample)
        mask = data.get("mask")
        corners = data.get("pixel_corners")
        center_pixel = data.get("center_pixel")
        center_mm = data.get("center_mm")
        self.set_axis_title_with_save(ax, f"{detector} {sample} axial symmetry cave", detector, "cave")
        if image is None or corners is None or center_pixel is None:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.5, 0.5, "missing geometry", ha="center", va="center", transform=ax.transAxes)
            return
        filled = self.axial_symmetry_cave(image, mask, center_pixel)
        self.draw_true_coordinate_image(ax, filled, None, corners, center_mm)

    def set_axis_title_with_save(self, ax, title, detector, kind):
        ax.set_title(title, pad=10)
        icon = ax.text(
            0.99,
            1.045,
            "💾",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            clip_on=False,
            zorder=20,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "#d1d5db",
                "linewidth": 0.8,
                "alpha": 0.95,
            },
        )
        self.save_icons[icon] = (detector, kind)

    def detector_axis(self, detector, kind):
        _detector = detector
        col = {"raw": 0, "real": 1, "cave": 2}[kind]
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
        cmap = colormaps["turbo"].copy()
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

    def draw_center_axes(self, ax, center_mm):
        if center_mm is None:
            return
        center_x_mm, center_y_mm = center_mm
        if center_x_mm is None or center_y_mm is None:
            return
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        ax.plot(
            [x_min, x_max],
            [center_y_mm, center_y_mm],
            color="red",
            linewidth=1.0,
            alpha=0.9,
            zorder=10,
        )
        ax.plot(
            [center_x_mm, center_x_mm],
            [y_min, y_max],
            color="red",
            linewidth=1.0,
            alpha=0.9,
            zorder=10,
        )
        ax.plot(
            center_x_mm,
            center_y_mm,
            marker="+",
            color="red",
            markersize=11,
            markeredgewidth=1.8,
            linestyle="None",
            zorder=11,
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

        center_col, center_row = center_pixel
        fillable = invalid_pixels & self.thin_mask_pixels(invalid_pixels)
        missing_rows, missing_cols = np.where(fillable)
        ny, nx = source.shape
        if missing_rows.size == 0:
            return filled

        vertical_cols = np.rint(2.0 * float(center_col) - missing_cols).astype(int)
        horizontal_rows = np.rint(2.0 * float(center_row) - missing_rows).astype(int)

        candidates = []
        valid_vertical = (vertical_cols >= 0) & (vertical_cols < nx)
        vertical_values = np.full(missing_rows.shape, np.nan, dtype=float)
        vertical_values[valid_vertical] = source[missing_rows[valid_vertical], vertical_cols[valid_vertical]]
        candidates.append(vertical_values)

        valid_horizontal = (horizontal_rows >= 0) & (horizontal_rows < ny)
        horizontal_values = np.full(missing_rows.shape, np.nan, dtype=float)
        horizontal_values[valid_horizontal] = source[horizontal_rows[valid_horizontal], missing_cols[valid_horizontal]]
        candidates.append(horizontal_values)

        valid_both = valid_vertical & valid_horizontal
        both_values = np.full(missing_rows.shape, np.nan, dtype=float)
        both_values[valid_both] = source[horizontal_rows[valid_both], vertical_cols[valid_both]]
        candidates.append(both_values)

        values = np.vstack(candidates)
        valid_counts = np.sum(np.isfinite(values), axis=0)
        replacement = np.full(missing_rows.shape, np.nan, dtype=float)
        replacement[valid_counts > 0] = (
            np.nansum(values[:, valid_counts > 0], axis=0)
            / valid_counts[valid_counts > 0]
        )
        local_estimate = self.local_nan_estimate(source)
        if local_estimate is not None:
            local_values = local_estimate[missing_rows, missing_cols]
            has_local = np.isfinite(local_values) & np.isfinite(replacement)
            replacement[has_local] = (0.85 * local_values[has_local]) + (0.15 * replacement[has_local])
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

    def on_motion(self, event):
        detector = self.detector_for_real_axis(event.inaxes)
        if detector is None or event.xdata is None or event.ydata is None:
            return

        data = self.loaded.get(detector, {})
        label = self.coordinate_labels.get(detector)
        if label is None:
            return

        row_centers = data.get("row_centers_mm")
        col_centers = data.get("col_centers_mm")
        image = data.get("5CB")
        q_map = data.get("q_nm")
        psi_map = data.get("psi_deg")
        mask = data.get("mask")
        if row_centers is None or col_centers is None or image is None:
            return

        row = self.nearest_sorted_index(row_centers, event.ydata)
        col = self.nearest_sorted_index(col_centers, event.xdata)
        if row is None or col is None or not (0 <= row < image.shape[0] and 0 <= col < image.shape[1]):
            label.setText(f"{detector}: x = - | y = - | q = - | I = - | psi = -")
            return

        value = float(image[row, col])
        masked = mask is not None and mask.shape == image.shape and bool(mask[row, col])
        if masked:
            intensity_text = "I = NaN"
        elif np.isfinite(value):
            intensity_text = f"I = {value:.6g}"
        else:
            intensity_text = "I = NaN"

        q_text = "q = -"
        if (
            not masked
            and np.isfinite(value)
            and value >= 0
            and q_map is not None
            and q_map.shape == image.shape
            and np.isfinite(q_map[row, col])
        ):
            q_text = f"q = {float(q_map[row, col]):.6g} nm⁻¹"

        psi_text = "psi = -"
        if psi_map is not None and psi_map.shape == image.shape and np.isfinite(psi_map[row, col]):
            psi_text = f"psi = {float(psi_map[row, col]):.3f}°"

        label.setText(f"{detector}: x = {col + 1} | y = {row + 1} | {q_text} | {intensity_text} | {psi_text}")

    def detector_for_real_axis(self, axis):
        if axis is self.axes[1]:
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
        detector = self.detector_for_axis(axis)
        if detector is None:
            return False

        if gesture_type == Qt.NativeGestureType.SmartZoomNativeGesture:
            bounds = self.default_detector_pixel_bounds(detector)
            if bounds is not None:
                self.apply_detector_pixel_view(detector, bounds)
                event.accept()
                return True
            return False

        if gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
            center = self.pixel_point_from_display(detector, axis, display_x, display_y)
            if center is None:
                return False
            value = float(event.value())
            if not np.isfinite(value) or abs(value) < 1e-9:
                return False
            factor = float(np.exp(-value))
            self.zoom_detector(detector, center, factor)
            event.accept()
            return True

        delta = event.delta()
        self.pan_detector_by_display_delta(detector, axis, float(delta.x()), float(delta.y()))
        event.accept()
        return True

    def wheelEvent(self, event):
        display_x, display_y = self.display_coords_from_qt_position(event.position())
        axis = self.axis_at_display_point(display_x, display_y)
        detector = self.detector_for_axis(axis)
        if detector is None:
            super().wheelEvent(event)
            return

        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        dx = float(pixel_delta.x())
        dy = float(pixel_delta.y())
        if dx == 0.0 and dy == 0.0:
            dx = float(angle_delta.x()) / 8.0
            dy = float(angle_delta.y()) / 8.0

        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and dy != 0.0:
            center = self.pixel_point_from_display(detector, axis, display_x, display_y)
            if center is not None:
                factor = 1.0 / 1.15 if dy > 0 else 1.15
                self.zoom_detector(detector, center, factor)
                event.accept()
                return

        self.pan_detector_by_display_delta(detector, axis, dx, dy)
        event.accept()

    def on_scroll(self, event):
        detector = self.detector_for_axis(event.inaxes)
        if detector is None or event.x is None or event.y is None:
            return
        center = self.pixel_point_from_display(detector, event.inaxes, event.x, event.y)
        if center is None:
            return
        factor = 1.0 / 1.25 if event.step > 0 else 1.25
        self.zoom_detector(detector, center, factor)

    def on_button_press(self, event):
        if self.handle_save_icon_click(event):
            return
        if event.button != 1:
            return
        detector = self.detector_for_axis(event.inaxes)
        if detector is None or event.x is None or event.y is None:
            return
        self._pan_detector = detector
        self._pan_last_pos = (float(event.x), float(event.y), event.inaxes)

    def handle_save_icon_click(self, event):
        if event.x is None or event.y is None:
            return False
        for icon, target in list(self.save_icons.items()):
            contains, _details = icon.contains(event)
            if contains:
                detector, kind = target
                self.save_requested.emit(detector, kind)
                return True
        return False

    def on_button_release(self, _event):
        self._pan_detector = None
        self._pan_last_pos = None

    def on_pan_motion(self, event):
        if self._pan_detector is None or self._pan_last_pos is None:
            return
        if event.x is None or event.y is None:
            return
        last_x, last_y, axis = self._pan_last_pos
        dx = float(event.x) - last_x
        dy = float(event.y) - last_y
        self.pan_detector_by_display_delta(self._pan_detector, axis, dx, dy)
        self._pan_last_pos = (float(event.x), float(event.y), axis)

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
        if kind in {"real", "cave"}:
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
            raw_axis = self.detector_axis(detector, "raw")
            raw_axis.set_xlim(col_min, col_max)
            raw_axis.set_ylim(row_max, row_min)

            for kind in ("real", "cave"):
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
        if kind in {"real", "cave"}:
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
            return "cave"
        return None


class DoubleDetectorProject(QWidget):
    folder_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.folder = DOUBLE_DETECTOR_DEFAULT_FOLDER
        self.loaded = {}
        self.build_ui()
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
        self.detector_combo.addItems(["Si4M", "WOS"])
        self.detector_combo.setCurrentText("WOS")
        self.detector_combo.currentTextChanged.connect(self.set_selected_detector)
        load_button = QPushButton("Load test files")
        load_button.clicked.connect(self.load_test_files)

        side_layout.addWidget(QLabel("Test folder"))
        side_layout.addWidget(self.folder_edit)
        side_layout.addWidget(browse_button)
        side_layout.addWidget(QLabel("Detector"))
        side_layout.addWidget(self.detector_combo)
        side_layout.addWidget(load_button)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(280)
        side_layout.addWidget(self.summary, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #4b5563;")
        side_layout.addWidget(self.status)

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        self.canvas = DoubleDetectorCanvas()
        self.canvas.save_requested.connect(self.save_panel_h5)
        preview_layout.addWidget(self.canvas, 1)

        coordinate_layout = QHBoxLayout()
        coordinate_layout.setContentsMargins(0, 0, 0, 0)
        coordinate_layout.setSpacing(8)
        self.si4m_coordinate_label = self.make_coordinate_label("Si4M")
        self.wos_coordinate_label = self.make_coordinate_label("WOS")
        coordinate_layout.addWidget(self.si4m_coordinate_label)
        coordinate_layout.addWidget(self.wos_coordinate_label)
        preview_layout.addLayout(coordinate_layout)
        self.canvas.set_coordinate_labels({
            "Si4M": self.si4m_coordinate_label,
            "WOS": self.wos_coordinate_label,
        })
        self.update_coordinate_label_visibility()

        layout.addWidget(side, 0)
        layout.addLayout(preview_layout, 1)

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
        if detector not in DOUBLE_DETECTOR_FILES:
            return
        self.canvas.selected_detector = detector
        self.update_coordinate_label_visibility()
        self.canvas.clear_coordinate_labels()
        if self.loaded:
            self.canvas.draw_detectors(self.loaded, detector)
            self.status.setText(f"Showing {detector}. Layout: raw + real coordinates, then axial-symmetry cave.")

    def update_coordinate_label_visibility(self):
        detector = self.selected_detector()
        if hasattr(self, "si4m_coordinate_label"):
            self.si4m_coordinate_label.setVisible(detector == "Si4M")
        if hasattr(self, "wos_coordinate_label"):
            self.wos_coordinate_label.setVisible(detector == "WOS")

    def set_folder(self, folder):
        folder = Path(folder).expanduser()
        self.folder = folder
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

    def load_test_files(self):
        try:
            self.loaded = {}
            for detector, config in DOUBLE_DETECTOR_FILES.items():
                detector_data = {}
                for sample in ("5CB",):
                    source_path = self.folder / config[sample]
                    detector_data[sample] = self.read_first_h5_image(source_path)
                    detector_data["source_path"] = source_path
                    detector_data["source_dataset"] = inspect_h5_image_dataset(source_path)[0]
                detector_data["mask"] = self.read_mask(self.folder / config["mask"])
                detector_data["mask_path"] = self.folder / config["mask"]
                detector_data["poni_path"] = self.folder / config["poni"]
                (
                    detector_data["pixel_corners"],
                    detector_data["center_mm"],
                    detector_data["center_pixel"],
                    detector_data["q_nm"],
                    detector_data["psi_deg"],
                    detector_data["row_centers_mm"],
                    detector_data["col_centers_mm"],
                ) = self.read_pyfai_geometry(
                    self.folder / config["poni"]
                )
                self.loaded[detector] = detector_data
            self.canvas.draw_detectors(self.loaded, self.selected_detector())
            self.status.setText(
                f"Loaded Si4M and WOS test files. Showing {self.selected_detector()}. Layout: raw + real coordinates, then axial-symmetry cave."
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
        image, *_ = read_edf_frame(path, 0)
        return np.asarray(image, dtype=float) > 0

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

    def panel_image_for_save(self, detector, kind):
        data = self.loaded.get(detector, {})
        image = data.get("5CB")
        if image is None:
            raise ValueError(f"No {detector} 5CB image is loaded.")

        mask = data.get("mask")
        if kind == "cave":
            center_pixel = data.get("center_pixel")
            if center_pixel is None:
                raise ValueError(f"No {detector} geometry center is loaded.")
            return self.canvas.axial_symmetry_cave(image, mask, center_pixel)

        output = np.asarray(image, dtype=float).copy()
        if mask is not None and mask.shape == output.shape:
            output[mask] = np.nan
        output[~np.isfinite(output)] = np.nan
        output[output < 0] = np.nan
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

    def prepare_panel_save_package(self, detector, kind, image, mask, data, config):
        base_image = np.asarray(image, dtype=float)
        base_mask = ~np.isfinite(base_image)
        q_nm = data.get("q_nm")
        psi_deg = data.get("psi_deg")

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
            "cave": "axial-symmetry cave image rectified to a regular detector grid from pyFAI real coordinates",
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
            center_mm = (float(integrator.poni2) * 1000.0, float(integrator.poni1) * 1000.0)
            center_pixel = (
                float(integrator.poni2) / float(integrator.detector.pixel2),
                float(integrator.poni1) / float(integrator.detector.pixel1),
            )
            shape = integrator.detector.shape
            q_nm = np.asarray(integrator.center_array(shape, unit="q_nm^-1"), dtype=float)
            psi_deg = np.rad2deg(np.asarray(integrator.center_array(shape, unit="chi_rad"), dtype=float))
            row_centers_mm = np.nanmedian(np.mean(corners[:, :, :, 1], axis=2), axis=1) * 1000.0
            col_centers_mm = np.nanmedian(np.mean(corners[:, :, :, 2], axis=2), axis=0) * 1000.0
            return corners, center_mm, center_pixel, q_nm, psi_deg, row_centers_mm, col_centers_mm
        finally:
            if ready_path != poni_path:
                try:
                    ready_path.unlink()
                except OSError:
                    pass
