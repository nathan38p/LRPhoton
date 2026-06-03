from pathlib import Path
import re

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication, QTableWidgetItem

from tabs.radial_tab import radial_average


class ImogoliteProjectMixin:
    def q_from_peak_radius_px(self, radius_px):
        pixel_size_um = self.geometry_value("pixel_size")
        distance_mm = self.geometry_value("distance")
        wavelength_a = self.geometry_value("wavelength")

        if radius_px is None or radius_px <= 0:
            raise ValueError("Peak radius must be > 0 px.")
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("Pixel size must be > 0 µm.")
        if distance_mm is None or distance_mm <= 0:
            raise ValueError("Distance must be > 0 mm.")
        if wavelength_a is None or wavelength_a <= 0:
            raise ValueError("Wavelength must be > 0 Å.")

        radius_m = float(radius_px) * pixel_size_um * 1e-6
        distance_m = distance_mm * 1e-3
        wavelength_nm = wavelength_a * 0.1
        two_theta = np.arctan2(radius_m, distance_m)
        return (4.0 * np.pi / wavelength_nm) * np.sin(two_theta / 2.0)


    def peak_radius_px_from_q(self, q_nm):
        pixel_size_um = self.geometry_value("pixel_size")
        distance_mm = self.geometry_value("distance")
        wavelength_a = self.geometry_value("wavelength")

        if q_nm is None or q_nm <= 0:
            raise ValueError("Peak q must be > 0 nm⁻¹.")
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("Pixel size must be > 0 µm.")
        if distance_mm is None or distance_mm <= 0:
            raise ValueError("Distance must be > 0 mm.")
        if wavelength_a is None or wavelength_a <= 0:
            raise ValueError("Wavelength must be > 0 Å.")

        wavelength_nm = wavelength_a * 0.1
        argument = float(q_nm) * wavelength_nm / (4.0 * np.pi)
        if abs(argument) > 1:
            raise ValueError("Peak q is incompatible with the wavelength.")
        two_theta = 2.0 * np.arcsin(argument)
        radius_m = distance_mm * 1e-3 * np.tan(two_theta)
        return radius_m / (pixel_size_um * 1e-6)


    def calculate_imogolite_distance(self):
        try:
            self.current_geometry = self.read_geometry_from_fields()
            q_nm = self.optional_float_from_text(self.imogolite_peak_q_edit.text())
            radius_px = self.optional_float_from_text(self.imogolite_peak_radius_edit.text())
            if q_nm is None:
                q_nm = self.q_from_peak_radius_px(radius_px)
            elif radius_px is None:
                radius_px = self.peak_radius_px_from_q(q_nm)
                self.imogolite_peak_radius_edit.setText(f"{radius_px:.6g}")
            d_nm = 2.0 * np.pi / q_nm
            volume_fraction = self.imogolite_volume_fraction_from_q(q_nm)
        except Exception as exc:
            self.imogolite_result_label.setText(f"Could not calculate:\n{exc}")
            return

        self.imogolite_peak_q_edit.setText(f"{q_nm:.6g}")
        self.imogolite_result_label.setText(
            f"q = {q_nm:.6g} nm⁻¹\n"
            f"r = {radius_px:.6g} px\n"
            f"d = 2π/q = {d_nm:.6g} nm\n"
            f"φ = {volume_fraction:.6g}"
        )


    def imogolite_tube_diameter_nm(self):
        diameter = self.optional_float_from_text(self.imogolite_tube_diameter_edit.text())
        if diameter is None or diameter <= 0:
            raise ValueError("Tube diameter must be > 0 nm.")
        return float(diameter)


    def imogolite_frame_step_um(self):
        step = self.optional_float_from_text(self.imogolite_frame_step_combo.currentText())
        if step is None or step <= 0:
            raise ValueError("Frame step must be > 0 µm.")
        return float(step)


    def imogolite_volume_fraction_from_q(self, q_nm):
        if q_nm is None or q_nm <= 0:
            raise ValueError("Peak q must be > 0 nm⁻¹.")
        d_nm = 2.0 * np.pi / float(q_nm)
        tube_diameter_nm = self.imogolite_tube_diameter_nm()
        swelling_prefactor = np.sqrt(np.pi * np.sqrt(3.0) / 8.0)
        return float(((swelling_prefactor * tube_diameter_nm) / d_nm) ** 2)


    def parabolic_peak_q(self, q, y):
        q = np.asarray(q, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(q) & np.isfinite(y) & (q > 0) & (q <= 1.0)
        if not np.any(valid):
            return None

        q_valid = q[valid]
        y_valid = y[valid]
        peak_index = int(np.nanargmax(y_valid))
        if peak_index == 0 or peak_index == len(q_valid) - 1:
            return float(q_valid[peak_index])

        x_fit = q_valid[peak_index - 1:peak_index + 2]
        y_fit = y_valid[peak_index - 1:peak_index + 2]
        try:
            a, b, _c = np.polyfit(x_fit, y_fit, 2)
        except Exception:
            return float(q_valid[peak_index])

        if not np.isfinite(a) or not np.isfinite(b) or a >= 0:
            return float(q_valid[peak_index])

        refined_q = float(-b / (2.0 * a))
        if refined_q < float(x_fit[0]) or refined_q > float(x_fit[-1]):
            return float(q_valid[peak_index])
        return refined_q


    def imogolite_metrics(self, q, intensity):
        q = np.asarray(q, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        qiq = q * intensity
        qmax = self.parabolic_peak_q(q, qiq)
        if qmax is None:
            return None

        d_nm = float(2.0 * np.pi / qmax)

        try:
            volume_fraction = self.imogolite_volume_fraction_from_q(qmax)
        except Exception:
            volume_fraction = np.nan

        return {
            "qmax": qmax,
            "d_nm": d_nm,
            "volume_fraction": volume_fraction,
        }


    def imogolite_output_folder(self):
        base_folder = self.current_folder or self.folder_path
        if base_folder is None and self.current_path is not None:
            base_folder = self.current_path.parent
        if base_folder is None:
            base_folder = Path.home()
        output_folder = Path(base_folder) / "imogolite_dat"
        output_folder.mkdir(exist_ok=True)
        return output_folder


    def frame_number_from_path(self, path):
        match = re.search(r"frame[_-]?(\d+)", Path(path).stem, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)", Path(path).stem)
        return int(match.group(1)) if match else None


    def add_imogolite_sample_positions(self, results):
        results = [dict(result) for result in results]
        if not results:
            return results

        try:
            frame_step_um = self.imogolite_frame_step_um()
        except Exception:
            frame_step_um = 5.0

        frames = [result.get("frame") for result in results if result.get("frame") is not None]
        reverse_origin = self.imogolite_origin_combo.currentText() == "last frame → first"

        if frames:
            first_frame = min(frames)
            last_frame = max(frames)
            for result in results:
                frame = result.get("frame")
                if frame is None:
                    result["sample_distance_um"] = None
                elif reverse_origin:
                    result["sample_distance_um"] = float(last_frame - frame) * frame_step_um
                else:
                    result["sample_distance_um"] = float(frame - first_frame) * frame_step_um
            return results

        last_index = len(results) - 1
        for index, result in enumerate(results):
            position_index = last_index - index if reverse_origin else index
            result["sample_distance_um"] = float(position_index) * frame_step_um
        return results


    def imogolite_curve_color(self, index, total):
        total = max(1, int(total))
        t = 0.0 if total == 1 else float(index) / float(total - 1)
        start = QColor(self.imogolite_gradient_start)
        end = QColor(self.imogolite_gradient_end)
        red = round(start.red() + (end.red() - start.red()) * t)
        green = round(start.green() + (end.green() - start.green()) * t)
        blue = round(start.blue() + (end.blue() - start.blue()) * t)
        return QColor(red, green, blue).name()


    def imogolite_dat_path_for_source(self, source_path):
        source_path = Path(source_path)
        return self.imogolite_output_folder() / f"{source_path.stem}_imogolite_qI.dat"


    def save_imogolite_dat(self, source_path, q, intensity, counts, metrics):
        output_path = self.imogolite_dat_path_for_source(source_path)
        q = np.asarray(q, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        counts = np.asarray(counts, dtype=float)
        if counts.shape != q.shape:
            counts = np.ones_like(q)

        data = np.column_stack([q, intensity, q * intensity, counts])
        with output_path.open("w", encoding="utf-8") as handle:
            handle.write("# LRPhoton imogolite integrated curve\n")
            handle.write(f"# source = {Path(source_path)}\n")
            if metrics is not None:
                handle.write(f"# qmax_nm-1 = {metrics['qmax']:.10g}\n")
                handle.write(f"# d_nm = {metrics['d_nm']:.10g}\n")
                handle.write(f"# volume_fraction = {metrics['volume_fraction']:.10g}\n")
            handle.write("# columns: q_nm-1 I_q qI_q counts\n")
            np.savetxt(handle, data, fmt="%.10g")
        return output_path


    def load_imogolite_dat(self, path):
        data = np.loadtxt(path, comments="#")
        data = np.atleast_2d(data)
        if data.shape[1] < 2:
            raise ValueError("DAT file must contain at least q and I(q).")
        q = np.asarray(data[:, 0], dtype=float)
        intensity = np.asarray(data[:, 1], dtype=float)
        counts = np.asarray(data[:, 3], dtype=float) if data.shape[1] >= 4 else np.ones_like(q)
        valid = np.isfinite(q) & np.isfinite(intensity)
        return q[valid], intensity[valid], counts[valid]


    def update_imogolite_results_table(self, results):
        self.imogolite_results = self.add_imogolite_sample_positions(results)
        self.imogolite_results_table.setRowCount(len(self.imogolite_results))

        for row, result in enumerate(self.imogolite_results):
            frame = result.get("frame")
            sample_distance_um = result.get("sample_distance_um")
            qmax = result.get("qmax")
            d_nm = result.get("d_nm")
            volume_fraction = result.get("volume_fraction")
            values = [
                "-" if frame is None else str(frame),
                "-" if sample_distance_um is None else f"{sample_distance_um:.6g}",
                "-" if qmax is None else f"{qmax:.6g}",
                "-" if d_nm is None else f"{d_nm:.6g}",
                "-" if volume_fraction is None or not np.isfinite(volume_fraction) else f"{volume_fraction:.6g}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignCenter)
                self.imogolite_results_table.setItem(row, column, item)

        self.imogolite_results_table.resizeColumnsToContents()


    def copy_imogolite_results_table(self):
        table = self.imogolite_results_table
        if table.rowCount() == 0:
            self.status_label.setText("No imogolite table to copy.")
            return

        selected_ranges = table.selectedRanges()
        if selected_ranges:
            selected_range = selected_ranges[0]
            row_start = selected_range.topRow()
            row_end = selected_range.bottomRow()
            column_start = selected_range.leftColumn()
            column_end = selected_range.rightColumn()
        else:
            row_start = 0
            row_end = table.rowCount() - 1
            column_start = 0
            column_end = table.columnCount() - 1

        lines = []
        headers = [
            table.horizontalHeaderItem(column).text()
            for column in range(column_start, column_end + 1)
        ]
        lines.append("\t".join(headers))

        for row in range(row_start, row_end + 1):
            values = []
            for column in range(column_start, column_end + 1):
                item = table.item(row, column)
                values.append("" if item is None else item.text())
            lines.append("\t".join(values))

        QApplication.clipboard().setText("\n".join(lines))
        self.status_label.setText(f"Copied {row_end - row_start + 1} table row(s) to clipboard.")


    def update_imogolite_table_only(self):
        if not self.imogolite_results:
            return
        refreshed = []
        for result in self.imogolite_results:
            updated = dict(result)
            updated.pop("sample_distance_um", None)
            qmax = updated.get("qmax")
            if qmax is not None and qmax > 0:
                metrics = self.imogolite_metrics(np.asarray([qmax]), np.asarray([1.0 / qmax]))
                if metrics is not None:
                    updated.update(metrics)
            refreshed.append(updated)
        self.update_imogolite_results_table(refreshed)


    def use_current_max_as_imogolite_peak(self):
        if self.imogolite_q is None or self.imogolite_intensity is None:
            self.plot_imogolite_iq(self.current_image)
        if self.imogolite_q is None or self.imogolite_intensity is None:
            self.imogolite_result_label.setText("Integrate I(q) first.")
            return

        valid = np.isfinite(self.imogolite_q) & np.isfinite(self.imogolite_intensity)
        if not np.any(valid):
            self.imogolite_result_label.setText("I(q) contains no finite points.")
            return

        q = self.imogolite_q[valid]
        intensity = self.imogolite_intensity[valid]
        peak_q = float(q[int(np.nanargmax(q * intensity))])
        self.set_imogolite_peak_q(peak_q)


    def set_imogolite_peak_q(self, q_nm):
        self.imogolite_peak_q_edit.setText(f"{float(q_nm):.6g}")
        self.imogolite_peak_radius_edit.clear()
        self.calculate_imogolite_distance()


    def imogolite_paths_for_stride(self, paths):
        stride = max(1, int(self.imogolite_file_stride_spinbox.value()))
        return list(paths)[::stride]


    def integrate_imogolite_selection(self):
        paths = self.selected_files() or self.imogolite_current_paths
        if paths:
            self.imogolite_current_paths = paths
            paths_to_integrate = self.imogolite_paths_for_stride(paths)
            if paths_to_integrate:
                self.plot_imogolite_files(paths_to_integrate, total_selected=len(paths))
            return

        self.replot_current_image()


    def imogolite_iq_profile(self, image):
        self.current_geometry = self.read_geometry_from_fields()
        center_x = self.geometry_value("center_x")
        center_y = self.geometry_value("center_y")
        pixel_size_um = self.geometry_value("pixel_size")
        distance_mm = self.geometry_value("distance")
        wavelength_a = self.geometry_value("wavelength")

        if center_x is None or center_y is None:
            raise ValueError("Set Center X and Center Y first.")
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("Pixel size must be > 0 µm.")
        if distance_mm is None or distance_mm <= 0:
            raise ValueError("Distance must be > 0 mm.")
        if wavelength_a is None or wavelength_a <= 0:
            raise ValueError("Wavelength must be > 0 Å.")

        image = np.asarray(image, dtype=float)
        q_min_limit = 0.0
        q_max_limit = 2.0
        n_bins = int(self.imogolite_bins_spinbox.value())
        distance_m = distance_mm * 1e-3
        pixel_mm = pixel_size_um * 1e-3

        try:
            try:
                from pyFAI.integrator.azimuthal import AzimuthalIntegrator
            except Exception:
                from pyFAI.azimuthalIntegrator import AzimuthalIntegrator

            invalid_mask = ~np.isfinite(image) | (image <= 0) | (image >= 4e9)
            integrator = AzimuthalIntegrator(
                dist=float(distance_m),
                poni1=float(center_y) * pixel_mm * 1e-3,
                poni2=float(center_x) * pixel_mm * 1e-3,
                pixel1=pixel_mm * 1e-3,
                pixel2=pixel_mm * 1e-3,
                wavelength=wavelength_a * 1e-10,
            )
            result = integrator.integrate1d(
                image.astype(np.float64),
                n_bins,
                unit="q_nm^-1",
                radial_range=(q_min_limit, q_max_limit),
                mask=invalid_mask,
                method=("bbox", "csr", "cython"),
                correctSolidAngle=True,
            )
            q = np.asarray(getattr(result, "radial", result[0]), dtype=float)
            intensity = np.asarray(getattr(result, "intensity", result[1]), dtype=float)

            valid = np.isfinite(q) & np.isfinite(intensity) & (q >= q_min_limit) & (q <= q_max_limit) & (intensity > 0)
            if not np.any(valid):
                raise ValueError("pyFAI returned no finite I(q) points.")
            return q[valid], intensity[valid], np.ones(np.count_nonzero(valid), dtype=int)
        except Exception:
            q, intensity, counts, _mask = radial_average(
                image,
                center_x,
                center_y,
                distance_m,
                pixel_mm,
                pixel_mm,
                wavelength_a,
                q_min_limit,
                q_max_limit,
                n_bins,
                False,
                0,
                360,
                1,
            )
            return q, intensity, counts


    def first_frame_from_path(self, path):
        stack = self.load_stack(path)
        if stack.shape[0] < 1:
            raise ValueError("No frame found")
        return stack[0]


    def plot_imogolite_files(self, paths, total_selected=None):
        paths = list(paths)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.ax = ax
        self.imogolite_q = None
        self.imogolite_intensity = None

        plot_mode = self.imogolite_plot_mode_combo.currentText()
        messages = []
        plotted_count = 0
        results = []
        saved_count = 0

        for path_index, path in enumerate(paths):
            image = None
            try:
                if path.suffix.lower() == ".dat":
                    q, intensity, counts = self.load_imogolite_dat(path)
                else:
                    image = self.first_frame_from_path(path)
                    q, intensity, counts = self.imogolite_iq_profile(image)
            except Exception as exc:
                messages.append(f"{path.name}: {exc}")
                continue

            plotted_intensity = q * intensity
            valid = np.isfinite(q) & np.isfinite(plotted_intensity)
            if plot_mode in {"log log", "log lin"}:
                valid &= q > 0
            if plot_mode in {"log log", "lin log"}:
                valid &= plotted_intensity > 0
            if not np.any(valid):
                messages.append(f"{path.name}: no finite qI(q) points for this scale")
                continue

            metrics = self.imogolite_metrics(q, intensity)
            if metrics is None:
                messages.append(f"{path.name}: could not determine qmax")
                continue

            if path.suffix.lower() != ".dat":
                try:
                    self.save_imogolite_dat(path, q, intensity, counts, metrics)
                    saved_count += 1
                except Exception as exc:
                    messages.append(f"{path.name}: could not save DAT ({exc})")

            color = self.imogolite_curve_color(path_index, len(paths))
            ax.plot(q[valid], plotted_intensity[valid], linewidth=1.1, color=color, label=path.stem)
            ax.axvline(metrics["qmax"], color=color, linestyle=":", linewidth=0.6, alpha=0.5)
            plotted_count += 1

            if self.imogolite_q is None:
                self.imogolite_q = q
                self.imogolite_intensity = intensity
                self.current_path = path
                if image is not None:
                    self.current_image = image

            results.append({
                "path": path,
                "frame": self.frame_number_from_path(path),
                "qmax": metrics["qmax"],
                "d_nm": metrics["d_nm"],
                "volume_fraction": metrics["volume_fraction"],
            })

        ax.set_xlabel("q / nm⁻¹")
        ax.set_ylabel("qI(q)")
        ax.set_xscale("log" if plot_mode in {"log log", "log lin"} else "linear")
        ax.set_yscale("log" if plot_mode in {"log log", "lin log"} else "linear")
        ax.set_xlim(0.0, 2.0)
        ax.set_title("Integrated qI(q)")
        ax.grid(True, alpha=0.25)
        if plotted_count and self.imogolite_legend_checkbox.isChecked():
            ax.legend(loc="best", frameon=True)

        peak_q = self.optional_float_from_text(self.imogolite_peak_q_edit.text())
        if peak_q is not None:
            ax.axvline(peak_q, color="#d62728", linestyle="--", linewidth=1.1)

        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.update_imogolite_results_table(results)
        selected_count = total_selected if total_selected is not None else len(paths)
        stride = self.imogolite_file_stride_spinbox.value()
        if messages:
            self.status_label.setText(
                f"Integrated/plotted {plotted_count} / {selected_count} selected file(s), every {stride}. "
                + " | ".join(messages[:3])
            )
        else:
            self.status_label.setText(
                f"Integrated/plotted {plotted_count} / {selected_count} selected file(s), every {stride}. "
                f"Saved {saved_count} DAT curve(s) in imogolite_dat."
            )


    def plot_imogolite_iq(self, image):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        self.ax = ax
        self.imogolite_q = None
        self.imogolite_intensity = None

        if image is None:
            self.status_label.setText("Open a cave EDF/H5 image first.")
            self.canvas.draw_idle()
            return

        try:
            q, intensity, counts = self.imogolite_iq_profile(image)
        except Exception as exc:
            self.status_label.setText(f"Could not integrate I(q): {exc}")
            self.canvas.draw_idle()
            return

        self.imogolite_q = q
        self.imogolite_intensity = intensity
        plotted_intensity = q * intensity
        metrics = self.imogolite_metrics(q, intensity)
        if metrics is not None and self.current_path is not None:
            try:
                self.save_imogolite_dat(self.current_path, q, intensity, counts, metrics)
            except Exception as exc:
                self.status_label.setText(f"Integrated I(q), but could not save DAT: {exc}")
        plot_mode = self.imogolite_plot_mode_combo.currentText()
        valid = np.isfinite(q) & np.isfinite(plotted_intensity)
        if plot_mode in {"log log", "log lin"}:
            valid &= q > 0
        if plot_mode in {"log log", "lin log"}:
            valid &= plotted_intensity > 0

        ax.plot(q[valid], plotted_intensity[valid], color=self.imogolite_curve_color(0, 1), linewidth=1.2)
        if metrics is not None:
            ax.axvline(metrics["qmax"], color=self.imogolite_curve_color(0, 1), linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("q / nm⁻¹")
        ax.set_ylabel("qI(q)")
        if plot_mode in {"log log", "log lin"}:
            ax.set_xscale("log")
        else:
            ax.set_xscale("linear")
        if plot_mode in {"log log", "lin log"}:
            ax.set_yscale("log")
        else:
            ax.set_yscale("linear")
        ax.set_xlim(0.0, 2.0)

        title = "Integrated qI(q)"
        if self.current_path is not None:
            title += f" - {self.current_path.name}"
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

        peak_q = self.optional_float_from_text(self.imogolite_peak_q_edit.text())
        if peak_q is not None:
            ax.axvline(peak_q, color="#d62728", linestyle="--", linewidth=1.1)

        self.figure.tight_layout()
        self.canvas.draw_idle()
        if metrics is not None:
            self.update_imogolite_results_table([{
                "path": self.current_path,
                "frame": self.frame_number_from_path(self.current_path) if self.current_path is not None else None,
                "qmax": metrics["qmax"],
                "d_nm": metrics["d_nm"],
                "volume_fraction": metrics["volume_fraction"],
            }])
        self.status_label.setText("Integrated qI(q), saved DAT when possible, and calculated qmax.")


    def on_canvas_click(self, event):
        if self.current_project() != "Imogolite distance":
            return
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if self.imogolite_q is None:
            return

        q_values = np.asarray(self.imogolite_q, dtype=float)
        if q_values.size == 0:
            return
        index = int(np.nanargmin(np.abs(q_values - float(event.xdata))))
        self.set_imogolite_peak_q(float(q_values[index]))
        paths = self.selected_files() or self.imogolite_current_paths
        if paths:
            self.plot_imogolite_files(self.imogolite_paths_for_stride(paths), total_selected=len(paths))
        else:
            self.plot_imogolite_iq(self.current_image)
