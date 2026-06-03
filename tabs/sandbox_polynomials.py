import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout

from tabs.ui_style import GROUP_BOX_STYLE


class PolynomialProjectMixin:
    def integrate_trapezoid(self, y, x):
        if hasattr(np, "trapezoid"):
            return np.trapezoid(y, x)
        return np.trapz(y, x)

    def build_polynomial_controls(self, parent_layout):
        self.polynomial_box = QGroupBox("Beidellite ODF")
        self.polynomial_box.setStyleSheet(GROUP_BOX_STYLE)
        layout = QVBoxLayout(self.polynomial_box)
        layout.setContentsMargins(8, 14, 8, 6)
        layout.setSpacing(4)

        self.polynomial_auto_peak_button = QPushButton("Auto first peak")
        self.polynomial_auto_peak_button.clicked.connect(self.auto_polynomial_peak)
        layout.addWidget(self.polynomial_auto_peak_button)

        self.polynomial_q_min_edit = QLineEdit("0.05")
        self.polynomial_q_min_edit.returnPressed.connect(self.auto_polynomial_peak)
        layout.addLayout(self.form_row("Peak q min", self.polynomial_q_min_edit))

        self.polynomial_q_max_edit = QLineEdit("")
        self.polynomial_q_max_edit.setPlaceholderText("auto")
        self.polynomial_q_max_edit.returnPressed.connect(self.auto_polynomial_peak)
        layout.addLayout(self.form_row("Peak q max", self.polynomial_q_max_edit))

        self.polynomial_peak_q_edit = QLineEdit("")
        self.polynomial_peak_q_edit.setPlaceholderText("auto")
        self.polynomial_peak_q_edit.textEdited.connect(self.set_polynomial_peak_manual)
        self.polynomial_peak_q_edit.returnPressed.connect(self.update_polynomial_analysis)
        layout.addLayout(self.form_row("Peak q", self.polynomial_peak_q_edit))

        self.polynomial_harmonics_spinbox = QSpinBox()
        self.polynomial_harmonics_spinbox.setRange(2, 20)
        self.polynomial_harmonics_spinbox.setValue(8)
        self.polynomial_harmonics_spinbox.valueChanged.connect(self.auto_polynomial_peak)
        layout.addLayout(self.form_row("Harmonics", self.polynomial_harmonics_spinbox))

        self.polynomial_harmonic_tolerance_edit = QLineEdit("4")
        self.polynomial_harmonic_tolerance_edit.returnPressed.connect(self.auto_polynomial_peak)
        layout.addLayout(self.form_row("Tolerance (%)", self.polynomial_harmonic_tolerance_edit))

        self.polynomial_peak_sensitivity_spinbox = QSpinBox()
        self.polynomial_peak_sensitivity_spinbox.setRange(1, 100)
        self.polynomial_peak_sensitivity_spinbox.setValue(20)
        self.polynomial_peak_sensitivity_spinbox.valueChanged.connect(self.auto_polynomial_peak)
        layout.addLayout(self.form_row("Sensitivity", self.polynomial_peak_sensitivity_spinbox))

        self.polynomial_ring_width_edit = QLineEdit("0.04")
        self.polynomial_ring_width_edit.returnPressed.connect(self.update_polynomial_analysis)
        layout.addLayout(self.form_row("Crown width", self.polynomial_ring_width_edit))

        self.polynomial_background_edit = QLineEdit("")
        self.polynomial_background_edit.setPlaceholderText("auto p5")
        self.polynomial_background_edit.returnPressed.connect(self.update_polynomial_analysis)
        layout.addLayout(self.form_row("Background", self.polynomial_background_edit))

        self.polynomial_radial_bins_spinbox = QSpinBox()
        self.polynomial_radial_bins_spinbox.setRange(50, 5000)
        self.polynomial_radial_bins_spinbox.setValue(600)
        self.polynomial_radial_bins_spinbox.valueChanged.connect(self.update_polynomial_analysis)
        layout.addLayout(self.form_row("q bins", self.polynomial_radial_bins_spinbox))

        self.polynomial_angle_bins_spinbox = QSpinBox()
        self.polynomial_angle_bins_spinbox.setRange(90, 1440)
        self.polynomial_angle_bins_spinbox.setValue(360)
        self.polynomial_angle_bins_spinbox.valueChanged.connect(self.update_polynomial_analysis)
        layout.addLayout(self.form_row("Angle bins", self.polynomial_angle_bins_spinbox))

        self.polynomial_result_label = QLabel("Open an EDF/H5 image.")
        self.polynomial_result_label.setAlignment(Qt.AlignTop)
        self.polynomial_result_label.setWordWrap(True)
        layout.addWidget(self.polynomial_result_label)

        parent_layout.addWidget(self.polynomial_box)
        self.polynomial_last_result = None
        self.polynomial_harmonic_matches = []
        self.polynomial_peak_auto = True

    def apply_polynomial_project(self):
        self.plot_box.setTitle("Beidellite ODF")
        self.update_polynomial_analysis()

    def auto_polynomial_peak(self):
        self.polynomial_peak_auto = True
        self.polynomial_peak_q_edit.clear()
        self.update_polynomial_analysis()

    def set_polynomial_peak_manual(self, *_args):
        self.polynomial_peak_auto = False

    def polynomial_geometry_values(self):
        self.current_geometry = self.read_geometry_from_fields()
        center_x = self.geometry_value("center_x")
        center_y = self.geometry_value("center_y")
        pixel_size_um = self.geometry_value("pixel_size")
        distance_mm = self.geometry_value("distance")
        wavelength_a = self.geometry_value("wavelength")
        if center_x is None or center_y is None:
            raise ValueError("Set Center X and Center Y.")
        if pixel_size_um is None or pixel_size_um <= 0:
            raise ValueError("Pixel size must be > 0 um.")
        if distance_mm is None or distance_mm <= 0:
            raise ValueError("Distance must be > 0 mm.")
        if wavelength_a is None or wavelength_a <= 0:
            raise ValueError("Wavelength must be > 0 A.")
        return center_x, center_y, pixel_size_um, distance_mm, wavelength_a

    def polynomial_q_chi_maps(self, image):
        center_x, center_y, pixel_size_um, distance_mm, wavelength_a = self.polynomial_geometry_values()
        y, x = np.indices(image.shape)
        dx_px = x - center_x
        dy_px = y - center_y
        dx_m = dx_px * pixel_size_um * 1e-6
        dy_m = dy_px * pixel_size_um * 1e-6
        r_m = np.sqrt(dx_m ** 2 + dy_m ** 2)
        two_theta = np.arctan2(r_m, distance_mm * 1e-3)
        q = (4.0 * np.pi / (wavelength_a * 0.1)) * np.sin(two_theta / 2.0)
        chi = (np.degrees(np.arctan2(dy_px, dx_px)) + 360.0) % 360.0
        return q, chi

    def binned_mean(self, x, y, edges):
        index = np.digitize(x, edges) - 1
        valid = (index >= 0) & (index < len(edges) - 1) & np.isfinite(y)
        sums = np.bincount(index[valid], weights=y[valid], minlength=len(edges) - 1)
        counts = np.bincount(index[valid], minlength=len(edges) - 1)
        result = np.full(len(edges) - 1, np.nan, dtype=float)
        ok = counts > 0
        result[ok] = sums[ok] / counts[ok]
        return result, counts

    def polynomial_radial_profile(self, image, q_map):
        valid = np.isfinite(image) & np.isfinite(q_map) & (image > 0) & (image < 4e9) & (q_map > 0)
        if not np.any(valid):
            raise ValueError("No valid pixels for radial integration.")
        q_values = q_map[valid]
        q_min = float(np.nanmin(q_values))
        q_max = float(np.nanmax(q_values))
        edges = np.linspace(q_min, q_max, self.polynomial_radial_bins_spinbox.value() + 1)
        q_axis = 0.5 * (edges[:-1] + edges[1:])
        intensity, counts = self.binned_mean(q_values, image[valid], edges)
        return q_axis, intensity, counts

    def polynomial_angle_mask(self, chi_map, ranges):
        mask = np.zeros(chi_map.shape, dtype=bool)
        for start, end in ranges:
            start = float(start) % 360.0
            end = float(end) % 360.0
            if start <= end:
                mask |= (chi_map >= start) & (chi_map <= end)
            else:
                mask |= (chi_map >= start) | (chi_map <= end)
        return mask

    def polynomial_sector_profiles(self, image, q_map, chi_map, q_axis):
        q_axis = np.asarray(q_axis, dtype=float)
        if q_axis.size < 2:
            return {}

        q_step = float(np.nanmedian(np.diff(q_axis[np.isfinite(q_axis)])))
        if not np.isfinite(q_step) or q_step <= 0:
            return {}

        edges = np.concatenate([
            [q_axis[0] - 0.5 * q_step],
            0.5 * (q_axis[:-1] + q_axis[1:]),
            [q_axis[-1] + 0.5 * q_step],
        ])
        base_valid = np.isfinite(image) & np.isfinite(q_map) & np.isfinite(chi_map) & (image > 0) & (image < 4e9) & (q_map > 0)

        sector_definitions = {
            "179-180 deg": [(179.0, 180.0), (359.0, 360.0)],
            "perpendicular": [(89.0, 90.0), (269.0, 270.0)],
        }
        profiles = {}
        for label, ranges in sector_definitions.items():
            mask = base_valid & self.polynomial_angle_mask(chi_map, ranges)
            if np.any(mask):
                profile, counts = self.binned_mean(q_map[mask], image[mask], edges)
            else:
                profile = np.full_like(q_axis, np.nan, dtype=float)
                counts = np.zeros_like(q_axis, dtype=int)
            profiles[label] = {"intensity": profile, "counts": counts, "ranges": ranges}
        return profiles

    def polynomial_find_first_peak(self, q_axis, intensity, sector_profiles=None):
        q2i = q_axis ** 2 * intensity
        q_min = self.optional_float_from_text(self.polynomial_q_min_edit.text())
        q_max = self.optional_float_from_text(self.polynomial_q_max_edit.text())
        if q_min is None:
            q_min = 0.0
        valid = np.isfinite(q_axis) & np.isfinite(q2i) & (q_axis >= q_min)
        if q_max is not None:
            valid &= q_axis <= q_max
        if not np.any(valid):
            raise ValueError("No radial point in peak search range.")

        peak_signal = self.polynomial_peak_signal(q_axis, q2i, valid)
        self.polynomial_last_peak_signal = peak_signal
        candidates = self.polynomial_peak_candidates_from_profiles(q_axis, intensity, sector_profiles, valid)
        self.polynomial_last_peak_candidates = candidates
        if not candidates:
            valid_indices = np.where(valid)[0]
            peak_index = valid_indices[int(np.nanargmax(peak_signal[valid]))]
            self.polynomial_harmonic_matches = [(1, float(q_axis[peak_index]), float(peak_signal[peak_index]))]
            return float(q_axis[peak_index])

        peak_q = self.polynomial_select_harmonic_q1(candidates)
        return peak_q

    def polynomial_peak_candidates_from_profiles(self, q_axis, intensity, sector_profiles, valid):
        profile_entries = [("all angles", np.asarray(intensity, dtype=float))]
        for label, profile_data in (sector_profiles or {}).items():
            profile = profile_data.get("intensity")
            if profile is not None:
                profile_entries.append((label, np.asarray(profile, dtype=float)))

        all_candidates = []
        signal_by_source = {}
        for label, profile in profile_entries:
            q2i = q_axis ** 2 * profile
            profile_valid = valid & np.isfinite(q2i)
            if not np.any(profile_valid):
                continue
            signal = self.polynomial_peak_signal(q_axis, q2i, profile_valid)
            signal_by_source[label] = signal
            for candidate in self.polynomial_radial_peak_candidates(q_axis, signal, profile_valid):
                enriched = dict(candidate)
                enriched["source"] = label
                all_candidates.append(enriched)

        self.polynomial_peak_signals_by_source = signal_by_source
        if not all_candidates:
            return []

        all_candidates.sort(key=lambda item: item["q"])
        q_step = float(np.nanmedian(np.diff(q_axis[np.isfinite(q_axis)])))
        merge_width = max(3.0 * q_step, 0.025)
        groups = []
        for candidate in all_candidates:
            if not groups or abs(candidate["q"] - groups[-1][-1]["q"]) > merge_width:
                groups.append([candidate])
            else:
                groups[-1].append(candidate)

        merged = []
        for group in groups:
            heights = np.asarray([max(item["height"], 1e-12) for item in group], dtype=float)
            q_values = np.asarray([item["q"] for item in group], dtype=float)
            sources = sorted({item["source"] for item in group})
            merged.append({
                "q": float(np.average(q_values, weights=heights)),
                "height": float(np.sum(heights) * (1.0 + 0.35 * (len(sources) - 1))),
                "index": int(max(group, key=lambda item: item["height"])["index"]),
                "sources": sources,
            })

        merged.sort(key=lambda item: item["q"])
        return merged[:120]

    def polynomial_peak_signal(self, q_axis, q2i, valid):
        q2i = np.asarray(q2i, dtype=float)
        valid_indices = np.where(valid & np.isfinite(q_axis) & np.isfinite(q2i))[0]
        if valid_indices.size < 5:
            return np.nan_to_num(q2i, nan=0.0)

        y = q2i.copy()
        finite_y = y[np.isfinite(y)]
        fill = float(np.nanmedian(finite_y)) if finite_y.size else 0.0
        y = np.nan_to_num(y, nan=fill)

        try:
            from scipy.ndimage import percentile_filter

            window = max(21, min(151, valid_indices.size // 8))
            if window % 2 == 0:
                window += 1
            background = percentile_filter(y, percentile=20, size=window, mode="nearest")
        except Exception:
            window = max(21, min(151, valid_indices.size // 8))
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window, dtype=float) / float(window)
            background = np.convolve(y, kernel, mode="same")

        corrected = y - background
        corrected[corrected < 0] = 0.0
        return corrected

    def polynomial_radial_peak_candidates(self, q_axis, peak_signal, valid):
        q_axis = np.asarray(q_axis, dtype=float)
        peak_signal = np.asarray(peak_signal, dtype=float)
        valid_indices = np.where(valid & np.isfinite(q_axis) & np.isfinite(peak_signal))[0]
        if valid_indices.size < 3:
            return []

        y = peak_signal.copy()
        finite_y = y[np.isfinite(y)]
        baseline = 0.0
        y = np.nan_to_num(y, nan=baseline)
        window = max(3, min(21, (valid_indices.size // 80) * 2 + 3))
        kernel = np.ones(window, dtype=float) / float(window)
        smooth = np.convolve(y, kernel, mode="same")
        valid_smooth = smooth[valid_indices]
        nonzero = valid_smooth[valid_smooth > 0]
        if nonzero.size == 0:
            return []
        spread = float(np.nanpercentile(nonzero, 95) - np.nanpercentile(nonzero, 10))
        sensitivity = float(self.polynomial_peak_sensitivity_spinbox.value()) / 100.0
        threshold = float(np.nanpercentile(nonzero, 10) + (0.24 * (1.0 - sensitivity)) * max(spread, 1e-12))

        q_step = float(np.nanmedian(np.diff(q_axis[np.isfinite(q_axis)])))
        merge_width = max(2.0 * q_step, 0.015)
        try:
            from scipy.signal import find_peaks

            min_distance = max(1, int(round(merge_width / max(q_step, 1e-12))))
            prominence = max(0.02 * sensitivity * max(spread, 1e-12), 1e-12)
            peak_indices, properties = find_peaks(smooth, height=threshold, prominence=prominence, distance=min_distance)
            local = [int(index) for index in peak_indices if valid[index]]
        except Exception:
            local = []
            for index in valid_indices:
                if index <= 0 or index >= len(smooth) - 1:
                    continue
                if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1] and smooth[index] >= threshold:
                    local.append(index)

        if not local:
            return []

        # Merge tiny shoulders produced by noisy SAXS/WAXS rings.
        groups = []
        for index in local:
            if not groups or abs(q_axis[index] - q_axis[groups[-1][-1]]) > merge_width:
                groups.append([index])
            else:
                groups[-1].append(index)

        candidates = []
        for group in groups:
            best = max(group, key=lambda item: smooth[item])
            candidates.append({
                "q": float(q_axis[best]),
                "height": float(max(smooth[best], 0.0)),
                "index": int(best),
            })

        candidates.sort(key=lambda item: item["q"])
        return candidates[:80]

    def polynomial_select_harmonic_q1(self, candidates):
        max_order = int(self.polynomial_harmonics_spinbox.value())
        tolerance_percent = self.optional_float_from_text(self.polynomial_harmonic_tolerance_edit.text())
        if tolerance_percent is None or tolerance_percent <= 0:
            tolerance_percent = 4.0
        relative_tolerance = max(tolerance_percent / 100.0, 0.08)

        q_values = np.asarray([candidate["q"] for candidate in candidates], dtype=float)
        heights = np.asarray([candidate["height"] for candidate in candidates], dtype=float)
        height_scale = float(np.nanmax(heights)) if np.any(np.isfinite(heights)) else 1.0
        height_scale = max(height_scale, 1e-12)
        first_candidate_q = float(np.nanmin(q_values)) if q_values.size else 0.0

        best = None
        hypotheses = []
        for candidate in candidates:
            for order in range(1, max_order + 1):
                q1_guess = candidate["q"] / float(order)
                if q1_guess > 0:
                    hypotheses.append(q1_guess)

        for q1_guess in hypotheses:
            raw_matches = []
            error_sum = 0.0

            for order in range(1, max_order + 1):
                target = order * q1_guess
                tolerance = max(abs(target) * relative_tolerance, 0.5 * q1_guess * relative_tolerance)
                distances = np.abs(q_values - target)
                nearest = int(np.nanargmin(distances))
                if distances[nearest] <= tolerance:
                    rel_error = float(distances[nearest] / max(tolerance, 1e-12))
                    raw_matches.append((order, float(q_values[nearest]), float(heights[nearest]), rel_error))
                    error_sum += rel_error

            harmonic_count = len(raw_matches)
            if harmonic_count < 2:
                continue

            q1_estimates = np.asarray([q_value / order for order, q_value, _height, _err in raw_matches], dtype=float)
            weights = np.asarray([max(height, 1e-12) / np.sqrt(order) for order, _q_value, height, _err in raw_matches], dtype=float)
            q1_refined = float(np.average(q1_estimates, weights=weights))
            matches = []
            refined_error_sum = 0.0
            for order in range(1, max_order + 1):
                target = order * q1_refined
                tolerance = max(abs(target) * relative_tolerance, 0.5 * q1_refined * relative_tolerance)
                distances = np.abs(q_values - target)
                nearest = int(np.nanargmin(distances))
                if distances[nearest] <= tolerance:
                    rel_error = float(distances[nearest] / max(tolerance, 1e-12))
                    matches.append((order, float(q_values[nearest]), float(heights[nearest])))
                    refined_error_sum += rel_error

            has_measured_q1 = any(order == 1 for order, _q_value, _height in matches)
            if not has_measured_q1:
                interpolated_height = float(np.interp(q1_refined, q_values, heights, left=0.0, right=0.0))
                matches.insert(0, (1, q1_refined, interpolated_height))

            unique_orders = {order for order, _q_value, _height in matches}
            score = 5.0 * len(unique_orders)
            score += sum((height / height_scale) / np.sqrt(order) for order, _q_value, height in matches)
            score -= 0.6 * refined_error_sum
            score -= 0.02 * q1_refined
            score -= 0.1 * error_sum
            if not has_measured_q1 and q1_refined < 1.25 * first_candidate_q:
                score -= 12.0

            if best is None or score > best["score"]:
                best = {"score": score, "q1": q1_refined, "matches": matches}

        if best is None:
            best_candidate = max(candidates, key=lambda item: item["height"])
            self.polynomial_harmonic_matches = [(1, best_candidate["q"], best_candidate["height"])]
            return float(best_candidate["q"])

        self.polynomial_harmonic_matches = best["matches"]
        return float(best["q1"])

    def polynomial_azimuthal_crown(self, image, q_map, chi_map, peak_q):
        width = self.optional_float_from_text(self.polynomial_ring_width_edit.text())
        if width is None or width <= 0:
            raise ValueError("Crown width must be > 0.")
        mask = (
            np.isfinite(image)
            & np.isfinite(q_map)
            & np.isfinite(chi_map)
            & (image > 0)
            & (image < 4e9)
            & (np.abs(q_map - peak_q) <= width / 2.0)
        )
        if not np.any(mask):
            raise ValueError("No valid pixels in the selected crown.")
        edges = np.linspace(0.0, 360.0, self.polynomial_angle_bins_spinbox.value() + 1)
        chi_axis = 0.5 * (edges[:-1] + edges[1:])
        intensity, counts = self.binned_mean(chi_map[mask], image[mask], edges)
        return chi_axis, intensity, counts, mask

    def polynomial_background_value(self, intensity_360):
        background = self.optional_float_from_text(self.polynomial_background_edit.text())
        if background is not None:
            return float(background)
        finite = intensity_360[np.isfinite(intensity_360)]
        if finite.size == 0:
            return 0.0
        return float(np.nanpercentile(finite, 5))

    def polynomial_azimuthal_peak_center(self, chi_axis, corrected):
        chi_axis = np.asarray(chi_axis, dtype=float)
        corrected = np.asarray(corrected, dtype=float)
        valid = np.isfinite(chi_axis) & np.isfinite(corrected) & (corrected > 0)
        if np.count_nonzero(valid) < 3:
            return 0.0

        chi = chi_axis[valid] % 180.0
        weights = corrected[valid]
        floor = float(np.nanpercentile(weights, 40))
        weights = np.maximum(weights - floor, 0.0)
        if not np.any(weights > 0):
            weights = corrected[valid]

        threshold = float(np.nanpercentile(weights[weights > 0], 65)) if np.any(weights > 0) else 0.0
        peak_weights = np.where(weights >= threshold, weights, 0.0)
        if np.count_nonzero(peak_weights) >= 3:
            weights = peak_weights

        doubled = np.deg2rad(2.0 * chi)
        vector = np.sum(weights * np.exp(1j * doubled))
        if not np.isfinite(vector.real) or not np.isfinite(vector.imag) or abs(vector) <= 0:
            return float(chi[int(np.nanargmax(corrected[valid]))] % 180.0)

        center = (0.5 * np.degrees(np.angle(vector))) % 180.0

        # Refine on the strongest lobe so weak shoulders do not drag the director.
        delta = ((chi - center + 90.0) % 180.0) - 90.0
        lobe = np.abs(delta) <= 20.0
        if np.count_nonzero(lobe) >= 3 and np.sum(weights[lobe]) > 0:
            lobe_delta_rad = np.deg2rad(delta[lobe])
            correction = np.degrees(np.angle(np.sum(weights[lobe] * np.exp(1j * lobe_delta_rad))))
            if np.isfinite(correction):
                center = (center + correction) % 180.0
        return float(center)

    def polynomial_fold_0_180(self, chi_axis, intensity_360):
        corrected = np.asarray(intensity_360, dtype=float)
        background = self.polynomial_background_value(corrected)
        corrected = np.maximum(corrected - background, 0.0)
        azimuth_center = self.polynomial_azimuthal_peak_center(chi_axis, corrected)
        valid = np.isfinite(chi_axis) & np.isfinite(corrected)
        theta_raw = (azimuth_center - chi_axis[valid]) % 180.0
        values = corrected[valid]
        edges = np.linspace(0.0, 180.0, 181)
        sums, _ = np.histogram(theta_raw, bins=edges, weights=values)
        counts, _ = np.histogram(theta_raw, bins=edges)
        theta_deg = 0.5 * (edges[:-1] + edges[1:])
        folded = np.full_like(theta_deg, np.nan, dtype=float)
        ok = counts > 0
        folded[ok] = sums[ok] / counts[ok]
        folded = np.nan_to_num(folded, nan=0.0)
        return theta_deg, folded, background, azimuth_center

    def polynomial_odf_and_moments(self, theta_deg, folded_intensity):
        theta = np.deg2rad(theta_deg)
        sin_theta = np.sin(theta)
        norm = self.integrate_trapezoid(folded_intensity * sin_theta, theta)
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("ODF normalization is zero after background subtraction.")
        odf = folded_intensity / norm
        mu = np.cos(theta)
        p0 = np.ones_like(mu)
        p2 = 0.5 * (3.0 * mu ** 2 - 1.0)
        p4 = (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0
        p6 = (231.0 * mu ** 6 - 315.0 * mu ** 4 + 105.0 * mu ** 2 - 5.0) / 16.0
        moments = {
            "P0": float(self.integrate_trapezoid(odf * p0 * sin_theta, theta)),
            "P2": float(self.integrate_trapezoid(odf * p2 * sin_theta, theta)),
            "P4": float(self.integrate_trapezoid(odf * p4 * sin_theta, theta)),
            "P6": float(self.integrate_trapezoid(odf * p6 * sin_theta, theta)),
        }
        return odf, moments

    def polynomial_legendre_basis(self, theta_deg):
        theta = np.deg2rad(theta_deg)
        mu = np.cos(theta)
        return {
            "P0": np.ones_like(mu),
            "P2": 0.5 * (3.0 * mu ** 2 - 1.0),
            "P4": (35.0 * mu ** 4 - 30.0 * mu ** 2 + 3.0) / 8.0,
            "P6": (231.0 * mu ** 6 - 315.0 * mu ** 4 + 105.0 * mu ** 2 - 5.0) / 16.0,
        }

    def polynomial_moments_from_odf(self, theta_deg, odf):
        theta = np.deg2rad(theta_deg)
        sin_theta = np.sin(theta)
        basis = self.polynomial_legendre_basis(theta_deg)
        return {
            name: float(self.integrate_trapezoid(odf * values * sin_theta, theta))
            for name, values in basis.items()
        }

    def polynomial_odf_quality_checks(self, chi_axis, intensity_360, background, theta_deg, odf, peak_q):
        theta = np.deg2rad(theta_deg)
        sin_theta = np.sin(theta)
        odf_norm = float(self.integrate_trapezoid(odf * sin_theta, theta))

        raw_corrected = np.asarray(intensity_360, dtype=float) - float(background)
        finite_raw = np.isfinite(raw_corrected)
        if np.any(finite_raw):
            clipped_fraction = float(np.count_nonzero(raw_corrected[finite_raw] < 0.0) / np.count_nonzero(finite_raw))
        else:
            clipped_fraction = np.nan

        corrected = np.maximum(raw_corrected, 0.0)
        finite = np.isfinite(chi_axis) & np.isfinite(corrected)
        centrosymmetry_error = np.nan
        if np.count_nonzero(finite) > 8:
            order = np.argsort(chi_axis[finite])
            chi_sorted = np.asarray(chi_axis[finite], dtype=float)[order]
            values_sorted = np.asarray(corrected[finite], dtype=float)[order]
            chi_extended = np.concatenate([chi_sorted, chi_sorted + 360.0])
            values_extended = np.concatenate([values_sorted, values_sorted])
            base = chi_sorted < 180.0
            if np.any(base):
                opposite = np.interp(chi_sorted[base] + 180.0, chi_extended, values_extended)
                direct = values_sorted[base]
                denominator = np.nanmean(0.5 * (np.abs(direct) + np.abs(opposite)))
                if np.isfinite(denominator) and denominator > 0:
                    centrosymmetry_error = float(np.nanmean(np.abs(direct - opposite)) / denominator)

        harmonic_errors = []
        for order, q_value, _height in getattr(self, "polynomial_harmonic_matches", []):
            expected = float(order) * float(peak_q)
            if expected > 0:
                harmonic_errors.append(abs(float(q_value) - expected) / expected)
        harmonic_rms_percent = np.nan
        if harmonic_errors:
            harmonic_rms_percent = float(100.0 * np.sqrt(np.nanmean(np.asarray(harmonic_errors) ** 2)))

        return {
            "odf_norm": odf_norm,
            "centrosymmetry_error": centrosymmetry_error,
            "clipped_fraction": clipped_fraction,
            "harmonic_count": len({order for order, _q_value, _height in getattr(self, "polynomial_harmonic_matches", [])}),
            "harmonic_rms_percent": harmonic_rms_percent,
        }

    def polynomial_maier_saupe_fit(self, theta_deg, odf):
        theta = np.deg2rad(theta_deg)
        valid = np.isfinite(theta) & np.isfinite(odf) & (odf > 0)
        if np.count_nonzero(valid) < 8:
            return None

        def model(theta_rad, k, m):
            return k * np.exp(m * np.cos(theta_rad) ** 2)

        try:
            from scipy.optimize import curve_fit

            params, _ = curve_fit(
                model,
                theta[valid],
                odf[valid],
                p0=(max(float(np.nanmedian(odf[valid])), 1e-12), 1.0),
                bounds=([0.0, -200.0], [np.inf, 200.0]),
                maxfev=20000,
            )
            k, m = float(params[0]), float(params[1])
        except Exception:
            mu2 = np.cos(theta) ** 2
            coeff = np.polyfit(mu2[valid], np.log(odf[valid]), 1)
            m = float(coeff[0])
            k = float(np.exp(coeff[1]))

        fitted = model(theta, k, m)
        return k, m, fitted

    def polynomial_mem_distribution(self, theta_deg, lambdas):
        theta = np.deg2rad(theta_deg)
        sin_theta = np.sin(theta)
        basis = self.polynomial_legendre_basis(theta_deg)
        exponent = (
            float(lambdas[0]) * basis["P2"]
            + float(lambdas[1]) * basis["P4"]
            + float(lambdas[2]) * basis["P6"]
        )
        shift = float(np.nanmax(exponent))
        raw = np.exp(exponent - shift)
        norm = self.integrate_trapezoid(raw * sin_theta, theta)
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("MEM normalization failed.")
        odf_mem = raw / norm
        log_k = -shift - float(np.log(norm))
        k = float(np.exp(log_k)) if log_k < 700.0 else np.inf
        achieved = self.polynomial_moments_from_odf(theta_deg, odf_mem)
        return odf_mem, achieved, k

    def polynomial_mem_fit(self, theta_deg, moments, maier_saupe=None):
        targets = np.asarray([moments["P2"], moments["P4"], moments["P6"]], dtype=float)
        if not np.all(np.isfinite(targets)):
            return None

        def residual(lambdas):
            try:
                _odf_mem, achieved, _k = self.polynomial_mem_distribution(theta_deg, lambdas)
            except Exception:
                return np.full(3, 1e6, dtype=float)
            return np.asarray([
                achieved["P2"] - targets[0],
                achieved["P4"] - targets[1],
                achieved["P6"] - targets[2],
            ], dtype=float)

        starts = [np.zeros(3, dtype=float)]
        if maier_saupe is not None:
            _ms_k, ms_m, _ms_fit = maier_saupe
            if np.isfinite(ms_m):
                starts.append(np.asarray([2.0 * ms_m / 3.0, 0.0, 0.0], dtype=float))

        best = None
        try:
            from scipy.optimize import least_squares

            for start in starts:
                result = least_squares(
                    residual,
                    start,
                    bounds=(-200.0, 200.0),
                    xtol=1e-11,
                    ftol=1e-11,
                    gtol=1e-11,
                    max_nfev=30000,
                )
                score = float(np.linalg.norm(result.fun))
                if best is None or score < best["score"]:
                    best = {"score": score, "lambdas": result.x, "success": result.success}
        except Exception:
            return None

        if best is None:
            return None

        odf_mem, achieved, k = self.polynomial_mem_distribution(theta_deg, best["lambdas"])
        return {
            "k": k,
            "lambda2": float(best["lambdas"][0]),
            "lambda4": float(best["lambdas"][1]),
            "lambda6": float(best["lambdas"][2]),
            "odf": odf_mem,
            "moments": achieved,
            "residual": best["score"],
            "success": bool(best["success"]),
        }

    def plot_polynomial_pattern_axis(self, image_ax, image, crown_mask=None):
        display = np.asarray(image, dtype=float)
        finite = np.isfinite(display)
        if np.any(finite):
            vmin = np.nanpercentile(display[finite], 1)
            vmax = np.nanpercentile(display[finite], 99)
            image_ax.imshow(display, origin="upper", cmap="jet", vmin=vmin, vmax=vmax)
            if crown_mask is not None and np.any(crown_mask):
                overlay = np.zeros((*display.shape, 4), dtype=float)
                overlay[..., :3] = 0.82
                overlay[..., 3] = np.where(crown_mask, 0.0, 0.58)
                image_ax.imshow(overlay, origin="upper")
            try:
                center_x = self.geometry_value("center_x")
                center_y = self.geometry_value("center_y")
            except Exception:
                center_x = center_y = None
            if center_x is not None and center_y is not None:
                height, width = display.shape
                zoom = 1.55
                half_width = 0.5 * width / zoom
                half_height = 0.5 * height / zoom
                left = max(0.0, min(float(center_x) - half_width, width - 2.0 * half_width))
                top = max(0.0, min(float(center_y) - half_height, height - 2.0 * half_height))
                image_ax.set_xlim(left, left + 2.0 * half_width)
                image_ax.set_ylim(top + 2.0 * half_height, top)
        image_ax.set_xticks([])
        image_ax.set_yticks([])

    def plot_polynomial_q2iq_axis(self, radial_ax, q_axis, radial, peak_q=None, sector_profiles=None):
        if q_axis is not None and radial is not None:
            q2i = q_axis ** 2 * radial
            valid = np.isfinite(q_axis) & np.isfinite(q2i)
            radial_ax.plot(q_axis[valid], q2i[valid], color="#1f77b4", linewidth=1.0, label="all angles")
            sector_styles = {
                "179-180 deg": ("#d62728", "-"),
                "perpendicular": ("#2ca02c", "--"),
            }
            for label, profile_data in (sector_profiles or {}).items():
                profile = profile_data.get("intensity")
                if profile is None:
                    continue
                sector_q2i = q_axis ** 2 * profile
                sector_valid = np.isfinite(q_axis) & np.isfinite(sector_q2i)
                if not np.any(sector_valid):
                    continue
                color, linestyle = sector_styles.get(label, ("#666666", "--"))
                radial_ax.plot(q_axis[sector_valid], sector_q2i[sector_valid], color=color, linestyle=linestyle, linewidth=1.0, label=label)
            for order, q_value, _height in getattr(self, "polynomial_harmonic_matches", []):
                if peak_q is not None and abs(q_value - peak_q) < 1e-12:
                    continue
                radial_ax.axvline(q_value, color="#888888", linestyle=":", linewidth=0.8, alpha=0.8)
                radial_ax.text(q_value, 0.96, f"{order:03d}", transform=radial_ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
            if peak_q is not None:
                radial_ax.axvline(peak_q, color="#d62728", linestyle="--", linewidth=0.9)
                radial_ax.text(peak_q, 0.96, "001", transform=radial_ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="#d62728")
            radial_ax.set_xlabel(r"$q$ / nm$^{-1}$")
            radial_ax.set_ylabel(r"$q^2 I(q)$ / a.u.")
            radial_ax.grid(True, alpha=0.25)
            radial_ax.legend(loc="best", frameon=True)

    def update_polynomial_analysis(self, *args):
        if self.current_image is None:
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)
            self.ax.set_axis_off()
            self.ax.text(0.5, 0.5, "Open an EDF/H5 image.", ha="center", va="center")
            self.canvas.draw_idle()
            self.status_label.setText("Open an EDF/H5 image for beidellite ODF.")
            return

        image = np.asarray(self.current_image, dtype=float)
        try:
            q_map, chi_map = self.polynomial_q_chi_maps(image)
            q_axis, radial, _counts = self.polynomial_radial_profile(image, q_map)
            sector_profiles = self.polynomial_sector_profiles(image, q_map, chi_map, q_axis)
            manual_peak = None if self.polynomial_peak_auto else self.optional_float_from_text(self.polynomial_peak_q_edit.text())
            peak_q = manual_peak if manual_peak is not None else self.polynomial_find_first_peak(q_axis, radial, sector_profiles)
            if manual_peak is None:
                self.polynomial_peak_q_edit.blockSignals(True)
                self.polynomial_peak_q_edit.setText(f"{peak_q:.6g}")
                self.polynomial_peak_q_edit.blockSignals(False)
            chi_axis, intensity_360, _angle_counts, crown_mask = self.polynomial_azimuthal_crown(image, q_map, chi_map, peak_q)
            theta_deg, folded, background, azimuth_center = self.polynomial_fold_0_180(chi_axis, intensity_360)
            odf, moments = self.polynomial_odf_and_moments(theta_deg, folded)
            quality = self.polynomial_odf_quality_checks(chi_axis, intensity_360, background, theta_deg, odf, peak_q)
            maier_saupe = self.polynomial_maier_saupe_fit(theta_deg, odf)
            mem = self.polynomial_mem_fit(theta_deg, moments, maier_saupe)
        except Exception as exc:
            self.status_label.setText(f"Could not calculate beidellite ODF: {exc}")
            self.canvas.draw_idle()
            return

        self.figure.clear()
        ax_pattern = self.figure.add_subplot(221)
        ax_radial = self.figure.add_subplot(222)
        ax_360 = self.figure.add_subplot(223)
        ax_odf = self.figure.add_subplot(224)
        self.ax = ax_odf

        self.plot_polynomial_pattern_axis(ax_pattern, image, crown_mask)
        self.plot_polynomial_q2iq_axis(ax_radial, q_axis, radial, peak_q, sector_profiles)

        corrected_360 = np.maximum(intensity_360 - background, 0.0)
        valid_360 = np.isfinite(chi_axis) & np.isfinite(corrected_360)
        ax_360.plot(chi_axis[valid_360], corrected_360[valid_360], color="#1f77b4", linewidth=1.0)
        for center_line in (azimuth_center, (azimuth_center + 180.0) % 360.0):
            ax_360.axvline(center_line, color="#d62728", linestyle="--", linewidth=0.9, alpha=0.9)
        ax_360.set_xlim(0, 360)
        ax_360.set_xlabel(r"$\psi$ / deg")
        ax_360.set_ylabel(r"$I(\psi) - I_\mathrm{bg}$ / a.u.")
        ax_360.grid(True, alpha=0.25)

        ax_odf.plot(theta_deg, odf, color="#111111", linewidth=1.2, label="ODF 0-180")
        if maier_saupe is not None:
            ms_k, ms_m, ms_fit = maier_saupe
            ax_odf.plot(theta_deg, ms_fit, color="#d62728", linestyle="--", linewidth=1.1, label="Maier-Saupe")
        else:
            ms_k = ms_m = np.nan
        if mem is not None:
            ax_odf.plot(theta_deg, mem["odf"], color="#2ca02c", linestyle="-.", linewidth=1.1, label="MEM")
        ax_odf.set_xlim(0, 180)
        ax_odf.set_xlabel(r"$\theta$ / deg")
        ax_odf.set_ylabel(r"$f(\theta)$")
        ax_odf.grid(True, alpha=0.25)
        ax_odf.legend(loc="best", frameon=True)
        self.figure.tight_layout()
        self.canvas.draw_idle()

        self.polynomial_last_result = {
            "q": q_axis,
            "radial_i": radial,
            "sector_profiles": sector_profiles,
            "peak_q": peak_q,
            "chi": chi_axis,
            "i360": corrected_360,
            "theta": theta_deg,
            "odf": odf,
            "moments": moments,
            "quality": quality,
            "azimuth_center": azimuth_center,
            "background": background,
            "maier_saupe_k": ms_k,
            "maier_saupe_m": ms_m,
            "mem": mem,
            "harmonics": list(getattr(self, "polynomial_harmonic_matches", [])),
        }
        if mem is None:
            mem_text = "MEM fit = failed"
        else:
            mem_text = (
                f"MEM k = {mem['k']:.6g}\n"
                f"MEM λ2 = {mem['lambda2']:.6g}\n"
                f"MEM λ4 = {mem['lambda4']:.6g}\n"
                f"MEM λ6 = {mem['lambda6']:.6g}"
            )
        self.polynomial_result_label.setText(
            f"File: {'-' if self.current_path is None else self.current_path.name}\n"
            f"<P0> = {moments['P0']:.6g}\n"
            f"<P2> = {moments['P2']:.6g}\n"
            f"<P4> = {moments['P4']:.6g}\n"
            f"<P6> = {moments['P6']:.6g}\n"
            f"MS k = {ms_k:.6g}\n"
            f"MS m = {ms_m:.6g}\n"
            f"{mem_text}"
        )
        self.status_label.setText("Beidellite crown ODF calculated.")
