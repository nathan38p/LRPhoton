import numpy as np

from matplotlib import cm, colors


class Saxs3DProjectMixin:
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

