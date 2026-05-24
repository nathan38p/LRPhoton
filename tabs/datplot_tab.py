import json
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt, QEvent, Signal, QMimeData, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QListWidget,
    QGroupBox,
    QCheckBox,
    QGridLayout,
    QLineEdit,
    QDoubleSpinBox,
    QScrollArea,
    QComboBox,
    QColorDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QAbstractItemView,
    QFrame,
    QSpinBox,
    QSlider,
    QSizePolicy,
    QStyle,
    QToolButton,
    QMenu,
    QStyledItemDelegate,
    QStyleOptionViewItem,
)
from PySide6.QtGui import QColor, QDrag

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from .file_ratings import install_file_rating_menu, is_file_rated_up, set_item_file_path
from .ui_style import (
    BLOCK_SPACING,
    FILE_BROWSER_WIDTH,
    FRAME_BUTTON_WIDTH,
    FRAME_COUNTER_WIDTH,
    FRAME_NAV_SPACING,
    FRAME_SPIN_WIDTH,
    GROUP_BOX_MARGINS,
    GROUP_BOX_STYLE,
    PAGE_MARGINS,
    apply_plot_display_style,
    clear_plot_canvas,
    finalize_plot_canvas,
    make_matplotlib_toolbar_block,
    make_plot_legend,
)


# ============================================================
# ========================== TOOLS ============================
# ============================================================

def read_dat_curve(file_path):
    file_path = Path(file_path)
    text = file_path.read_text(encoding="utf-8", errors="ignore")
    text = text.replace(",", ".")

    data = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if not any(char.isdigit() for char in line):
            continue

        for separator in [";", "\t", ","]:
            line = line.replace(separator, " ")

        values = []
        for part in line.split():
            try:
                values.append(float(part))
            except ValueError:
                pass

        if len(values) >= 2:
            data.append([values[0], values[1]])

    if not data:
        raise ValueError("No valid numerical data found in this file.")

    array = np.asarray(data, dtype=float)
    valid = np.isfinite(array[:, 0]) & np.isfinite(array[:, 1])
    array = array[valid]

    if array.size == 0:
        raise ValueError("No finite numerical data found in this file.")

    order = np.argsort(array[:, 0])
    array = array[order]
    return array[:, 0], array[:, 1]


def default_color(index):
    palette = [
        "#e91e63", "#9c27b0", "#f44336", "#4caf50", "#2196f3",
        "#000000", "#ff9800", "#009688", "#795548", "#607d8b",
    ]
    return palette[index % len(palette)]


def _legend_store_path():
    return Path.home() / ".lrphoton" / "datplot_legends.json"


# ============================================================
# ======================== CUSTOM TABLE =======================
# ============================================================

class CurveTableWidget(QTableWidget):
    """Custom table widget with better drag-drop handling for curves."""
    
    def __init__(self, rows, cols, parent=None):
        super().__init__(rows, cols, parent)
        self._drag_row = None
        # Enable drag-drop
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDragDropOverwriteMode(False)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
    
    def mousePressEvent(self, event):
        """Record which row is being dragged."""
        index = self.indexAt(event.pos())
        if index.isValid():
            self._drag_row = index.row()
        super().mousePressEvent(event)
    
    def dragMoveEvent(self, event):
        """Allow drop on rows."""
        if event.source() is self:
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            super().dragMoveEvent(event)
    
    def dropEvent(self, event):
        """Move the underlying curve row instead of relying on QTableWidget internals."""
        if event.source() is self:
            parent_tab = self.parent()
            while parent_tab and not hasattr(parent_tab, 'refresh_curve_table'):
                parent_tab = parent_tab.parent()

            drop_row = self.indexAt(event.position().toPoint()).row()
            if drop_row < 0:
                drop_row = self.rowCount()
            elif self._drag_row is not None:
                rect = self.visualRect(self.model().index(drop_row, 0))
                if event.position().toPoint().y() > rect.center().y():
                    drop_row += 1

            if parent_tab and hasattr(parent_tab, "move_curve_row") and self._drag_row is not None:
                parent_tab.move_curve_row(self._drag_row, drop_row)
                event.setDropAction(Qt.DropAction.MoveAction)
                event.accept()
                self._drag_row = None
                return

            self._drag_row = None
            event.ignore()
        else:
            super().dropEvent(event)


class ColorCellDelegate(QStyledItemDelegate):
    """Keep color swatches visible even when their row is selected."""

    def paint(self, painter, option, index):
        item_option = QStyleOptionViewItem(option)
        self.initStyleOption(item_option, index)

        widget = option.widget
        style = widget.style() if widget is not None else None
        if style is not None:
            style.drawControl(QStyle.CE_ItemViewItem, item_option, painter, widget)

        color = index.data(Qt.BackgroundRole)
        if not isinstance(color, QColor):
            color = QColor(index.data(Qt.ToolTipRole) or "")
        if not color.isValid():
            return

        painter.save()
        painter.fillRect(option.rect.adjusted(1, 1, -1, -1), color)
        painter.restore()


# ============================================================
# =========================== CANVAS ==========================
# ============================================================

class PlotCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(dpi=150)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(620, 420)
        self.fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.18)

        try:
            self.grabGesture(Qt.PinchGesture)
        except Exception:
            pass

    def event(self, event):
        try:
            if event.type() == QEvent.NativeGesture:
                gesture_type = event.gestureType()
                value = event.value()
                if gesture_type == Qt.ZoomNativeGesture and value != 0:
                    scale = 1.0 / (1.0 + value) if value > -0.95 else 1.25
                    self.zoom_at_qpoint(self._event_center_point(event), scale)
                    event.accept()
                    return True

                if gesture_type == Qt.SmartZoomNativeGesture:
                    self.ax.relim()
                    self.ax.autoscale_view()
                    self.draw_idle()
                    event.accept()
                    return True

            if event.type() == QEvent.Gesture:
                pinch = event.gesture(Qt.PinchGesture)
                if pinch is not None:
                    factor = pinch.scaleFactor()
                    if factor and factor > 0:
                        self.zoom_at_qpoint(self._event_center_point(event), 1.0 / factor)
                        event.accept()
                        return True
        except Exception:
            pass

        return super().event(event)

    def wheelEvent(self, event):
        delta = event.pixelDelta()
        if delta.isNull():
            delta = event.angleDelta()
            dx = delta.x() / 120.0
            dy = delta.y() / 120.0
        else:
            dx = delta.x() / 80.0
            dy = delta.y() / 80.0

        if event.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            if dy != 0:
                scale = 0.88 if dy > 0 else 1.14
                self.zoom_at_qpoint(event.position(), scale)
        else:
            self.pan_by_trackpad(dx, dy)

        event.accept()

    def _event_center_point(self, event):
        try:
            position = event.position()
            if position is not None:
                return position
        except Exception:
            pass

        return self.rect().center()

    def _qpoint_to_data_pos(self, qpoint):
        try:
            x_widget = float(qpoint.x())
            y_widget = float(qpoint.y())
        except Exception:
            x_widget = self.width() / 2
            y_widget = self.height() / 2

        bbox = self.ax.get_window_extent()
        x_fig = bbox.x0 + (x_widget / max(self.width(), 1)) * bbox.width
        y_fig = bbox.y1 - (y_widget / max(self.height(), 1)) * bbox.height
        xdata, ydata = self.ax.transData.inverted().transform((x_fig, y_fig))

        if not np.isfinite(xdata) or not np.isfinite(ydata):
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            xdata = (xlim[0] + xlim[1]) / 2
            ydata = (ylim[0] + ylim[1]) / 2

        return xdata, ydata

    def _scaled_limits(self, limits, center, scale, is_log):
        left, right = limits
        if is_log and left > 0 and right > 0 and center > 0:
            left_l, right_l, center_l = np.log10([left, right, center])
            return (
                10 ** (center_l + (left_l - center_l) * scale),
                10 ** (center_l + (right_l - center_l) * scale),
            )

        return (
            center + (left - center) * scale,
            center + (right - center) * scale,
        )

    def zoom_at_qpoint(self, qpoint, scale):
        if scale <= 0:
            return

        xdata, ydata = self._qpoint_to_data_pos(qpoint)
        self.ax.set_xlim(
            self._scaled_limits(
                self.ax.get_xlim(),
                xdata,
                scale,
                self.ax.get_xscale() == "log",
            )
        )
        self.ax.set_ylim(
            self._scaled_limits(
                self.ax.get_ylim(),
                ydata,
                scale,
                self.ax.get_yscale() == "log",
            )
        )
        self.draw_idle()

    def _panned_limits(self, limits, delta, is_log):
        left, right = limits
        if is_log and left > 0 and right > 0:
            left_l, right_l = np.log10([left, right])
            span = right_l - left_l
            shift = -delta * span * 0.08
            return 10 ** (left_l + shift), 10 ** (right_l + shift)

        span = right - left
        shift = -delta * span * 0.08
        return left + shift, right + shift

    def pan_by_trackpad(self, dx, dy):
        self.ax.set_xlim(
            self._panned_limits(
                self.ax.get_xlim(),
                dx,
                self.ax.get_xscale() == "log",
            )
        )
        self.ax.set_ylim(
            self._panned_limits(
                self.ax.get_ylim(),
                -dy,
                self.ax.get_yscale() == "log",
            )
        )
        self.draw_idle()


# ============================================================
# ========================= DAT PLOT TAB ======================
# ============================================================

class DatPlotTab(QWidget):
    """Plot tab: display and compare .dat curves."""

    folder_changed = Signal(Path)

    def __init__(self):
        super().__init__()

        self.current_folder = Path("/Users/nathanpiaget/Documents/Thèse LRP/Expériences/XENOCS")
        self.curves = {}
        self.guide_bars = []
        self.saved_legends = self.load_saved_legends()
        self._syncing_folder = False
        self._refreshing_curve_table = False
        self._refreshing_guide_table = False

        self.build_ui()
        self.refresh_files()

    def load_saved_legends(self):
        path = _legend_store_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

        legends = data.get("legends", {}) if isinstance(data, dict) else {}
        return {
            str(file_path): str(legend)
            for file_path, legend in legends.items()
            if isinstance(file_path, str) and isinstance(legend, str)
        }

    def save_saved_legends(self):
        path = _legend_store_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        data = {"version": 1, "legends": self.saved_legends}
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def saved_legend_for_file(self, file_path):
        try:
            key = str(Path(file_path).expanduser().resolve())
        except (OSError, RuntimeError):
            key = str(file_path)
        return self.saved_legends.get(key)

    def remember_legend_for_file(self, file_path, legend):
        try:
            key = str(Path(file_path).expanduser().resolve())
        except (OSError, RuntimeError):
            key = str(file_path)
        self.saved_legends[key] = legend
        self.save_saved_legends()

    def create_matplotlib_toolbar_block(
        self,
        title,
        toolbar,
        option_widgets=None,
        save_callback=None,
        save_tooltip="Save",
        toolbar_width=340,
    ):
        return make_matplotlib_toolbar_block(self, title, toolbar, option_widgets=option_widgets, save_callback=save_callback, save_tooltip=save_tooltip, toolbar_width=toolbar_width)

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(*PAGE_MARGINS)
        main_layout.setSpacing(BLOCK_SPACING)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(BLOCK_SPACING)
        main_layout.addLayout(content_layout, stretch=1)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(FILE_BROWSER_WIDTH)
        left_scroll.setFrameShape(QFrame.NoFrame)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(BLOCK_SPACING)
        left_scroll.setWidget(left_panel)
        content_layout.addWidget(left_scroll, stretch=0)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        content_layout.addWidget(right_panel, stretch=1)

        curve_scroll = QScrollArea()
        curve_scroll.setWidgetResizable(True)
        curve_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        curve_scroll.setFixedWidth(FILE_BROWSER_WIDTH)
        curve_scroll.setFrameShape(QFrame.NoFrame)

        curve_panel = QWidget()
        curve_panel_layout = QVBoxLayout(curve_panel)
        curve_panel_layout.setContentsMargins(0, 0, 0, 0)
        curve_panel_layout.setSpacing(BLOCK_SPACING)
        curve_scroll.setWidget(curve_panel)
        content_layout.addWidget(curve_scroll, stretch=0)

        file_box = QGroupBox("File browser")
        self.style_top_group_box(file_box)
        file_layout = QVBoxLayout(file_box)
        file_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        file_layout.setSpacing(6)
        file_box.setMinimumHeight(220)
        left_layout.addWidget(file_box, stretch=1)

        self.folder_path = QLineEdit(str(self.current_folder))
        self.folder_path.returnPressed.connect(self.refresh_files)
        file_layout.addWidget(self.folder_path)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_folder)
        file_layout.addWidget(self.browse_button)

        filters_layout = QGridLayout()
        self.extensions_filter = QLineEdit("*.dat")
        self.name_filter = QLineEdit("**")
        self.extensions_filter.textChanged.connect(self.refresh_files)
        self.name_filter.textChanged.connect(self.refresh_files)
        filters_layout.addWidget(QLabel("Name:"), 0, 0)
        filters_layout.addWidget(self.name_filter, 0, 1)
        filters_layout.addWidget(QLabel("Extensions:"), 1, 0)
        filters_layout.addWidget(self.extensions_filter, 1, 1)
        file_layout.addLayout(filters_layout)

        self.show_subfolders = QCheckBox("Show subfolders")
        self.show_subfolders.setChecked(False)
        self.show_subfolders.stateChanged.connect(self.refresh_files)
        self.only_thumbs_up_checkbox = QCheckBox("Only 👍")
        self.only_thumbs_up_checkbox.setChecked(False)
        self.only_thumbs_up_checkbox.stateChanged.connect(self.refresh_files)
        file_options_layout = QHBoxLayout()
        file_options_layout.setContentsMargins(0, 0, 0, 0)
        file_options_layout.addWidget(self.show_subfolders)
        file_options_layout.addWidget(self.only_thumbs_up_checkbox)
        file_options_layout.addStretch(1)
        file_layout.addLayout(file_options_layout)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_files)
        file_layout.addWidget(self.refresh_button)

        self.file_list = QListWidget()
        install_file_rating_menu(self.file_list)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemSelectionChanged.connect(self.selection_changed)
        file_layout.addWidget(self.file_list, stretch=1)

        # Plot settings widgets (previously in settings_box, now just created here)
        self.plot_mode = QComboBox()
        self.plot_mode.addItems(["linear linear", "linear log", "log linear", "log log", "Kratky"])
        self.plot_mode.setCurrentText("log log")
        self.plot_mode.currentTextChanged.connect(self.update_plot)

        self.auto_limits = QCheckBox("Auto limits")
        self.auto_limits.setChecked(True)
        self.auto_limits.stateChanged.connect(self.update_limit_state)

        self.x_min = self.double_spin(0.0)
        self.x_max = self.double_spin(1.0)
        self.y_min = self.double_spin(0.0)
        self.y_max = self.double_spin(1.0)

        self.x_label = QLineEdit("q / nm⁻¹")
        self.y_label = QLineEdit("Intensity / a.u.")
        self.title_edit = QLineEdit("")

        self.x_label.textChanged.connect(self.update_plot)
        self.y_label.textChanged.connect(self.update_plot)
        self.title_edit.textChanged.connect(self.update_plot)

        for spin in [self.x_min, self.x_max, self.y_min, self.y_max]:
            spin.valueChanged.connect(self.update_plot)

        curve_box = QGroupBox("Curves")
        self.style_top_group_box(curve_box)
        curve_layout = QVBoxLayout(curve_box)
        curve_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        curve_layout.setSpacing(6)
        curve_box.setMinimumHeight(170)
        curve_panel_layout.addWidget(curve_box, stretch=1)

        self.curve_table = CurveTableWidget(0, 4)
        self.curve_table.setMinimumHeight(140)
        self.curve_table.setHorizontalHeaderLabels(["File", "Legend", "Color", ""])
        self.curve_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.curve_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.curve_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.curve_table.setColumnWidth(2, 44)
        self.curve_table.setColumnWidth(3, 30)
        self.curve_table.verticalHeader().setVisible(False)
        self.curve_table.verticalHeader().setDefaultSectionSize(28)
        self.curve_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.curve_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.curve_table.setDropIndicatorShown(True)
        self.curve_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.curve_table.setItemDelegateForColumn(2, ColorCellDelegate(self.curve_table))
        self.curve_table.cellChanged.connect(self.curve_table_changed)
        self.curve_table.cellDoubleClicked.connect(self.curve_table_double_clicked)
        self.curve_table.customContextMenuRequested.connect(self.open_curve_table_menu)
        curve_layout.addWidget(self.curve_table, stretch=1)

        mask_buttons_layout = QHBoxLayout()
        mask_buttons_layout.setContentsMargins(0, 0, 0, 0)
        mask_buttons_layout.setSpacing(4)
        self.mask_range_button = QPushButton("Mask range...")
        self.mask_range_button.setToolTip("Mask a q or psi range in selected curves, or all curves if none is selected.")
        self.mask_range_button.clicked.connect(self.open_mask_range_dialog)
        self.mask_range_button.setEnabled(False)
        self.reset_masks_button = QPushButton("Reset masks")
        self.reset_masks_button.setToolTip("Restore original data for selected curves, or all curves if none is selected.")
        self.reset_masks_button.clicked.connect(self.reset_curve_masks)
        self.reset_masks_button.setEnabled(False)
        mask_buttons_layout.addWidget(self.mask_range_button)
        mask_buttons_layout.addWidget(self.reset_masks_button)
        curve_layout.addLayout(mask_buttons_layout)

        guide_box = QGroupBox("Dashed bars")
        self.style_top_group_box(guide_box)
        guide_layout = QVBoxLayout(guide_box)
        guide_layout.setContentsMargins(*GROUP_BOX_MARGINS)
        guide_layout.setSpacing(6)
        curve_panel_layout.addWidget(guide_box, stretch=0)

        self.guide_table = QTableWidget(0, 4)
        self.guide_table.setMinimumHeight(96)
        self.guide_table.setMaximumHeight(150)
        self.guide_table.setHorizontalHeaderLabels(["Axis", "Value", "Color", ""])
        self.guide_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.guide_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.guide_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.guide_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.guide_table.setColumnWidth(0, 42)
        self.guide_table.setColumnWidth(2, 44)
        self.guide_table.setColumnWidth(3, 30)
        self.guide_table.verticalHeader().setVisible(False)
        self.guide_table.verticalHeader().setDefaultSectionSize(26)
        self.guide_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.guide_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.guide_table.setItemDelegateForColumn(2, ColorCellDelegate(self.guide_table))
        self.guide_table.cellChanged.connect(self.guide_table_changed)
        self.guide_table.cellDoubleClicked.connect(self.guide_table_double_clicked)
        guide_layout.addWidget(self.guide_table)

        guide_buttons_layout = QHBoxLayout()
        guide_buttons_layout.setContentsMargins(0, 0, 0, 0)
        guide_buttons_layout.setSpacing(4)
        self.add_x_bar_button = QPushButton("+ X")
        self.add_x_bar_button.setToolTip("Add a vertical dashed bar")
        self.add_x_bar_button.clicked.connect(lambda: self.add_guide_bar("x"))
        self.add_y_bar_button = QPushButton("+ Y")
        self.add_y_bar_button.setToolTip("Add a horizontal dashed bar")
        self.add_y_bar_button.clicked.connect(lambda: self.add_guide_bar("y"))
        guide_buttons_layout.addWidget(self.add_x_bar_button)
        guide_buttons_layout.addWidget(self.add_y_bar_button)
        guide_layout.addLayout(guide_buttons_layout)

        self.clear_header_button = QPushButton("−", self.curve_table.horizontalHeader())
        self.clear_header_button.setFixedSize(22, 18)
        self.clear_header_button.setToolTip("Clear all curves")
        self.clear_header_button.clicked.connect(self.clear_curves)
        self.clear_header_button.setStyleSheet("""
            QPushButton {
                background: #ffecec;
                color: #b00020;
                border: 1px solid #ffb3b3;
                border-radius: 8px;
                font-weight: bold;
                font-size: 11px;
                padding: 0px;
            }
            QPushButton:hover {
                background: #ffd6d6;
            }
        """)
        self.curve_table.horizontalHeader().sectionResized.connect(self.update_clear_header_button_position)
        self.update_clear_header_button_position()


        self.canvas = PlotCanvas()
        self.canvas.setContentsMargins(0, 0, 0, 0)
        clear_plot_canvas(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.plot_mode.setFixedWidth(120)

        self.show_legend = QCheckBox("Legend")
        self.show_legend.setChecked(True)
        self.show_legend.stateChanged.connect(self.update_plot)

        graph_box, self.toolbar_extra_layout, self.save_plot_button = self.create_matplotlib_toolbar_block(
            title="Plot",
            toolbar=self.toolbar,
            option_widgets=[
                self.plot_mode,
                self.show_legend,
            ],
            save_callback=self.save_plot_high_quality,
            save_tooltip="Save plot",
            toolbar_width=320,
        )
        right_layout.addWidget(graph_box, stretch=0)
        right_layout.addWidget(self.canvas, stretch=1)

        self.graph_coordinate_label = QLabel("q = - | I = -")
        self.graph_coordinate_label.setMinimumHeight(28)
        self.graph_coordinate_label.setAlignment(Qt.AlignCenter)
        self.graph_coordinate_label.setStyleSheet("""
            QLabel {
                background-color: #f4f4f4;
                border-radius: 8px;
                padding: 6px;
                font-family: Menlo, Monaco, monospace;
                font-size: 11px;
            }
        """)
        right_layout.addWidget(self.graph_coordinate_label, stretch=0)
        self.update_graph_toolbar_enabled()

        self.canvas.mpl_connect("motion_notify_event", self.update_graph_coordinates)
        self.canvas.mpl_connect("button_press_event", self.graph_button_press)
        self.canvas.mpl_connect("axes_leave_event", self.clear_graph_coordinates)

        frame_nav = QHBoxLayout()
        frame_nav.setContentsMargins(0, 0, 0, 0)
        frame_nav.setSpacing(FRAME_NAV_SPACING)

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

        self.frame_counter_label = QLabel("1 / 1")
        self.frame_counter_label.setMinimumWidth(FRAME_COUNTER_WIDTH)
        self.frame_counter_label.setAlignment(Qt.AlignCenter)

        frame_nav.addWidget(QLabel("Start:"))
        frame_nav.addWidget(self.frame_start_spin)
        frame_nav.addWidget(self.prev_frame_button)
        frame_nav.addWidget(self.frame_slider, stretch=1)
        frame_nav.addWidget(self.next_frame_button)
        frame_nav.addWidget(QLabel("End:"))
        frame_nav.addWidget(self.frame_end_spin)
        frame_nav.addWidget(self.frame_counter_label)
        main_layout.addLayout(frame_nav, stretch=0)

        for widget in [
            self.frame_start_spin, self.frame_end_spin, self.prev_frame_button,
            self.next_frame_button, self.frame_slider,
        ]:
            widget.setEnabled(False)

        self.update_limit_state()

    def style_top_group_box(self, box):
        box.setStyleSheet(GROUP_BOX_STYLE)

    def double_spin(self, value):
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(-1e12, 1e12)
        spin.setValue(value)
        spin.setFixedHeight(24)
        spin.setFixedWidth(90)
        return spin

    def save_plot_high_quality(self):
        if not self.curves:
            return

        default_name = "plot_1d.png"
        if self.curves:
            first_curve = next(iter(self.curves.values()))
            default_name = f"{first_curve['path'].stem}_plot_1d.png"

        start_path = str(self.current_folder / default_name)
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save plot",
            start_path,
            "PNG image (*.png);;TIFF image (*.tif *.tiff);;PDF vector (*.pdf);;SVG vector (*.svg)",
        )

        if not file_path:
            return

        suffix = Path(file_path).suffix.lower()
        if not suffix:
            if "TIFF" in selected_filter:
                file_path += ".tif"
                suffix = ".tif"
            elif "PDF" in selected_filter:
                file_path += ".pdf"
                suffix = ".pdf"
            elif "SVG" in selected_filter:
                file_path += ".svg"
                suffix = ".svg"
            else:
                file_path += ".png"
                suffix = ".png"

        save_kwargs = {
            "bbox_inches": "tight",
            "pad_inches": 0.04,
            "facecolor": "white",
        }

        if suffix in [".png", ".tif", ".tiff"]:
            save_kwargs["dpi"] = 600
        else:
            save_kwargs["dpi"] = 300

        try:
            self.canvas.fig.savefig(file_path, **save_kwargs)
        except Exception as error:
            QMessageBox.warning(self, "Save plot error", f"Could not save plot:\n\n{error}")

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder", str(self.current_folder))
        if folder:
            self.current_folder = Path(folder)
            self.folder_path.setText(str(self.current_folder))
            self.refresh_files()

    def set_folder_from_external_tab(self, folder):
        folder = Path(folder).expanduser().resolve()
        if self.current_folder.expanduser().resolve() == folder:
            return
        self._syncing_folder = True
        self.current_folder = folder
        self.folder_path.setText(str(self.current_folder))
        self.refresh_files()
        self._syncing_folder = False

    def refresh_files(self):
        folder = Path(self.folder_path.text()).expanduser()
        if not folder.exists():
            return

        self.current_folder = folder
        if not self._syncing_folder:
            self.folder_changed.emit(self.current_folder)

        patterns = self.extensions_filter.text().split()
        if not patterns:
            patterns = ["*.dat"]

        name_filter = self.name_filter.text().strip()
        if not name_filter:
            name_filter = "**"

        from fnmatch import fnmatch
        files = []
        glob_method = folder.rglob if self.show_subfolders.isChecked() else folder.glob

        for pattern in patterns:
            files.extend(glob_method(pattern))

        files = sorted(set(files))
        files = [file for file in files if file.is_file() and fnmatch(file.name, name_filter)]
        if self.only_thumbs_up_checkbox.isChecked():
            files = [file for file in files if is_file_rated_up(file)]

        self.file_list.blockSignals(True)
        self.file_list.clear()
        for file in files:
            display_name = str(file.relative_to(folder)) if self.show_subfolders.isChecked() else file.name
            self.file_list.addItem(display_name)
            item = self.file_list.item(self.file_list.count() - 1)
            set_item_file_path(item, file)
        self.file_list.blockSignals(False)

    def selection_changed(self):
        selected = self.selected_files()
        if not selected:
            self.update_graph_toolbar_enabled()
            return

        for file_path in selected:
            key = file_path.name
            if key in self.curves:
                continue

            try:
                x, y = read_dat_curve(file_path)
            except Exception as error:
                QMessageBox.warning(self, "File reading error", f"{file_path.name}\n\n{error}")
                continue

            index = len(self.curves)
            self.curves[key] = {
                "path": file_path,
                "x": x,
                "y": y,
                "original_y": y.copy(),
                "legend": self.saved_legend_for_file(file_path) or file_path.stem,
                "color": default_color(index),
            }

        self.refresh_curve_table()
        self.apply_default_plot_mode()
        self.update_plot()

    def selected_files(self):
        files = []
        for item in self.file_list.selectedItems():
            stored_path = item.data(Qt.UserRole)
            files.append(Path(stored_path) if stored_path else self.current_folder / item.text())
        return files

    def update_clear_header_button_position(self):
        header = self.curve_table.horizontalHeader()
        column = 3
        x = header.sectionViewportPosition(column)
        width = header.sectionSize(column)
        y = max(0, (header.height() - self.clear_header_button.height()) // 2)
        self.clear_header_button.move(
            x + max(0, (width - self.clear_header_button.width()) // 2),
            y,
        )
        self.clear_header_button.raise_()

    def refresh_curve_table(self):
        self._refreshing_curve_table = True
        self.curve_table.blockSignals(True)
        self.curve_table.setRowCount(0)
        self.update_clear_header_button_position()

        for row, (key, curve) in enumerate(self.curves.items()):
            self.curve_table.insertRow(row)

            file_item = QTableWidgetItem(key)
            file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
            file_item.setToolTip(str(curve["path"].name))
            self.curve_table.setItem(row, 0, file_item)

            legend_item = QTableWidgetItem(curve["legend"])
            legend_item.setToolTip(str(curve["path"].name))
            self.curve_table.setItem(row, 1, legend_item)

            color_item = QTableWidgetItem("")
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            color_item.setBackground(QColor(curve["color"]))
            color_item.setToolTip(curve["color"])
            self.curve_table.setItem(row, 2, color_item)

            remove_button = QPushButton("−")
            remove_button.setFixedSize(22, 18)
            remove_button.setToolTip("Remove this curve from the plot")
            remove_button.setStyleSheet("""
                QPushButton {
                    background: #ffecec;
                    color: #b00020;
                    border: 1px solid #ffb3b3;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 11px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #ffd6d6;
                }
            """)
            remove_button.clicked.connect(lambda checked=False, curve_key=key: self.remove_curve(curve_key))

            remove_holder = QWidget()
            remove_layout = QHBoxLayout(remove_holder)
            remove_layout.setContentsMargins(0, 0, 0, 0)
            remove_layout.setSpacing(0)
            remove_layout.addWidget(remove_button, alignment=Qt.AlignCenter)
            self.curve_table.setCellWidget(row, 3, remove_holder)

        self.curve_table.blockSignals(False)
        self._refreshing_curve_table = False

    def curve_table_changed(self, row, column):
        if self._refreshing_curve_table:
            return
        file_item = self.curve_table.item(row, 0)
        if file_item is None:
            return

        key = file_item.text()
        if key not in self.curves:
            return

        if column == 1:
            item = self.curve_table.item(row, column)
            self.curves[key]["legend"] = item.text() if item else self.curves[key]["legend"]
            self.remember_legend_for_file(self.curves[key]["path"], self.curves[key]["legend"])
            self.update_plot_legend_only()
            return
        elif column == 2:
            item = self.curve_table.item(row, column)
            if item:
                self.curves[key]["color"] = item.text()
                self.update_curve_color_only(key)
                return

        self.update_plot()

    def update_plot_legend_only(self):
        ax = self.canvas.ax
        previous_xlim = ax.get_xlim()
        previous_ylim = ax.get_ylim()
        previous_xscale = ax.get_xscale()
        previous_yscale = ax.get_yscale()

        for key, curve in self.curves.items():
            first = True
            for line in ax.lines:
                if line.get_gid() != key:
                    continue
                line.set_label(curve["legend"] if first else "_nolegend_")
                first = False

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

        if self.show_legend.isChecked() and self.curves:
            make_plot_legend(ax)

        ax.set_xscale(previous_xscale)
        ax.set_yscale(previous_yscale)
        ax.set_xlim(previous_xlim)
        ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def update_curve_color_only(self, key):
        if key not in self.curves:
            return

        ax = self.canvas.ax
        previous_xlim = ax.get_xlim()
        previous_ylim = ax.get_ylim()
        previous_xscale = ax.get_xscale()
        previous_yscale = ax.get_yscale()

        for line in ax.lines:
            if line.get_gid() == key:
                line.set_color(self.curves[key]["color"])

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if self.show_legend.isChecked() and self.curves:
            make_plot_legend(ax)

        ax.set_xscale(previous_xscale)
        ax.set_yscale(previous_yscale)
        ax.set_xlim(previous_xlim)
        ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def update_plot_preserving_view(self):
        ax = self.canvas.ax
        previous_xlim = ax.get_xlim()
        previous_ylim = ax.get_ylim()
        previous_xscale = ax.get_xscale()
        previous_yscale = ax.get_yscale()

        self.update_plot()

        self.canvas.ax.set_xscale(previous_xscale)
        self.canvas.ax.set_yscale(previous_yscale)
        self.canvas.ax.set_xlim(previous_xlim)
        self.canvas.ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def default_guide_value(self, axis):
        ax = self.canvas.ax
        limits = ax.get_xlim() if axis == "x" else ax.get_ylim()
        scale = ax.get_xscale() if axis == "x" else ax.get_yscale()
        left, right = limits
        if scale == "log" and left > 0 and right > 0:
            return float(np.sqrt(left * right))
        return float((left + right) / 2)

    def add_guide_bar(self, axis):
        self.guide_bars.append({
            "axis": axis,
            "value": self.default_guide_value(axis),
            "color": "#444444",
        })
        self.refresh_guide_table()
        self.update_plot_preserving_view()

    def refresh_guide_table(self):
        self._refreshing_guide_table = True
        self.guide_table.blockSignals(True)
        self.guide_table.setRowCount(0)

        for row, bar in enumerate(self.guide_bars):
            self.guide_table.insertRow(row)

            axis_item = QTableWidgetItem(bar["axis"].upper())
            axis_item.setFlags(axis_item.flags() & ~Qt.ItemIsEditable)
            self.guide_table.setItem(row, 0, axis_item)

            value_item = QTableWidgetItem(f"{bar['value']:.6g}")
            self.guide_table.setItem(row, 1, value_item)

            color_item = QTableWidgetItem("")
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            color_item.setBackground(QColor(bar["color"]))
            color_item.setToolTip(bar["color"])
            self.guide_table.setItem(row, 2, color_item)

            remove_button = QPushButton("−")
            remove_button.setFixedSize(22, 18)
            remove_button.setToolTip("Remove this dashed bar")
            remove_button.setStyleSheet("""
                QPushButton {
                    background: #ffecec;
                    color: #b00020;
                    border: 1px solid #ffb3b3;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 11px;
                    padding: 0px;
                }
                QPushButton:hover {
                    background: #ffd6d6;
                }
            """)
            remove_button.clicked.connect(lambda checked=False, bar_row=row: self.remove_guide_bar(bar_row))

            remove_holder = QWidget()
            remove_layout = QHBoxLayout(remove_holder)
            remove_layout.setContentsMargins(0, 0, 0, 0)
            remove_layout.setSpacing(0)
            remove_layout.addWidget(remove_button, alignment=Qt.AlignCenter)
            self.guide_table.setCellWidget(row, 3, remove_holder)

        self.guide_table.blockSignals(False)
        self._refreshing_guide_table = False

    def guide_table_changed(self, row, column):
        if self._refreshing_guide_table or column != 1:
            return
        if not 0 <= row < len(self.guide_bars):
            return

        item = self.guide_table.item(row, column)
        if item is None:
            return

        text = item.text().strip().replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            self.refresh_guide_table()
            return

        self.guide_bars[row]["value"] = value
        self.update_plot_preserving_view()

    def guide_table_double_clicked(self, row, column):
        if column != 2 or not 0 <= row < len(self.guide_bars):
            return

        color = QColorDialog.getColor(QColor(self.guide_bars[row]["color"]), self, "Choose dashed bar color")
        if not color.isValid():
            return

        self.guide_bars[row]["color"] = color.name()
        self.refresh_guide_table()
        self.update_plot_preserving_view()

    def remove_guide_bar(self, row):
        if not 0 <= row < len(self.guide_bars):
            return

        del self.guide_bars[row]
        self.refresh_guide_table()
        self.update_plot_preserving_view()

    def draw_guide_bars(self, ax):
        for bar in self.guide_bars:
            value = float(bar.get("value", 0.0))
            if not np.isfinite(value):
                continue

            color = bar.get("color", "#444444")
            axis = bar.get("axis", "x")
            if axis == "x":
                line = ax.axvline(value, color=color, linestyle="--", linewidth=1.2, alpha=0.9, label="_nolegend_")
            else:
                line = ax.axhline(value, color=color, linestyle="--", linewidth=1.2, alpha=0.9, label="_nolegend_")
            line.set_gid("guide_bar")

    def plot_curve_segments(self, ax, key, curve, x, y, mode):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)

        if mode in ("log linear", "log log"):
            valid &= x > 0
        if mode in ("linear log", "log log"):
            valid &= y > 0

        ranges = []
        start = None
        for index, is_valid in enumerate(valid):
            if is_valid and start is None:
                start = index
            elif not is_valid and start is not None:
                ranges.append((start, index))
                start = None

        if start is not None:
            ranges.append((start, len(valid)))

        if not ranges:
            line, = ax.plot([], [], linewidth=1.6, label=curve["legend"], color=curve["color"])
            line.set_gid(key)
            return

        for segment_index, (start, end) in enumerate(ranges):
            line, = ax.plot(
                x[start:end],
                y[start:end],
                linewidth=1.6,
                label=curve["legend"] if segment_index == 0 else "_nolegend_",
                color=curve["color"],
                antialiased=True,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            line.set_gid(key)

    def move_curve_row(self, source_row, destination_row):
        previous_xlim = self.canvas.ax.get_xlim()
        previous_ylim = self.canvas.ax.get_ylim()
        previous_xscale = self.canvas.ax.get_xscale()
        previous_yscale = self.canvas.ax.get_yscale()

        keys = list(self.curves.keys())
        if not 0 <= source_row < len(keys):
            return

        destination_row = max(0, min(destination_row, len(keys)))
        key = keys.pop(source_row)
        if destination_row > source_row:
            destination_row -= 1
        keys.insert(destination_row, key)

        self.curves = {curve_key: self.curves[curve_key] for curve_key in keys}
        self.refresh_curve_table()
        self.curve_table.selectRow(destination_row)
        self.update_plot()
        self.canvas.ax.set_xscale(previous_xscale)
        self.canvas.ax.set_yscale(previous_yscale)
        self.canvas.ax.set_xlim(previous_xlim)
        self.canvas.ax.set_ylim(previous_ylim)
        finalize_plot_canvas(self.canvas)

    def curve_rows_moved(self, parent, start, end, destination, row):
        """Handle curve reordering when dragged in the table."""
        if self._refreshing_curve_table:
            return

        # Rebuild the curves dict based on the current table order
        reordered_curves = {}
        
        # Iterate through all rows in the table and preserve their order
        for table_row in range(self.curve_table.rowCount()):
            file_item = self.curve_table.item(table_row, 0)
            
            # Skip if item is None
            if file_item is None:
                continue
            
            key = file_item.text()
            
            # Only add keys that actually exist in our curves dict
            if key in self.curves:
                reordered_curves[key] = self.curves[key]
        
        # Update the curves dict
        self.curves = reordered_curves
        
        # Clear selection to fix visual glitches
        self.curve_table.clearSelection()
        
        # Update the plot
        self.update_plot()

    def curve_table_double_clicked(self, row, column):
        if column != 2:
            return

        file_item = self.curve_table.item(row, 0)
        if file_item is None:
            return

        key = file_item.text()
        if key not in self.curves:
            return

        color = QColorDialog.getColor(QColor(self.curves[key]["color"]), self, "Choose curve color")
        if not color.isValid():
            return

        self.curves[key]["color"] = color.name()
        self.refresh_curve_table()
        self.update_curve_color_only(key)

    def remove_curve(self, key):
        if key in self.curves:
            del self.curves[key]
        self.refresh_curve_table()
        self.update_plot()

    def clear_curves(self):
        self.curves.clear()
        self.refresh_curve_table()
        clear_plot_canvas(self.canvas)
        self.clear_graph_coordinates()
        self.update_graph_toolbar_enabled()

    def selected_curve_keys(self):
        keys = []
        for index in self.curve_table.selectionModel().selectedRows():
            item = self.curve_table.item(index.row(), 0)
            if item is not None and item.text() in self.curves:
                keys.append(item.text())
        return keys

    def open_curve_table_menu(self, position):
        item = self.curve_table.itemAt(position)
        if item is None:
            return

        row = item.row()
        key_item = self.curve_table.item(row, 0)
        if key_item is None:
            return

        curve_key = key_item.text()
        if curve_key not in self.curves:
            return

        self.curve_table.selectRow(row)
        legend = self.curves[curve_key]["legend"]

        menu = QMenu(self.curve_table)
        mask_action = menu.addAction(f"Mask range on {legend}")
        reset_action = menu.addAction(f"Reset mask on {legend}")

        action = menu.exec(self.curve_table.viewport().mapToGlobal(position))
        if action is mask_action:
            self.open_mask_range_dialog(curve_key=curve_key)
        elif action is reset_action:
            curve = self.curves[curve_key]
            if "original_y" in curve:
                curve["y"] = np.asarray(curve["original_y"], dtype=float).copy()
                self.update_plot()

    def open_mask_range_dialog(self, curve_key=None, center_x=None):
        if not self.curves:
            return

        x_label, _ = self.graph_coordinate_labels()
        dialog = QDialog(self)
        dialog.setWindowTitle("Mask data range")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        if curve_key is not None and curve_key in self.curves:
            target_text = f"Target: {self.curves[curve_key]['legend']}"
        else:
            selected_count = len(self.selected_curve_keys())
            target_text = f"Target: {selected_count} selected curve(s)" if selected_count else "Target: all curves"
        target_label = QLabel(target_text)
        layout.addWidget(target_label)

        range_layout = QGridLayout()
        xlim = self.canvas.ax.get_xlim()
        if center_x is not None and np.isfinite(center_x):
            span = abs(xlim[1] - xlim[0])
            half_width = span * 0.01
            if self.canvas.ax.get_xscale() == "log" and center_x > 0:
                factor = 10 ** 0.01
                default_min = center_x / factor
                default_max = center_x * factor
            else:
                default_min = center_x - half_width
                default_max = center_x + half_width
        else:
            default_min, default_max = xlim

        min_spin = self.double_spin(default_min)
        max_spin = self.double_spin(default_max)
        min_spin.setFixedWidth(130)
        max_spin.setFixedWidth(130)
        range_layout.addWidget(QLabel(f"{x_label} min:"), 0, 0)
        range_layout.addWidget(min_spin, 0, 1)
        range_layout.addWidget(QLabel(f"{x_label} max:"), 1, 0)
        range_layout.addWidget(max_spin, 1, 1)
        layout.addLayout(range_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() != QDialog.Accepted:
            return

        self.mask_data_range(min_spin.value(), max_spin.value(), curve_key=curve_key)

    def mask_data_range(self, x_min, x_max, curve_key=None):
        if not self.curves:
            return

        if x_max < x_min:
            x_min, x_max = x_max, x_min

        keys = [curve_key] if curve_key in self.curves else (self.selected_curve_keys() or list(self.curves.keys()))
        total_masked = 0

        for key in keys:
            curve = self.curves[key]
            x = np.asarray(curve["x"], dtype=float)
            y = np.asarray(curve["y"], dtype=float).copy()
            mask = np.isfinite(x) & (x >= x_min) & (x <= x_max)
            if np.any(mask):
                y[mask] = np.nan
                curve["y"] = y
                total_masked += int(np.count_nonzero(mask))

        if total_masked == 0:
            QMessageBox.information(self, "Mask range", "No point was found in this range.")
            return

        self.update_plot()

    def reset_curve_masks(self):
        if not self.curves:
            return

        keys = self.selected_curve_keys() or list(self.curves.keys())
        for key in keys:
            curve = self.curves[key]
            if "original_y" in curve:
                curve["y"] = np.asarray(curve["original_y"], dtype=float).copy()

        self.update_plot()

    def nearest_curve_key_at(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return None

        click_display = self.canvas.ax.transData.transform((event.xdata, event.ydata))
        best_key = None
        best_distance = float("inf")

        for key, curve in self.curves.items():
            x = np.asarray(curve["x"], dtype=float)
            y = np.asarray(self.make_plot_y(x, curve["y"]), dtype=float)
            valid = np.isfinite(x) & np.isfinite(y)
            if self.canvas.ax.get_xscale() == "log":
                valid &= x > 0
            if self.canvas.ax.get_yscale() == "log":
                valid &= y > 0
            if not np.any(valid):
                continue

            points = self.canvas.ax.transData.transform(np.column_stack((x[valid], y[valid])))
            distances = np.hypot(points[:, 0] - click_display[0], points[:, 1] - click_display[1])
            distance = float(np.nanmin(distances))
            if distance < best_distance:
                best_distance = distance
                best_key = key

        return best_key if best_distance <= 18 else None

    def graph_button_press(self, event):
        if event.button != 3:
            return

        curve_key = self.nearest_curve_key_at(event)
        if curve_key is None:
            return

        menu = QMenu(self)
        legend = self.curves[curve_key]["legend"]
        mask_action = menu.addAction(f"Mask range on {legend}")
        reset_action = menu.addAction(f"Reset mask on {legend}")

        try:
            global_pos = event.guiEvent.globalPosition().toPoint()
        except Exception:
            global_pos = self.canvas.mapToGlobal(self.canvas.rect().center())

        action = menu.exec(global_pos)
        if action is mask_action:
            self.open_mask_range_dialog(curve_key=curve_key, center_x=event.xdata)
        elif action is reset_action:
            curve = self.curves[curve_key]
            if "original_y" in curve:
                curve["y"] = np.asarray(curve["original_y"], dtype=float).copy()
                self.update_plot()

    def update_graph_toolbar_enabled(self):
        enabled = bool(self.curves)
        for widget in [
            getattr(self, "plot_mode", None),
            getattr(self, "show_legend", None),
            getattr(self, "save_plot_button", None),
            getattr(self, "mask_range_button", None),
            getattr(self, "reset_masks_button", None),
        ]:
            if widget is not None:
                widget.setEnabled(enabled)

    def make_plot_y(self, x, y):
        if self.plot_mode.currentText() == "Kratky":
            return x ** 2 * y
        return y

    def graph_coordinate_labels(self):
        if self.curves_are_really_0_to_360():
            return "ψ", "I"
        if self.plot_mode.currentText() == "Kratky":
            return "q", "q²I(q)"
        return "q", "I"

    def update_graph_coordinates(self, event):
        if event.inaxes != self.canvas.ax or event.xdata is None or event.ydata is None:
            return

        try:
            x_name, y_name = self.graph_coordinate_labels()
            x_suffix = "°" if x_name == "ψ" else ""
            self.graph_coordinate_label.setText(
                f"{x_name} = {event.xdata:.6g}{x_suffix} | {y_name} = {event.ydata:.6g}"
            )
        except Exception:
            self.clear_graph_coordinates()

    def clear_graph_coordinates(self, event=None):
        x_name, y_name = self.graph_coordinate_labels()
        self.graph_coordinate_label.setText(f"{x_name} = - | {y_name} = -")

    def curves_are_really_0_to_360(self):
        if not self.curves:
            return False

        if any("azimprof" in curve["path"].name.lower() for curve in self.curves.values()):
            return True

        for curve in self.curves.values():
            x = curve["x"]
            valid = x[np.isfinite(x)]
            if valid.size == 0:
                return False

            x_min = float(np.nanmin(valid))
            x_max = float(np.nanmax(valid))
            if not (abs(x_min - 0.0) <= 1e-6 and abs(x_max - 360.0) <= 1e-6):
                return False

        return True

    def apply_default_plot_mode(self):
        mode = "linear linear" if self.curves_are_really_0_to_360() else "log log"

        if self.plot_mode.currentText() == mode:
            return

        self.plot_mode.blockSignals(True)
        self.plot_mode.setCurrentText(mode)
        self.plot_mode.blockSignals(False)

    def update_limit_state(self):
        auto = self.auto_limits.isChecked()

        if auto:
            self.update_limit_fields_from_current_data()

        for widget in [self.x_min, self.x_max, self.y_min, self.y_max]:
            widget.setEnabled(not auto)

        self.update_plot()

    def update_limit_fields_from_current_data(self):
        if not self.curves:
            return

        all_x = []
        all_y = []

        for curve in self.curves.values():
            x = curve["x"]
            y = self.make_plot_y(x, curve["y"])

            valid = np.isfinite(x) & np.isfinite(y)
            if np.any(valid):
                all_x.append(x[valid])
                all_y.append(y[valid])

        if not all_x or not all_y:
            return

        x_values = np.concatenate(all_x)
        has_azim_profile = self.curves_are_really_0_to_360()
        y_values = np.concatenate(all_y)

        mode = self.plot_mode.currentText()
        if mode in ["log linear", "log log"]:
            x_values = x_values[x_values > 0]
        if mode in ["linear log", "log log"]:
            y_values = y_values[y_values > 0]

        if x_values.size == 0 or y_values.size == 0:
            return

        if has_azim_profile:
            x_min = 0.0
            x_max = 360.0
        else:
            x_min = float(np.nanmin(x_values))
            x_max = float(np.nanmax(x_values))
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))

        if x_max == x_min:
            x_max = x_min + 1
        if y_max == y_min:
            y_max = y_min + 1

        for spin, value in [
            (self.x_min, x_min),
            (self.x_max, x_max),
            (self.y_min, y_min),
            (self.y_max, y_max),
        ]:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

    def update_plot(self):
        ax = self.canvas.ax
        ax.clear()

        if not self.curves:
            self.clear_graph_coordinates()
            clear_plot_canvas(self.canvas)
            self.update_graph_toolbar_enabled()
            return

        ax.set_axis_on()
        self.update_graph_toolbar_enabled()

        mode = self.plot_mode.currentText()
        if self.auto_limits.isChecked():
            self.update_limit_fields_from_current_data()

        for key, curve in self.curves.items():
            x = curve["x"]
            y = self.make_plot_y(x, curve["y"])
            self.plot_curve_segments(ax, key, curve, x, y, mode)

        self.draw_guide_bars(ax)

        if mode == "linear linear" or mode == "Kratky":
            ax.set_xscale("linear")
            ax.set_yscale("linear")
        elif mode == "linear log":
            ax.set_xscale("linear")
            ax.set_yscale("log")
        elif mode == "log linear":
            ax.set_xscale("log")
            ax.set_yscale("linear")
        elif mode == "log log":
            ax.set_xscale("log")
            ax.set_yscale("log")

        has_azim_profile = self.curves_are_really_0_to_360()

        if has_azim_profile:
            default_x_label = "ψ / °"
            ax.set_xlim(0, 360)
            ax.set_xlabel(default_x_label)
        else:
            default_x_label = "q / nm⁻¹"
            ax.set_xlabel(self.x_label.text() or default_x_label)
        ax.set_ylabel("q²I(q)" if mode == "Kratky" else (self.y_label.text() or "Intensity / a.u."))
        ax.set_title(self.title_edit.text())
        apply_plot_display_style(ax)
        if self.show_legend.isChecked():
            make_plot_legend(ax)

        if not self.auto_limits.isChecked():
            if self.x_max.value() > self.x_min.value():
                ax.set_xlim(self.x_min.value(), self.x_max.value())
            if self.y_max.value() > self.y_min.value():
                ax.set_ylim(self.y_min.value(), self.y_max.value())

        finalize_plot_canvas(self.canvas)
