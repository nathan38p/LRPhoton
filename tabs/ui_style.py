GROUP_BOX_STYLE = """
    QGroupBox {
        background-color: #f4f4f4;
        border: 0px;
        border-radius: 10px;
        margin-top: 14px;
        padding: 4px;
        font-family: Arial;
        font-size: 12px;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        left: 8px;
        padding: 0px 4px;
        color: #222222;
        font-family: Arial;
        font-size: 12px;
    }

    QPushButton {
        background-color: #e2e2e2;
        border: 0px;
        border-radius: 5px;
        padding: 4px;
    }

    QPushButton:hover {
        background-color: #d8d8d8;
    }
"""


TOOL_GROUP_BOX_STYLE = GROUP_BOX_STYLE + """
    QToolBar {
        background-color: #f4f4f4;
        border: 0px;
        spacing: 8px;
    }

    QToolButton {
        background-color: #f4f4f4;
        border: 0px;
        padding: 4px;
    }

    QToolButton:hover {
        background-color: #e5e5e5;
        border-radius: 5px;
    }
"""


GROUP_BOX_MARGINS = (8, 20, 8, 8)
BLOCK_SPACING = 8
PAGE_MARGINS = (4, 4, 4, 4)
PANEL_MARGINS = (0, 0, 0, 0)
FRAME_NAV_SPACING = 6
FRAME_SPIN_WIDTH = 70
FRAME_BUTTON_WIDTH = 44
FRAME_COUNTER_WIDTH = 56
FILE_BROWSER_WIDTH = 320
MATPLOTLIB_TOOLBAR_ICON_SCALE = 0.8
MATPLOTLIB_TOOLBAR_MAX_HEIGHT = 42
