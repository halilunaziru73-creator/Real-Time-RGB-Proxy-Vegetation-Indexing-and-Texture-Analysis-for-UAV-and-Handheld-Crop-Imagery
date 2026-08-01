"""QSS stylesheet giving the app a professional, QGIS-like desktop-GIS appearance."""

QGIS_STYLE = """
* {
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
    font-size: 12px;
}

QMainWindow {
    background: #eef1ee;
}

QMenuBar {
    background: #2a4a2c;
    color: #ffffff;
    padding: 3px;
    font-weight: 500;
}
QMenuBar::item {
    background: transparent;
    padding: 5px 12px;
    border-radius: 3px;
}
QMenuBar::item:selected {
    background: #3f6b40;
}
QMenu {
    background: #ffffff;
    border: 1px solid #b9c7b6;
    padding: 3px;
}
QMenu::item {
    padding: 5px 20px;
    border-radius: 2px;
}
QMenu::item:selected {
    background: #cfe3cd;
    color: #1f2d1e;
}

QToolBar {
    background: #35603a;
    border: none;
    spacing: 6px;
    padding: 6px;
}
QToolButton {
    color: #ffffff;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 500;
}
QToolButton:hover {
    background: #4a7a4d;
    border: 1px solid #6a994e;
}
QToolButton:pressed {
    background: #24421f;
}
QToolButton:disabled {
    color: #a9c2ab;
}

QDockWidget {
    color: #1f2d1e;
    font-weight: 600;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background: #d7e4d3;
    padding: 7px 8px;
    border-bottom: 2px solid #6a994e;
}

QTreeWidget, QTableWidget, QListWidget {
    background: #ffffff;
    alternate-background-color: #f4f8f2;
    border: 1px solid #c3d0bf;
    border-radius: 3px;
    gridline-color: #e1e9de;
    selection-background-color: #cfe3cd;
    selection-color: #1f2d1e;
}
QTreeWidget::item, QTableWidget::item, QListWidget::item {
    padding: 3px;
}
QHeaderView::section {
    background: #dfe9db;
    color: #1f2d1e;
    padding: 5px;
    border: none;
    border-right: 1px solid #c3d0bf;
    border-bottom: 1px solid #c3d0bf;
    font-weight: 600;
}

QTabWidget::pane {
    border: 1px solid #c3d0bf;
    background: #ffffff;
    border-radius: 3px;
}
QTabBar::tab {
    background: #dfe9db;
    color: #3a4a38;
    padding: 7px 16px;
    border: 1px solid #c3d0bf;
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 1px;
}
QTabBar::tab:selected {
    background: #ffffff;
    color: #1f2d1e;
    font-weight: 600;
    border-bottom: 2px solid #6a994e;
}
QTabBar::tab:hover:!selected {
    background: #ecf3ea;
}

QTextEdit {
    background: #101613;
    color: #b8f2b0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
    border: 1px solid #c3d0bf;
    border-radius: 3px;
}

QPushButton {
    background: #6a994e;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background: #578140;
}
QPushButton:pressed {
    background: #476b34;
}
QPushButton:disabled {
    background: #b7c4b3;
    color: #eef1ee;
}

QStatusBar {
    background: #d7e4d3;
    color: #2a3a28;
    border-top: 1px solid #b9c7b6;
}

QComboBox {
    background: white;
    border: 1px solid #b9c7b6;
    padding: 4px 6px;
    border-radius: 3px;
    min-height: 20px;
}
QComboBox:hover {
    border: 1px solid #6a994e;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #c3d0bf;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #3b6b40;
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}

QLabel {
    color: #1f2d1e;
}

QScrollBar:vertical {
    background: #eef1ee;
    width: 12px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #b7c4b3;
    border-radius: 5px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #8fa48a;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: #eef1ee;
    height: 12px;
}
QScrollBar::handle:horizontal {
    background: #b7c4b3;
    border-radius: 5px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #8fa48a;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

QProgressDialog {
    background: #eef1ee;
}

QDialog {
    background: #eef1ee;
}
"""
