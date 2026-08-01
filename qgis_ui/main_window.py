"""
QGIS-style main window for the Naziru Image Analysis Pipeline.

Core design: every ANALYSIS RESULT (Map Canvas layers, Charts & Reports)
is computed on the AGGREGATE COMPOSITE built by aligning and averaging
every loaded image (up to 2000) -- never on one specific image in
isolation. A lightweight per-image descriptor (8 real numbers) is kept
for every successfully processed image to support cross-image analyses
(PCA, trend, similarity) without holding thousands of full images in
memory. A separate "Raw Preview" tab lets you still browse individual
photos for QC, clearly apart from the measured outputs.
"""
from __future__ import annotations

import os
import traceback

import numpy as np
from PIL import Image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

from PyQt6.QtWidgets import (
    QMainWindow, QToolBar, QFileDialog, QMessageBox, QDockWidget, QTabWidget,
    QScrollArea, QLabel, QWidget, QVBoxLayout, QStatusBar, QApplication,
    QProgressDialog,
)
from PyQt6.QtGui import QAction, QDesktopServices
from PyQt6.QtCore import Qt, QUrl

from core import texture, ml_models, vegetation, batch as batch_mod, indices
from core import spectral, texture_features, embedding_metrics, vegetation_indices
from core import ml_classifier, morphology_profile, pdf_export, synthetic_reference_model
from core.alignment import BatchAligner, ImageRecord
from core.multispectral import load_multispectral, is_multispectral_file, RASTERIO_AVAILABLE, build_capture_from_band_files
from core.raster_layers import REPORT_LAYER_DEFS
from .styles import QGIS_STYLE
from .layer_panel import LayerPanel
from .canvas_widget import MapCanvas
from .xai_panel import XAIPanel
from .attribute_table import AttributeTable
from .log_panel import LogPanel
from .field_data_panel import FieldDataPanel
from .image_manager_panel import ImageManagerPanel, MAX_IMAGES
from .preview_panel import PreviewPanel
from .band_assignment_dialog import BandAssignmentDialog
from .gacl_panel import GACLPanel
from PyQt6.QtWidgets import QDialog

SPATIAL_REPORT_KEYS = (
    "Cross-Crop Alignment", "Geometric Transfer", "Loss Functions", "Faster R-CNN",
    "ResNet101", "Pathology Localization", "Root Architecture", "Colour Histogram",
    "Spectral Reflectance & Band Ratios", "GLCM Texture Features", "Invariance Tests",
    "InfoNCE (self-supervised)", "Morphology & Environment Profile",
    "Synthetic Reference Classifier (DEMO)",
)
BATCH_REPORT_KEYS = (
    "Time-Series Trend (linear)", "Feature-Space Projection (PCA)", "Batch Summary",
    "Spectral Index Availability", "Embedding & Similarity Metrics",
    "Cross-Image Pathology Transfer Recommendation", "Classification / Clustering (automatic)",
)


class QGISMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Naziru Image Analysis Pipeline — GIS Workbench")
        self.resize(1600, 980)
        self.setStyleSheet(QGIS_STYLE)

        self.image_paths: list[str] = []
        self.image_names: list[str] = []
        self.multispectral_paths: set[str] = set()
        self.multispectral_groups: dict[str, dict] = {}

        self.composite = None
        self.records: list[ImageRecord] = []
        self.descriptors: np.ndarray | None = None
        self.included_names: list[str] = []

        self.report_figure: Figure | None = None
        self.charts_canvas: FigureCanvas | None = None
        self._last_export_path: str | None = None

        self._build_docks()
        self._build_central()
        self._build_toolbar()
        self._build_menu()
        self._build_statusbar()

        self.map_canvas.clear()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_docks(self) -> None:
        self.layer_panel = LayerPanel()
        self.layer_panel.layersChanged.connect(self._on_layers_changed)
        self.layers_dock = QDockWidget("Layers", self)
        self.layers_dock.setWidget(self.layer_panel)
        self.layers_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                      QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.layers_dock.setMinimumHeight(180)
        self.layers_dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.layers_dock)

        self.image_manager = ImageManagerPanel()
        self.image_manager.directoryLoaded.connect(self._on_directory_loaded)
        self.image_manager.imagesDeleted.connect(self._on_images_deleted)
        self.image_manager.imageRenamed.connect(self._on_image_renamed)
        self.image_manager.refreshRequested.connect(self._on_refresh_requested)
        self.images_dock = QDockWidget("Image Manager", self)
        self.images_dock.setWidget(self.image_manager)
        self.images_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                      QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.images_dock.setMinimumHeight(180)
        self.images_dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.images_dock)
        # Stacked (not tabbed) with Layers so delete/refresh/rename controls are always
        # visible, rather than hidden behind a tab a user might not notice.
        self.splitDockWidget(self.layers_dock, self.images_dock, Qt.Orientation.Vertical)

        # GACL Measurements dock -- placed as a tab next to Image Manager (i.e. right
        # next to the file-management docks near the top of the window), per request.
        self.gacl_panel = GACLPanel()
        self.gacl_dock = QDockWidget("GACL Measurements", self)
        self.gacl_dock.setWidget(self.gacl_panel)
        self.gacl_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                    QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.gacl_dock.setMinimumHeight(220)
        self.gacl_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.gacl_dock)
        self.tabifyDockWidget(self.images_dock, self.gacl_dock)

        self.xai_panel = XAIPanel()
        self.xai_dock = QDockWidget("XAI & Recommendations", self)
        self.xai_dock.setWidget(self.xai_panel)
        self.xai_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                   QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.xai_dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.xai_dock)

        self.attribute_table = AttributeTable()
        self.log_panel = LogPanel()
        self.field_panel = FieldDataPanel()
        bottom_tabs = QTabWidget()
        bottom_tabs.addTab(self.attribute_table, "Attribute Table")
        bottom_tabs.addTab(self.field_panel, "Field && Sensor Data")
        bottom_tabs.addTab(self.log_panel, "Execution Log")
        self.bottom_dock = QDockWidget("Data && Log", self)
        self.bottom_dock.setWidget(bottom_tabs)
        self.bottom_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable |
                                      QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.bottom_dock.setMinimumHeight(180)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.bottom_dock)

    def _build_central(self) -> None:
        self.central_tabs = QTabWidget()
        self.setCentralWidget(self.central_tabs)

        map_tab = QWidget()
        map_tab_layout = QVBoxLayout(map_tab)
        map_tab_layout.setContentsMargins(0, 0, 0, 0)
        map_tab_layout.setSpacing(0)

        self.map_canvas = MapCanvas()
        map_tab_layout.addWidget(self.map_canvas)
        self.central_tabs.addTab(map_tab, "Map Canvas (aggregate)")

        self.charts_scroll = QScrollArea()
        # IMPORTANT: setWidgetResizable(True) here was the cause of report panels
        # appearing cut off at the top or bottom of the window. With it enabled,
        # Qt forcibly stretches/squishes the matplotlib canvas to exactly match
        # the visible viewport size, ignoring the canvas's own rendered pixel
        # dimensions -- so instead of scrolling to see an oversized report, parts
        # of it were being squeezed out of view entirely. Disabling it restores
        # normal scroll-area behaviour: the canvas keeps its real size, and a
        # scrollbar appears whenever a report is taller/wider than the visible
        # area, so nothing is ever clipped.
        self.charts_scroll.setWidgetResizable(False)
        placeholder = QLabel("Tick items under \u201cReports & Charts\u201d in the Layers panel,\n"
                              "then click \u201cRun Reports\u201d to generate this report.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #666666; font-size: 12px;")
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(placeholder)
        self.charts_scroll.setWidget(container)

        charts_tab = QWidget()
        charts_tab_layout = QVBoxLayout(charts_tab)
        charts_tab_layout.setContentsMargins(0, 0, 0, 0)
        self.charts_toolbar_holder = QWidget()
        self.charts_toolbar_layout = QVBoxLayout(self.charts_toolbar_holder)
        self.charts_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        charts_tab_layout.addWidget(self.charts_toolbar_holder)
        charts_tab_layout.addWidget(self.charts_scroll)
        self.central_tabs.addTab(charts_tab, "Charts && Reports")

        self.preview_panel = PreviewPanel()
        self.central_tabs.addTab(self.preview_panel, "Raw Preview")

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        open_action = QAction("Browse Images", self)
        open_action.triggered.connect(self.select_files)
        toolbar.addAction(open_action)

        uav_action = QAction("Load UAV Multispectral...", self)
        uav_action.triggered.connect(self.select_multispectral_files)
        toolbar.addAction(uav_action)

        toolbar.addSeparator()
        self.build_action = QAction("Align && Build Composite", self)
        self.build_action.setEnabled(False)
        self.build_action.triggered.connect(self.build_composite)
        toolbar.addAction(self.build_action)

        self.run_action = QAction("Run Reports", self)
        self.run_action.setEnabled(False)
        self.run_action.triggered.connect(self.run_analysis)
        toolbar.addAction(self.run_action)

        toolbar.addSeparator()
        save_map_action = QAction("Save Map", self)
        save_map_action.triggered.connect(self.save_map)
        toolbar.addAction(save_map_action)

        save_report_action = QAction("Save Report", self)
        save_report_action.triggered.connect(self.save_report)
        toolbar.addAction(save_report_action)

        export_pdf_action = QAction("Export PDF Report", self)
        export_pdf_action.triggered.connect(self.export_pdf_report)
        toolbar.addAction(export_pdf_action)

        # Dedicated toolbar for showing/hiding (and recovering, if one ever gets
        # squeezed out of view, or hidden behind another dock's tab) the main
        # panels -- always visible at the top, near the file-management actions.
        #
        # IMPORTANT: these are NOT simple checkable toggles. A dock can be
        # "visible" (isVisible() == True) per Qt while still not being the
        # active tab in a tabified stack, so a plain checked/unchecked toggle
        # can silently do nothing when clicked (toggling True->True fires no
        # signal at all) -- exactly the "I clicked it and nothing happened"
        # bug this fixes. Every click below unconditionally shows AND raises
        # the target dock so the user always lands on the panel they asked for.
        panels_toolbar = QToolBar("Panels")
        panels_toolbar.setMovable(False)
        self.addToolBar(panels_toolbar)

        self.panel_toggle_actions = {}
        for dock, label in (
            (self.layers_dock, "Layers"),
            (self.images_dock, "Image Manager"),
            (self.gacl_dock, "GACL Measurements"),
            (self.bottom_dock, "Data && Log"),
            (self.xai_dock, "XAI Recommendations"),
        ):
            action = QAction(label, self)
            action.setCheckable(False)   # plain "show & bring to front" button, not a toggle
            action.triggered.connect(lambda checked=False, d=dock: self._show_and_raise_dock(d))
            panels_toolbar.addAction(action)
            self.panel_toggle_actions[label] = action

        panels_toolbar.addSeparator()
        show_all_action = QAction("Show All Panels", self)
        show_all_action.triggered.connect(self._show_all_panels)
        panels_toolbar.addAction(show_all_action)

        panels_toolbar.addSeparator()
        share_email_action = QAction("Share Results by Email...", self)
        share_email_action.triggered.connect(self.share_results_by_email)
        panels_toolbar.addAction(share_email_action)

    def _show_and_raise_dock(self, dock: QDockWidget) -> None:
        """Unconditionally show and focus a dock, even if it is tabbed behind
        another dock or was otherwise not the currently visible tab. This is
        the fix for panel buttons that appeared to do nothing when clicked."""
        dock.setVisible(True)
        dock.raise_()
        dock.widget().setFocus()

    def _show_all_panels(self) -> None:
        """Recovery action: forces all main panels visible again, in case one
        was ever hidden/closed or squeezed out of the layout."""
        for dock in (self.layers_dock, self.images_dock, self.gacl_dock, self.bottom_dock, self.xai_dock):
            dock.setVisible(True)
            dock.raise_()

    def share_results_by_email(self) -> None:
        """Opens the user's default email client with a pre-filled subject/body
        referencing the most recently exported report. Uses the standard
        mailto: scheme (QDesktopServices) rather than sending mail directly,
        since directly sending mail would require the user's SMTP credentials
        to be entered and stored somewhere in this app -- mailto: works with
        whatever mail client is already configured on the machine, with no
        credentials handled by this application at all."""
        subject = "Agronomic Image Analysis / GACL results"
        body_lines = [
            "Attach your exported report (Save Report / Export PDF Report) to this email before sending.",
            "",
        ]
        if getattr(self, "_last_export_path", None):
            body_lines.insert(0, f"Exported file: {self._last_export_path}")
        body = "%0D%0A".join(body_lines)
        mailto_url = QUrl(f"mailto:?subject={subject.replace(' ', '%20')}&body={body}")
        QDesktopServices.openUrl(mailto_url)

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        file_menu.addAction("Browse Images...", self.select_files)
        file_menu.addAction("Load UAV Multispectral...", self.select_multispectral_files)
        file_menu.addSeparator()
        file_menu.addAction("Save Map As...", self.save_map)
        file_menu.addAction("Save Report As...", self.save_report)
        file_menu.addAction("Export PDF Report...", self.export_pdf_report)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        analysis_menu = menubar.addMenu("&Analysis")
        analysis_menu.addAction("Align && Build Composite", self.build_composite)
        analysis_menu.addAction("Run Reports", self.run_analysis)

        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.panel_toggle_actions["Layers"])
        view_menu.addAction(self.panel_toggle_actions["Image Manager"])
        view_menu.addAction(self.panel_toggle_actions["Data && Log"])
        view_menu.addAction(self.panel_toggle_actions["XAI Recommendations"])
        view_menu.addSeparator()
        view_menu.addAction("Show All Panels", self._show_all_panels)

        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("Open README", self._open_readme)
        help_menu.addAction("About", self._show_about)

    def _build_statusbar(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready. Browse Images (or set a working directory) to begin.")

    # ------------------------------------------------------------------ #
    # Error handling helper
    # ------------------------------------------------------------------ #
    def _report_error(self, context: str, exc: Exception) -> None:
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.log_panel.log(f"ERROR during {context}: {exc}")
        QMessageBox.critical(self, f"Error: {context}", f"{exc}\n\nSee Execution Log for the full traceback.")
        print(detail)

    # ------------------------------------------------------------------ #
    # Image ingestion
    # ------------------------------------------------------------------ #
    def select_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Agronomic Images", "",
            "Images (*.jpg *.jpeg *.png *.webp *.tif *.tiff)",
        )
        if paths:
            self._append_paths(list(paths))

    def select_multispectral_files(self) -> None:
        if not RASTERIO_AVAILABLE:
            QMessageBox.information(
                self, "rasterio not installed",
                "UAV multispectral GeoTIFFs need the 'rasterio' package for best results.\n\n"
                "Install with: pip install rasterio\n\n"
                "Single-band TIFF files can still often be read via Pillow as a fallback.")
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select UAV Multispectral Image(s) (GeoTIFF, or separate band files)", "",
            "GeoTIFF / TIFF (*.tif *.tiff)")
        if not paths:
            return

        if len(paths) == 1:
            # Could be one combined multi-band GeoTIFF -- handled per-file in build_composite.
            self.multispectral_paths.add(paths[0])
            self._append_paths(list(paths))
            return

        # Multiple files selected: very likely separate single-band files (common for
        # MicaSense/DJI-style sensors) -- ask the user to confirm which file is which band
        # rather than silently guessing and risking wrong-but-not-obviously-wrong results.
        dialog = BandAssignmentDialog(list(paths), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        role_to_path = dialog.role_to_path()
        capture = build_capture_from_band_files(role_to_path)
        if not capture.get("available"):
            QMessageBox.warning(self, "Could not build UAV capture", capture.get("note", ""))
            return

        group_id = f"__ms_group_{len(self.multispectral_groups) + 1}__"
        bands_found = [b for b in ("red", "green", "blue", "rededge", "nir", "thermal") if b in capture or b in role_to_path]
        capture["display_name"] = f"UAV Capture {len(self.multispectral_groups) + 1} ({'+'.join(bands_found)})"
        self.multispectral_groups[group_id] = capture
        self.multispectral_paths.add(group_id)
        self._append_paths([group_id])
        self.log_panel.log(f"Combined {len(paths)} UAV band file(s) into one capture "
                            f"({capture['display_name']}): {list(role_to_path.keys())}")

    def _display_name_for_path(self, path: str) -> str:
        if path in self.multispectral_groups:
            return self.multispectral_groups[path].get("display_name", path)
        return os.path.basename(path)

    def _on_directory_loaded(self, paths: list[str]) -> None:
        self._set_paths(paths)

    def _append_paths(self, paths: list[str]) -> None:
        combined = self.image_paths + [p for p in paths if p not in self.image_paths]
        if len(combined) > MAX_IMAGES:
            QMessageBox.warning(self, "Too many images",
                                 f"That would exceed the {MAX_IMAGES}-image cap; only the first "
                                 f"{MAX_IMAGES} will be kept.")
            combined = combined[:MAX_IMAGES]
        self._set_paths(combined)

    def _set_paths(self, paths: list[str]) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.image_paths = paths
            self.image_names = [self._display_name_for_path(p) for p in paths]
            self.composite = None
            self.records = []
            self.descriptors = None
            self.included_names = []

            self.image_manager.set_names(self.image_names)
            self.attribute_table.populate_names_only(self.image_names)
            self.field_panel.populate(self.image_names)
            self.preview_panel.set_images(self.image_names, self.image_paths, self.multispectral_groups)
            self.map_canvas.clear()
            self.xai_panel.clear()

            self.build_action.setEnabled(bool(self.image_paths))
            self.run_action.setEnabled(False)

            self.log_panel.log(f"{len(self.image_paths)} image(s) registered "
                                f"({len(self.multispectral_paths)} flagged as UAV multispectral).")
            self.status.showMessage(f"{len(self.image_paths)} image(s) registered. "
                                     f"Click \u201cAlign & Build Composite\u201d next.")
        except Exception as e:
            self._report_error("registering images", e)
        finally:
            QApplication.restoreOverrideCursor()

    # ------------------------------------------------------------------ #
    # Image Manager wiring
    # ------------------------------------------------------------------ #
    def _on_images_deleted(self, rows: list[int]) -> None:
        rows = sorted(set(rows), reverse=True)
        for row in rows:
            if 0 <= row < len(self.image_paths):
                path = self.image_paths.pop(row)
                self.image_names.pop(row)
                self.multispectral_paths.discard(path)
                self.multispectral_groups.pop(path, None)
        self.attribute_table.populate_names_only(self.image_names)
        self.field_panel.populate(self.image_names)
        self.preview_panel.set_images(self.image_names, self.image_paths, self.multispectral_groups)
        self.composite = None
        self.records = []
        self.descriptors = None
        self.run_action.setEnabled(False)
        self.build_action.setEnabled(bool(self.image_paths))
        self.map_canvas.clear()
        self.log_panel.log(f"Removed {len(rows)} image(s) from the session. "
                            f"Re-run \u201cAlign & Build Composite\u201d to update results.")

    def _on_image_renamed(self, index: int, new_name: str) -> None:
        if 0 <= index < len(self.image_names):
            self.image_names[index] = new_name
            self.log_panel.log(f"Renamed image #{index + 1} to '{new_name}' (display name only).")

    def _on_refresh_requested(self) -> None:
        self.log_panel.log("No working directory set; nothing to refresh from. "
                            "Use 'Set Working Directory...' first, or Browse Images.")

    # ------------------------------------------------------------------ #
    # Alignment / composite building
    # ------------------------------------------------------------------ #
    def build_composite(self) -> None:
        if not self.image_paths:
            QMessageBox.warning(self, "No images", "Browse Images or set a working directory first.")
            return

        n = len(self.image_paths)
        progress = QProgressDialog("Aligning and aggregating images...", "Cancel", 0, n, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        aligner = BatchAligner(target_long_edge=700)
        try:
            for i, path in enumerate(self.image_paths):
                if progress.wasCanceled():
                    self.log_panel.log(f"Composite build cancelled after {i}/{n} images.")
                    break
                name = self.image_names[i]
                try:
                    extra_bands = None
                    if path in self.multispectral_groups:
                        capture = self.multispectral_groups[path]
                        pil_img = Image.fromarray(capture["rgb"])
                        extra_bands = {k: capture[k] for k in ("nir", "rededge") if k in capture}
                    elif path in self.multispectral_paths:
                        ms = load_multispectral(path)
                        if ms.get("available"):
                            pil_img = Image.fromarray(ms["rgb"])
                            extra_bands = {k: ms[k] for k in ("nir", "rededge") if k in ms}
                        else:
                            self.log_panel.log(f"{name}: {ms.get('note')} Falling back to RGB proxy.")
                            pil_img = Image.open(path).convert("RGB")
                    else:
                        pil_img = Image.open(path).convert("RGB")
                    aligner.add_image(name, path, pil_img, extra_bands)
                except Exception as e:
                    aligner.records.append(ImageRecord(
                        name=name, path=path, orig_width=0, orig_height=0,
                        mean_intensity=float("nan"), shift_y=float("nan"), shift_x=float("nan"),
                        included=False, error=str(e)))
                    self.log_panel.log(f"Skipped {name}: {e}")

                progress.setValue(i + 1)
                if i % 5 == 0:
                    QApplication.processEvents()
        finally:
            progress.close()

        composite = aligner.finalize()
        if composite is None:
            QMessageBox.warning(self, "No composite", "No images could be processed successfully.")
            return

        self.composite = composite
        self.records = aligner.records
        included = [r for r in self.records if r.included and r.descriptor is not None]
        self.descriptors = np.array([r.descriptor for r in included]) if included else None
        self.included_names = [r.name for r in included]

        self.attribute_table.populate_from_records(self.records)
        self.run_action.setEnabled(True)
        self._refresh_map_and_xai()

        skipped = n - composite.count
        ms_note = (f" ({composite.multispectral_count} with real multispectral bands)"
                   if composite.multispectral_count else "")
        self.log_panel.log(f"Built composite from {composite.count}/{n} image(s){ms_note}. "
                            f"{skipped} skipped/failed (see Attribute Table for details). "
                            f"Alignment is translation-only (phase correlation) -- rotation/scale/"
                            f"perspective differences between photos are not corrected.")
        self.status.showMessage(f"Composite built from {composite.count} image(s).")

    # ------------------------------------------------------------------ #
    # Reactivity
    # ------------------------------------------------------------------ #
    def _on_layers_changed(self) -> None:
        self._refresh_map_and_xai(update_xai=False)

    def _refresh_map_and_xai(self, update_xai: bool = True) -> None:
        if self.composite is None:
            return
        try:
            raster_layers = self.layer_panel.active_raster_layers()
            vector_layers = self.layer_panel.active_vector_layers()
            title = f"Aggregate composite of {self.composite.count} aligned image(s)"
            self.map_canvas.render(self.composite, raster_layers, vector_layers, title=title)
            if update_xai:
                self.xai_panel.update_for_composite(self.composite, self.included_names, self.descriptors)
        except Exception as e:
            self._report_error("rendering the map canvas", e)

    # ------------------------------------------------------------------ #
    # Analysis / reports
    # ------------------------------------------------------------------ #
    def run_analysis(self) -> None:
        if self.composite is None:
            QMessageBox.warning(self, "No composite", "Click \u201cAlign & Build Composite\u201d first.")
            return
        selected = self.layer_panel.checked_report_layers()
        if not selected:
            QMessageBox.warning(self, "No analyses selected",
                                 "Tick at least one item under \u201cReports & Charts\u201d.")
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.status.showMessage("Running analysis...")
        QApplication.processEvents()
        try:
            spatial_choices = [c for c in selected if c in SPATIAL_REPORT_KEYS]
            batch_choices = [c for c in selected if c in BATCH_REPORT_KEYS]
            items = spatial_choices + batch_choices
            n = len(items)
            n_cols = min(3, n) if n else 1
            n_rows = max(1, -(-n // n_cols))

            # Figure size now scales with content and is allowed to exceed a single
            # screen's height/width when there are many panels -- the scroll area
            # (fixed above) correctly shows scrollbars for the overflow instead of
            # squeezing everything into a fixed box, which was the source of
            # content being clipped at the top or bottom of the window.
            fig_width = 6.5 * n_cols
            fig_height = 4.5 * n_rows
            figure = Figure(figsize=(fig_width, fig_height))
            gs = gridspec.GridSpec(n_rows, n_cols, figure=figure)
            composite_img = self.composite.mean_rgb

            for idx, choice in enumerate(items):
                ax = figure.add_subplot(gs[idx // n_cols, idx % n_cols])
                if choice in SPATIAL_REPORT_KEYS:
                    self._dispatch_spatial_report(choice, composite_img, ax)
                else:
                    self._dispatch_batch_report(choice, ax)
                ax.set_title(choice, fontsize=9)

            figure.suptitle(f"Aggregate report — {self.composite.count} aligned image(s)", fontsize=12)
            figure.tight_layout(rect=[0, 0, 1, 0.96])

            self.report_figure = figure
            self.charts_canvas = FigureCanvas(figure)
            # Explicit pixel size matching the figure's real inches x DPI, so the
            # (now non-resizable) scroll area knows the canvas's true size and
            # shows scrollbars for it correctly, instead of the previous behaviour
            # where the canvas was silently squeezed to the viewport and content
            # got clipped off the top or bottom.
            canvas_dpi = figure.get_dpi()
            self.charts_canvas.setMinimumSize(int(fig_width * canvas_dpi), int(fig_height * canvas_dpi))

            while self.charts_toolbar_layout.count():
                child = self.charts_toolbar_layout.takeAt(0).widget()
                if child:
                    child.deleteLater()
            self.charts_toolbar_layout.addWidget(NavigationToolbar(self.charts_canvas, self))

            self.charts_scroll.setWidget(self.charts_canvas)
            self.central_tabs.setCurrentWidget(self.charts_scroll.parentWidget())

            self.log_panel.log(f"Ran analysis on the aggregate composite: {', '.join(selected)}")
            self.status.showMessage("Analysis complete.")
        except Exception as e:
            self._report_error("running analysis", e)
            self.status.showMessage("Analysis failed -- see Execution Log.")
        finally:
            QApplication.restoreOverrideCursor()

    def _dispatch_spatial_report(self, choice: str, composite_img: np.ndarray, ax) -> None:
        if choice == "Cross-Crop Alignment":
            labels, matrix = texture.cross_crop_alignment_matrix(composite_img)
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, fontsize=7)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=7)
            ax.figure.colorbar(im, ax=ax)
        elif choice == "Geometric Transfer":
            texture.geometric_transfer_metrics(composite_img, ax)
        elif choice == "Loss Functions":
            texture.image_quantization_loss(composite_img, ax)
        elif choice == "Faster R-CNN":
            ml_models.faster_rcnn_detection(composite_img, ax)
        elif choice == "ResNet101":
            ml_models.resnet101_classification(composite_img, ax)
        elif choice == "Pathology Localization":
            vegetation.pathology_localization_demo(composite_img, ax)
        elif choice == "Root Architecture":
            texture.root_architecture(composite_img, ax)
        elif choice == "Colour Histogram":
            indices.color_histogram(composite_img, ax)
        elif choice == "Spectral Reflectance & Band Ratios":
            self._render_spectral_panel(composite_img, ax)
        elif choice == "GLCM Texture Features":
            texture_features.glcm_report(composite_img, ax)
        elif choice == "Invariance Tests":
            self._render_invariance_panel(composite_img, ax)
        elif choice == "InfoNCE (self-supervised)":
            self._render_infonce_panel(composite_img, ax)
        elif choice == "Morphology & Environment Profile":
            self._render_morphology_panel(composite_img, ax)
        elif choice == "Synthetic Reference Classifier (DEMO)":
            self._render_synthetic_reference_panel(composite_img, ax)

    def _dispatch_batch_report(self, choice: str, ax) -> None:
        have_records = self.descriptors is not None and len(self.included_names) > 1
        if choice == "Time-Series Trend (linear)":
            if have_records:
                batch_mod.plot_time_series_trend_from_records(self.included_names, self.descriptors, ax)
            else:
                ax.axis("off"); ax.text(0.1, 0.5, "Needs 2+ successfully aligned images.", fontsize=10)
        elif choice == "Feature-Space Projection (PCA)":
            if have_records:
                batch_mod.plot_feature_space_from_records(self.included_names, self.descriptors, ax)
            else:
                ax.axis("off"); ax.text(0.1, 0.5, "Needs 2+ successfully aligned images.", fontsize=10)
        elif choice == "Batch Summary":
            if have_records:
                batch_mod.plot_batch_summary_from_records(self.included_names, self.descriptors, ax)
            else:
                ax.axis("off"); ax.text(0.1, 0.5, "Needs 2+ successfully aligned images.", fontsize=10)
        elif choice == "Spectral Index Availability":
            self._render_index_availability_panel(ax)
        elif choice == "Embedding & Similarity Metrics":
            self._render_embedding_panel(ax)
        elif choice == "Cross-Image Pathology Transfer Recommendation":
            self._render_cross_image_recommendation_panel(ax)
        elif choice == "Classification / Clustering (automatic)":
            self._render_classification_panel(ax)

    # ------------------------------------------------------------------ #
    # New report panel renderers
    # ------------------------------------------------------------------ #
    def _render_spectral_panel(self, img: np.ndarray, ax) -> None:
        refl = spectral.relative_reflectance_proxy(img)
        ratios = spectral.band_ratios(img)
        pix = spectral.pixel_intensity(img)
        lines = [
            "Spectral Reflectance (uncalibrated proxy, 0-1):",
            f"  Blue : {refl['Blue']:.3f}   Green: {refl['Green']:.3f}   Red: {refl['Red']:.3f}",
            "  Red-edge: N/A   NIR: N/A  (unless real UAV multispectral bands were loaded)",
            "", "Spectral Radiance: UNAVAILABLE (needs a calibrated sensor + known exposure)",
            "", "Pixel Intensity (grayscale, 0-255):",
            f"  mean={pix['mean']:.2f}  std={pix['std']:.2f}  min={pix['min']:.0f}  max={pix['max']:.0f}",
            "", "Band Ratios:",
        ] + [f"  {k:26s}: {v:.3f}" for k, v in ratios.items()]
        ax.axis("off")
        ax.text(0.02, 0.5, "\n".join(lines), fontsize=8, va="center", family="monospace")

    def _render_invariance_panel(self, img: np.ndarray, ax) -> None:
        results = embedding_metrics.invariance_tests(img)
        lines = ["Invariance Tests (cosine similarity of the real descriptor,",
                 "composite vs. transformed composite; 1.0 = fully invariant):", ""]
        for name, value in results.items():
            lines.append(f"  {name:22s}: {value:.3f}")
        ax.axis("off")
        ax.text(0.02, 0.5, "\n".join(lines), fontsize=8, va="center", family="monospace")

    def _render_infonce_panel(self, img: np.ndarray, ax) -> None:
        loss = embedding_metrics.info_nce_self_supervised(img)
        lines = [
            "Contrastive Loss (InfoNCE-style, self-supervised):", "",
            f"  Loss value: {loss:.4f}", "",
            "Computed on the composite's real 8-d descriptor using a flip-",
            "augmentation positive pair and jittered negatives. NOT the loss",
            "of a trained deep contrastive/SSL model -- none is trained here.",
        ]
        ax.axis("off")
        ax.text(0.02, 0.5, "\n".join(lines), fontsize=8, va="center", family="monospace")

    def _render_synthetic_reference_panel(self, img: np.ndarray, ax) -> None:
        result = synthetic_reference_model.predict(img)
        ax.axis("off")
        if not result.get("available"):
            ax.text(0.05, 0.5, result.get("note", "Unavailable"), fontsize=8, va="center", wrap=True)
            return

        tr = result["train_report"]
        ax.set_title(
            f"Synthetic Crop & Pathology-Class Reference Classifier — SIMULATED DEMO\n"
            f"(trained on {tr['n_rows']} fabricated rows; CV acc: crop={tr['crop_cv_accuracy']:.2f}, "
            f"class={tr['class_cv_accuracy']:.2f})", fontsize=7.5, color="#a33")

        # Layout note: both inset axes and the disclaimer text are kept strictly
        # within [0, 1] of this panel's own coordinate space (previously the
        # disclaimer was placed at a negative y-coordinate, which put it outside
        # this panel's bounds and let it visually collide with whatever panel sits
        # below it in the report grid -- fixed here, wording unchanged).
        crop_ax = ax.inset_axes([0.0, 0.58, 1.0, 0.37])
        class_ax = ax.inset_axes([0.0, 0.16, 1.0, 0.37])

        top_crops = sorted(result["crop_probabilities"].items(), key=lambda x: -x[1])[:6]
        top_classes = sorted(result["class_probabilities"].items(), key=lambda x: -x[1])[:6]

        crop_labels, crop_values = [k for k, _ in top_crops][::-1], [v for _, v in top_crops][::-1]
        crop_y = np.arange(len(crop_labels))
        crop_ax.barh(crop_y, crop_values, color="#6a994e")
        crop_ax.set_yticks(crop_y)
        crop_ax.set_yticklabels(crop_labels)
        crop_ax.set_title("Predicted 'crop' category (simulated)", fontsize=7)
        crop_ax.tick_params(labelsize=6)

        class_labels, class_values = [k for k, _ in top_classes][::-1], [v for _, v in top_classes][::-1]
        class_y = np.arange(len(class_labels))
        class_ax.barh(class_y, class_values, color="#c62828")
        class_ax.set_yticks(class_y)
        class_ax.set_yticklabels(class_labels)
        class_ax.set_title("Predicted 'class' (pathology-like) category (simulated)", fontsize=7)
        class_ax.tick_params(labelsize=6)

        ax.text(0.02, 0.02, "SIMULATED DEMO -- fabricated categories/data, not real crop or disease\n"
                             "identities. Not for real agronomic decisions.",
                transform=ax.transAxes, fontsize=6.5, color="#a33", va="bottom", ha="left", wrap=True)

    def _render_morphology_panel(self, img: np.ndarray, ax) -> None:
        profile = morphology_profile.morphology_environment_profile(img, path=None)
        lines = ["Morphology & Environment Attribute Profile", "(classical CV proxies -- NOT a trained",
                 "contrastive/GACL model; see Execution Log note)", ""]
        for category, attrs in profile.items():
            lines.append(f"[{category}]")
            for attr_name, values in attrs.items():
                if isinstance(values, dict):
                    summary = ", ".join(f"{k}={v:.3g}" for k, v in values.items()
                                         if isinstance(v, (int, float)))
                    lines.append(f"  {attr_name}: {summary}" if summary else f"  {attr_name}: n/a")
            lines.append("")
        ax.axis("off")
        ax.text(0.01, 0.99, "\n".join(lines), fontsize=6.5, va="top", family="monospace", wrap=True)

    def _render_index_availability_panel(self, ax) -> None:
        lines = ["Spectral Index Availability:", ""]
        for key, func in vegetation_indices.UNAVAILABLE_INDEX_FUNCS.items():
            result = func()
            lines.append(f"{key}: {result.note}")
            lines.append("")
        ax.axis("off")
        ax.text(0.01, 0.5, "\n".join(lines), fontsize=8, va="center", family="monospace", wrap=True)

    def _render_embedding_panel(self, ax) -> None:
        dim = embedding_metrics.embedding_dimensionality()
        if self.descriptors is None or len(self.descriptors) < 2:
            ax.axis("off")
            ax.text(0.1, 0.5, "Needs 2+ successfully aligned images.", fontsize=10)
            return
        variance = embedding_metrics.latent_feature_variance(self.descriptors)
        labels = self.field_panel.get_group_labels_for_names(self.included_names)
        intra_inter = embedding_metrics.intra_inter_similarity(self.descriptors, labels)

        # Layout note: lines are kept short (hard-wrapped below rather than relying
        # on matplotlib's unreliable wrap=True) and confined to the left ~50% of
        # the panel, so this text can never visually collide with the similarity-
        # matrix inset docked on the right half -- previously both used loose,
        # unconstrained placement, which is what produced the scattered/overlapping
        # look reported.
        lines = [
            "Embedding & Similarity Metrics",
            "(real, on this pipeline's 8-d",
            "descriptor -- not a trained",
            "deep embedding):",
            "",
            f"Dimensionality  : {dim}",
            f"Latent variance : {variance:.4f}",
        ]
        if intra_inter.get("available"):
            lines.append(f"Intra-class sim : {intra_inter['intra_class_similarity']:.3f}")
            lines.append(f"Inter-class sim : {intra_inter['inter_class_similarity']:.3f}")
        else:
            note = intra_inter["note"]
            # hard-wrap long notes to the same ~26-char column width as the lines above
            import textwrap
            lines.extend(textwrap.wrap(note, width=26))

        pw = embedding_metrics.pairwise_metrics(self.descriptors)
        lines.append("")
        lines.append("Similarity matrix at right.")
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join(lines), fontsize=7.5, va="top", ha="left",
                family="monospace", linespacing=1.4, transform=ax.transAxes)

        n = len(self.included_names)
        if n <= 60:
            inset = ax.inset_axes([0.58, 0.08, 0.40, 0.55])
            im = inset.imshow(pw["cosine_similarity"], cmap="viridis", vmin=-1, vmax=1)
            step = max(1, n // 15)
            ticks = list(range(0, n, step))
            inset.set_xticks(ticks); inset.set_xticklabels([str(t + 1) for t in ticks], fontsize=6)
            inset.set_yticks(ticks); inset.set_yticklabels([str(t + 1) for t in ticks], fontsize=6)
            inset.set_title("Cosine similarity", fontsize=7)
            try:
                ax.figure.colorbar(im, ax=inset, fraction=0.046, pad=0.04)
            except Exception:
                pass
        else:
            ax.text(0.58, 0.3, f"(similarity matrix omitted for {n} images;\nsee PCA panel for a visual summary)",
                    fontsize=7, transform=ax.transAxes)

    def _render_cross_image_recommendation_panel(self, ax) -> None:
        if self.descriptors is None or len(self.included_names) < 2:
            ax.axis("off")
            ax.text(0.1, 0.5, "Needs 2+ successfully aligned images.", fontsize=10)
            return
        rec = embedding_metrics.cross_image_similarity_recommendation(self.included_names, self.descriptors)
        n = len(self.included_names)

        # Circular node layout -- each image is a node; an edge connects it to its
        # single most-similar match, coloured/thickened by similarity strength.
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs, ys = np.cos(angles), np.sin(angles)
        name_to_idx = {name: i for i, name in enumerate(self.included_names)}

        for r in rec["recommendations"]:
            i, j = name_to_idx[r["image"]], name_to_idx[r["most_similar_image"]]
            sim = max(0.0, r["similarity"])
            ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color=cm.viridis(sim), linewidth=0.5 + 2.5 * sim, alpha=0.7)

        ax.scatter(xs, ys, s=60, c="#2f5233", zorder=3, edgecolor="white", linewidths=0.8)
        if n <= 40:
            for i in range(n):
                ax.annotate(str(i + 1), (xs[i], ys[i]), fontsize=6, ha="center", va="center", color="white", zorder=4)

        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title("Cross-Image Pathology Transfer Recommendation\n"
                      "(real descriptor-similarity graph -- not a trained pathotype model)", fontsize=8)
        ax.text(0, -1.45, "Edge = 'most similar to' link; brighter/thicker = higher similarity",
                fontsize=6.5, ha="center", color="#555555")

    def _render_classification_panel(self, ax) -> None:
        ax.axis("off")
        if self.descriptors is None or len(self.included_names) < 2:
            ax.text(0.1, 0.5, "Needs 2+ successfully aligned images.", fontsize=10)
            return

        labels = self.field_panel.get_group_labels_for_names(self.included_names)
        if labels:
            result = ml_classifier.auto_classify_or_cluster(self.descriptors, labels)
        else:
            y_true, y_pred = self.field_panel.get_classification_labels()
            if y_true and y_pred:
                result = embedding_metrics.classification_metrics_from_labels(y_true, y_pred)
                result["mode"] = "csv_external"
            else:
                result = ml_classifier.auto_classify_or_cluster(self.descriptors, None)

        mode = result.get("mode", "unavailable")
        if mode == "unavailable":
            ax.text(0.02, 0.5, "Classification / Clustering: N/A\n\n" + result.get("note", ""),
                    fontsize=8, va="center", family="monospace", wrap=True)
            return
        if mode == "supervised_knn":
            lines = [
                f"Real k-NN classifier (k={result['k_neighbors']}), leave-one-out CV,",
                f"trained on your {result['n_classes']}-class group labels:", "",
                f"  Accuracy    : {result['accuracy']:.3f}",
                f"  Precision   : {result['precision']:.3f}",
                f"  Recall      : {result['recall']:.3f}",
                f"  F1-score    : {result['f1_score']:.3f}",
                f"  Cohen Kappa : {result['cohens_kappa']:.3f}",
                f"  MCC         : {result['matthews_corrcoef']:.3f}",
                "", result["note"],
            ]
        elif mode == "unsupervised_clustering":
            lines = [
                f"No group labels found -- ran real automatic K-Means clustering",
                f"(k={result['n_clusters']}, chosen by silhouette score):", "",
                f"  Silhouette Score        : {result['silhouette_score']:.3f}  (higher = better separated)",
                f"  Davies-Bouldin Index    : {result['davies_bouldin_index']:.3f}  (lower = better separated)",
                f"  Calinski-Harabasz Index : {result['calinski_harabasz_index']:.1f}  (higher = better separated)",
                "", result["note"],
            ]
        else:  # csv_external
            if not result.get("available"):
                lines = ["Classification Metrics: N/A", "", result.get("note", "")]
            else:
                lines = [
                    "Classification metrics from your loaded labels CSV:", "",
                    f"  Accuracy   : {result['accuracy']:.3f}",
                    f"  Precision  : {result['precision']:.3f}",
                    f"  Recall     : {result['recall']:.3f}",
                    f"  F1-score   : {result['f1_score']:.3f}",
                    f"  Kappa      : {result['cohens_kappa']:.3f}",
                    f"  MCC        : {result['matthews_corrcoef']:.3f}",
                ]
        ax.text(0.02, 0.5, "\n".join(lines), fontsize=8, va="center", family="monospace", wrap=True)

    # ------------------------------------------------------------------ #
    # Save / export
    # ------------------------------------------------------------------ #
    def save_map(self) -> None:
        if self.composite is None:
            QMessageBox.warning(self, "No map", "Build the composite first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Map As", "", "PNG Image (*.png);;PDF Document (*.pdf)")
        if path:
            try:
                self.map_canvas.save(path)
                self._last_export_path = path
                self.log_panel.log(f"Map saved to {path}")
                self.status.showMessage(f"Map saved to {path}")
            except Exception as e:
                self._report_error("saving the map", e)

    def save_report(self) -> None:
        if self.report_figure is None:
            QMessageBox.warning(self, "No report", "Run Reports first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Report As", "", "PNG Image (*.png);;PDF Document (*.pdf)")
        if path:
            try:
                self.report_figure.savefig(path, dpi=200, bbox_inches="tight")
                self._last_export_path = path
                self.log_panel.log(f"Report saved to {path}")
                self.status.showMessage(f"Report saved to {path}")
            except Exception as e:
                self._report_error("saving the report", e)

    def export_pdf_report(self) -> None:
        figures, titles = [], []
        if self.composite is not None:
            figures.append(self.map_canvas.figure); titles.append("Map Canvas")
        if self.report_figure is not None:
            figures.append(self.report_figure); titles.append("Charts & Reports")
        if not figures:
            QMessageBox.warning(self, "Nothing to export", "Build a composite and/or run reports first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export PDF Report", "", "PDF Document (*.pdf)")
        if not path:
            return
        try:
            pdf_export.export_figures_to_pdf(path, figures, titles)
            self._last_export_path = path
            self.log_panel.log(f"Exported multi-page PDF report to {path}")
            self.status.showMessage(f"PDF report exported to {path}")
        except Exception as e:
            self._report_error("exporting the PDF report", e)

    # ------------------------------------------------------------------ #
    # Help
    # ------------------------------------------------------------------ #
    def _readme_path(self) -> str:
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(here, "README.md")

    def _open_readme(self) -> None:
        path = self._readme_path()
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.information(self, "README not found", f"Expected README.md at:\n{path}")

    def _show_about(self) -> None:
        QMessageBox.information(
            self, "About",
            "Naziru Image Analysis Pipeline — GIS Workbench\n\n"
            "A desktop tool for exploratory agronomic image analysis, styled\n"
            "after professional desktop-GIS applications. All results are\n"
            "computed on the aggregate composite of your loaded images.\n\n"
            f"See the README (Help > Open README) for full details.\n\n"
            "Author: Naziru Halilu",
        )
