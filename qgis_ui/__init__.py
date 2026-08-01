"""
QGIS-style desktop interface for the Naziru Image Analysis Pipeline (PyQt6).

Modules
-------
styles          : QSS stylesheet giving the QGIS look and feel.
layer_panel     : Left dock -- Table of Contents / Layer Controller with symbology.
canvas_widget   : Central Map Canvas -- base image + raster layer overlays.
xai_panel       : Right dock -- live descriptive stats & rule-based recommendations.
attribute_table : Bottom dock (tab 1) -- per-image attribute table.
log_panel       : Bottom dock (tab 2) -- execution log.
main_window     : Assembles everything into the QMainWindow application.
"""
