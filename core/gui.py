"""
Naziru Image Analysis Pipeline — Tkinter GUI.

Flow (matches the on-screen app):
  1. "Browse Images" -> pick one or more images from disk (the INPUT).
  2. Tick which analyses to run from the checklist.
  3. "Run Analysis" -> builds a matplotlib figure with one row per image
     (per-image analyses) plus one full-width row per batch-level analysis
     (Time-Series Trend, Feature-Space Projection, Batch Summary).
  4. A save dialog lets you choose where the OUTPUT (PNG/JPEG/PDF) is written,
     and the result is previewed inline in the scrollable output area.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk

from . import theme
from .indices import color_histogram, ndvi, ndwi, plot_index_with_colorbar
from .texture import (
    cross_crop_alignment_matrix,
    geometric_transfer_metrics,
    idw_demo,
    image_quantization_loss,
    invariant_texture_map,
    root_architecture,
)
from .ml_models import faster_rcnn_detection, resnet101_classification
from .vegetation import (
    canopy_structure,
    pathology_localization_demo,
    susceptible_spot_detection,
    variable_rate_map,
    vis_nir_graph,
    xai_recommendation,
)
from .batch import plot_batch_summary, plot_feature_space, plot_time_series_trend


class AgronomicGUI:
    """Top-level Tkinter application window."""

    #: Checklist labels that are cross-image (batch-level) summaries rather
    #: than one panel per image.
    BATCH_LEVEL_CHOICES = ("Time-Series Trend (linear)", "Feature-Space Projection (PCA)", "Batch Summary")

    ANALYSIS_OPTIONS = [
        "Invariant Texture", "Cross-Crop Alignment", "Geometric Transfer", "Loss Functions",
        "Faster R-CNN", "ResNet101", "Pathology Localization", "NDVI", "NDWI", "IDW",
        "Root Architecture", "Colour Histogram", "Susceptible Spots (heuristic)",
        "Canopy Structure", "Time-Series Trend (linear)", "VIS/NIR Graph",
        "XAI Recommendation", "Variable Rate Map", "Feature-Space Projection (PCA)",
        "Batch Summary",
    ]

    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Naziru Image Analysis Pipeline")
        master.geometry("1300x900")
        master.configure(bg=theme.BG_MAIN)

        self.files: tuple[str, ...] = ()
        self._build_header()
        self._build_input_section()
        self._build_checklist()
        self._build_run_button()
        self._build_output_area()
        self._build_footer()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _build_header(self) -> None:
        header = tk.Frame(self.master, bg=theme.BG_HEADER)
        header.pack(fill="x", side="top")
        try:
            logo_img = Image.open("assets/sustagri_logo.png")
            logo_img.thumbnail((110, 110))
            self.logo = ImageTk.PhotoImage(logo_img)
            tk.Label(header, image=self.logo, bg=theme.BG_HEADER).pack(side="left", anchor="nw", padx=15, pady=10)
        except Exception:
            tk.Label(header, text="[EMJMD\nSUSTAGRI]", bg=theme.BG_HEADER, fg=theme.FG_HEADER,
                     font=("Arial", 9, "bold"), justify="left").pack(side="left", anchor="nw", padx=15, pady=10)
        tk.Label(header, text="Cutting Edge\nTechnologies For Sustainable Agriculture",
                 bg=theme.BG_HEADER, fg=theme.FG_HEADER, font=("Arial", 12, "bold"),
                 justify="left").pack(side="left", padx=10, pady=10)

    def _build_input_section(self) -> None:
        """This is the image INPUT control ('Browse Images')."""
        tk.Label(self.master, text="Select images to process:", bg=theme.BG_MAIN,
                 font=("Arial", 10, "bold")).pack(pady=10)
        tk.Button(self.master, text="Browse Images", command=self.select_files,
                  bg=theme.ACCENT, fg=theme.BTN_FG, activebackground="#557a3d",
                  relief="flat", padx=10, pady=4).pack(pady=5)
        self.files_status = tk.Label(self.master, text="No images selected yet.",
                                      bg=theme.BG_MAIN, fg="#555555", font=("Arial", 9))
        self.files_status.pack()

    def _build_checklist(self) -> None:
        checkbox_frame = tk.Frame(self.master, bg=theme.BG_PANEL, padx=10, pady=8)
        checkbox_frame.pack(fill="x", padx=20, pady=5)
        self.analysis_vars: dict[str, tk.BooleanVar] = {}
        for opt in self.ANALYSIS_OPTIONS:
            var = tk.BooleanVar()
            tk.Checkbutton(checkbox_frame, text=opt, variable=var, bg=theme.BG_PANEL,
                            activebackground=theme.BG_PANEL, anchor="w").pack(anchor="w")
            self.analysis_vars[opt] = var

    def _build_run_button(self) -> None:
        self.run_button = tk.Button(self.master, text="Run Analysis", command=self.run_pipeline,
                                     state=tk.DISABLED, bg=theme.ACCENT, fg=theme.BTN_FG,
                                     activebackground="#557a3d", relief="flat", padx=10, pady=4)
        self.run_button.pack(pady=10)

    def _build_output_area(self) -> None:
        """This is the result/OUTPUT preview area (scrollable image)."""
        self.output_frame = tk.Frame(self.master, bg=theme.BG_MAIN)
        self.output_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.output_frame, bg=theme.BG_MAIN, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.output_frame, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.inner_frame = tk.Frame(self.canvas, bg=theme.BG_MAIN)
        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.inner_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.image_label = tk.Label(self.inner_frame, bg=theme.BG_MAIN)
        self.image_label.pack()

    def _build_footer(self) -> None:
        tk.Label(self.master, text="Author: Naziru Halilu", font=("Arial", 12, "italic"),
                 bg=theme.BG_HEADER, fg=theme.FG_HEADER, pady=8).pack(side="bottom", fill="x")

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def select_files(self) -> None:
        """Open the file picker and register the chosen images as INPUT."""
        self.files = filedialog.askopenfilenames(
            title="Select Agronomic Images",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp")],
        )
        if self.files:
            self.files_status.config(text=f"{len(self.files)} image(s) selected.")
            messagebox.showinfo("Files Selected", f"{len(self.files)} image(s) selected.")
            self.run_button.config(state=tk.NORMAL)
        else:
            self.files_status.config(text="No images selected yet.")

    # ------------------------------------------------------------------ #
    # Pipeline execution
    # ------------------------------------------------------------------ #
    def run_pipeline(self) -> None:
        selected = [opt for opt, var in self.analysis_vars.items() if var.get()]
        if not self.files or not selected:
            messagebox.showerror("Error", "No files or analyses selected.")
            return

        per_image_choices = [c for c in selected if c not in self.BATCH_LEVEL_CHOICES]
        batch_choices = [c for c in selected if c in self.BATCH_LEVEL_CHOICES]

        n_rows = len(self.files)
        n_cols = max(len(per_image_choices), 1)
        n_batch_rows = len(batch_choices)

        fig = plt.figure(figsize=(6 * n_cols, 4 * n_rows + 4 * n_batch_rows))
        gs = gridspec.GridSpec(n_rows + n_batch_rows, n_cols, figure=fig)

        loaded_images = [np.array(Image.open(fp).convert("RGB")) for fp in self.files]

        if per_image_choices:
            for i, np_img in enumerate(loaded_images):
                for j, choice in enumerate(per_image_choices):
                    ax = fig.add_subplot(gs[i, j])
                    self.run_analysis_choice(choice, np_img, ax)
                    title = choice if n_rows == 1 else f"{choice} (Image {i + 1})"
                    ax.set_title(title)

        for r_offset, choice in enumerate(batch_choices):
            ax = fig.add_subplot(gs[n_rows + r_offset, :])
            if choice == "Time-Series Trend (linear)":
                if len(loaded_images) > 1:
                    plot_time_series_trend(loaded_images, ax)
                else:
                    ax.axis("off")
                    ax.text(0.15, 0.5, "Time-Series Trend needs more than one image.", fontsize=10)
            elif choice == "Feature-Space Projection (PCA)":
                plot_feature_space(loaded_images, ax)
            elif choice == "Batch Summary":
                plot_batch_summary(loaded_images, ax)

        plt.tight_layout()

        # Ask user where/how to save the OUTPUT
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("PDF Document", "*.pdf")],
            title="Save analysis output as...",
        )
        if file_path:
            plt.savefig(file_path, dpi=200)
            messagebox.showinfo("Saved", f"Output saved as {file_path}")
        else:
            messagebox.showwarning("Cancelled", "Save cancelled.")

        plt.close(fig)

        if file_path and not file_path.lower().endswith(".pdf"):
            img_preview = Image.open(file_path)
            img_preview.thumbnail((1200, 800))
            tk_img = ImageTk.PhotoImage(img_preview)
            self.image_label.config(image=tk_img)
            self.image_label.image = tk_img
        elif file_path and file_path.lower().endswith(".pdf"):
            messagebox.showinfo("Preview", "PDF saved. Preview not available in GUI.")

    def run_analysis_choice(self, choice: str, np_img: np.ndarray, ax) -> None:
        """Dispatch a single per-image analysis choice to its plotting function."""
        if choice == "Invariant Texture":
            ax.imshow(invariant_texture_map(np_img), cmap="viridis")
        elif choice == "Cross-Crop Alignment":
            labels, matrix = cross_crop_alignment_matrix(np_img)
            im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels)
            plt.colorbar(im, ax=ax)
        elif choice == "Geometric Transfer":
            geometric_transfer_metrics(np_img, ax)
        elif choice == "Loss Functions":
            image_quantization_loss(np_img, ax)
        elif choice == "Faster R-CNN":
            faster_rcnn_detection(np_img, ax)
        elif choice == "ResNet101":
            resnet101_classification(np_img, ax)
        elif choice == "Pathology Localization":
            pathology_localization_demo(np_img, ax)
        elif choice == "NDVI":
            plot_index_with_colorbar(ndvi(np_img), ax, "RdYlGn", "NDVI")
        elif choice == "NDWI":
            plot_index_with_colorbar(ndwi(np_img), ax, "Blues", "NDWI")
        elif choice == "IDW":
            idw_demo(np_img, ax)
        elif choice == "Root Architecture":
            root_architecture(np_img, ax)
        elif choice == "Colour Histogram":
            color_histogram(np_img, ax)
        elif choice == "Susceptible Spots (heuristic)":
            susceptible_spot_detection(np_img, ax)
        elif choice == "Canopy Structure":
            canopy_structure(np_img, ax)
        elif choice == "VIS/NIR Graph":
            vis_nir_graph(np_img, ax)
        elif choice == "XAI Recommendation":
            xai_recommendation(np_img, ax)
        elif choice == "Variable Rate Map":
            variable_rate_map(np_img, ax)
