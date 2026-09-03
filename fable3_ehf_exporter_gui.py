#!/usr/bin/env python3
"""
Fable 3 EHF Terrain Exporter - GUI + Batch Export

Single-file and batch EHF terrain exporter.

Input:
    Fable 3 HeightFieldGraphicsFile (.ehf)

Outputs:
    OBJ terrain
    8-bit PGM heightmap named <obj_stem>_height.pgm

EHF format:
    magic                  HeightFieldGraphicsFile
    patch header            6 little-endian float32 + 2 uint32
                           min_x,min_y,min_z,max_x,max_y,max_z,width,height
    vertex records          width*height records, 8 bytes each
                           float32 elevation + 4-byte auxiliary value

Axis conversion:
    EHF X = horizontal X
    EHF Y = horizontal Y
    EHF Z = elevation

    OBJ X = EHF X
    OBJ Y = EHF Z
    OBJ Z = EHF Y

PGM:
    Raw EHF elevation is mapped directly to 8-bit grayscale.

        0.0   -> 0
        15.0  -> 15
        60.0  -> 60
        128.0 -> 128
        255.0 -> 255

    No min/max normalization is performed.

    Values below 0 are clamped to 0.
    Values above 255 are clamped to 255.

Batch mode:
    Every *.ehf file in the selected input folder is exported.

    For example:

        Input folder:
            C:\Fable3\EHF\

        Output folder:
            C:\Fable3\Terrain\

    produces:

        C:\Fable3\Terrain\area01.obj
        C:\Fable3\Terrain\area01_height.pgm
        C:\Fable3\Terrain\area02.obj
        C:\Fable3\Terrain\area02_height.pgm
        ...

Requirements:
    Python 3.x
    Tkinter
"""

from __future__ import annotations

import math
import struct
import threading
import traceback
import tkinter as tk

from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ---------------------------------------------------------------------------
# EHF FORMAT
# ---------------------------------------------------------------------------

MAGIC = b"HeightFieldGraphicsFile"
PATCH_HEADER_SIZE = 32
VERTEX_STRIDE = 8
HEIGHT_OFFSET = 0


@dataclass
class Patch:
    offset: int
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    width: int
    height: int
    heights: list[float]

    @property
    def dx(self):
        return (self.max_x - self.min_x) / (self.width - 1)

    @property
    def dy(self):
        return (self.max_y - self.min_y) / (self.height - 1)


def valid_float(v):
    return math.isfinite(v)


def find_patch_headers(data: bytes):
    """
    Scan the EHF data for plausible patch headers.
    """
    hits = []

    limit = len(data) - PATCH_HEADER_SIZE

    for off in range(limit):
        try:
            (
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
                w,
                h,
            ) = struct.unpack_from("<6f2I", data, off)

        except struct.error:
            break

        if not (2 <= w <= 2048 and 2 <= h <= 2048):
            continue

        if not all(
            valid_float(v)
            for v in (
                min_x,
                min_y,
                min_z,
                max_x,
                max_y,
                max_z,
            )
        ):
            continue

        if max_x <= min_x:
            continue

        if max_y <= min_y:
            continue

        if max_z < min_z:
            continue

        data_start = off + PATCH_HEADER_SIZE
        data_end = data_start + w * h * VERTEX_STRIDE

        if data_end > len(data):
            continue

        good = 0

        sample_count = min(32, w * h)

        for i in range(sample_count):
            v = struct.unpack_from(
                "<f",
                data,
                data_start + i * VERTEX_STRIDE + HEIGHT_OFFSET,
            )[0]

            if (
                valid_float(v)
                and min_z - 1e-3 <= v <= max_z + 1e-3
            ):
                good += 1

        if good >= max(4, sample_count // 2):
            hits.append(off)

    return sorted(set(hits))


def discover_patches(data: bytes):
    """
    Locate and validate all terrain patches in an EHF file.
    """

    if not data.startswith(MAGIC):
        raise ValueError(
            "This is not a HeightFieldGraphicsFile."
        )

    candidates = find_patch_headers(data)

    patches = []

    for off in candidates:

        (
            min_x,
            min_y,
            min_z,
            max_x,
            max_y,
            max_z,
            w,
            h,
        ) = struct.unpack_from("<6f2I", data, off)

        start = off + PATCH_HEADER_SIZE

        heights = [
            struct.unpack_from(
                "<f",
                data,
                start + i * VERTEX_STRIDE + HEIGHT_OFFSET,
            )[0]
            for i in range(w * h)
        ]

        hmin = min(heights)
        hmax = max(heights)

        if hmin < min_z - 1e-2:
            continue

        if hmax > max_z + 1e-2:
            continue

        next_candidates = [
            x for x in candidates if x > off
        ]

        next_off = (
            next_candidates[0]
            if next_candidates
            else len(data)
        )

        if start + w * h * VERTEX_STRIDE > next_off:
            continue

        patches.append(
            Patch(
                offset=off,
                min_x=min_x,
                min_y=min_y,
                min_z=min_z,
                max_x=max_x,
                max_y=max_y,
                max_z=max_z,
                width=w,
                height=h,
                heights=heights,
            )
        )

    patches.sort(
        key=lambda p: (p.min_y, p.min_x)
    )

    if not patches:
        raise ValueError(
            "No valid EHF terrain patches found."
        )

    return patches


def stitch(patches):
    """
    Stitch overlapping EHF patches into one regular height grid.
    """

    dx = patches[0].dx
    dy = patches[0].dy

    for p in patches:

        if abs(p.dx - dx) > 1e-3:
            raise ValueError(
                "Terrain patches use inconsistent X grid spacing."
            )

        if abs(p.dy - dy) > 1e-3:
            raise ValueError(
                "Terrain patches use inconsistent Y grid spacing."
            )

    min_x = min(p.min_x for p in patches)
    min_y = min(p.min_y for p in patches)

    max_x = max(p.max_x for p in patches)
    max_y = max(p.max_y for p in patches)

    nx = int(
        round((max_x - min_x) / dx)
    ) + 1

    ny = int(
        round((max_y - min_y) / dy)
    ) + 1

    grid = [
        [None] * nx
        for _ in range(ny)
    ]

    for p in patches:

        bx = int(
            round((p.min_x - min_x) / dx)
        )

        by = int(
            round((p.min_y - min_y) / dy)
        )

        for row in range(p.height):

            for col in range(p.width):

                gx = bx + col
                gy = by + row

                value = p.heights[
                    row * p.width + col
                ]

                if not (
                    0 <= gx < nx
                    and
                    0 <= gy < ny
                ):
                    raise ValueError(
                        "Patch lies outside the stitched grid."
                    )

                if grid[gy][gx] is None:

                    grid[gy][gx] = value

                else:

                    old_value = grid[gy][gx]

                    if abs(old_value - value) > 0.05:

                        print(
                            f"WARNING: seam mismatch at "
                            f"{gx},{gy}: "
                            f"{old_value:.5f} vs "
                            f"{value:.5f}"
                        )

                    grid[gy][gx] = (
                        old_value + value
                    ) * 0.5

    missing = sum(
        v is None
        for row in grid
        for v in row
    )

    if missing:
        raise ValueError(
            f"{missing} vertices were not filled."
        )

    return (
        grid,
        min_x,
        min_y,
        dx,
        dy,
    )


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def write_obj(
    path,
    grid,
    min_x,
    min_y,
    dx,
    dy,
    scale=1.0,
    flip_y=False,
):
    """
    Write the stitched terrain as an OBJ.
    """

    ny = len(grid)
    nx = len(grid[0])

    with Path(path).open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:

        f.write(
            "# Fable 3 EHF terrain\n"
        )

        f.write(
            f"# stitched grid: {nx} x {ny}\n"
        )

        # Vertices
        for row in range(ny):

            src_row = (
                ny - 1 - row
                if flip_y
                else row
            )

            world_y = (
                min_y + src_row * dy
            )

            for col in range(nx):

                world_x = (
                    min_x + col * dx
                )

                elevation = (
                    grid[src_row][col]
                )

                ox = world_x * scale
                oy = elevation * scale
                oz = world_y * scale

                f.write(
                    f"v {ox:.6f} "
                    f"{oy:.6f} "
                    f"{oz:.6f}\n"
                )

        # Faces
        for row in range(ny - 1):

            for col in range(nx - 1):

                a = (
                    row * nx
                    + col
                    + 1
                )

                b = (
                    row * nx
                    + col
                    + 2
                )

                c = (
                    (row + 1) * nx
                    + col
                    + 2
                )

                d = (
                    (row + 1) * nx
                    + col
                    + 1
                )

                f.write(
                    f"f {a} {c} {b}\n"
                )

                f.write(
                    f"f {a} {d} {c}\n"
                )


def elevation_to_gray(elevation: float) -> int:
    """
    Direct mapping:

        elevation 0   -> grayscale 0
        elevation 255 -> grayscale 255

    Values outside the 0..255 range are clamped.
    """

    value = int(round(elevation))

    if value < 0:
        value = 0

    elif value > 255:
        value = 255

    return value


def write_pgm(
    path,
    grid,
    flip_y=False,
):
    """
    Write an 8-bit binary PGM heightmap.

    No normalization or OBJ scaling is applied.
    """

    ny = len(grid)
    nx = len(grid[0])

    with Path(path).open("wb") as f:

        f.write(
            (
                f"P5\n"
                f"{nx} {ny}\n"
                f"255\n"
            ).encode("ascii")
        )

        for row in range(ny):

            src_row = (
                ny - 1 - row
                if flip_y
                else row
            )

            for col in range(nx):

                elevation = (
                    grid[src_row][col]
                )

                value = elevation_to_gray(
                    elevation
                )

                f.write(
                    bytes((value,))
                )


def analyse_grid(grid):
    """
    Return elevation statistics.
    """

    values = [
        v
        for row in grid
        for v in row
    ]

    minimum = min(values)
    maximum = max(values)

    below_zero = sum(
        v < 0
        for v in values
    )

    above_255 = sum(
        v > 255
        for v in values
    )

    return (
        minimum,
        maximum,
        below_zero,
        above_255,
    )


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class EHFExporterGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Fable 3 EHF Terrain Exporter"
        )

        self.root.geometry(
            "850x700"
        )

        self.root.minsize(
            750,
            600
        )

        self.input_var = (
            tk.StringVar()
        )

        self.output_var = (
            tk.StringVar()
        )

        self.batch_input_var = (
            tk.StringVar()
        )

        self.batch_output_var = (
            tk.StringVar()
        )

        self.scale_var = (
            tk.StringVar(
                value="1.0"
            )
        )

        self.flip_y_var = (
            tk.BooleanVar(
                value=False
            )
        )

        self.export_obj_var = (
            tk.BooleanVar(
                value=True
            )
        )

        self.export_pgm_var = (
            tk.BooleanVar(
                value=True
            )
        )

        self.status_var = (
            tk.StringVar(
                value="Ready."
            )
        )

        self.build_gui()


    # -----------------------------------------------------------------------
    # GUI CONSTRUCTION
    # -----------------------------------------------------------------------

    def build_gui(self):

        main = ttk.Frame(
            self.root,
            padding=18
        )

        main.pack(
            fill="both",
            expand=True
        )

        title = ttk.Label(
            main,
            text="Fable 3 EHF Terrain Exporter",
            font=(
                "Segoe UI",
                18,
                "bold"
            )
        )

        title.pack(
            anchor="w",
            pady=(0, 5)
        )

        subtitle = ttk.Label(
            main,
            text=(
                "Extract stitched EHF terrain "
                "to OBJ and direct elevation PGM."
            )
        )

        subtitle.pack(
            anchor="w",
            pady=(0, 15)
        )


        # -------------------------------------------------------------------
        # NOTEBOOK
        # -------------------------------------------------------------------

        notebook = ttk.Notebook(main)

        notebook.pack(
            fill="x",
            pady=(0, 12)
        )

        single_tab = ttk.Frame(
            notebook,
            padding=12
        )

        batch_tab = ttk.Frame(
            notebook,
            padding=12
        )

        notebook.add(
            single_tab,
            text="Single File"
        )

        notebook.add(
            batch_tab,
            text="Batch Export"
        )


        # -------------------------------------------------------------------
        # SINGLE FILE TAB
        # -------------------------------------------------------------------

        ttk.Label(
            single_tab,
            text="Input EHF:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=5
        )

        self.input_entry = ttk.Entry(
            single_tab,
            textvariable=self.input_var
        )

        self.input_entry.grid(
            row=0,
            column=1,
            sticky="ew",
            pady=5
        )

        ttk.Button(
            single_tab,
            text="Browse...",
            command=self.browse_input
        ).grid(
            row=0,
            column=2,
            padx=(8, 0),
            pady=5
        )


        ttk.Label(
            single_tab,
            text="Output OBJ:"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=5
        )

        self.output_entry = ttk.Entry(
            single_tab,
            textvariable=self.output_var
        )

        self.output_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            pady=5
        )

        ttk.Button(
            single_tab,
            text="Browse...",
            command=self.browse_output
        ).grid(
            row=1,
            column=2,
            padx=(8, 0),
            pady=5
        )

        single_tab.columnconfigure(
            1,
            weight=1
        )


        # -------------------------------------------------------------------
        # BATCH TAB
        # -------------------------------------------------------------------

        ttk.Label(
            batch_tab,
            text="Input folder:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=5
        )

        self.batch_input_entry = ttk.Entry(
            batch_tab,
            textvariable=self.batch_input_var
        )

        self.batch_input_entry.grid(
            row=0,
            column=1,
            sticky="ew",
            pady=5
        )

        ttk.Button(
            batch_tab,
            text="Browse...",
            command=self.browse_batch_input
        ).grid(
            row=0,
            column=2,
            padx=(8, 0),
            pady=5
        )


        ttk.Label(
            batch_tab,
            text="Output folder:"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=5
        )

        self.batch_output_entry = ttk.Entry(
            batch_tab,
            textvariable=self.batch_output_var
        )

        self.batch_output_entry.grid(
            row=1,
            column=1,
            sticky="ew",
            pady=5
        )

        ttk.Button(
            batch_tab,
            text="Browse...",
            command=self.browse_batch_output
        ).grid(
            row=1,
            column=2,
            padx=(8, 0),
            pady=5
        )

        batch_tab.columnconfigure(
            1,
            weight=1
        )


        ttk.Label(
            batch_tab,
            text=(
                "All .ehf files directly inside the input "
                "folder will be exported."
            )
        ).grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(10, 0)
        )


        # -------------------------------------------------------------------
        # OPTIONS
        # -------------------------------------------------------------------

        options_frame = ttk.LabelFrame(
            main,
            text="Export Options",
            padding=12
        )

        options_frame.pack(
            fill="x",
            pady=(0, 12)
        )


        ttk.Label(
            options_frame,
            text="OBJ Scale:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8)
        )

        self.scale_entry = ttk.Entry(
            options_frame,
            textvariable=self.scale_var,
            width=15
        )

        self.scale_entry.grid(
            row=0,
            column=1,
            sticky="w"
        )

        ttk.Label(
            options_frame,
            text="1.0 = original EHF coordinate scale"
        ).grid(
            row=0,
            column=2,
            sticky="w",
            padx=(12, 0)
        )


        ttk.Checkbutton(
            options_frame,
            text="Flip Y",
            variable=self.flip_y_var
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(10, 0)
        )

        ttk.Label(
            options_frame,
            text="Applies to both OBJ and PGM."
        ).grid(
            row=1,
            column=2,
            sticky="w",
            padx=(12, 0),
            pady=(10, 0)
        )


        ttk.Checkbutton(
            options_frame,
            text="Export OBJ",
            variable=self.export_obj_var
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(10, 0)
        )


        ttk.Checkbutton(
            options_frame,
            text="Export PGM",
            variable=self.export_pgm_var
        ).grid(
            row=2,
            column=1,
            sticky="w",
            pady=(10, 0)
        )


        # -------------------------------------------------------------------
        # ACTIONS
        # -------------------------------------------------------------------

        action_frame = ttk.Frame(
            main
        )

        action_frame.pack(
            fill="x",
            pady=(0, 12)
        )


        self.export_button = ttk.Button(
            action_frame,
            text="EXPORT TERRAIN",
            command=self.start_export
        )

        self.export_button.pack(
            side="left"
        )


        self.batch_button = ttk.Button(
            action_frame,
            text="BATCH EXPORT",
            command=self.start_batch_export
        )

        self.batch_button.pack(
            side="left",
            padx=(8, 0)
        )


        self.progress = ttk.Progressbar(
            action_frame,
            mode="indeterminate"
        )

        self.progress.pack(
            side="left",
            fill="x",
            expand=True,
            padx=(15, 0)
        )


        # -------------------------------------------------------------------
        # STATUS / LOG
        # -------------------------------------------------------------------

        status_frame = ttk.LabelFrame(
            main,
            text="Export Information",
            padding=10
        )

        status_frame.pack(
            fill="both",
            expand=True
        )


        self.status_text = tk.Text(
            status_frame,
            height=15,
            wrap="word",
            state="disabled",
            font=("Consolas", 10)
        )

        self.status_text.pack(
            side="left",
            fill="both",
            expand=True
        )


        scrollbar = ttk.Scrollbar(
            status_frame,
            orient="vertical",
            command=self.status_text.yview
        )

        scrollbar.pack(
            side="right",
            fill="y"
        )


        self.status_text.configure(
            yscrollcommand=scrollbar.set
        )


        status_bar = ttk.Label(
            main,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w",
            padding=(6, 4)
        )

        status_bar.pack(
            fill="x",
            pady=(10, 0)
        )


    # -----------------------------------------------------------------------
    # LOGGING
    # -----------------------------------------------------------------------

    def log(self, text=""):
        self.root.after(
            0,
            self._log_now,
            text
        )


    def _log_now(self, text):

        self.status_text.configure(
            state="normal"
        )

        self.status_text.insert(
            "end",
            text + "\n"
        )

        self.status_text.see(
            "end"
        )

        self.status_text.configure(
            state="disabled"
        )


    def set_status(self, text):

        self.root.after(
            0,
            self.status_var.set,
            text
        )


    def clear_log(self):

        self.status_text.configure(
            state="normal"
        )

        self.status_text.delete(
            "1.0",
            "end"
        )

        self.status_text.configure(
            state="disabled"
        )


    # -----------------------------------------------------------------------
    # BROWSERS
    # -----------------------------------------------------------------------

    def browse_input(self):

        filename = filedialog.askopenfilename(
            title="Select Fable 3 EHF file",
            filetypes=[
                ("EHF files", "*.ehf"),
                ("All files", "*.*"),
            ]
        )

        if not filename:
            return

        self.input_var.set(
            filename
        )

        if not self.output_var.get():

            input_path = Path(
                filename
            )

            self.output_var.set(
                str(
                    input_path.with_name(
                        input_path.stem + ".obj"
                    )
                )
            )


    def browse_output(self):

        filename = filedialog.asksaveasfilename(
            title="Choose OBJ output",
            defaultextension=".obj",
            filetypes=[
                ("OBJ files", "*.obj"),
                ("All files", "*.*"),
            ]
        )

        if filename:
            self.output_var.set(
                filename
            )


    def browse_batch_input(self):

        folder = filedialog.askdirectory(
            title="Select folder containing EHF files"
        )

        if folder:
            self.batch_input_var.set(
                folder
            )

            if not self.batch_output_var.get():

                input_path = Path(
                    folder
                )

                self.batch_output_var.set(
                    str(
                        input_path / "exported"
                    )
                )


    def browse_batch_output(self):

        folder = filedialog.askdirectory(
            title="Select batch output folder"
        )

        if folder:
            self.batch_output_var.set(
                folder
            )


    # -----------------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------------

    def get_scale(self):

        try:

            scale = float(
                self.scale_var.get()
            )

        except ValueError:

            raise ValueError(
                "Scale must be a valid number."
            )

        if not math.isfinite(scale):

            raise ValueError(
                "Scale must be a finite number."
            )

        return scale


    def outputs_selected(self):

        if not (
            self.export_obj_var.get()
            or
            self.export_pgm_var.get()
        ):

            messagebox.showerror(
                "No outputs selected",
                "Please select OBJ, PGM, or both."
            )

            return False

        return True


    # -----------------------------------------------------------------------
    # SINGLE EXPORT
    # -----------------------------------------------------------------------

    def start_export(self):

        input_name = (
            self.input_var.get().strip()
        )

        output_name = (
            self.output_var.get().strip()
        )

        if not input_name:

            messagebox.showerror(
                "Missing input",
                "Please select an EHF file."
            )

            return

        if not output_name:

            messagebox.showerror(
                "Missing output",
                "Please select an output OBJ file."
            )

            return

        if not self.outputs_selected():
            return

        input_path = Path(
            input_name
        )

        output_path = Path(
            output_name
        )

        if not input_path.is_file():

            messagebox.showerror(
                "Input not found",
                f"The input file does not exist:\n\n"
                f"{input_path}"
            )

            return

        try:

            scale = self.get_scale()

        except ValueError as exc:

            messagebox.showerror(
                "Invalid scale",
                str(exc)
            )

            return

        self.clear_log()

        self.set_controls_enabled(
            False
        )

        self.progress.start(
            10
        )

        self.set_status(
            "Exporting..."
        )

        thread = threading.Thread(
            target=self.export_worker,
            args=(
                input_path,
                output_path,
                scale,
                self.flip_y_var.get(),
                self.export_obj_var.get(),
                self.export_pgm_var.get(),
            ),
            daemon=True
        )

        thread.start()


    def export_worker(
        self,
        input_path,
        output_path,
        scale,
        flip_y,
        export_obj,
        export_pgm,
    ):

        try:

            self.log(
                "Fable 3 EHF Terrain Exporter"
            )

            self.log(
                "=" * 70
            )

            self.log()

            self.log(
                f"Input:  {input_path}"
            )

            self.log(
                f"Output: {output_path}"
            )

            self.log(
                f"Scale:  {scale}"
            )

            self.log(
                f"Flip Y: {flip_y}"
            )

            self.log()

            result = self.process_ehf(
                input_path,
                output_path,
                scale,
                flip_y,
                export_obj,
                export_pgm,
                log_patches=True,
            )

            self.log()

            self.log(
                "=" * 70
            )

            self.log(
                "EXPORT COMPLETE"
            )

            self.root.after(
                0,
                self.export_finished,
                result
            )

        except Exception as exc:

            error_text = (
                f"{type(exc).__name__}: {exc}"
            )

            self.log()

            self.log(
                "=" * 70
            )

            self.log(
                "EXPORT FAILED"
            )

            self.log(
                error_text
            )

            self.log()

            self.log(
                traceback.format_exc()
            )

            self.root.after(
                0,
                self.export_failed,
                error_text
            )


    # -----------------------------------------------------------------------
    # BATCH EXPORT
    # -----------------------------------------------------------------------

    def start_batch_export(self):

        input_name = (
            self.batch_input_var.get().strip()
        )

        output_name = (
            self.batch_output_var.get().strip()
        )

        if not input_name:

            messagebox.showerror(
                "Missing input folder",
                "Please select the folder containing the EHF files."
            )

            return

        if not output_name:

            messagebox.showerror(
                "Missing output folder",
                "Please select an output folder."
            )

            return

        if not self.outputs_selected():
            return

        input_folder = Path(
            input_name
        )

        output_folder = Path(
            output_name
        )

        if not input_folder.is_dir():

            messagebox.showerror(
                "Input folder not found",
                f"The input folder does not exist:\n\n"
                f"{input_folder}"
            )

            return

        ehf_files = sorted(
            [
                p
                for p in input_folder.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".ehf"
            ],
            key=lambda p: p.name.lower()
        )

        if not ehf_files:

            messagebox.showerror(
                "No EHF files",
                "No .ehf files were found directly inside:\n\n"
                f"{input_folder}"
            )

            return

        try:

            scale = self.get_scale()

        except ValueError as exc:

            messagebox.showerror(
                "Invalid scale",
                str(exc)
            )

            return


        # Confirm before starting a potentially large batch.

        answer = messagebox.askyesno(
            "Confirm batch export",
            (
                f"Found {len(ehf_files)} EHF file(s).\n\n"
                f"Input:\n{input_folder}\n\n"
                f"Output:\n{output_folder}\n\n"
                "Continue?"
            )
        )

        if not answer:
            return


        self.clear_log()

        self.set_controls_enabled(
            False
        )

        self.progress.configure(
            mode="determinate",
            maximum=len(ehf_files),
            value=0
        )

        self.set_status(
            f"Batch export: 0 / {len(ehf_files)}"
        )


        thread = threading.Thread(
            target=self.batch_worker,
            args=(
                ehf_files,
                output_folder,
                scale,
                self.flip_y_var.get(),
                self.export_obj_var.get(),
                self.export_pgm_var.get(),
            ),
            daemon=True
        )

        thread.start()


    def batch_worker(
        self,
        ehf_files,
        output_folder,
        scale,
        flip_y,
        export_obj,
        export_pgm,
    ):

        successful = []
        failed = []

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )


        self.log(
            "Fable 3 EHF Terrain Exporter - BATCH MODE"
        )

        self.log(
            "=" * 70
        )

        self.log()

        self.log(
            f"Input EHF files: {len(ehf_files)}"
        )

        self.log(
            f"Output folder:   {output_folder}"
        )

        self.log(
            f"Scale:            {scale}"
        )

        self.log(
            f"Flip Y:           {flip_y}"
        )

        self.log(
            f"Export OBJ:       {export_obj}"
        )

        self.log(
            f"Export PGM:       {export_pgm}"
        )

        self.log()


        total = len(ehf_files)


        for index, input_path in enumerate(
            ehf_files,
            start=1
        ):

            self.log(
                ""
            )

            self.log(
                "#" * 70
            )

            self.log(
                f"FILE {index} / {total}: "
                f"{input_path.name}"
            )

            self.log(
                "#" * 70
            )

            self.root.after(
                0,
                self.update_batch_progress,
                index - 1,
                total,
            )


            output_path = (
                output_folder
                /
                f"{input_path.stem}.obj"
            )


            try:

                result = self.process_ehf(
                    input_path,
                    output_path,
                    scale,
                    flip_y,
                    export_obj,
                    export_pgm,
                    log_patches=False,
                )

                successful.append(
                    (
                        input_path,
                        result,
                    )
                )

                self.log()

                self.log(
                    f"SUCCESS: {input_path.name}"
                )

            except Exception as exc:

                error_text = (
                    f"{type(exc).__name__}: {exc}"
                )

                failed.append(
                    (
                        input_path,
                        error_text,
                    )
                )

                self.log()

                self.log(
                    f"FAILED: {input_path.name}"
                )

                self.log(
                    error_text
                )

                self.log(
                    traceback.format_exc()
                )


            self.root.after(
                0,
                self.update_batch_progress,
                index,
                total,
            )


        # ---------------------------------------------------------------
        # Batch summary
        # ---------------------------------------------------------------

        self.log()

        self.log(
            "=" * 70
        )

        self.log(
            "BATCH EXPORT COMPLETE"
        )

        self.log(
            "=" * 70
        )

        self.log()

        self.log(
            f"Total files: {total}"
        )

        self.log(
            f"Successful:  {len(successful)}"
        )

        self.log(
            f"Failed:      {len(failed)}"
        )


        if failed:

            self.log()

            self.log(
                "FAILED FILES:"
            )

            for input_path, error_text in failed:

                self.log(
                    f"  {input_path.name}: "
                    f"{error_text}"
                )


        # Write batch log.

        log_path = (
            output_folder
            /
            "batch_export_log.txt"
        )

        try:

            with log_path.open(
                "w",
                encoding="utf-8"
            ) as log_file:

                log_file.write(
                    "Fable 3 EHF Batch Export Log\n"
                )

                log_file.write(
                    "=" * 60 + "\n\n"
                )

                log_file.write(
                    f"Input folder: {ehf_files[0].parent}\n"
                )

                log_file.write(
                    f"Output folder: {output_folder}\n"
                )

                log_file.write(
                    f"Scale: {scale}\n"
                )

                log_file.write(
                    f"Flip Y: {flip_y}\n"
                )

                log_file.write(
                    f"OBJ: {export_obj}\n"
                )

                log_file.write(
                    f"PGM: {export_pgm}\n\n"
                )

                log_file.write(
                    f"Total: {total}\n"
                )

                log_file.write(
                    f"Successful: {len(successful)}\n"
                )

                log_file.write(
                    f"Failed: {len(failed)}\n\n"
                )

                if failed:

                    log_file.write(
                        "FAILED FILES\n"
                    )

                    log_file.write(
                        "-" * 60 + "\n"
                    )

                    for (
                        input_path,
                        error_text,
                    ) in failed:

                        log_file.write(
                            f"{input_path.name}: "
                            f"{error_text}\n"
                        )

        except Exception as exc:

            self.log()

            self.log(
                "WARNING: Could not write "
                "batch_export_log.txt"
            )

            self.log(
                str(exc)
            )

            log_path = None


        self.root.after(
            0,
            self.batch_finished,
            total,
            successful,
            failed,
            log_path,
        )


    # -----------------------------------------------------------------------
    # COMMON EHF PROCESSING
    # -----------------------------------------------------------------------

    def process_ehf(
        self,
        input_path,
        output_path,
        scale,
        flip_y,
        export_obj,
        export_pgm,
        log_patches=False,
    ):
        """
        Process one EHF file.

        This is deliberately shared by single-file and batch mode.
        """

        self.log(
            "Reading EHF file..."
        )

        data = input_path.read_bytes()

        self.log(
            f"File size: {len(data):,} bytes"
        )


        self.log(
            "Discovering terrain patches..."
        )

        patches = discover_patches(
            data
        )

        self.log(
            f"Terrain patches: {len(patches)}"
        )


        if log_patches:

            self.log()

            for i, p in enumerate(
                patches,
                1
            ):

                self.log(
                    f"Patch {i}: "
                    f"0x{p.offset:X}, "
                    f"{p.width}x{p.height}, "
                    f"AABB=("
                    f"{p.min_x:.3f}, "
                    f"{p.min_y:.3f}, "
                    f"{p.min_z:.3f}"
                    f") -> ("
                    f"{p.max_x:.3f}, "
                    f"{p.max_y:.3f}, "
                    f"{p.max_z:.3f}"
                    f")"
                )


        self.log(
            "Stitching terrain patches..."
        )

        (
            grid,
            min_x,
            min_y,
            dx,
            dy,
        ) = stitch(
            patches
        )


        nx = len(
            grid[0]
        )

        ny = len(
            grid
        )


        (
            elevation_min,
            elevation_max,
            below_zero,
            above_255,
        ) = analyse_grid(
            grid
        )


        self.log(
            f"Stitched grid: "
            f"{nx} x {ny}"
        )

        self.log(
            f"Spacing: "
            f"X={dx:.6f}, "
            f"Y={dy:.6f}"
        )

        self.log(
            f"Elevation: "
            f"{elevation_min:.6f} .. "
            f"{elevation_max:.6f}"
        )


        # ---------------------------------------------------------------
        # OBJ
        # ---------------------------------------------------------------

        if export_obj:

            self.log(
                "Writing OBJ..."
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            write_obj(
                output_path,
                grid,
                min_x,
                min_y,
                dx,
                dy,
                scale=scale,
                flip_y=flip_y,
            )

            self.log(
                f"Wrote OBJ: "
                f"{output_path}"
            )


        # ---------------------------------------------------------------
        # PGM
        # ---------------------------------------------------------------

        pgm_path = None

        if export_pgm:

            pgm_path = (
                output_path.with_name(
                    output_path.stem
                    +
                    "_height.pgm"
                )
            )

            self.log(
                "Writing direct-mapped "
                "PGM heightmap..."
            )

            write_pgm(
                pgm_path,
                grid,
                flip_y=flip_y,
            )

            self.log(
                f"Wrote PGM: "
                f"{pgm_path}"
            )

            self.log(
                "PGM mapping:"
            )

            self.log(
                "  Raw EHF elevation -> "
                "grayscale 0..255"
            )

            self.log(
                "  No min/max normalization"
            )

            self.log(
                "  No contrast stretching"
            )

            self.log(
                "  No OBJ scale applied"
            )


            if below_zero:

                self.log(
                    f"  WARNING: "
                    f"{below_zero:,} samples "
                    f"below 0 were clamped."
                )


            if above_255:

                self.log(
                    f"  WARNING: "
                    f"{above_255:,} samples "
                    f"above 255 were clamped."
                )


        return {
            "input": input_path,
            "obj": output_path
                if export_obj
                else None,
            "pgm": pgm_path,
            "patches": len(patches),
            "nx": nx,
            "ny": ny,
            "dx": dx,
            "dy": dy,
            "elevation_min": elevation_min,
            "elevation_max": elevation_max,
        }


    # -----------------------------------------------------------------------
    # GUI STATE
    # -----------------------------------------------------------------------

    def set_controls_enabled(
        self,
        enabled
    ):

        state = (
            "normal"
            if enabled
            else
            "disabled"
        )

        self.export_button.configure(
            state=state
        )

        self.batch_button.configure(
            state=state
        )

        self.input_entry.configure(
            state=state
        )

        self.output_entry.configure(
            state=state
        )

        self.batch_input_entry.configure(
            state=state
        )

        self.batch_output_entry.configure(
            state=state
        )

        self.scale_entry.configure(
            state=state
        )


    def update_batch_progress(
        self,
        completed,
        total
    ):

        self.progress.configure(
            value=completed
        )

        self.set_status(
            f"Batch export: "
            f"{completed} / {total}"
        )


    # -----------------------------------------------------------------------
    # COMPLETION
    # -----------------------------------------------------------------------

    def export_finished(
        self,
        result
    ):

        self.progress.stop()

        self.progress.configure(
            mode="indeterminate"
        )

        self.set_controls_enabled(
            True
        )

        self.set_status(
            "Export complete."
        )


        outputs = []

        if result["obj"]:

            outputs.append(
                f"OBJ:\n{result['obj']}"
            )

        if result["pgm"]:

            outputs.append(
                f"Heightmap:\n{result['pgm']}"
            )


        messagebox.showinfo(
            "Export complete",
            (
                "Terrain export completed successfully.\n\n"
                +
                "\n\n".join(outputs)
            )
        )


    def batch_finished(
        self,
        total,
        successful,
        failed,
        log_path,
    ):

        self.set_controls_enabled(
            True
        )

        self.set_status(
            f"Batch complete: "
            f"{len(successful)} / {total} succeeded."
        )


        log_message = ""

        if log_path:

            log_message = (
                f"\n\nLog:\n{log_path}"
            )


        if failed:

            messagebox.showwarning(
                "Batch export completed",
                (
                    f"Batch export finished.\n\n"
                    f"Total files: {total}\n"
                    f"Successful: {len(successful)}\n"
                    f"Failed: {len(failed)}"
                    f"{log_message}"
                )
            )

        else:

            messagebox.showinfo(
                "Batch export complete",
                (
                    f"All {total} EHF files exported "
                    f"successfully."
                    f"{log_message}"
                )
            )


    def export_failed(
        self,
        error_text
    ):

        self.progress.stop()

        self.progress.configure(
            mode="indeterminate"
        )

        self.set_controls_enabled(
            True
        )

        self.set_status(
            "Export failed."
        )

        messagebox.showerror(
            "Export failed",
            error_text
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    root = tk.Tk()

    try:

        style = ttk.Style(
            root
        )

        if "vista" in style.theme_names():

            style.theme_use(
                "vista"
            )

    except Exception:

        pass


    EHFExporterGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()
