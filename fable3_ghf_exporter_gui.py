#!/usr/bin/env python3
"""
Fable 3 GHF Terrain Exporter - GUI + Batch Export

Supports:

    - Single GHF export
    - Batch GHF export
    - Decompressed GHF files
    - gzip/zlib-compressed GHF files
    - Variable terrain dimensions

Outputs:

    OBJ terrain
    PGM terrain visualisation
    TXT diagnostic information

Fable 3 GHF format:

    28-byte little-endian header

        +00 float32
        +04 float32
        +08 uint32
        +0C uint32     width
        +10 uint32     height
        +14 float32    elevation/base value
        +18 uint32

    Followed by width * height records.

    Each record is exactly 14 bytes:

        +00..+03    float32 LE   terrain elevation
        +04..+07    float32 LE   secondary field
        +08..+0B    packed data
        +0C..+0D    flags

IMPORTANT:

    Terrain elevation is ONLY the float32 at record +00.

    Do NOT:

      - reconstruct elevation from Photoshop channels
      - combine bytes +02/+03 as a 16-bit height
      - multiply +00 by +04
      - treat +07 as an alpha channel
      - apply smoothing/interpolation to elevation

Horizontal scale:

    X = column * 0.5
    Z = row    * 0.5
    Y = raw elevation

OBJ axes:

    X = map column
    Y = elevation
    Z = map row

PGM:

    The PGM uses the absolute raw elevation values.

    Raw elevation is rounded to the nearest integer and
    clamped to the 0..255 PGM range.

    There is NO minimum/maximum normalization or contrast
    stretching.

    Examples:

        0   -> 0
        15  -> 15
        60  -> 60
        128 -> 128
        255 -> 255

    Values below 0 are clamped to 0.
    Values above 255 are clamped to 255.

    This conversion does NOT affect the OBJ.

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
import zlib

from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ============================================================================
# GHF FORMAT CONSTANTS
# ============================================================================

HEADER_SIZE = 28
RECORD_SIZE = 14

HORIZONTAL_SCALE = 0.5

HDR_WIDTH = 0x0C
HDR_HEIGHT = 0x10
HDR_BASE_HEIGHT = 0x14

REC_HEIGHT = 0x00


# ============================================================================
# LOW-LEVEL READING
# ============================================================================

def read_u32_le(data, offset):
    return struct.unpack_from(
        "<I",
        data,
        offset
    )[0]


def read_f32_le(data, offset):
    return struct.unpack_from(
        "<f",
        data,
        offset
    )[0]


def load_ghf(path):
    """
    Load either compressed or already-decompressed GHF.

    First attempts gzip/zlib decompression.
    If decompression fails, the file is treated as raw/decompressed GHF.
    """

    raw = Path(path).read_bytes()

    try:

        decompressor = zlib.decompressobj(
            15 + 32
        )

        data = (
            decompressor.decompress(raw)
            +
            decompressor.flush()
        )

        if len(data) >= HEADER_SIZE:
            return data

    except zlib.error:
        pass

    return raw


def parse_header(data):
    """
    Read and validate the 28-byte GHF header.
    """

    if len(data) < HEADER_SIZE:

        raise ValueError(
            "File is smaller than the "
            "28-byte GHF header."
        )

    width = read_u32_le(
        data,
        HDR_WIDTH
    )

    height = read_u32_le(
        data,
        HDR_HEIGHT
    )

    base_height = read_f32_le(
        data,
        HDR_BASE_HEIGHT
    )

    if width <= 0 or height <= 0:

        raise ValueError(
            f"Invalid dimensions: "
            f"{width} x {height}"
        )

    record_count = (
        width * height
    )

    expected_size = (
        HEADER_SIZE
        +
        record_count * RECORD_SIZE
    )

    if len(data) != expected_size:

        raise ValueError(
            "\n"
            "Unexpected decompressed file size.\n\n"
            f"Actual size:   {len(data):,} bytes\n"
            f"Expected size: {expected_size:,} bytes\n"
            f"Header:        {HEADER_SIZE} bytes\n"
            f"Dimensions:    {width} x {height}\n"
            f"Records:       {record_count:,}\n"
            f"Record size:   {RECORD_SIZE} bytes\n"
        )

    if not math.isfinite(base_height):

        raise ValueError(
            f"Invalid header +14 value: "
            f"{base_height}"
        )

    return (
        width,
        height,
        base_height
    )


def read_heights(
    data,
    width,
    height
):
    """
    Extract the raw terrain elevation from record +00.

    Returns:

        heights
        minimum
        maximum
    """

    count = (
        width * height
    )

    heights = [
        0.0
    ] * count

    offset = HEADER_SIZE

    min_height = float("inf")
    max_height = float("-inf")

    for i in range(count):

        height_value = read_f32_le(
            data,
            offset + REC_HEIGHT
        )

        if not (
            math.isfinite(height_value)
            and
            -1e30 < height_value < 1e30
        ):

            raise ValueError(
                f"Invalid elevation at "
                f"record {i}: "
                f"{height_value}"
            )

        heights[i] = (
            height_value
        )

        if height_value < min_height:
            min_height = (
                height_value
            )

        if height_value > max_height:
            max_height = (
                height_value
            )

        offset += RECORD_SIZE

    return (
        heights,
        min_height,
        max_height
    )


# ============================================================================
# OUTPUT
# ============================================================================

def write_obj(
    output_path,
    heights,
    width,
    height,
    horizontal_scale=HORIZONTAL_SCALE,
    elevation_scale=1.0,
    elevation_offset=0.0,
    flip_y=False,
):
    """
    Write the terrain as a regular OBJ grid.

    Default:

        X = x * 0.5
        Y = raw elevation
        Z = y * 0.5

    elevation_scale and elevation_offset are retained as optional
    controls, but default to the established raw values.

    flip_y changes the exported row order without changing elevation.
    """

    with open(
        output_path,
        "w",
        encoding="ascii",
        newline="\n"
    ) as f:

        f.write(
            "# Fable 3 GHF terrain\n"
        )

        f.write(
            f"# Width: {width}\n"
        )

        f.write(
            f"# Height: {height}\n"
        )

        f.write(
            f"# Horizontal sample spacing: "
            f"{horizontal_scale}\n"
        )

        f.write(
            "# Elevation: record +00 "
            "float32 LE\n"
        )

        f.write("\n")

        # --------------------------------------------------------------------
        # Vertices
        # --------------------------------------------------------------------

        for output_y in range(height):

            if flip_y:

                source_y = (
                    height - 1 - output_y
                )

            else:

                source_y = output_y

            row = (
                source_y * width
            )

            for x in range(width):

                elevation = (
                    heights[row + x]
                )

                px = (
                    x
                    *
                    horizontal_scale
                )

                py = (
                    elevation
                    *
                    elevation_scale
                    +
                    elevation_offset
                )

                pz = (
                    output_y
                    *
                    horizontal_scale
                )

                f.write(
                    f"v {px:.9f} "
                    f"{py:.9f} "
                    f"{pz:.9f}\n"
                )

        f.write("\n")

        # --------------------------------------------------------------------
        # Faces
        # --------------------------------------------------------------------

        for y in range(height - 1):

            row0 = (
                y * width
            )

            row1 = (
                (y + 1) * width
            )

            for x in range(width - 1):

                a = (
                    row0
                    +
                    x
                    +
                    1
                )

                b = (
                    row0
                    +
                    x
                    +
                    2
                )

                c = (
                    row1
                    +
                    x
                    +
                    1
                )

                d = (
                    row1
                    +
                    x
                    +
                    2
                )

                f.write(
                    f"f {a} {c} {b}\n"
                )

                f.write(
                    f"f {b} {c} {d}\n"
                )


def write_pgm(
    output_path,
    heights,
    width,
    height,
    flip_y=False,
):
    """
    Write an 8-bit PGM using the ABSOLUTE raw GHF elevation values.

    Mapping:

        0   -> 0
        15  -> 15
        60  -> 60
        128 -> 128
        255 -> 255

    Values below 0 are clamped to 0.
    Values above 255 are clamped to 255.

    Fractional values are rounded to the nearest integer.

    There is NO min/max normalization or contrast stretching.

    The PGM therefore represents the absolute raw elevation values,
    rather than merely visualizing the relative elevation range.

    Flip Y changes only the row orientation.

    Returns:

        clamp_low
        clamp_high
    """

    pixels = bytearray(
        width * height
    )

    clamp_low = 0
    clamp_high = 0

    for output_y in range(height):

        if flip_y:

            source_y = (
                height - 1 - output_y
            )

        else:

            source_y = output_y

        source_row = (
            source_y * width
        )

        output_row = (
            output_y * width
        )

        for x in range(width):

            value = (
                heights[
                    source_row + x
                ]
            )

            # --------------------------------------------------------------
            # DIRECT ABSOLUTE VALUE MAPPING
            #
            # Do NOT normalize against the terrain minimum/maximum.
            # --------------------------------------------------------------

            gray = int(
                round(value)
            )

            if gray < 0:

                gray = 0
                clamp_low += 1

            elif gray > 255:

                gray = 255
                clamp_high += 1

            pixels[
                output_row + x
            ] = gray

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            (
                f"P5\n"
                f"{width} {height}\n"
                f"255\n"
            ).encode("ascii")
        )

        f.write(
            pixels
        )

    return (
        clamp_low,
        clamp_high
    )


def write_info(
    output_path,
    width,
    height,
    base_height,
    minimum,
    maximum,
):
    """
    Write diagnostic GHF information.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FABLE 3 GHF TERRAIN INFORMATION\n"
        )

        f.write(
            "=" * 60
            +
            "\n\n"
        )

        f.write(
            f"Width:                  "
            f"{width}\n"
        )

        f.write(
            f"Height:                 "
            f"{height}\n"
        )

        f.write(
            f"Samples:                "
            f"{width * height:,}\n"
        )

        f.write(
            f"Record size:            "
            f"{RECORD_SIZE} bytes\n"
        )

        f.write(
            f"Header size:            "
            f"{HEADER_SIZE} bytes\n"
        )

        f.write("\n")

        f.write(
            f"Horizontal scale:       "
            f"{HORIZONTAL_SCALE}\n"
        )

        f.write(
            f"Terrain width:          "
            f"{(width - 1) * HORIZONTAL_SCALE}\n"
        )

        f.write(
            f"Terrain depth:          "
            f"{(height - 1) * HORIZONTAL_SCALE}\n"
        )

        f.write("\n")

        f.write(
            f"Header +14:             "
            f"{base_height:.12f}\n"
        )

        f.write(
            f"Minimum record +00:     "
            f"{minimum:.12f}\n"
        )

        f.write(
            f"Maximum record +00:     "
            f"{maximum:.12f}\n"
        )

        f.write(
            f"Elevation range:        "
            f"{maximum - minimum:.12f}\n"
        )

        f.write("\n")

        f.write(
            "ELEVATION SOURCE\n"
        )

        f.write(
            "-" * 60
            +
            "\n"
        )

        f.write(
            "Record offset +00, "
            "little-endian float32.\n"
        )

        f.write("\n")

        f.write(
            "OTHER RECORD FIELDS\n"
        )

        f.write(
            "-" * 60
            +
            "\n"
        )

        f.write(
            "+04..+07 : secondary float\n"
        )

        f.write(
            "+08..+0B : packed data\n"
        )

        f.write(
            "+0C..+0D : flags\n"
        )


# ============================================================================
# GUI
# ============================================================================

class GHFExporterGUI:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Fable 3 GHF Terrain Exporter"
        )

        self.root.geometry(
            "900x720"
        )

        self.root.minsize(
            760,
            620
        )

        # Single-file variables

        self.input_var = (
            tk.StringVar()
        )

        self.output_var = (
            tk.StringVar()
        )

        # Batch variables

        self.batch_input_var = (
            tk.StringVar()
        )

        self.batch_output_var = (
            tk.StringVar()
        )

        # Options

        self.scale_var = (
            tk.StringVar(
                value="1.0"
            )
        )

        self.horizontal_scale_var = (
            tk.StringVar(
                value="0.5"
            )
        )

        self.elevation_offset_var = (
            tk.StringVar(
                value="0.0"
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

        self.export_info_var = (
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


    # =========================================================================
    # GUI CONSTRUCTION
    # =========================================================================

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
            text="Fable 3 GHF Terrain Exporter",
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
                "Extract Fable 3 GHF terrain "
                "to OBJ, PGM and diagnostic information."
            )
        )

        subtitle.pack(
            anchor="w",
            pady=(0, 15)
        )

        # ---------------------------------------------------------------------
        # NOTEBOOK
        # ---------------------------------------------------------------------

        notebook = ttk.Notebook(
            main
        )

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

        # ---------------------------------------------------------------------
        # SINGLE FILE
        # ---------------------------------------------------------------------

        ttk.Label(
            single_tab,
            text="Input GHF:"
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

        # ---------------------------------------------------------------------
        # BATCH
        # ---------------------------------------------------------------------

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

        ttk.Label(
            batch_tab,
            text=(
                "All .ghf files directly inside the selected "
                "input folder will be exported."
            )
        ).grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(10, 0)
        )

        batch_tab.columnconfigure(
            1,
            weight=1
        )

        # ---------------------------------------------------------------------
        # OPTIONS
        # ---------------------------------------------------------------------

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
            text="OBJ Elevation Scale:"
        ).grid(
            row=0,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3
        )

        self.scale_entry = ttk.Entry(
            options_frame,
            textvariable=self.scale_var,
            width=12
        )

        self.scale_entry.grid(
            row=0,
            column=1,
            sticky="w",
            pady=3
        )

        ttk.Label(
            options_frame,
            text="1.0 = raw GHF elevation"
        ).grid(
            row=0,
            column=2,
            sticky="w",
            padx=(12, 0),
            pady=3
        )

        ttk.Label(
            options_frame,
            text="Horizontal Scale:"
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3
        )

        self.horizontal_scale_entry = ttk.Entry(
            options_frame,
            textvariable=self.horizontal_scale_var,
            width=12
        )

        self.horizontal_scale_entry.grid(
            row=1,
            column=1,
            sticky="w",
            pady=3
        )

        ttk.Label(
            options_frame,
            text="Established value = 0.5"
        ).grid(
            row=1,
            column=2,
            sticky="w",
            padx=(12, 0),
            pady=3
        )

        ttk.Label(
            options_frame,
            text="Elevation Offset:"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=3
        )

        self.elevation_offset_entry = ttk.Entry(
            options_frame,
            textvariable=self.elevation_offset_var,
            width=12
        )

        self.elevation_offset_entry.grid(
            row=2,
            column=1,
            sticky="w",
            pady=3
        )

        ttk.Label(
            options_frame,
            text="0.0 = no vertical offset"
        ).grid(
            row=2,
            column=2,
            sticky="w",
            padx=(12, 0),
            pady=3
        )

        ttk.Checkbutton(
            options_frame,
            text="Flip Y",
            variable=self.flip_y_var
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Label(
            options_frame,
            text="Changes OBJ and PGM row orientation."
        ).grid(
            row=3,
            column=2,
            sticky="w",
            padx=(12, 0),
            pady=(8, 0)
        )

        ttk.Checkbutton(
            options_frame,
            text="Export OBJ",
            variable=self.export_obj_var
        ).grid(
            row=4,
            column=0,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Checkbutton(
            options_frame,
            text="Export PGM",
            variable=self.export_pgm_var
        ).grid(
            row=4,
            column=1,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Checkbutton(
            options_frame,
            text="Export Info TXT",
            variable=self.export_info_var
        ).grid(
            row=4,
            column=2,
            sticky="w",
            pady=(8, 0)
        )

        # ---------------------------------------------------------------------
        # ACTIONS
        # ---------------------------------------------------------------------

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

        # ---------------------------------------------------------------------
        # LOG
        # ---------------------------------------------------------------------

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


    # =========================================================================
    # LOGGING
    # =========================================================================

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


    # =========================================================================
    # BROWSERS
    # =========================================================================

    def browse_input(self):

        filename = filedialog.askopenfilename(
            title="Select Fable 3 GHF file",
            filetypes=[
                ("GHF files", "*.ghf"),
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
                        input_path.stem
                        +
                        "_terrain.obj"
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
            title="Select folder containing GHF files"
        )

        if not folder:
            return

        self.batch_input_var.set(
            folder
        )

        if not self.batch_output_var.get():

            input_path = Path(
                folder
            )

            self.batch_output_var.set(
                str(
                    input_path
                    /
                    "exported"
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


    # =========================================================================
    # OPTIONS
    # =========================================================================

    def get_options(self):

        try:

            elevation_scale = float(
                self.scale_var.get()
            )

        except ValueError:

            raise ValueError(
                "OBJ Elevation Scale must be "
                "a valid number."
            )

        try:

            horizontal_scale = float(
                self.horizontal_scale_var.get()
            )

        except ValueError:

            raise ValueError(
                "Horizontal Scale must be "
                "a valid number."
            )

        try:

            elevation_offset = float(
                self.elevation_offset_var.get()
            )

        except ValueError:

            raise ValueError(
                "Elevation Offset must be "
                "a valid number."
            )

        for name, value in (
            (
                "OBJ Elevation Scale",
                elevation_scale
            ),
            (
                "Horizontal Scale",
                horizontal_scale
            ),
            (
                "Elevation Offset",
                elevation_offset
            ),
        ):

            if not math.isfinite(value):

                raise ValueError(
                    f"{name} must be finite."
                )

        return (
            elevation_scale,
            horizontal_scale,
            elevation_offset,
        )


    def outputs_selected(self):

        if not (
            self.export_obj_var.get()
            or
            self.export_pgm_var.get()
            or
            self.export_info_var.get()
        ):

            messagebox.showerror(
                "No outputs selected",
                "Please select OBJ, PGM, or Info TXT."
            )

            return False

        return True


    # =========================================================================
    # SINGLE EXPORT
    # =========================================================================

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
                "Please select a GHF file."
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

            (
                elevation_scale,
                horizontal_scale,
                elevation_offset,
            ) = self.get_options()

        except ValueError as exc:

            messagebox.showerror(
                "Invalid option",
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
                elevation_scale,
                horizontal_scale,
                elevation_offset,
                self.flip_y_var.get(),
                self.export_obj_var.get(),
                self.export_pgm_var.get(),
                self.export_info_var.get(),
            ),
            daemon=True
        )

        thread.start()


    def export_worker(
        self,
        input_path,
        output_path,
        elevation_scale,
        horizontal_scale,
        elevation_offset,
        flip_y,
        export_obj,
        export_pgm,
        export_info,
    ):

        try:

            self.log(
                "Fable 3 GHF Terrain Exporter"
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
                f"Elevation scale:   "
                f"{elevation_scale}"
            )

            self.log(
                f"Horizontal scale:  "
                f"{horizontal_scale}"
            )

            self.log(
                f"Elevation offset:   "
                f"{elevation_offset}"
            )

            self.log(
                f"Flip Y:             "
                f"{flip_y}"
            )

            self.log()

            result = self.process_ghf(
                input_path,
                output_path,
                elevation_scale,
                horizontal_scale,
                elevation_offset,
                flip_y,
                export_obj,
                export_pgm,
                export_info,
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
                f"{type(exc).__name__}: "
                f"{exc}"
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


    # =========================================================================
    # BATCH EXPORT
    # =========================================================================

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
                "Please select the folder containing "
                "the GHF files."
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

        ghf_files = sorted(
            [
                p
                for p in input_folder.iterdir()
                if p.is_file()
                and p.suffix.lower() == ".ghf"
            ],
            key=lambda p: p.name.lower()
        )

        if not ghf_files:

            messagebox.showerror(
                "No GHF files",
                "No .ghf files were found directly inside:\n\n"
                f"{input_folder}"
            )

            return

        try:

            (
                elevation_scale,
                horizontal_scale,
                elevation_offset,
            ) = self.get_options()

        except ValueError as exc:

            messagebox.showerror(
                "Invalid option",
                str(exc)
            )

            return

        answer = messagebox.askyesno(
            "Confirm batch export",
            (
                f"Found {len(ghf_files)} GHF file(s).\n\n"
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
            maximum=len(ghf_files),
            value=0
        )

        self.set_status(
            f"Batch export: "
            f"0 / {len(ghf_files)}"
        )

        thread = threading.Thread(
            target=self.batch_worker,
            args=(
                ghf_files,
                output_folder,
                elevation_scale,
                horizontal_scale,
                elevation_offset,
                self.flip_y_var.get(),
                self.export_obj_var.get(),
                self.export_pgm_var.get(),
                self.export_info_var.get(),
            ),
            daemon=True
        )

        thread.start()


    def batch_worker(
        self,
        ghf_files,
        output_folder,
        elevation_scale,
        horizontal_scale,
        elevation_offset,
        flip_y,
        export_obj,
        export_pgm,
        export_info,
    ):

        successful = []

        failed = []

        output_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        total = len(
            ghf_files
        )

        self.log(
            "FABLE 3 GHF TERRAIN EXPORTER"
        )

        self.log(
            "BATCH MODE"
        )

        self.log(
            "=" * 70
        )

        self.log()

        self.log(
            f"Input GHF files:   {total}"
        )

        self.log(
            f"Output folder:     {output_folder}"
        )

        self.log(
            f"Elevation scale:   {elevation_scale}"
        )

        self.log(
            f"Horizontal scale:  {horizontal_scale}"
        )

        self.log(
            f"Elevation offset:  {elevation_offset}"
        )

        self.log(
            f"Flip Y:             {flip_y}"
        )

        self.log(
            f"Export OBJ:         {export_obj}"
        )

        self.log(
            f"Export PGM:         {export_pgm}"
        )

        self.log(
            f"Export Info TXT:    {export_info}"
        )

        self.log()

        for index, input_path in enumerate(
            ghf_files,
            start=1
        ):

            self.log()

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
                total
            )

            output_path = (
                output_folder
                /
                f"{input_path.stem}_terrain.obj"
            )

            try:

                result = self.process_ghf(
                    input_path,
                    output_path,
                    elevation_scale,
                    horizontal_scale,
                    elevation_offset,
                    flip_y,
                    export_obj,
                    export_pgm,
                    export_info,
                )

                successful.append(
                    (
                        input_path,
                        result
                    )
                )

                self.log()

                self.log(
                    f"SUCCESS: "
                    f"{input_path.name}"
                )

            except Exception as exc:

                error_text = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                failed.append(
                    (
                        input_path,
                        error_text
                    )
                )

                self.log()

                self.log(
                    f"FAILED: "
                    f"{input_path.name}"
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
                self.update_batch_progress,
                index,
                total
            )

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

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

            for (
                input_path,
                error_text
            ) in failed:

                self.log(
                    f"  {input_path.name}: "
                    f"{error_text}"
                )

        # ---------------------------------------------------------------------
        # BATCH LOG
        # ---------------------------------------------------------------------

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
                    "Fable 3 GHF Batch Export Log\n"
                )

                log_file.write(
                    "=" * 60
                    +
                    "\n\n"
                )

                log_file.write(
                    f"Input folder: "
                    f"{ghf_files[0].parent}\n"
                )

                log_file.write(
                    f"Output folder: "
                    f"{output_folder}\n"
                )

                log_file.write(
                    f"Elevation scale: "
                    f"{elevation_scale}\n"
                )

                log_file.write(
                    f"Horizontal scale: "
                    f"{horizontal_scale}\n"
                )

                log_file.write(
                    f"Elevation offset: "
                    f"{elevation_offset}\n"
                )

                log_file.write(
                    f"Flip Y: "
                    f"{flip_y}\n"
                )

                log_file.write(
                    f"OBJ: "
                    f"{export_obj}\n"
                )

                log_file.write(
                    f"PGM: "
                    f"{export_pgm}\n"
                )

                log_file.write(
                    f"Info TXT: "
                    f"{export_info}\n\n"
                )

                log_file.write(
                    "PGM mapping: absolute raw elevation "
                    "rounded and clamped to 0..255.\n\n"
                )

                log_file.write(
                    f"Total: "
                    f"{total}\n"
                )

                log_file.write(
                    f"Successful: "
                    f"{len(successful)}\n"
                )

                log_file.write(
                    f"Failed: "
                    f"{len(failed)}\n\n"
                )

                if failed:

                    log_file.write(
                        "FAILED FILES\n"
                    )

                    log_file.write(
                        "-" * 60
                        +
                        "\n"
                    )

                    for (
                        input_path,
                        error_text
                    ) in failed:

                        log_file.write(
                            f"{input_path.name}: "
                            f"{error_text}\n"
                        )

        except Exception as exc:

            log_path = None

            self.log()

            self.log(
                "WARNING: Could not write "
                "batch_export_log.txt"
            )

            self.log(
                str(exc)
            )

        self.root.after(
            0,
            self.batch_finished,
            total,
            successful,
            failed,
            log_path
        )


    # =========================================================================
    # COMMON PROCESSING
    # =========================================================================

    def process_ghf(
        self,
        input_path,
        output_path,
        elevation_scale,
        horizontal_scale,
        elevation_offset,
        flip_y,
        export_obj,
        export_pgm,
        export_info,
    ):
        """
        Process one GHF file.

        Used by both single-file and batch modes.
        """

        self.log(
            "Reading GHF file..."
        )

        raw_size = (
            input_path.stat().st_size
        )

        data = load_ghf(
            input_path
        )

        self.log(
            f"Input size: "
            f"{raw_size:,} bytes"
        )

        self.log(
            f"Decompressed/loaded size: "
            f"{len(data):,} bytes"
        )

        (
            width,
            height,
            base_height,
        ) = parse_header(
            data
        )

        self.log(
            f"Dimensions: "
            f"{width} x {height}"
        )

        self.log(
            f"Samples: "
            f"{width * height:,}"
        )

        self.log(
            f"Header +14: "
            f"{base_height:.12f}"
        )

        (
            heights,
            minimum,
            maximum,
        ) = read_heights(
            data,
            width,
            height
        )

        self.log(
            f"Minimum +00: "
            f"{minimum:.12f}"
        )

        self.log(
            f"Maximum +00: "
            f"{maximum:.12f}"
        )

        self.log(
            f"Elevation range: "
            f"{maximum - minimum:.12f}"
        )

        # ---------------------------------------------------------------------
        # OBJ
        # ---------------------------------------------------------------------

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
                heights,
                width,
                height,
                horizontal_scale=horizontal_scale,
                elevation_scale=elevation_scale,
                elevation_offset=elevation_offset,
                flip_y=flip_y,
            )

            self.log(
                f"Wrote OBJ: "
                f"{output_path}"
            )

        # ---------------------------------------------------------------------
        # PGM
        # ---------------------------------------------------------------------

        pgm_path = None

        if export_pgm:

            pgm_path = (
                output_path.with_name(
                    output_path.stem
                    +
                    ".pgm"
                )
            )

            self.log(
                "Writing absolute-value PGM..."
            )

            (
                clamp_low,
                clamp_high
            ) = write_pgm(
                pgm_path,
                heights,
                width,
                height,
                flip_y=flip_y,
            )

            self.log(
                f"Wrote PGM: "
                f"{pgm_path}"
            )

            self.log(
                "PGM uses direct raw elevation "
                "values: 0..255."
            )

            self.log(
                f"PGM values clamped below 0: "
                f"{clamp_low:,}"
            )

            self.log(
                f"PGM values clamped above 255: "
                f"{clamp_high:,}"
            )

        # ---------------------------------------------------------------------
        # INFO
        # ---------------------------------------------------------------------

        info_path = None

        if export_info:

            info_path = (
                output_path.with_name(
                    output_path.stem
                    +
                    "_info.txt"
                )
            )

            self.log(
                "Writing diagnostic information..."
            )

            write_info(
                info_path,
                width,
                height,
                base_height,
                minimum,
                maximum,
            )

            self.log(
                f"Wrote Info: "
                f"{info_path}"
            )

        return {
            "input": input_path,
            "obj": (
                output_path
                if export_obj
                else None
            ),
            "pgm": pgm_path,
            "info": info_path,
            "width": width,
            "height": height,
            "base_height": base_height,
            "minimum": minimum,
            "maximum": maximum,
        }


    # =========================================================================
    # GUI STATE
    # =========================================================================

    def set_controls_enabled(
        self,
        enabled
    ):

        state = (
            "normal"
            if enabled
            else "disabled"
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

        self.horizontal_scale_entry.configure(
            state=state
        )

        self.elevation_offset_entry.configure(
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


    # =========================================================================
    # COMPLETION
    # =========================================================================

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
                f"OBJ:\n"
                f"{result['obj']}"
            )

        if result["pgm"]:

            outputs.append(
                f"PGM:\n"
                f"{result['pgm']}"
            )

        if result["info"]:

            outputs.append(
                f"Info:\n"
                f"{result['info']}"
            )

        messagebox.showinfo(
            "Export complete",
            (
                "GHF terrain export completed "
                "successfully.\n\n"
                +
                "\n\n".join(outputs)
            )
        )


    def batch_finished(
        self,
        total,
        successful,
        failed,
        log_path
    ):

        self.set_controls_enabled(
            True
        )

        self.set_status(
            f"Batch complete: "
            f"{len(successful)} / "
            f"{total} succeeded."
        )

        log_message = ""

        if log_path:

            log_message = (
                f"\n\nLog:\n"
                f"{log_path}"
            )

        if failed:

            messagebox.showwarning(
                "Batch export completed",
                (
                    "Batch export finished.\n\n"
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
                    f"All {total} GHF files "
                    f"exported successfully."
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


# ============================================================================
# MAIN
# ============================================================================

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

    GHFExporterGUI(
        root
    )

    root.mainloop()


if __name__ == "__main__":
    main()