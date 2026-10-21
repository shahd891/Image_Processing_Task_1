"""
app_ui.py  -  ALL the interface code (customtkinter)
====================================================
This file only builds the window and calls the functions of image_processing.py.
It contains no image-processing math.

Install:  pip install customtkinter opencv-python numpy pillow matplotlib
          pip install tkinterdnd2        (optional - enables drag & drop of image files)
Run:      python app_ui.py          (keep it in the same folder as main.py)

Sections
    1) Settings and theme colors
    2) Help texts shown in the tip bar
    3) Small reusable widgets
         ParamSlider  - title + value badge + slider (double-click the badge to reset)
         Section      - collapsible block of the sidebar
         Accordion    - keeps only one Section open at a time
         OperationPanel - one side (A or B) of the "A vs B" comparison: pick any filter or edge
                          detector + its sliders
         EmptyState   - the "drop or open an image" screen shown before an image is loaded
    Helpers: read_image / write_image (work with Arabic / non-ASCII folder names too)
    4) App          - the main window (sidebar, image area, actions)

Speed: padding and FFT results of an image are remembered by CACHE (see main.py), so changing a
slider or a view on the SAME image never repeats them. The status bar shows how often it helped.
"""
import os
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image

try:                                                  # optional: drag & drop of files
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except Exception:                                     # not installed -> the app still works
    HAS_DND = False

from main import (uniform_noise, gaussian_noise, salt_pepper_noise,   # requirement 1
                              apply_filter,                                       # requirement 2
                              gradient_edges, canny_edges, to_gray,               # requirement 3
                              histogram, cdf, equalize, plot_histograms,          # requirement 4/5/8
                              normalize,                                          # requirement 6
                              frequency_filter, hybrid_image,                     # requirement 9/10
                              detect_shapes, draw_shapes,                         # requirement 11
                              psnr, fmt_db,                                       # helpers
                              CACHE)                                              # remembered results


# ======================================================================
# 1) SETTINGS AND THEME COLORS
# ======================================================================

MAX_SIDE = 800   # loaded images are scaled to at most this many pixels (longest side) for speed

BG     = "#12110f"   # window background
PANEL  = "#1c1a17"   # sidebar and image cards
CARD   = "#2a2622"   # small elements: value chips, switches, badges
ACCENT = "#f97316"   # orange accent (buttons, sliders, selected items)
HOVER  = "#fb923c"   # lighter orange when hovering
MUTED  = "#a39e96"   # secondary text
TEXT   = "#f3efe9"   # main text
SOFT   = "#38332e"   # hover colour of cards / buttons
B_COL  = "#38bdf8"   # sky blue = everything that belongs to "B" (A uses ACCENT orange)
B_HOV  = "#7dd3fc"
A_BGR  = (22, 115, 249)    # ACCENT as BGR (for text drawn into pictures with OpenCV)
B_BGR  = (248, 189, 56)    # B_COL as BGR

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
IMG_FILTER = [("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")]


# ======================================================================
# 2) HELP TEXTS (tip bar under the images)
# ======================================================================

FILTER_INFO = {
    "Average": "Average filter: replaces each pixel by the mean of its k x k neighbours. "
               "Removes Gaussian and Uniform noise well but blurs edges. A bigger kernel "
               "smooths more and also blurs more.",
    "Median": "Median filter: replaces each pixel by the middle value of its k x k neighbours. "
              "Extreme pixels are ignored, so it removes Salt & Pepper while keeping edges "
              "fairly sharp. Use kernel 3 for light noise and a larger kernel for heavy noise "
              "(a larger kernel also erases fine detail).",
    "Gaussian": "Gaussian filter: a weighted mean where pixels near the centre count more. "
                "Sigma sets the blur strength (bigger = blurrier); the kernel size only sets "
                "where the bell curve is cut off. Gentler on edges than the Average filter.",
}


EDGE_INFO = {
    "Sobel": "Sobel: 3x3 masks that count the centre row/column twice, so it smooths a little "
             "while finding edges. The X mask finds vertical edges, the Y mask horizontal ones; "
             "magnitude = sqrt(Gx^2 + Gy^2).",
    "Prewitt": "Prewitt: like Sobel but all weights are equal (no extra weight on the centre), "
               "so it is slightly more sensitive to noise.",
    "Roberts": "Roberts cross: tiny 2x2 diagonal masks. Sharp and fast, but the most sensitive "
               "to noise, so filter first.",
    "Canny": "Canny (OpenCV): finds gradients, thins the edges, then keeps strong edges "
             "(above the high threshold) plus weak edges connected to them (above the low "
             "threshold). It does NOT blur the image itself, so run it on the Filtered image. Raise both thresholds for fewer edges.",
}


HIST_INFO = ("Histogram (solid, left axis) = how many pixels have each intensity 0-255. "
             "CDF (dashed, right axis) = running total of the histogram, 0 to 1. Equalization uses the "
             "CDF as the mapping new = (CDF - CDF_min) / (1 - CDF_min) * 255, which spreads the "
             "intensities so the CDF becomes almost a straight line (flat histogram, more contrast). "
             "Per channel: B, G, R equalized separately (colours may shift). "
             "Luminance: only brightness is equalized, colours are kept.")


NORM_INFO = {
    "Min-Max": "Min-Max normalization: new = (v - min) / (max - min) * 255, per channel. The darkest "
               "pixel becomes 0 and the brightest 255. It is LINEAR, so the histogram keeps its shape "
               "(unlike equalization). One extreme pixel (e.g. salt & pepper noise) already uses the "
               "full range, so the Noisy image barely changes.",
    "Z-score": "Z-score normalization: z = (v - mean) / std, shown with -3..+3 mapped to 0..255 "
               "(mean becomes mid-gray). Less sensitive to a single extreme pixel than Min-Max, "
               "but values beyond 3 std are clipped.",
}

FREQ_INFO = {
    "Ideal": "Ideal filter: H = 1 inside the circle of radius D0 around the centre of the spectrum "
             "and 0 outside (Low pass keeps the inside, High pass keeps the outside). The sharp cut "
             "creates ringing (ripples) around edges. Smaller D0 = blurrier (low pass) / only the "
             "finest edges (high pass).",
    "Gaussian": "Gaussian filter: H = exp(-D^2 / (2 D0^2)). A smooth transition, so there is NO ringing. "
                "D0 is the radius where the response is 0.61. Low pass = blur, High pass = 1 - low "
                "pass = edges and detail (shown on a mid-gray background).",
    "Butterworth": "Butterworth filter: H = 1 / (1 + (D/D0)^(2n)). The order n controls the sharpness "
                   "of the cut: n = 1 is very smooth like Gaussian, a large n approaches the Ideal "
                   "filter (with some ringing).",
}

FREQ_STEPS = ("Steps: FFT (library) -> shift the centre -> spectrum x mask H -> inverse FFT (library). "
              "Bottom row = log-magnitude spectrum before and after the mask (centre = low frequencies).")

SHAPES_INFO = ("Shapes: Canny edges first, then three detectors. LINES (green) = Hough transform: every "
               "edge pixel votes for all lines through it (rho, theta); strong peaks become line segments. "
               "CIRCLES (red) = Hough transform: every edge pixel votes for the centres lying along its "
               "gradient direction, for each radius; sensitivity = how much of the outline must exist. "
               "ELLIPSES (blue) = edge contours + ellipse fitting, kept only if the points really lie on "
               "the ellipse. Edges explained by a circle or ellipse are not reported again as lines. "
               "Tip: noisy picture -> keep 'Smooth first' on, or run it on the Filtered image.")

AB_INFO = ("A vs B: pick ANY two operations - including 'Original (no change)' to keep the untouched "
           "picture as a reference (spatial low-pass, frequency low / high pass, Sobel, "
           "Prewitt, Roberts, Canny), each with its own settings, and run both on the same image. "
           "Layout '4 panels' shows the input image, A, B and the difference |A - B| (stretched so small differences "
           "are visible; 'Heatmap' colours it: dark = same, bright = very different). Layout 'Wipe' "
           "overlays A and B - drag the divider to flip between them. Use the presets to start fast, "
           "and double-click any picture to enlarge it.")

# one-line explanation under each operation of an OperationPanel
OP_HINT = {
    "Equalize - Luminance": "Histogram equalization of brightness only",
    "Equalize - Per channel": "Equalize B, G, R separately (colours may shift)",
    "Normalize - Min-Max": "Linear stretch to the full 0-255 range",
    "Normalize - Z-score": "Mean to mid-gray, +-3 std mapped to 0-255",
    "Original (no change)": "The untouched image - use it as a reference",
    "Average": "Mean of the neighbours - blurs evenly",
    "Median": "Middle value - best for salt & pepper",
    "Gaussian": "Weighted blur - gentle on edges",
    "Low pass - Ideal": "Hard circular cut - can ring",
    "Low pass - Gaussian": "Smooth blur, no ringing",
    "Low pass - Butterworth": "Between Ideal and Gaussian",
    "High pass - Ideal": "Hard cut, keeps fine detail",
    "High pass - Gaussian": "Edges and detail, no ringing",
    "High pass - Butterworth": "Edges, sharpness set by n",
    "Sobel": "Gradient, smooths a little",
    "Prewitt": "Gradient, equal weights",
    "Roberts": "Tiny 2x2 masks, noise-sensitive",
    "Canny": "Thin, clean edges (two thresholds)",
}

# quick starting points for A vs B: (label, (op A, settings A), (op B, settings B))
AB_PRESETS = [
    ("Original vs Gaussian blur", ("Original (no change)", {}),  ("Gaussian", {"k": 5, "sigma": 1.5})),
    ("Original vs Sobel edges", ("Original (no change)", {}),    ("Sobel", {})),
    ("Original vs Low pass",    ("Original (no change)", {}),    ("Low pass - Gaussian", {"d0": 30})),
    ("Equalize vs Min-Max",    ("Equalize - Luminance", {}),       ("Normalize - Min-Max", {})),
    ("Min-Max vs Z-score",     ("Normalize - Min-Max", {}),        ("Normalize - Z-score", {})),
    ("Sobel vs Prewitt",        ("Sobel", {}),                      ("Prewitt", {})),
    ("Sobel vs Canny",          ("Sobel", {}),                      ("Canny", {})),
    ("Average vs Median",       ("Average", {"k": 5}),              ("Median", {"k": 5})),
    ("Gaussian vs Median",      ("Gaussian", {"k": 5, "sigma": 1.5}), ("Median", {"k": 5})),
    ("Low pass vs High pass",   ("Low pass - Gaussian", {"d0": 30}), ("High pass - Gaussian", {"d0": 30})),
    ("Ideal vs Gaussian (LP)",  ("Low pass - Ideal", {"d0": 30}),   ("Low pass - Gaussian", {"d0": 30})),
]

# every operation that can be chosen in an OperationPanel, and which sliders it uses
OPERATIONS = ["Original (no change)", "Average", "Median", "Gaussian",
              "Equalize - Luminance", "Equalize - Per channel",
              "Normalize - Min-Max", "Normalize - Z-score",
              "Low pass - Ideal", "Low pass - Gaussian", "Low pass - Butterworth",
              "High pass - Ideal", "High pass - Gaussian", "High pass - Butterworth",
              "Sobel", "Prewitt", "Roberts", "Canny"]


HYBRID_INFO = ("Hybrid image = LowPass(A) + HighPass(B). A is the main image, B is the second image you "
               "open (it is resized to A). For each one choose Original / Noisy / Filtered - B gets the "
               "same noise and filter settings as the main image. Up close you see the sharp "
               "details of B; from far away - or in the small copy at the bottom right - you see the "
               "smooth shape of A. Try a similar cut-off for both; a larger cut-off keeps more detail.")


BEST_FILTER = {"Uniform": "Average or Gaussian", "Gaussian": "Gaussian or Average",
               "Salt & Pepper": "Median"}

VIEWS = ["Side by side", "Compare", "Edges", "Histogram", "Normalize", "Frequency", "Hybrid",
         "Shapes", "A vs B"]

# which item of the "Save as..." menu belongs to each view (used by Ctrl+S)
SAVE_FOR_VIEW = {"Side by side": "Filtered", "Compare": "Filtered", "Edges": "Edges",
                 "Histogram": "Equalized", "Normalize": "Normalized",
                 "Frequency": "Freq. filtered", "Hybrid": "Hybrid", "Shapes": "Shapes",
                 "A vs B": "Result A"}

SHORTCUTS_HINT = ("Ctrl+O open  -  Ctrl+S save  -  Ctrl+R new noise\n"
                  "Ctrl+1 ... 9 switch view\n"
                  "Double-click a picture to enlarge it, or a value badge to reset its slider.")


# ======================================================================
# 3) PARAMSLIDER - title + value badge + slider
# ======================================================================


def read_image(path):
    """
    Read an image file -> (uint8 array, is_gray).  Raises ValueError with a friendly message.
    Uses np.fromfile + imdecode instead of cv2.imread, because cv2.imread FAILS on Windows when the
    path contains Arabic (or any non-ASCII) letters.
    """
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except OSError as e:
        raise ValueError(f"Can't read the file:\n{e}")
    if img is None:
        raise ValueError("OpenCV couldn't decode this file - is it a real image?")
    if img.dtype != np.uint8:                           # 16-bit -> 8-bit
        img = (img / 257).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 2:             # gray + alpha
        img = img[..., 0]
    if img.ndim == 3 and img.shape[2] == 4:             # drop alpha
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    # work on a smaller copy: filtering a 12-megapixel photo would take many seconds
    # (change MAX_SIDE at the top of the file)
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # "grayscale" = 1 channel, or 3 (almost) identical channels
    is_gray = img.ndim == 2 or (np.abs(img[..., 0].astype(int) - img[..., 1]).max() < 3
                                and np.abs(img[..., 1].astype(int) - img[..., 2]).max() < 3)
    if is_gray and img.ndim == 3:
        img = img[..., 0]
    return np.ascontiguousarray(img), is_gray


def write_image(path, img):
    """cv2.imwrite replacement that also works with non-ASCII paths."""
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise ValueError("Unsupported file type - use .png, .jpg or .bmp")
    buf.tofile(path)


def short(text, n=24):
    """Shorten a long file name for a small label."""
    return text if len(text) <= n else text[:n - 3] + "..."


def knob_only(slider, lo, hi):
    """
    Make a CTkSlider move ONLY while you press on its knob and drag it.
    By default customtkinter jumps the knob to wherever the mouse clicks on the track, which
    feels like the slider moves by itself. Here a press that does not start on the knob is ignored.
    """
    state = {"grab": False}

    def press(event):                                   # did the press land on the knob?
        radius = slider.winfo_height() / 2
        frac = (slider.get() - lo) / (hi - lo)
        knob_x = radius + frac * (slider.winfo_width() - 2 * radius)
        state["grab"] = abs(event.x - knob_x) <= radius * 1.8 + 6

    def drag(event):                                    # move only if the knob was grabbed
        if state["grab"]:
            slider._clicked(event)                      # customtkinter's own "set value" code

    def release(event):
        state["grab"] = False

    slider._canvas.bind("<Button-1>", press)            # replace the default handlers
    slider._canvas.bind("<B1-Motion>", drag)
    slider._canvas.bind("<ButtonRelease-1>", release, add=True)


class ParamSlider(ctk.CTkFrame):
    def __init__(self, parent, title, lo, hi, val, unit, on_change, odd=False, div=1,
                 color=ACCENT, hover=HOVER):
        """
        odd : only odd numbers are returned (kernel sizes must be odd)
        div : slider value is divided by this (div=10 gives steps of 0.1, used for sigma)
        """
        super().__init__(parent, fg_color="transparent")
        self.unit, self.odd, self.div = unit, odd, div
        self.default, self.on_change = val, on_change

        head = ctk.CTkFrame(self, fg_color="transparent")        # title (left) + value (right)
        head.pack(fill="x")
        ctk.CTkLabel(head, text=title, text_color=MUTED).pack(side="left")
        self.chip = ctk.CTkLabel(head, text="", fg_color=CARD, corner_radius=8, width=56,
                                 text_color=color, font=ctk.CTkFont(weight="bold"))
        self.chip.pack(side="right")
        self.chip.bind("<Double-Button-1>", lambda e: self.reset())      # double-click = default

        self.s = ctk.CTkSlider(self, from_=lo, to=hi, number_of_steps=hi - lo,
                               progress_color=color, button_color=color,
                               button_hover_color=hover, fg_color=CARD,
                               command=lambda v: (self._show(), on_change()))
        self.s.set(val)
        knob_only(self.s, lo, hi)                          # drag the knob to change it
        self.s.pack(fill="x", pady=(4, 10))
        self._show()

    def get(self):
        """Current value (odd-only / divided if requested)."""
        v = int(round(self.s.get()))
        if self.odd and v % 2 == 0:
            v += 1
        return v / self.div if self.div != 1 else v

    def _show(self):
        v = self.get()
        self.chip.configure(text=(f"{v:.1f}" if self.div != 1 else f"{v}") + self.unit)

    def set_value(self, v):
        """Set the slider from a value in the same units get() returns (no callback)."""
        self.s.set(round(v * self.div) if self.div != 1 else v)
        self._show()

    def reset(self):
        """Put the slider back to its starting value and run the same action as a drag would."""
        if str(self.s.cget("state")) == "disabled":      # locked (e.g. auto kernel size)
            return
        self.s.set(self.default)
        self._show()
        self.on_change()


class Section(ctk.CTkFrame):
    """
    Collapsible block of the sidebar: a clickable header and a `body` frame that holds the
    controls. Clicking the header opens the section (and closes the others, see Accordion)
    and runs `on_open`, which switches the main area to the matching view.
    """
    def __init__(self, parent, title, group, on_open=None, expanded=False, hint=""):
        super().__init__(parent, fg_color="transparent")
        self.title, self.group, self.on_open = title, group, on_open
        self.is_open = False
        self.header = ctk.CTkButton(self, text="", anchor="w", height=42, corner_radius=12,
                                    fg_color=CARD, hover_color=SOFT, text_color=TEXT,
                                    font=ctk.CTkFont(size=14, weight="bold"), command=self.toggle)
        self.header.pack(fill="x")
        self.body = ctk.CTkFrame(self, fg_color="transparent")
        if hint:                                           # one friendly sentence: what is this for?
            ctk.CTkLabel(self.body, text=hint, text_color=MUTED, anchor="w", justify="left",
                         wraplength=270, font=ctk.CTkFont(size=12)).pack(fill="x", padx=6,
                                                                         pady=(0, 10))
        group.add(self)
        self.set_open(expanded)

    def set_open(self, flag):
        """Show / hide the body (no callbacks - used by Accordion and by view syncing)."""
        self.is_open = flag
        self.header.configure(text=("\u25be  " if flag else "\u25b8  ") + self.title,
                              text_color=ACCENT if flag else TEXT)
        if flag:
            self.body.pack(fill="x", pady=(10, 4))
        else:
            self.body.pack_forget()

    def toggle(self):
        if self.is_open:
            self.set_open(False)
        else:
            self.group.open(self)
            if self.on_open:
                self.on_open()


class Accordion:
    """Keeps track of the Sections of the sidebar so that at most one is open at a time."""
    def __init__(self):
        self.sections = []

    def add(self, section):
        self.sections.append(section)

    def open(self, section):
        for s in self.sections:
            s.set_open(s is section)


class OperationPanel(ctk.CTkFrame):
    """
    One side of the "A vs B" comparison: a coloured card (A = orange, B = blue) with a drop-down of
    every filter / edge detector and the sliders that the chosen operation needs.
    run(src) returns (picture, caption). Results are remembered by CACHE, so redrawing the same
    A and B costs nothing. get_state / set_state let the App swap or copy the two sides.
    """
    def __init__(self, parent, title, default_op, on_change, color=ACCENT, hover=HOVER):
        super().__init__(parent, fg_color="#221f1b", corner_radius=14, border_width=2,
                         border_color=color)
        self.on_change = on_change
        head = ctk.CTkFrame(self, fg_color="transparent")
        head.pack(fill="x", padx=12, pady=(12, 0))
        ctk.CTkLabel(head, text=title, width=30, height=30, corner_radius=15, fg_color=color,
                     text_color="#161310",
                     font=ctk.CTkFont(size=15, weight="bold")).pack(side="left")
        self.op = ctk.CTkOptionMenu(head, values=OPERATIONS, command=self._chosen, height=32,
                                    dynamic_resizing=False, fg_color=CARD, button_color=color,
                                    button_hover_color=hover, text_color=TEXT,
                                    dropdown_fg_color=PANEL, dropdown_text_color=TEXT,
                                    dropdown_hover_color=CARD)
        self.op.set(default_op)
        self.op.pack(side="left", fill="x", expand=True, padx=(10, 0))
        self.hint = ctk.CTkLabel(self, text="", text_color=MUTED, anchor="w",
                                 font=ctk.CTkFont(size=11))
        self.hint.pack(fill="x", padx=14, pady=(4, 8))
        self.box = ctk.CTkFrame(self, fg_color="transparent")
        self.box.pack(fill="x", padx=12, pady=(0, 4))
        go = lambda: self.on_change()
        self.sliders = {
            "k":     ParamSlider(self.box, "Kernel size", 3, 15, 5, "", go, odd=True, color=color, hover=hover),
            "sigma": ParamSlider(self.box, "Sigma", 1, 50, 15, "", go, div=10, color=color, hover=hover),
            "d0":    ParamSlider(self.box, "Cut-off D0", 1, 200, 30, "", go, color=color, hover=hover),
            "order": ParamSlider(self.box, "Butterworth order n", 1, 10, 2, "", go, color=color, hover=hover),
            "thr":   ParamSlider(self.box, "Edge threshold (0 = off)", 0, 255, 0, "", go, color=color, hover=hover),
            "low":   ParamSlider(self.box, "Canny low threshold", 0, 255, 80, "", go, color=color, hover=hover),
            "high":  ParamSlider(self.box, "Canny high threshold", 0, 255, 160, "", go, color=color, hover=hover),
        }
        self._layout()

    def used(self):
        """Names of the sliders the current operation needs."""
        op = self.op.get()
        if op.startswith(("Original", "Equalize", "Normalize")):
            return []
        if op in ("Average", "Median"):
            return ["k"]
        if op == "Gaussian":
            return ["k", "sigma"]
        if op.endswith("Butterworth"):
            return ["d0", "order"]
        if op.startswith(("Low pass", "High pass")):
            return ["d0"]
        if op == "Canny":
            return ["low", "high"]
        return ["thr"]

    def _layout(self):
        for w in self.sliders.values():
            w.pack_forget()
        for name in self.used():
            self.sliders[name].pack(fill="x")
        self.hint.configure(text=OP_HINT.get(self.op.get(), ""))

    def _chosen(self, _value):
        self._layout()
        self.on_change()

    # ---- used by the swap / copy / preset buttons (they redraw once afterwards) ----
    def get_state(self):
        return {"op": self.op.get(), "vals": {n: w.get() for n, w in self.sliders.items()}}

    def set_state(self, state):
        self.op.set(state["op"])
        for n, v in state["vals"].items():
            self.sliders[n].set_value(v)
        self._layout()

    def set_op(self, op, **vals):
        """Choose an operation and (optionally) some slider values, e.g. set_op("Gaussian", k=5)."""
        self.op.set(op)
        for n, v in vals.items():
            self.sliders[n].set_value(v)
        self._layout()

    def run(self, src):
        """(result picture, short caption) for this panel's settings on the image `src`."""
        op = self.op.get()
        vals = tuple(self.sliders[n].get() for n in self.used())
        key = ("panel", CACHE.fingerprint(src), op, vals)
        return CACHE.get(key, lambda: self._compute(src, op))

    def _compute(self, src, op):
        s = self.sliders
        if op.startswith("Original"):
            return src, "Original"
        if op.startswith("Equalize"):
            mode = op.split(" - ")[1]
            return equalize(src, mode), f"Equalized ({mode.lower()})"
        if op.startswith("Normalize"):
            method = op.split(" - ")[1]
            return normalize(src, method), f"Normalized ({method})"
        if op in ("Average", "Median", "Gaussian"):
            k, sigma = s["k"].get(), s["sigma"].get()
            out = apply_filter(src, op, k, sigma)
            return out, f"{op} {k}x{k}" + (f", sigma {sigma:.1f}" if op == "Gaussian" else "")
        if op.startswith(("Low pass", "High pass")):
            kind, shape = op.split(" - ")
            d0, n = s["d0"].get(), s["order"].get()
            out = frequency_filter(src, kind, shape, d0, n)[0]
            return out, f"{shape} {kind.lower()}, D0 = {d0}" + (f", n = {n}" if shape == "Butterworth" else "")
        gray = to_gray(src)
        if op == "Canny":
            low = min(s["low"].get(), 254)
            high = min(max(s["high"].get(), low + 1), 255)
            return canny_edges(gray, low, high), f"Canny {low} / {high}"
        thr = s["thr"].get()
        return gradient_edges(gray, op, thr)[2], f"{op}" + (f", T = {thr}" if thr else "")


class EmptyState(ctk.CTkFrame):
    """
    Shown over the image area until an image is loaded. The WHOLE card is a big drop zone:
    click it to browse, or drag an image file onto the window (needs tkinterdnd2).
    highlight(True) lights the border while a file is being dragged over the window.
    """
    def __init__(self, parent, on_open, can_drop):
        super().__init__(parent, fg_color=PANEL, corner_radius=18, border_width=2, border_color=CARD,
                         cursor="hand2")
        self.can_drop = can_drop
        inner = ctk.CTkFrame(self, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner, text="+", width=84, height=84, corner_radius=22, fg_color=CARD,
                     text_color=ACCENT, font=ctk.CTkFont(size=48, weight="bold")).pack()
        ctk.CTkLabel(inner, text="Drop an image here", text_color=TEXT,
                     font=ctk.CTkFont(size=26, weight="bold")).pack(pady=(16, 0))
        self.msg = ctk.CTkLabel(inner, text="", text_color=MUTED, font=ctk.CTkFont(size=14))
        self.msg.pack(pady=(6, 14))
        ctk.CTkButton(inner, text="Browse files   (Ctrl+O)", height=44, width=220, corner_radius=12,
                      fg_color=ACCENT, hover_color=HOVER, text_color="#2a1203",
                      font=ctk.CTkFont(size=14, weight="bold"), command=on_open).pack()
        ctk.CTkLabel(inner, text="PNG  -  JPG  -  BMP  -  TIFF  -  WEBP", text_color=MUTED,
                     font=ctk.CTkFont(size=11)).pack(pady=(14, 18))
        steps = ctk.CTkFrame(inner, fg_color="transparent")        # tiny "how it works" guide
        steps.pack()
        for n, (head, sub) in enumerate((("Open", "pick any picture"),
                                         ("Play", "add noise, filter, detect edges"),
                                         ("Compare", "use A vs B to see the difference")), start=1):
            step = ctk.CTkFrame(steps, fg_color=CARD, corner_radius=12)
            step.grid(row=0, column=n - 1, padx=6)
            ctk.CTkLabel(step, text=f"{n}  {head}", text_color=ACCENT,
                         font=ctk.CTkFont(size=13, weight="bold")).pack(padx=16, pady=(10, 0))
            ctk.CTkLabel(step, text=sub, text_color=MUTED,
                         font=ctk.CTkFont(size=11)).pack(padx=16, pady=(0, 10))
        for w in (self, inner) + tuple(inner.winfo_children()):     # click anywhere = browse
            if not isinstance(w, ctk.CTkButton):
                w.bind("<Button-1>", lambda e: on_open(), add=True)
        self.bind("<Enter>", lambda e: self.highlight(True), add=True)
        self.bind("<Leave>", lambda e: self.highlight(False), add=True)

    def highlight(self, on):
        self.configure(border_color=ACCENT if on else CARD)

    def show(self, mode):
        how = "drag a file here or click to browse" if self.can_drop else "click to browse"
        if mode == "RGB":
            self.msg.configure(text=f"Any image works - {how}.\nColour and grayscale are detected automatically.")
        else:
            self.msg.configure(text=f"Gray mode has no image yet - {how}.\nOpen a grayscale image.")
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()

    def hide(self):
        self.place_forget()


# ======================================================================
# 4) APP - THE MAIN WINDOW
# ======================================================================

# with tkinterdnd2 installed the window can receive dropped files (the mixin adds the drop methods)
_BASES = (ctk.CTk, TkinterDnD.DnDWrapper) if HAS_DND else (ctk.CTk,)


class App(*_BASES):
    def __init__(self):
        super().__init__(fg_color=BG)
        ctk.set_appearance_mode("dark")
        self.title("Noise Lab")
        self.fit_to_screen()
        self.setup_dnd()

        self.imgs = {"RGB": None, "Gray": None}   # one image per mode (never converted)
        self.names = {"RGB": "", "Gray": ""}      # file name of each one
        self.name2 = ""                           # file name of the 2nd (hybrid) image
        self.tip_open = True
        self.original = self.noisy = self.filtered = None
        self.img2 = None                          # second image of the hybrid view
        self.noisy2 = self.filtered2 = None       # image B after noise / after noise + filter
        self._fargs = None                        # (filter name, k, sigma) last used
        self._resize_job = self._slider_job = self._edge_job = self._status_job = None
        self._shape_job = None   # pending timers (debouncing)
        self.blank = ctk.CTkImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)), size=(1, 1))

        self.build_sidebar()
        self.build_stage()
        self.bind_shortcuts()
        self.set_view()
        self.show_params()                      # also generates the first noise/filter
        self.stage.bind("<Configure>", lambda e: self.schedule(self.render, 80, "_resize_job"))

    def fit_to_screen(self):
        """Never open a window bigger than the screen (Windows display scaling)."""
        try:
            scale = self._get_window_scaling()
        except Exception:
            scale = 1.0
        w = min(1360, int(self.winfo_screenwidth() / scale * 0.95))
        h = min(780, int(self.winfo_screenheight() / scale * 0.90))
        self.geometry(f"{w}x{h}+20+20")
        self.minsize(min(1000, w), min(600, h))

    # ---------------- drag & drop ----------------
    def setup_dnd(self):
        """Let image files be dropped on the window (only if tkinterdnd2 is installed)."""
        self.dnd_ok = False
        if not HAS_DND:
            return
        try:
            self.TkdndVersion = TkinterDnD._require(self)
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self.on_drop)
            self.dnd_bind("<<DropEnter>>", self.on_drop_enter)
            self.dnd_bind("<<DropLeave>>", self.on_drop_leave)
            self.dnd_ok = True
        except Exception:
            self.dnd_ok = False

    def on_drop_enter(self, event):
        self.empty.highlight(True)
        return event.action

    def on_drop_leave(self, event):
        self.empty.highlight(False)
        return event.action

    def on_drop(self, event):
        """Dropped file -> main image (or image B while the Hybrid view is open)."""
        self.empty.highlight(False)
        files = [f for f in self.tk.splitlist(event.data) if f.lower().endswith(IMG_EXT)]
        if not files:
            self.notify("That is not an image file (use png, jpg, bmp, tif or webp)")
        elif self.view.get() == "Hybrid" and self.original is not None:
            self.load_second(files[0])
        else:
            self.load_main(files[0])
        return event.action

    def schedule(self, fn, delay=60, job="_slider_job"):
        """Run fn after `delay` ms; a newer call cancels the older one. Keeps sliders smooth
        (the heavy median filter doesn't run on every single pixel of slider movement)."""
        if getattr(self, job):
            self.after_cancel(getattr(self, job))
        setattr(self, job, self.after(delay, fn))

    # ---------------- sidebar ----------------
    def build_sidebar(self):
        # scrollable, because it holds many controls; they live in collapsible sections
        side = ctk.CTkScrollableFrame(self, width=320, fg_color=PANEL, corner_radius=0)
        self.side = side
        side.pack(side="left", fill="y")
        top_pad = {"padx": 20}                              # padding of the always-visible header
        pad = {"padx": 6}                                   # padding inside a section

        ctk.CTkLabel(side, text="Noise Lab", font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=TEXT).pack(anchor="w", pady=(18, 0), **top_pad)
        ctk.CTkLabel(side, text="Add noise to any image and see it live.",
                     text_color=MUTED).pack(anchor="w", pady=(0, 12), **top_pad)

        # current image: thumbnail + file name + size (refreshed by update_file_card)
        fc = ctk.CTkFrame(side, fg_color=CARD, corner_radius=14)
        fc.pack(fill="x", pady=(0, 10), **top_pad)
        fc.columnconfigure(1, weight=1)
        self.thumb = ctk.CTkLabel(fc, text="", width=64, height=64, corner_radius=10, fg_color=BG,
                                  text_color=MUTED, font=ctk.CTkFont(size=11))
        self.thumb.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        self.file_name = ctk.CTkLabel(fc, text="No image loaded", anchor="w", text_color=TEXT,
                                      font=ctk.CTkFont(size=13, weight="bold"))
        self.file_name.grid(row=0, column=1, sticky="sw", padx=(0, 10), pady=(12, 0))
        self.file_info = ctk.CTkLabel(fc, text="", anchor="w", text_color=MUTED,
                                      font=ctk.CTkFont(size=11))
        self.file_info.grid(row=1, column=1, sticky="nw", padx=(0, 10), pady=(0, 12))

        self.open_btn = ctk.CTkButton(side, text="Open image   (Ctrl+O)", height=44, corner_radius=12,
                                      fg_color=ACCENT, hover_color=HOVER, text_color="#2a1203",
                                      font=ctk.CTkFont(size=14, weight="bold"),
                                      command=self.open_image)
        self.open_btn.pack(fill="x", pady=(0, 14), **top_pad)

        # RGB / Gray
        self.caption(side, "Read as", padx=20)
        self.mode = ctk.CTkSegmentedButton(side, values=["RGB", "Gray"], command=lambda v: self.apply())
        self.mode.set("RGB")
        self.style_seg(self.mode)
        self.mode.pack(fill="x", pady=(0, 8), **top_pad)
        ctk.CTkLabel(side, text=SHORTCUTS_HINT, text_color=MUTED, justify="left", wraplength=282,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", pady=(0, 14), **top_pad)

        # every group of controls is a collapsible section; opening one closes the others
        self.accordion = Accordion()
        self.view_sections = {}                             # view name -> section that controls it

        def section(title, views, on_open, expanded=False, hint=""):
            """Add a collapsible section and return its body frame (the controls go inside)."""
            sec = Section(side, title, self.accordion, on_open, expanded, hint)
            sec.pack(fill="x", padx=14, pady=(0, 8))
            for v in views:
                self.view_sections[v] = sec
            return sec.body

        # ---------------- 1) noise + low-pass filter ----------------
        body = section("1.  Noise & low-pass filter", ("Side by side", "Compare"), self.show_main,
                       expanded=True,
                       hint="Damage the picture with noise, then try to clean it with a filter.")

        # noise type + its two sliders (only one pair visible at a time)
        self.caption(body, "Noise type")
        self.kind = ctk.CTkSegmentedButton(body, values=["Uniform", "Gaussian", "Salt & Pepper"],
                                           command=lambda v: self.show_params())
        self.kind.set("Gaussian")
        self.style_seg(self.kind)
        self.kind.pack(fill="x", pady=(0, 12), **pad)

        self.box = ctk.CTkFrame(body, fg_color="transparent")
        self.box.pack(fill="x", **pad)
        go = lambda: self.schedule(self.apply)             # noise slider moved -> new noise
        self.params = {
            "Uniform":       [ParamSlider(self.box, "Low",       -100, 0,  -40, "",  go),
                              ParamSlider(self.box, "High",         0, 100, 40, "",  go)],
            "Gaussian":      [ParamSlider(self.box, "Mean",       -50, 50,   0, "",  go),
                              ParamSlider(self.box, "Sigma",        0, 100, 25, "",  go)],
            "Salt & Pepper": [ParamSlider(self.box, "Amount",       0, 100,  5, "%", go),
                              ParamSlider(self.box, "Salt share",   0, 100, 50, "%", go)],
        }

        ctk.CTkButton(body, text="New random noise   (Ctrl+R)", height=30, corner_radius=10,
                      fg_color=CARD, hover_color=SOFT, text_color=TEXT,
                      command=self.apply).pack(fill="x", pady=(0, 14), **pad)

        # filter type + kernel size (+ sigma for the Gaussian filter)
        self.caption(body, "Low-pass filter")
        self.fkind = ctk.CTkSegmentedButton(body, values=["Average", "Median", "Gaussian"],
                                            command=lambda v: self.show_fparams())
        self.fkind.set("Average")
        self.style_seg(self.fkind)
        self.fkind.pack(fill="x", pady=(0, 12), **pad)

        self.fbox = ctk.CTkFrame(body, fg_color="transparent")
        self.fbox.pack(fill="x", **pad)
        gf = lambda: self.schedule(self.apply_filter, 200)      # filter slider moved -> re-filter
        self.k_slider = ParamSlider(self.fbox, "Kernel size", 3, 15, 5, "", gf, odd=True)
        self.sig_slider = ParamSlider(self.fbox, "Sigma", 1, 50, 15, "", gf, div=10)
        self.k_slider.pack(fill="x")                       # sigma + auto switch: "Gaussian" only
        self.auto_k = ctk.CTkSwitch(self.fbox, text="Auto kernel size from sigma", text_color=MUTED,
                                    progress_color=ACCENT, button_color="#ffffff", fg_color=CARD,
                                    command=self.apply_filter)

        # ---------------- 2) edge detection ----------------
        # operator, which image to run it on, and the Canny thresholds
        body = section("2.  Edge detection", ("Edges",), self.show_edges,
                       hint="Find the outlines of objects in the picture.")
        self.caption(body, "Operator")
        self.ekind = ctk.CTkSegmentedButton(body, values=["Sobel", "Prewitt", "Roberts", "Canny"],
                                            command=lambda v: self.show_eparams())
        self.ekind.set("Sobel")
        self.style_seg(self.ekind)
        self.ekind.pack(fill="x", pady=(0, 12), **pad)
        self.caption(body, "Run it on")
        self.esrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                           command=lambda v: self.show_edges())
        self.esrc.set("Filtered")                 # edges of a denoised image are much cleaner
        self.style_seg(self.esrc)
        self.esrc.pack(fill="x", pady=(0, 12), **pad)
        self.ebox = ctk.CTkFrame(body, fg_color="transparent")
        self.ebox.pack(fill="x", **pad)
        ge = lambda: self.schedule(self.render, 80, "_edge_job")
        # Sobel / Prewitt / Roberts parameter
        self.thr_slider = ParamSlider(self.ebox, "Edge threshold (0 = off)", 0, 255, 0, "", ge)
        # Canny parameters
        self.low_slider = ParamSlider(self.ebox, "Canny low threshold", 0, 255, 80, "", ge)
        self.high_slider = ParamSlider(self.ebox, "Canny high threshold", 0, 255, 160, "", ge)
        self.ap_slider = ParamSlider(self.ebox, "Canny aperture size", 3, 7, 3, "", ge, odd=True)
        self.l2_switch = ctk.CTkSwitch(self.ebox, text="Accurate gradient (L2)", text_color=MUTED,
                                       progress_color=ACCENT, button_color="#ffffff", fg_color=CARD,
                                       command=ge)
        self.thr_slider.pack(fill="x")                    # Sobel is selected at start

        # ---------------- 3) histogram + equalization ----------------
        # which image, and how to equalize a color image
        body = section("3.  Histogram & equalization", ("Histogram",), self.show_hist,
                       hint="See how bright the pixels are, and boost the contrast.")
        self.caption(body, "Image")
        self.hsrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                           command=lambda v: self.show_hist())
        self.hsrc.set("Original")
        self.style_seg(self.hsrc)
        self.hsrc.pack(fill="x", pady=(0, 8), **pad)
        self.caption(body, "Colour handling")
        self.hmode = ctk.CTkSegmentedButton(body, values=["Per channel", "Luminance"],
                                            command=lambda v: self.show_hist())
        self.hmode.set("Luminance")                   # only used for RGB images
        self.style_seg(self.hmode)
        self.hmode.pack(fill="x", pady=(0, 8), **pad)
        self.hgray = ctk.CTkSwitch(body, text="Convert to gray first", text_color=MUTED,
                                   progress_color=ACCENT, button_color="#ffffff", fg_color=CARD,
                                   command=self.show_hist)
        self.hgray.pack(anchor="w", pady=(0, 6), **pad)

        # ---------------- 4) normalization ----------------
        body = section("4.  Normalization", ("Normalize",), self.show_norm,
                       hint="Stretch the brightness range of the picture (Min-Max or Z-score).")
        self.caption(body, "Method")
        self.nmethod = ctk.CTkSegmentedButton(body, values=["Min-Max", "Z-score"],
                                              command=lambda v: self.show_norm())
        self.nmethod.set("Min-Max")
        self.style_seg(self.nmethod)
        self.nmethod.pack(fill="x", pady=(0, 8), **pad)
        self.caption(body, "Image")
        self.nsrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                           command=lambda v: self.show_norm())
        self.nsrc.set("Original")
        self.style_seg(self.nsrc)
        self.nsrc.pack(fill="x", pady=(0, 6), **pad)

        # ---------------- 5) frequency-domain filters ----------------
        # which image, low / high pass, mask shape, cut-off radius
        body = section("5.  Frequency domain filter", ("Frequency",), self.show_freq,
                       hint="Blur or sharpen by editing the picture's frequencies (FFT).")
        self.caption(body, "Image")
        self.fsrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                           command=lambda v: self.show_freq())
        self.fsrc.set("Original")
        self.style_seg(self.fsrc)
        self.fsrc.pack(fill="x", pady=(0, 8), **pad)
        self.caption(body, "Pass")
        self.fpass = ctk.CTkSegmentedButton(body, values=["Low pass", "High pass"],
                                            command=lambda v: self.show_freq())
        self.fpass.set("Low pass")
        self.style_seg(self.fpass)
        self.fpass.pack(fill="x", pady=(0, 8), **pad)
        self.caption(body, "Mask shape")
        self.fshape = ctk.CTkSegmentedButton(body, values=["Ideal", "Gaussian", "Butterworth"],
                                             command=lambda v: self.show_fqparams())
        self.fshape.set("Gaussian")
        self.style_seg(self.fshape)
        self.fshape.pack(fill="x", pady=(0, 8), **pad)
        gq = lambda: self.schedule(self.show_freq, 80, "_edge_job")
        self.fqbox = ctk.CTkFrame(body, fg_color="transparent")
        self.fqbox.pack(fill="x", **pad)
        self.fd0_slider = ParamSlider(self.fqbox, "Cut-off D0", 1, 200, 30, "", gq)
        self.forder_slider = ParamSlider(self.fqbox, "Butterworth order n", 1, 10, 2, "", gq)
        self.fd0_slider.pack(fill="x")                    # the order slider: "Butterworth" only

        # ---------------- 6) hybrid image ----------------
        # A = Original image, B = a second image you open here
        body = section("6.  Hybrid image", ("Hybrid",), self.show_hyb,
                       hint="Mix two pictures: one is seen up close, the other from far away.")
        self.open2_btn = ctk.CTkButton(body, text="Open 2nd image (high frequencies)", height=36,
                                       corner_radius=12, fg_color=CARD, hover_color="#38332e",
                                       text_color=TEXT, command=self.open_second_image)
        self.open2_btn.pack(fill="x", pady=(0, 4), **pad)
        self.b_info = ctk.CTkLabel(body, text="No 2nd image yet  (you can also drop one here)",
                                   text_color=MUTED, anchor="w", font=ctk.CTkFont(size=11))
        self.b_info.pack(fill="x", pady=(0, 8), **pad)
        self.caption(body, "Image A  (low frequencies)")
        self.hasrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                            command=lambda v: self.show_hyb())
        self.hasrc.set("Original")
        self.style_seg(self.hasrc)
        self.hasrc.pack(fill="x", pady=(0, 8), **pad)
        self.caption(body, "Image B  (high frequencies)")
        self.hbsrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                            command=lambda v: self.show_hyb())
        self.hbsrc.set("Original")
        self.style_seg(self.hbsrc)
        self.hbsrc.pack(fill="x", pady=(0, 8), **pad)
        gy = lambda: self.schedule(self.show_hyb, 80, "_edge_job")
        self.hbox = ctk.CTkFrame(body, fg_color="transparent")
        self.hbox.pack(fill="x", pady=(0, 6), **pad)
        self.hlow_slider = ParamSlider(self.hbox, "Low-pass cut-off (A)", 1, 100, 15, "", gy)
        self.hhigh_slider = ParamSlider(self.hbox, "High-pass cut-off (B)", 1, 100, 15, "", gy)
        self.hlow_slider.pack(fill="x")
        self.hhigh_slider.pack(fill="x")

        # ---------------- 7) shape detection: lines, circles, ellipses ----------------
        body = section("7.  Shapes (lines, circles, ellipses)", ("Shapes",), self.show_shapes,
                       hint="Canny edges, then lines, circles and ellipses drawn on the picture.")
        gs = lambda: self.schedule(self.show_shapes, 250, "_shape_job")   # slider moved -> re-detect

        def switch(text, on=True):
            sw = ctk.CTkSwitch(body, text=text, text_color=TEXT, progress_color=ACCENT,
                               button_color="#ffffff", fg_color=CARD, command=self.show_shapes)
            if on:
                sw.select()
            return sw

        self.caption(body, "Detect on")
        self.shsrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                            command=lambda v: self.show_shapes())
        self.shsrc.set("Original")
        self.style_seg(self.shsrc)
        self.shsrc.pack(fill="x", pady=(0, 8), **pad)
        self.sh_smooth = switch("Smooth first (Gaussian 5x5)")
        self.sh_smooth.pack(anchor="w", pady=(0, 8), **pad)
        self.sh_low = ParamSlider(body, "Canny low threshold", 0, 254, 60, "", gs)
        self.sh_high = ParamSlider(body, "Canny high threshold", 1, 255, 150, "", gs)
        self.sh_low.pack(fill="x")
        self.sh_high.pack(fill="x")

        self.sh_lines = switch("Lines  (green)")
        self.sh_lines.pack(anchor="w", pady=(4, 6), **pad)
        self.sh_lmin = ParamSlider(body, "Min line length", 10, 300, 60, " px", gs)
        self.sh_lgap = ParamSlider(body, "Max gap inside a line", 1, 50, 10, " px", gs)
        self.sh_lmin.pack(fill="x")
        self.sh_lgap.pack(fill="x")

        self.sh_circles = switch("Circles  (red)")
        self.sh_circles.pack(anchor="w", pady=(4, 6), **pad)
        self.sh_rmin = ParamSlider(body, "Min radius", 5, 200, 15, " px", gs)
        self.sh_rmax = ParamSlider(body, "Max radius", 10, 300, 100, " px", gs)
        self.sh_csens = ParamSlider(body, "Outline needed", 20, 100, 55, "%", gs)
        for sl in (self.sh_rmin, self.sh_rmax, self.sh_csens):
            sl.pack(fill="x")

        self.sh_ell = switch("Ellipses  (blue)")
        self.sh_ell.pack(anchor="w", pady=(4, 6), **pad)
        self.sh_etol = ParamSlider(body, "Fit tolerance", 1, 30, 6, "%", gs)
        self.sh_emin = ParamSlider(body, "Min edge length", 10, 300, 40, " px", gs)
        self.sh_etol.pack(fill="x")
        self.sh_emin.pack(fill="x")

        # ---------------- 8) compare two results (A vs B) ----------------
        body = section("8.  Compare two results (A vs B)", ("A vs B",), self.show_ab,
                       hint="Pick two operations - or the Original - and see them side by side. "
                            "Start with a preset.")
        self.caption(body, "Run both on")
        self.absrc = ctk.CTkSegmentedButton(body, values=["Original", "Noisy", "Filtered"],
                                            command=lambda v: self.show_ab())
        self.absrc.set("Noisy")
        self.style_seg(self.absrc)
        self.absrc.pack(fill="x", pady=(0, 12), **pad)

        self.caption(body, "Quick start")
        self.preset_menu = ctk.CTkOptionMenu(body, values=[p[0] for p in AB_PRESETS],
                                             command=self.apply_preset, height=32,
                                             dynamic_resizing=False, fg_color=CARD,
                                             button_color=ACCENT, button_hover_color=HOVER,
                                             text_color=TEXT, dropdown_fg_color=PANEL,
                                             dropdown_text_color=TEXT, dropdown_hover_color=CARD)
        self.preset_menu.set("Choose a preset...")
        self.preset_menu.pack(fill="x", pady=(0, 12), **pad)
        ctk.CTkLabel(body, text="The first picture is always the input that A and B are run on - choose it with 'Run both on' (Original / Noisy / Filtered).",
                     text_color=MUTED, anchor="w", justify="left", wraplength=270,
                     font=ctk.CTkFont(size=11)).pack(fill="x", pady=(0, 10), **pad)

        gab = lambda: self.schedule(self.render, 80, "_edge_job")
        self.panel_a = OperationPanel(body, "A", "Gaussian", gab, ACCENT, HOVER)
        self.panel_a.pack(fill="x", pady=(0, 8), **pad)
        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", pady=(0, 8), **pad)
        row.columnconfigure((0, 1), weight=1, uniform="sw")
        ctk.CTkButton(row, text="Swap A and B", height=30, corner_radius=10, fg_color=CARD,
                      hover_color=SOFT, text_color=TEXT,
                      command=self.swap_ab).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(row, text="Copy A to B", height=30, corner_radius=10, fg_color=CARD,
                      hover_color=SOFT, text_color=TEXT,
                      command=self.copy_ab).grid(row=0, column=1, sticky="ew", padx=(4, 0))
        self.panel_b = OperationPanel(body, "B", "Median", gab, B_COL, B_HOV)
        self.panel_b.pack(fill="x", pady=(0, 6), **pad)

    def caption(self, parent, text, padx=6):
        ctk.CTkLabel(parent, text=text, text_color=MUTED).pack(anchor="w", padx=padx, pady=(0, 4))

    def style_seg(self, seg):
        seg.configure(selected_color=ACCENT, selected_hover_color=HOVER, unselected_color=CARD,
                      unselected_hover_color="#38332e", fg_color=CARD, text_color=TEXT, height=34)

    # ---------------- right side ----------------
    def build_stage(self):
        right = ctk.CTkFrame(self, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=22, pady=18)

        # top bar: PSNR badges (left) | save buttons + view switch (right)
        top = ctk.CTkFrame(right, fg_color="transparent")
        top.pack(fill="x", pady=(0, 12))
        self.m_size = self.badge(top, width=140, color=MUTED)        # image size + mode
        self.m_noisy = self.badge(top)
        self.m_filt = self.badge(top)
        top2 = ctk.CTkFrame(right, fg_color="transparent")            # second row: the view switch
        top2.pack(fill="x", pady=(0, 12))
        self.view = ctk.CTkSegmentedButton(top2, values=VIEWS, command=lambda v: self.set_view())
        self.view.set("Side by side")
        self.style_seg(self.view)
        self.view.pack(fill="x")
        self.save_menu = ctk.CTkOptionMenu(top, values=["Noisy", "Filtered", "Edges", "Equalized", "Normalized",
                                                   "Freq. filtered", "Hybrid", "Shapes", "Result A", "Result B"], width=130,
                                           height=34, dynamic_resizing=False, fg_color=CARD,
                                           button_color=ACCENT, button_hover_color=HOVER,
                                           text_color=TEXT, dropdown_fg_color=PANEL,
                                           dropdown_text_color=TEXT, dropdown_hover_color=CARD,
                                           command=self.save)
        self.save_menu.set("Save as...")
        self.save_menu.pack(side="right")
        self.save_btn = ctk.CTkButton(top, text="Save   (Ctrl+S)", width=130, height=34,
                                      corner_radius=10, fg_color=ACCENT, hover_color=HOVER,
                                      text_color="#2a1203", font=ctk.CTkFont(weight="bold"),
                                      command=lambda: self.save(SAVE_FOR_VIEW[self.view.get()]))
        self.save_btn.pack(side="right", padx=(0, 8))

        # status bar (lowest row): busy / saved messages and how much the cache has saved
        bar = ctk.CTkFrame(right, fg_color="transparent")
        bar.pack(side="bottom", fill="x", pady=(8, 0))
        self.status = ctk.CTkLabel(bar, text="", text_color=MUTED, anchor="w", font=ctk.CTkFont(size=12))
        self.status.pack(side="left")
        ctk.CTkButton(bar, text="Clear cache", width=96, height=26, corner_radius=8, fg_color=CARD,
                      hover_color="#38332e", text_color=TEXT, font=ctk.CTkFont(size=12),
                      command=self.clear_cache).pack(side="right")

        # tip bar: explains the selected filter and suggests the best one for the noise type
        tipbox = ctk.CTkFrame(right, fg_color=CARD, corner_radius=12)
        tipbox.pack(side="bottom", fill="x", pady=(12, 0))
        self.tip_head = ctk.CTkButton(tipbox, text="\u25be  What am I looking at?", anchor="w", height=28,
                                      corner_radius=10, fg_color="transparent", hover_color=SOFT,
                                      text_color=ACCENT, font=ctk.CTkFont(size=12, weight="bold"),
                                      command=self.toggle_tip)
        self.tip_head.pack(fill="x", padx=6, pady=(4, 0))
        self.tip = ctk.CTkLabel(tipbox, text="", text_color=MUTED, anchor="w", justify="left",
                                wraplength=800)
        self.tip.pack(fill="x", padx=14, pady=(0, 10))

        # the stage holds both views stacked; only one is visible
        self.stage = ctk.CTkFrame(right, fg_color="transparent")
        self.stage.pack(fill="both", expand=True)

        # view A: Original | Noisy | Filtered
        self.sbs = ctk.CTkFrame(self.stage, fg_color="transparent")
        self.sbs.columnconfigure((0, 1, 2), weight=1, uniform="c")
        self.sbs.rowconfigure(0, weight=1)
        self.l_orig = self.card(self.sbs, 0, "Original")
        self.l_noisy = self.card(self.sbs, 1, "With noise")
        self.l_filt = self.card(self.sbs, 2, "Filtered", gap=False)

        # view B: noisy vs filtered with a draggable divider
        self.cmp = ctk.CTkFrame(self.stage, fg_color=PANEL, corner_radius=18)
        self.l_cmp = ctk.CTkLabel(self.cmp, text="", text_color=MUTED)
        self.l_cmp.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        row = ctk.CTkFrame(self.cmp, fg_color="transparent")
        row.pack(fill="x", padx=24, pady=(6, 14))
        ctk.CTkLabel(row, text="Noisy", text_color=MUTED).pack(side="left")
        ctk.CTkLabel(row, text="Filtered", text_color=MUTED).pack(side="right")
        self.split = ctk.CTkSlider(row, from_=0, to=1, progress_color=CARD, fg_color=ACCENT,
                                   button_color="#ffffff", command=lambda v: self.render())
        self.split.set(0.5)
        knob_only(self.split, 0, 1)
        self.split.pack(fill="x", padx=14)

        # view C: edge detection  (gray input | X | Y | magnitude)  or  (gray input | Canny)
        self.edg = ctk.CTkFrame(self.stage, fg_color="transparent")
        self.edg.rowconfigure(0, weight=1)
        self.e_cards = [self.card(self.edg, c, "", gap=(c < 3)) for c in range(4)]

        # views D, E: 2 x 2 grids (top: two images, bottom: two plots / images)
        self.hst, self.h_img, self.h_plot = self.grid_view(("Before", "Equalized"), ("", ""))
        self.nrm, self.n_img, self.n_plot = self.grid_view(("Before", "Normalized"), ("", ""))

        # views G, H: frequency-domain filter and hybrid image (also 2 x 2 grids)
        self.frq, self.f_top, self.f_bot = self.grid_view(("Before", "After"), ("", ""))
        self.hyb, self.y_top, self.y_bot = self.grid_view(("Image A", "Image B"), ("", ""))
        # view: shapes = edges | everything superimposed / lines only | circles + ellipses only
        self.shp, self.s_top, self.s_bot = self.grid_view(
            ("Canny edges", "All shapes"), ("Lines", "Circles and ellipses"))

        # view I: A vs B = a toolbar (layout, difference style, numbers) + either
        #         4 panels (input | result A | result B | difference) or one "wipe" picture
        self.abv = ctk.CTkFrame(self.stage, fg_color="transparent")
        self.abv.columnconfigure(0, weight=1)
        self.abv.rowconfigure(1, weight=1)
        bar = ctk.CTkFrame(self.abv, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.ab_layout = ctk.CTkSegmentedButton(bar, values=["4 panels", "Wipe"], width=190,
                                                command=lambda v: self.render())
        self.ab_layout.set("4 panels")
        self.style_seg(self.ab_layout)
        self.ab_layout.pack(side="left")
        self.ab_diff = ctk.CTkSegmentedButton(bar, values=["Gray", "Heatmap"], width=170,
                                              command=lambda v: self.render())
        self.ab_diff.set("Heatmap")
        self.style_seg(self.ab_diff)
        self.ab_diff.pack(side="left", padx=(10, 0))
        self.ab_stats = ctk.CTkLabel(bar, text="", fg_color=CARD, corner_radius=10, height=34,
                                     text_color=HOVER, font=ctk.CTkFont(size=12, weight="bold"))
        self.ab_stats.pack(side="right")

        body = ctk.CTkFrame(self.abv, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.ab_row = ctk.CTkFrame(body, fg_color="transparent")          # layout "4 panels"
        self.ab_row.grid(row=0, column=0, sticky="nsew")
        self.ab_row.columnconfigure((0, 1, 2, 3), weight=1, uniform="ab")
        self.ab_row.rowconfigure(0, weight=1)
        # 4 cards: the image both operations run on | result A | result B | difference
        self.ab_cards = [self.card(self.ab_row, c, "", gap=(c < 3)) for c in range(4)]
        for lbl, col in zip(self.ab_cards[1:3], (ACCENT, B_COL)):         # colour = A / B
            lbl.master.configure(border_width=2, border_color=col)
            lbl.caption.configure(text_color=col)

        self.ab_wipe = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=18)   # layout "Wipe"
        self.ab_wipe.grid(row=0, column=0, sticky="nsew")
        self.ab_wipe.grid_remove()
        self.l_wipe = ctk.CTkLabel(self.ab_wipe, text="", text_color=MUTED)
        self.l_wipe.pack(fill="both", expand=True, padx=10, pady=(10, 0))
        self.l_wipe.bind("<Double-Button-1>", lambda e: self.open_preview(self.l_wipe))
        wrow = ctk.CTkFrame(self.ab_wipe, fg_color="transparent")
        wrow.pack(fill="x", padx=24, pady=(6, 14))
        ctk.CTkLabel(wrow, text="A", text_color=ACCENT, font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkLabel(wrow, text="B", text_color=B_COL, font=ctk.CTkFont(weight="bold")).pack(side="right")
        self.ab_split = ctk.CTkSlider(wrow, from_=0, to=1, progress_color=ACCENT, fg_color=B_COL,
                                      button_color="#ffffff", command=lambda v: self.render())
        self.ab_split.set(0.5)
        knob_only(self.ab_split, 0, 1)
        self.ab_split.pack(fill="x", padx=14)

        # "open an image" screen, laid over everything until an image is loaded
        self.empty = EmptyState(self.stage, self.open_image, self.dnd_ok)

    def grid_view(self, top_caps, bottom_caps):
        """Frame with 2 x 2 cards inside the stage. Returns (frame, top two cards, bottom two cards)."""
        f = ctk.CTkFrame(self.stage, fg_color="transparent")
        f.columnconfigure((0, 1), weight=1, uniform="g")
        f.rowconfigure((0, 1), weight=1, uniform="r")
        cards = [self.card(f, c, cap, gap=(c == 0), row=r)
                 for r, caps in enumerate((top_caps, bottom_caps)) for c, cap in enumerate(caps)]
        for lbl in cards:                                # small gap between the two rows
            lbl.master.grid_configure(pady=(0, 10))
        return f, cards[:2], cards[2:]

    def badge(self, parent, width=170, color=HOVER):
        """Small rounded label used for the PSNR values and the image size."""
        lbl = ctk.CTkLabel(parent, text="", fg_color=CARD, corner_radius=10, height=34, width=width,
                           text_color=color, font=ctk.CTkFont(weight="bold"))
        lbl.pack(side="left", padx=(0, 10))
        return lbl

    def card(self, parent, col, caption, gap=True, row=0):
        """Rounded card = image area + caption. Returns the label that shows the image."""
        c = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=18)
        c.grid(row=row, column=col, sticky="nsew", padx=(0, 12) if gap else 0)
        lbl = ctk.CTkLabel(c, text="", text_color=MUTED)
        lbl.pack(fill="both", expand=True, padx=8, pady=(10, 0))
        cap = ctk.CTkLabel(c, text=caption, text_color=TEXT, font=ctk.CTkFont(weight="bold"))
        cap.pack(pady=(4, 12))
        lbl.caption = cap                      # so the caption text can be changed later
        lbl.bind("<Double-Button-1>", lambda e, l=lbl: self.open_preview(l))
        return lbl

    # ---------------- showing / hiding controls ----------------
    def set_view(self):
        for f in (self.sbs, self.cmp, self.edg, self.hst, self.nrm, self.frq, self.hyb, self.shp, self.abv):
            f.place_forget()
        shown = {"Side by side": self.sbs, "Compare": self.cmp, "Edges": self.edg, "Histogram": self.hst,
                 "Normalize": self.nrm, "Frequency": self.frq,
                 "Hybrid": self.hyb, "Shapes": self.shp, "A vs B": self.abv}[self.view.get()]
        shown.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.sync_section()
        self.update_tip()
        self.render()

    def sync_section(self):
        """Open the sidebar section that belongs to the current view (e.g. after a Ctrl+3)."""
        sec = self.view_sections.get(self.view.get())
        if sec is not None and not sec.is_open:
            self.accordion.open(sec)
        if sec is not None and sec is not getattr(self, "_last_sec", None):
            self._last_sec = sec
            self.after(80, lambda: self.scroll_sidebar_to(sec))

    def scroll_sidebar_to(self, sec):
        """Scroll the sidebar so the header of `sec` is at the top (A vs B is far down the list)."""
        try:
            self.update_idletasks()
            total = max(1, self.side._parent_canvas.bbox("all")[3])
            self.side._parent_canvas.yview_moveto(max(0.0, (sec.winfo_y() - 6) / total))
        except Exception:
            pass

    def go_view(self, name):
        self.view.set(name)
        self.set_view()

    def show_main(self):
        """Back to the noise / filter views (keeps 'Compare' if that is what is showing)."""
        if self.view.get() not in ("Side by side", "Compare"):
            self.view.set("Side by side")
        self.set_view()

    # ---------------- shortcuts + status bar ----------------
    def bind_shortcuts(self):
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-r>", lambda e: self.apply())            # new random noise
        self.bind("<Control-s>", lambda e: self.save(SAVE_FOR_VIEW[self.view.get()]))
        for i, name in enumerate(VIEWS, start=1):
            self.bind(f"<Control-Key-{i}>", lambda e, n=name: self.go_view(n))

    def toggle_tip(self):
        """Hide / show the explanation bar to give the pictures more room."""
        self.tip_open = not self.tip_open
        if self.tip_open:
            self.tip.pack(fill="x", padx=14, pady=(0, 10))
        else:
            self.tip.pack_forget()
        self.tip_head.configure(text=("\u25be  " if self.tip_open else "\u25b8  ") + "What am I looking at?")

    def open_preview(self, lbl):
        """Double-click on a picture -> a resizable window with a big version of it (Esc closes)."""
        arr = getattr(lbl, "arr", None)
        if arr is None:
            return
        win = ctk.CTkToplevel(self, fg_color=BG)
        cap = getattr(lbl, "caption", None)
        win.title(cap.cget("text") if cap is not None and cap.cget("text") else "Preview")
        win.geometry("980x740")
        pic = ctk.CTkLabel(win, text="")
        pic.pack(fill="both", expand=True, padx=12, pady=12)
        job = {"id": None}

        def redraw():
            self.show(pic, self.fit(arr, max(120, win.winfo_width() - 24),
                                    max(120, win.winfo_height() - 24)))

        def later(e):
            if e.widget is not win:                         # ignore events of the child widgets
                return
            if job["id"]:
                win.after_cancel(job["id"])
            job["id"] = win.after(60, redraw)

        win.bind("<Configure>", later)
        win.bind("<Escape>", lambda e: win.destroy())
        win.after(150, lambda: (win.lift(), win.focus_force(), redraw()))

    def update_status(self):
        self.status.configure(text=CACHE.summary())

    def busy(self, text):
        """Show a message right now, before a slow calculation blocks the window."""
        self.status.configure(text=text)
        self.update_idletasks()

    def notify(self, text, ms=3000):
        """Show a message for a few seconds, then go back to the cache summary."""
        self.status.configure(text=text)
        self.schedule(self.update_status, ms, "_status_job")

    def clear_cache(self):
        CACHE.clear()
        self.notify("Cache cleared - the next change recomputes everything once")

    def show_params(self):
        """Show only the sliders of the selected noise type, then make new noise."""
        self.update_tip()
        for sliders in self.params.values():
            for w in sliders:
                w.pack_forget()
        for w in self.params[self.kind.get()]:
            w.pack(fill="x")
        self.apply()

    def show_fparams(self):
        """Sigma slider + auto-kernel switch only make sense for the Gaussian filter."""
        self.update_tip()
        self.sig_slider.pack_forget()
        self.auto_k.pack_forget()
        if self.fkind.get() == "Gaussian":
            self.sig_slider.pack(fill="x")
            self.auto_k.pack(anchor="w", pady=(0, 6))
        self.apply_filter()

    def update_tip(self):
        """Explain the selected filter and recommend one for the selected noise type."""
        if self.view.get() == "Edges":
            self.tip.configure(text=f"{EDGE_INFO[self.ekind.get()]}\nTip: edge detectors amplify "
                                    "noise, so run them on the Filtered image (see 'Run it on'). "
                                    "Raise the threshold(s) to keep only the strongest edges.")
            return
        if self.view.get() == "Histogram":
            self.tip.configure(text=HIST_INFO)
            return
        if self.view.get() == "Normalize":
            self.tip.configure(text=NORM_INFO[self.nmethod.get()])
            return
        if self.view.get() == "Frequency":
            self.tip.configure(text=f"{FREQ_INFO[self.fshape.get()]}\n{FREQ_STEPS}")
            return
        if self.view.get() == "Hybrid":
            self.tip.configure(text=HYBRID_INFO)
            return
        if self.view.get() == "Shapes":
            self.tip.configure(text=SHAPES_INFO)
            return
        if self.view.get() == "A vs B":
            self.tip.configure(text=AB_INFO)
            return
        noise = self.kind.get()
        self.tip.configure(text=f"{FILTER_INFO[self.fkind.get()]}\n"
                                f"Tip: for {noise} noise, {BEST_FILTER[noise]} usually works best.")

    # ---------------- loading ----------------
    def open_image(self):
        """Ask for a file and load it (RGB or grayscale is detected automatically)."""
        path = filedialog.askopenfilename(title="Open an image", filetypes=IMG_FILTER)
        if path:
            self.load_main(path)

    def load_main(self, path):
        """Load the main image. A colour file goes to RGB mode and a gray one to Gray mode - the
        'Read as' switch follows the file, so no 'wrong mode' error box is needed any more."""
        try:
            img, is_gray = read_image(path)
        except ValueError as e:
            messagebox.showerror("Can't open file", str(e))
            return
        mode = "Gray" if is_gray else "RGB"
        switched = mode != self.mode.get()
        self.mode.set(mode)
        self.imgs[mode] = img
        self.names[mode] = os.path.basename(path)
        self.apply()
        self.notify(f"Loaded {self.names[mode]}" + (f"  -  switched to {mode} mode" if switched else ""))

    def update_file_card(self):
        """Thumbnail + name + size of the image of the current mode (top of the sidebar)."""
        mode = self.mode.get()
        img = self.imgs[mode]
        if img is None:
            self.thumb.configure(image=self.blank, text="No\nimage")
            self.thumb.image = None
            self.file_name.configure(text="No image loaded")
            self.file_info.configure(text=f"{mode} mode")
            self.open_btn.configure(text="Open image   (Ctrl+O)")
            self.title("Noise Lab")
            return
        small = self.fit(img, 64, 64)
        pil = Image.fromarray(small if small.ndim == 2 else small[..., ::-1])     # BGR -> RGB
        im = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        self.thumb.configure(image=im, text="")
        self.thumb.image = im
        h, w = img.shape[:2]
        self.file_name.configure(text=short(self.names[mode], 22))
        self.file_info.configure(text=f"{w} x {h}   -   {mode}")
        self.open_btn.configure(text="Open another image   (Ctrl+O)")
        self.title(f"Noise Lab  -  {self.names[mode]}")

    # ---------------- processing ----------------
    def apply(self):
        """STEP 1 - make the noisy image (new random noise), then filter it."""
        mode = self.mode.get()
        self.original = self.imgs[mode]
        self.update_file_card()

        if self.original is None:                          # nothing loaded for this mode yet
            self.noisy = self.filtered = None
            self.m_size.configure(text="No image")
            self.m_noisy.configure(text="Noisy  -")
            self.m_filt.configure(text="Filtered  -")
            for lbl in (self.l_orig, self.l_noisy, self.l_filt, self.l_cmp):
                lbl.configure(image=self.blank, text="")
                lbl.arr = None
            self.empty.show(mode)
            return
        self.empty.hide()
        oh, ow = self.original.shape[:2]
        self.m_size.configure(text=f"{ow} x {oh}  {mode}")

        self.noisy = self.make_noise(self.original)
        # image B (hybrid view) gets its own noise with the same settings
        self.noisy2 = self.make_noise(self.img2) if self.img2 is not None else None
        self.apply_filter()

    def make_noise(self, img):
        """Add the noise chosen in the sidebar (new random values) to any image."""
        kind = self.kind.get()
        a, b = self.params[kind]
        if kind == "Uniform":
            return uniform_noise(img.astype(np.float32), a.get(), b.get())
        if kind == "Gaussian":
            return gaussian_noise(img.astype(np.float32), a.get(), b.get())
        return salt_pepper_noise(img, a.get() / 100, b.get() / 100)

    def refresh_b(self):
        """Image B -> noisy B -> filtered B, using the CURRENT noise / filter settings."""
        if self.img2 is None:
            self.noisy2 = self.filtered2 = None
            return
        if self.noisy2 is None:
            self.noisy2 = self.make_noise(self.img2)
        if self._fargs is not None:
            self.filtered2 = apply_filter(self.noisy2, *self._fargs)

    def apply_filter(self):
        """STEP 2 - filter the CURRENT noisy image (noise is NOT regenerated here, so changing
        the filter or kernel size is a fair comparison on the same noisy image)."""
        if self.noisy is None:
            return
        name, sigma = self.fkind.get(), self.sig_slider.get()
        if name == "Gaussian" and self.auto_k.get() == 1:
            # AUTO: keep the kernel wide enough to hold +-3 sigma (99.7% of the bell curve):
            #       k = 2 * ceil(3 * sigma) + 1   (always odd, capped at the slider maximum 15)
            k = min(15, max(3, 2 * int(np.ceil(3 * sigma)) + 1))
            self.k_slider.s.set(k)                       # show the chosen size on the slider
            self.k_slider._show()
            self.k_slider.s.configure(state="disabled")  # slider is locked while auto is on
        else:
            self.k_slider.s.configure(state="normal")
            k = self.k_slider.get()
        self.busy(f"Filtering ({name} {k}x{k})...")
        self.filtered = apply_filter(self.noisy, name, k, sigma)
        self._fargs = (name, k, sigma)
        self.filtered2 = (apply_filter(self.noisy2, name, k, sigma)
                          if self.noisy2 is not None else None)
        # the Filtered card caption states exactly which parameters were used
        self.l_filt.caption.configure(
            text=f"Filtered  -  {name} {k}x{k}" + (f", sigma {sigma:.1f}" if name == "Gaussian" else ""))
        self.m_noisy.configure(text=f"Noisy  {fmt_db(psnr(self.original, self.noisy))}")
        self.m_filt.configure(text=f"Filtered  {fmt_db(psnr(self.original, self.filtered))}")
        self.render()

    # ---------------- drawing ----------------
    @staticmethod
    def fit(img, w, h):
        """Resize to fit inside w x h, keeping the aspect ratio."""
        k = min(w / img.shape[1], h / img.shape[0])
        size = (max(1, int(img.shape[1] * k)), max(1, int(img.shape[0] * k)))
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)

    @staticmethod
    def show(label, arr):
        """Display a numpy image (gray or BGR) in a label."""
        pil = Image.fromarray(arr if arr.ndim == 2 else arr[..., ::-1])    # BGR -> RGB
        im = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
        label.configure(image=im, text="")
        label.image = im                                   # keep a reference or it disappears
        label.arr = arr                                    # remembered so a double-click can enlarge it

    def render(self):
        self._draw()
        self.update_status()

    def _draw(self):
        self.tip.configure(wraplength=max(300, self.stage.winfo_width() - 40))   # wrap to width
        if self.original is None or self.filtered is None:
            return
        W, H = self.stage.winfo_width(), self.stage.winfo_height()
        if self.view.get() == "Edges":
            self.render_edges(W, H)
            return
        if self.view.get() == "Histogram":
            self.render_hist(W, H)
            return
        if self.view.get() == "Normalize":
            self.render_norm(W, H)
            return
        if self.view.get() == "Frequency":
            self.render_freq(W, H)
            return
        if self.view.get() == "Hybrid":
            self.render_hyb(W, H)
            return
        if self.view.get() == "Shapes":
            self.render_shapes(W, H)
            return
        if self.view.get() == "A vs B":
            self.render_ab(W, H)
            return
        if self.view.get() == "Side by side":
            w, h = (W - 24) // 3 - 24, H - 90
            self.show(self.l_orig, self.fit(self.original, w, h))
            self.show(self.l_noisy, self.fit(self.noisy, w, h))
            self.show(self.l_filt, self.fit(self.filtered, w, h))
        else:
            left = self.fit(self.noisy, W - 40, H - 110)               # noisy on the left
            right = cv2.resize(self.filtered, (left.shape[1], left.shape[0]))
            x = int(left.shape[1] * self.split.get())                  # divider position
            comp = right.copy()
            comp[:, :x] = left[:, :x]                                  # left of line: noisy
            comp[:, max(x - 1, 0):x + 1] = 255                         # white divider line
            self.show(self.l_cmp, comp)

    # ---------------- edge detection ----------------
    def show_eparams(self):
        """Canny thresholds only matter for Canny. Choosing an operator opens the Edges view."""
        for w in (self.thr_slider, self.low_slider, self.high_slider, self.ap_slider, self.l2_switch):
            w.pack_forget()
        if self.ekind.get() == "Canny":
            for w in (self.low_slider, self.high_slider, self.ap_slider):
                w.pack(fill="x")
            self.l2_switch.pack(anchor="w", pady=(0, 6))
        else:
            self.thr_slider.pack(fill="x")
        self.show_edges()

    def show_edges(self):
        self.view.set("Edges")
        self.set_view()

    def edge_panels(self):
        """List of (caption, picture) for the Edges view, computed from the chosen source."""
        src = {"Original": self.original, "Noisy": self.noisy,
               "Filtered": self.filtered}[self.esrc.get()]
        gray = to_gray(src)
        name = self.ekind.get()
        if name == "Canny":                                  # OpenCV
            low = self.low_slider.get()
            low = min(low, 254)                              # keep room for high = low + 1 <= 255
            high = min(max(self.high_slider.get(), low + 1), 255)   # high threshold must be > low
            edges = canny_edges(gray, low, high, self.ap_slider.get(), self.l2_switch.get() == 1)
            return [("Gray input", gray), ("Canny edges (OpenCV)", edges)]
        ex, ey, mag = gradient_edges(gray, name, self.thr_slider.get())   # from scratch
        return [("Gray input", gray), (f"{name} - X direction", ex),
                (f"{name} - Y direction", ey), (f"{name} - edges (magnitude)", mag)]

    def render_edges(self, W, H):
        panels = self.edge_panels()
        n = len(panels)
        cols = [0, 3] if n == 2 else [0, 1, 2, 3]            # Canny only has 2 panels
        for c, lbl in enumerate(self.e_cards):               # show / hide the unused cards
            on = c in cols
            self.edg.columnconfigure(c, weight=1 if on else 0, uniform="e" if on else "")
            (lbl.master.grid if on else lbl.master.grid_remove)()
        w, h = (W - 12 * (n - 1)) // n - 24, H - 90
        for c, (caption, im) in zip(cols, panels):
            self.show(self.e_cards[c], self.fit(im, w, h))
            self.e_cards[c].caption.configure(text=caption)

    # ---------------- histogram + equalization ----------------
    def show_hist(self):
        self.view.set("Histogram")
        self.set_view()

    def hist_pair(self):
        """(source image, equalized image) for the chosen source and equalization mode."""
        src = self.pick(self.hsrc.get())
        if self.hgray.get() == 1:                          # requirement 8: colour -> gray first
            src = to_gray(src)
        return src, equalize(src, self.hmode.get())

    def render_hist(self, W, H):
        src, eq = self.hist_pair()
        w, h = (W - 12) // 2 - 24, (H - 10) // 2 - 56
        pw, ph = max(200, w), max(120, h)
        mode = self.hmode.get() if src.ndim == 3 else "gray"
        for lbl, im in zip(self.h_img, (src, eq)):
            self.show(lbl, self.fit(im, w, h))
        self.h_img[0].caption.configure(text=f"Before  ({self.hsrc.get()})")
        self.h_img[1].caption.configure(text=f"Equalized  ({mode})")
        for lbl, im, t in zip(self.h_plot, (src, eq), ("Histogram + CDF - before", "Histogram + CDF - equalized")):
            self.show(lbl, plot_histograms(im, pw, ph, t))
            lbl.caption.configure(text="")

    # ---------------- normalization ----------------
    def pick(self, which):
        """The Original / Noisy / Filtered image."""
        return {"Original": self.original, "Noisy": self.noisy, "Filtered": self.filtered}[which]

    def show_norm(self):
        self.view.set("Normalize")
        self.set_view()

    def norm_pair(self):
        src = self.pick(self.nsrc.get())
        return src, normalize(src, self.nmethod.get())

    @staticmethod
    def grid_size(W, H):
        """Image size (w, h) that fits one card of a 2 x 2 grid."""
        return (W - 12) // 2 - 24, (H - 10) // 2 - 56

    def render_norm(self, W, H):
        src, out = self.norm_pair()
        w, h = self.grid_size(W, H)
        pw, ph = max(200, w), max(120, h)
        for lbl, im in zip(self.n_img, (src, out)):
            self.show(lbl, self.fit(im, w, h))
        self.n_img[0].caption.configure(text=f"Before  ({self.nsrc.get()})")
        self.n_img[1].caption.configure(text=f"Normalized  ({self.nmethod.get()})")
        for lbl, im, t in zip(self.n_plot, (src, out), ("Histogram + CDF - before", "Histogram + CDF - normalized")):
            self.show(lbl, plot_histograms(im, pw, ph, t))
            lbl.caption.configure(text="")

    # ---------------- frequency-domain filters ----------------
    def show_fqparams(self):
        """The Butterworth order slider only matters for the Butterworth mask."""
        self.forder_slider.pack_forget()
        if self.fshape.get() == "Butterworth":
            self.forder_slider.pack(fill="x")
        self.show_freq()

    def show_freq(self):
        self.view.set("Frequency")
        self.set_view()

    def freq_results(self):
        """(filtered image, spectrum before, spectrum after, mask) for the chosen settings."""
        src = self.pick(self.fsrc.get())
        self.busy("Computing FFT...")
        return frequency_filter(src, self.fpass.get(), self.fshape.get(),
                                self.fd0_slider.get(), self.forder_slider.get())

    def render_freq(self, W, H):
        out, sp_before, sp_after, _ = self.freq_results()
        src = self.pick(self.fsrc.get())
        w, h = self.grid_size(W, H)
        for lbl, im in zip(self.f_top + self.f_bot, (src, out, sp_before, sp_after)):
            self.show(lbl, self.fit(im, w, h))
        shape = self.fshape.get()
        extra = f", n = {self.forder_slider.get()}" if shape == "Butterworth" else ""
        self.f_top[0].caption.configure(text=f"Before  ({self.fsrc.get()})")
        self.f_top[1].caption.configure(text=f"After - {shape} {self.fpass.get()}  "
                                             f"(D0 = {self.fd0_slider.get()}{extra})")
        self.f_bot[0].caption.configure(text="Spectrum before  (log magnitude)")
        self.f_bot[1].caption.configure(text="Spectrum after  (spectrum x mask)")

    # ---------------- hybrid images ----------------
    def open_second_image(self):
        """Ask for image B (the one that gives the high frequencies)."""
        path = filedialog.askopenfilename(title="Open the 2nd image (image B)", filetypes=IMG_FILTER)
        if path:
            self.load_second(path)

    def load_second(self, path):
        """Load image B and open the Hybrid view."""
        try:
            img, _ = read_image(path)
        except ValueError as e:
            messagebox.showerror("Can't open file", str(e))
            return
        self.img2, self.name2 = img, os.path.basename(path)
        self.noisy2 = self.filtered2 = None
        self.refresh_b()
        self.b_info.configure(text=f"B: {short(self.name2, 28)}   ({img.shape[1]} x {img.shape[0]})",
                              text_color=TEXT)
        self.show_hyb()

    def show_hyb(self):
        self.view.set("Hybrid")
        self.set_view()

    # ---------------- shapes: lines, circles, ellipses ----------------
    def show_shapes(self):
        self.view.set("Shapes")
        self.set_view()

    def shape_results(self):
        """(source image, detection dict, overlay with everything, lines only, circles + ellipses)."""
        self.busy("Detecting lines, circles and ellipses...")
        src = self.pick(self.shsrc.get())
        r = detect_shapes(src, self.sh_low.get(), self.sh_high.get(), self.sh_smooth.get() == 1,
                          self.sh_lines.get() == 1, self.sh_lmin.get(), self.sh_lgap.get(),
                          self.sh_circles.get() == 1, self.sh_rmin.get(), self.sh_rmax.get(),
                          self.sh_csens.get() / 100.0,
                          self.sh_ell.get() == 1, self.sh_emin.get(), self.sh_etol.get() / 100.0)
        allv = draw_shapes(src, r["lines"], r["circles"], r["ellipses"])
        lines = draw_shapes(src, r["lines"])
        rounds = draw_shapes(src, (), r["circles"], r["ellipses"])
        return src, r, allv, lines, rounds

    def render_shapes(self, W, H):
        src, r, allv, lines, rounds = self.shape_results()
        w, h = (W - 12) // 2 - 24, (H - 10) // 2 - 56
        w, h = max(100, w), max(80, h)
        cards = (self.s_top[0], self.s_top[1], self.s_bot[0], self.s_bot[1])
        images = (r["edges"], allv, lines, rounds)
        n_l, n_c, n_e = len(r["lines"]), len(r["circles"]), len(r["ellipses"])
        caps = (f"Canny edges  ({self.shsrc.get()})",
                f"All shapes:  {n_l} lines, {n_c} circles, {n_e} ellipses",
                f"Lines  ({n_l})", f"Circles ({n_c}) and ellipses ({n_e})")
        for lbl, im, cap in zip(cards, images, caps):
            self.show(lbl, self.fit(im, w, h))
            lbl.caption.configure(text=cap)

    def hybrid_results(self):
        """(A low-passed, B high-passed, hybrid) - needs the second image."""
        self.busy("Building hybrid image...")
        a = self.pick(self.hasrc.get())
        b = {"Original": self.img2, "Noisy": self.noisy2,
             "Filtered": self.filtered2}[self.hbsrc.get()]
        return hybrid_image(a, b, self.hlow_slider.get(), self.hhigh_slider.get())

    def render_hyb(self, W, H):
        cards = self.y_top + self.y_bot
        if self.img2 is None:                              # image B not opened yet
            for lbl in cards:
                lbl.configure(image=self.blank, text="")
                lbl.caption.configure(text="")
                lbl.arr = None
            self.y_top[0].configure(text="Click here (or press 'Open 2nd image' in the sidebar)\n"
                                         "to choose image B and build a hybrid image")
            self.y_top[0].bind("<Button-1>", lambda e: self.img2 is None and self.open_second_image())
            return
        low, high, hyb = self.hybrid_results()
        w, h = self.grid_size(W, H)
        big = self.fit(hyb, w, h)
        small = cv2.resize(big, (max(1, big.shape[1] // 4), max(1, big.shape[0] // 4)),
                           interpolation=cv2.INTER_AREA)   # what the hybrid looks like from far away
        for lbl, im in zip(cards, (self.fit(low, w, h), self.fit(high, w, h), big, small)):
            self.show(lbl, im)
        self.y_top[0].caption.configure(text=f"Image A ({self.hasrc.get()}) - low pass  (D0 = {self.hlow_slider.get()})")
        self.y_top[1].caption.configure(text=f"Image B ({self.hbsrc.get()}) - high pass  (D0 = {self.hhigh_slider.get()})")
        self.y_bot[0].caption.configure(text="Hybrid  (A low + B high)")
        self.y_bot[1].caption.configure(text="Hybrid seen from far away  (1/4 size)")

    # ---------------- A vs B comparison ----------------
    def show_ab(self):
        self.view.set("A vs B")
        self.set_view()

    def ab_results(self):
        """((picture A, caption A), (picture B, caption B)) on the chosen source image."""
        src = self.pick(self.absrc.get())
        self.busy("Computing A and B...")
        return self.panel_a.run(src), self.panel_b.run(src)

    def apply_preset(self, name):
        """Fill panels A and B with a ready-made pair of operations."""
        for label, (op_a, va), (op_b, vb) in AB_PRESETS:
            if label == name:
                self.panel_a.set_op(op_a, **va)
                self.panel_b.set_op(op_b, **vb)
        self.preset_menu.set("Choose a preset...")
        self.show_ab()

    def swap_ab(self):
        sa, sb = self.panel_a.get_state(), self.panel_b.get_state()
        self.panel_a.set_state(sb)
        self.panel_b.set_state(sa)
        self.show_ab()

    def copy_ab(self):
        """B gets the same operation and settings as A - then change just one thing in B."""
        self.panel_b.set_state(self.panel_a.get_state())
        self.show_ab()

    @staticmethod
    def to_bgr(img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    @staticmethod
    def wipe(left, right, frac):
        """`left` up to the divider, `right` after it, a white line and small A / B tags."""
        x = int(left.shape[1] * frac)
        out = right.copy()
        out[:, :x] = left[:, :x]
        out[:, max(x - 1, 0):x + 1] = 255

        def tag(text, color, at_right):
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            pw, ph = tw + 18, th + 14
            x0 = out.shape[1] - pw - 8 if at_right else 8
            cv2.rectangle(out, (x0, 8), (x0 + pw, 8 + ph), color, -1)
            cv2.putText(out, text, (x0 + 9, 8 + ph - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (20, 20, 20), 2, cv2.LINE_AA)

        if x > 44:                                       # only tag a side that is visible
            tag("A", A_BGR, False)
        if out.shape[1] - x > 44:
            tag("B", B_BGR, True)
        return out

    def render_ab(self, W, H):
        (a, cap_a), (b, cap_b) = self.ab_results()
        ga, gb = to_gray(a), to_gray(b)                      # compare brightness only
        raw = np.abs(ga.astype(np.int16) - gb.astype(np.int16)).astype(np.uint8)
        peak = int(raw.max())
        self.ab_stats.configure(
            text=f"   Similarity (PSNR) {fmt_db(psnr(ga, gb))}     avg. difference {raw.mean():.1f}"
                 f"     changed pixels {(raw > 0).mean() * 100:.0f}%   ")
        src = self.absrc.get()

        if self.ab_layout.get() == "Wipe":                   # A and B on top of each other
            self.ab_row.grid_remove()
            self.ab_wipe.grid()
            left = self.fit(self.to_bgr(a), W - 40, H - 170)
            right = cv2.resize(self.to_bgr(b), (left.shape[1], left.shape[0]))
            self.show(self.l_wipe, self.wipe(left, right, self.ab_split.get()))
            return

        self.ab_wipe.grid_remove()                           # 4 panels: input | A | B | difference
        self.ab_row.grid()
        stretched = raw if peak == 0 else np.rint(raw * (255.0 / peak)).astype(np.uint8)
        diff = (cv2.applyColorMap(stretched, cv2.COLORMAP_INFERNO)
                if self.ab_diff.get() == "Heatmap" else stretched)
        w, h = (W - 36) // 4 - 24, H - 140
        source = self.pick(src)                              # the picture A and B were made from
        for lbl, im in zip(self.ab_cards, (source, a, b, diff)):
            self.show(lbl, self.fit(im, w, h))
        self.ab_cards[0].caption.configure(text=f"Input  -  {src}")
        self.ab_cards[1].caption.configure(text=f"A  -  {cap_a}")
        self.ab_cards[2].caption.configure(text=f"B  -  {cap_b}")
        self.ab_cards[3].caption.configure(
            text=f"|A - B|  identical ({src})" if peak == 0
            else f"|A - B|  stretched x{255.0 / peak:.1f}   ({src})")

    # ---------------- saving ----------------
    def save(self, which):
        """Save the noisy image, the filtered image or the final edge map."""
        self.save_menu.set("Save as...")                     # menu shows its title again
        if self.filtered is None:
            return
        if which == "Edges":
            img = self.edge_panels()[-1][1]                  # last panel = the final edge map
        elif which == "Equalized":
            img = self.hist_pair()[1]
        elif which == "Normalized":
            img = self.norm_pair()[1]
        elif which == "Freq. filtered":
            img = self.freq_results()[0]
        elif which in ("Result A", "Result B"):
            img = self.ab_results()[0 if which == "Result A" else 1][0]
        elif which == "Shapes":
            img = self.shape_results()[2]                    # the picture with all shapes drawn on it
        elif which == "Hybrid":
            if self.img2 is None:
                messagebox.showinfo("No hybrid image", "Open the 2nd image first (Hybrid image section).")
                return
            img = self.hybrid_results()[2]
        else:
            img = self.noisy if which == "Noisy" else self.filtered
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
            initialfile=which.lower().replace(".", "").replace(" ", "_") + ".png")
        if path:
            try:
                write_image(path, img)
            except (ValueError, OSError) as e:
                messagebox.showerror("Can't save", str(e))
                return
            self.notify(f"Saved: {os.path.basename(path)}")


if __name__ == "__main__":
    App().mainloop()