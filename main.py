"""
image_processing.py  -  ALL the assignment requirements (no user-interface code here)
=====================================================================================

  Requirement 1  Add additive noise
                   Uniform ........ OpenCV   (cv2.randu)
                   Gaussian ....... FROM SCRATCH (Box-Muller transform)
                   Salt & Pepper .. FROM SCRATCH (random mask)

  Requirement 2  Filter the noisy image with low-pass filters (FROM SCRATCH, from the equations)
                   Average ........ equal weights 1/k^2
                   Gaussian ....... weights from the Gaussian equation
                   Median ......... middle value of the k x k neighbourhood

  Requirement 3  Detect edges
                   Sobel, Prewitt, Roberts ... FROM SCRATCH (masks + correlation)
                   Canny ..................... OpenCV (cv2.Canny)

  Requirement 4/5/8  Histogram, cumulative distribution (CDF) and histogram equalization
                   all FROM SCRATCH (np.bincount, np.cumsum, lookup table)
                   plot_histograms() only DRAWS the curves (matplotlib)

  Requirement 6  Normalization (min-max and z-score)                     FROM SCRATCH

  Requirement 9  Frequency-domain filters (low pass / high pass: Ideal, Gaussian, Butterworth)
                 FFT and inverse FFT come from the LIBRARY (np.fft); the masks and the
                 filtering step (spectrum x mask) are FROM SCRATCH
  Requirement 10 Hybrid images (low frequencies of image A + high frequencies of image B)
                 built with the same frequency filters of requirement 9

  Requirement 11 Shape detection: Canny edges -> lines, circles, ellipses -> drawn on the image
                   Lines ........ Hough transform FROM SCRATCH (rho/theta voting, then line segments)
                   Circles ...... Hough transform FROM SCRATCH (votes along the gradient direction)
                   Ellipses ..... edge contours + least-squares fit (cv2.fitEllipse) + our own
                                  validation (fit error and how much of the outline is present)
                   Drawing ...... draw_shapes() superimposes everything on the picture

  Helpers        to_gray (luminance equation) and PSNR (quality in dB)

  Cache          ImageCache remembers the expensive intermediate results of the SAME image
                 (mirrored padding, FFT spectra, finished filter outputs) so they are computed
                 only once. One shared instance, CACHE, is used by every function below.

Every function takes numpy images (uint8; gray = H x W, color = H x W x 3 in OpenCV's BGR
order) and returns new images. Nothing here touches the screen, so it can be tested alone:
    python image_processing.py
"""
import hashlib
from collections import OrderedDict

import cv2
import numpy as np

__all__ = ["ImageCache", "CACHE",
           "uniform_noise", "gaussian_noise", "salt_pepper_noise",
           "apply_filter", "gradient_edges", "canny_edges", "to_gray", "psnr", "fmt_db",
           "histogram", "cdf", "equalize", "plot_histograms",
           "normalize",
           "frequency_filter", "spectrum_image", "hybrid_image",
           "hough_lines", "hough_circles", "detect_ellipses", "detect_shapes", "draw_shapes"]


# ======================================================================
# CACHE - never repeat an expensive calculation on the same image
# ======================================================================

class ImageCache:
    """
    Remembers expensive results so that working on the SAME image never repeats them:
        padding   (mirrored border used by Average / Gaussian / Median)
        FFT       (centred spectrum of every channel - the slowest step of requirement 9 / 10)
        outputs   (finished low-pass / frequency / hybrid results)

    How an image is recognised: by its CONTENT (shape + dtype + a 128-bit blake2b hash of its
    pixels), not by its name. So re-opening the same picture, or getting identical pixels from
    another path, still hits the cache, while a new noisy image (different pixels) misses.
    The hash of an array object is itself remembered, so it is computed only once per array.

    Memory: a least-recently-used (LRU) list with a byte budget. When the budget is exceeded
    the entries that were used longest ago are dropped first.

    Cached arrays are SHARED between callers - treat everything you get from here as read-only.
    """

    def __init__(self, max_bytes=400 * 1024 * 1024):
        self.max_bytes = max_bytes
        self.enabled = True
        self.hits = 0
        self.misses = 0
        self.bytes = 0
        self._store = OrderedDict()      # key -> (value, size in bytes)       (LRU order)
        self._fp = OrderedDict()         # id(array) -> (array, fingerprint)   (hash memo)

    # ---------- identifying an image ----------
    def fingerprint(self, img):
        """Hashable identity of an image's CONTENT. Arrays must not be modified in place."""
        memo = self._fp.get(id(img))
        if memo is not None and memo[0] is img:          # same array object seen before
            self._fp.move_to_end(id(img))
            return memo[1]
        flat = np.ascontiguousarray(img).reshape(-1).view(np.uint8)
        fp = (img.shape, img.dtype.str, hashlib.blake2b(flat, digest_size=16).digest())
        self._fp[id(img)] = (img, fp)                    # keeping `img` alive keeps its id valid
        while len(self._fp) > 16:
            self._fp.popitem(last=False)
        return fp

    # ---------- storing / retrieving ----------
    def get(self, key, compute):
        """Return the remembered value for `key`, or call compute() once and remember it."""
        if self.enabled and key in self._store:
            self._store.move_to_end(key)                 # now the most recently used
            self.hits += 1
            return self._store[key][0]
        self.misses += 1
        value = compute()
        if self.enabled:
            self._put(key, value)
        return value

    def _put(self, key, value):
        size = self._size_of(value)
        if size > self.max_bytes:                        # too big to ever fit: don't keep it
            return
        if key in self._store:
            self.bytes -= self._store.pop(key)[1]
        self._store[key] = (value, size)
        self.bytes += size
        while self.bytes > self.max_bytes and len(self._store) > 1:
            _, (_, old) = self._store.popitem(last=False)   # drop the least recently used
            self.bytes -= old

    @classmethod
    def _size_of(cls, value):
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, (tuple, list)):
            return sum(cls._size_of(v) for v in value)
        return 64

    # ---------- housekeeping ----------
    def clear(self):
        self._store.clear()
        self._fp.clear()
        self.bytes = self.hits = self.misses = 0

    def summary(self):
        """One short line for the status bar."""
        return (f"Cache: {self.hits} reused  |  {self.misses} computed  |  "
                f"{len(self._store)} items, {self.bytes / 1048576:.0f} MB")


CACHE = ImageCache()          # the one shared instance used by all functions below


# ======================================================================
# REQUIREMENT 1 - ADDITIVE NOISE
# ======================================================================

def uniform_noise(img, low, high):
    """
    Add uniformly distributed random noise in [low, high)
    to every pixel of a grayscale or color image.
    """
    noise = np.empty(img.shape, dtype=np.float32)

    if img.ndim == 2:  # Grayscale image
        cv2.randu(noise, float(low), float(high))
    else:  # Color image: specify bounds for each channel
        n = img.shape[2]
        cv2.randu(
            noise,
            (float(low),) * n,
            (float(high),) * n
        )

    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


# def uniform_noise(img, low, high):
#     # Step 1: Generate uniform random values in [0, 1)
#     random_values = np.random.random(img.shape)

#     # Step 2: Scale the values to [low, high)
#     noise = low + (high - low) * random_values

#     # Step 3: Convert the image to float
#     img_float = img.astype(np.float32)

#     # Step 4: Add noise to every pixel
#     noisy = img_float + noise

#     # Step 5: Keep pixel values within [0, 255]
#     noisy = np.clip(noisy, 0, 255)

#     # Step 6: Convert back to 8-bit image
#     return noisy.astype(np.uint8)


def gaussian_noise(img, mean, sigma):
    """
    Generate Gaussian noise from scratch using the Box-Muller transform.
    Two uniform random numbers u1 and u2 generate a standard normal value.
    The noise follows N(mean, sigma^2).
    """

    u1 = 1.0 - np.random.rand(*img.shape)  # Avoid log(0)
    u2 = np.random.rand(*img.shape)

    z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    noise = mean + sigma * z

    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)



def salt_pepper_noise(img, amount, salt_share):
    """
    Salt & Pepper noise (FROM SCRATCH).
    amount     : fraction of pixels corrupted (0..1)
    salt_share : of those, the fraction that turn white (the rest turn black)
    Each pixel gets one random r in [0,1):
        r < amount*(1-salt_share)  -> pepper (0)
        r > =1 - amount*salt_share  -> salt (255)
    """
    out = img.copy()
    r = np.random.rand(*img.shape[:2])
    out[r < amount * (1 - salt_share)] = 0
    out[r >= 1 - amount * salt_share] = 255
    return out


# ======================================================================
# REQUIREMENT 2 - LOW-PASS FILTERS (Average, Gaussian, Median)
# ======================================================================


def pad_image(img, r):
    """
    Add a border of r pixels (mirrored) so the filter also works at the image edges.
    The padded copy is REMEMBERED (CACHE): Average, Gaussian and Median with the same kernel
    size on the same image all share one padded array instead of building it again.
    """
    return CACHE.get(("pad", CACHE.fingerprint(img), r),
                     lambda: np.pad(img, [(r, r), (r, r)] + [(0, 0)] * (img.ndim - 2),
                                    mode="reflect"))


def convolve_1d(img, k1d, axis):
    """
    Apply a 1D convolution along the specified axis.
    axis = 1: horizontal; axis = 0: vertical.
    """

    k1d = k1d[::-1]  # Flip the kernel for convolution
    k = len(k1d)
    r = k // 2

    widths = [(0, 0)] * img.ndim
    widths[axis] = (r, r)
    padded = np.pad(img, widths, mode="reflect")

    n = img.shape[axis]
    out = np.zeros(img.shape, dtype=np.float32)

    for j in range(k):
        shifted = (
            padded[:, j:j + n]
            if axis == 1
            else padded[j:j + n]
        )
        out += k1d[j] * shifted

    return out



def filter_separable(img, k1d):
    """
    The Average and Gaussian k x k kernels are SEPARABLE: kernel(i, j) = k1d(i) * k1d(j).
    So filtering with the 2D kernel equals one horizontal pass followed by one vertical pass.
    Same result, but 2k instead of k*k operations per pixel (15x15 kernel: 30 instead of 225).
    """
    f = img.astype(np.float32)
    out = convolve_1d(convolve_1d(f, k1d, 1), k1d, 0)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def average_kernel(k):
    """Average (box) filter: every weight is equal.   1D weight 1/k   ->   2D weight 1/k^2"""
    return np.full(k, 1.0 / k, np.float32)


def gaussian_kernel(k, sigma):
    """
    Gaussian filter - weights from the Gaussian equation. The 2D equation

        G(x, y) = 1 / (2 * pi * sigma^2) * exp( -(x^2 + y^2) / (2 * sigma^2) )

    factorizes:  G(x, y) = G(x) * G(y)   with   G(x) = exp( -x^2 / (2 * sigma^2) ) / (sqrt(2 pi) sigma)
    so we build the 1D weights for x = -r .. +r and apply them horizontally then vertically.
    The weights are divided by their sum so they add up to exactly 1 (brightness unchanged);
    this also cancels the constant in front of the exponential.
    """
    r = k // 2
    x = np.arange(-r, r + 1)
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def median_filter(img, k):
    """
    Median filter - NON-linear: each pixel becomes the MEDIAN of its k x k neighbourhood.
    (Sort the k*k values, take the middle one.) Great against salt & pepper because the
    extreme black/white values end up at the ends of the sorted list and are ignored.

    We stack the k*k shifted copies of the image on a new last axis, then np.partition puts
    the value that belongs in the MIDDLE position in place (much cheaper than a full sort).
    Rows are processed in chunks to keep memory low.
    """
    r = k // 2
    padded = pad_image(img, r)
    H, W = img.shape[:2]
    out = np.empty_like(img)
    mid = (k * k) // 2                                       # k*k is odd -> exact median
    for y0 in range(0, H, 64):                               # 64 rows at a time
        y1 = min(y0 + 64, H)
        stack = np.stack([padded[y0 + i:y1 + i, j:j + W] for i in range(k) for j in range(k)],
                         axis=-1)                            # shape (rows, W, [3], k*k)
        out[y0:y1] = np.partition(stack, mid, axis=-1)[..., mid]
    return out


def apply_filter(img, name, k, sigma):
    """Select and apply the requested low-pass filter."""

    if k <= 0 or k % 2 == 0:
        raise ValueError("Kernel size must be a positive odd number.")

    if name == "Gaussian" and sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")
    if name not in ("Average", "Gaussian", "Median"):
        raise ValueError("Unknown filter name.")

    # Same image + same settings -> reuse the finished result (the Median filter is slow).
    # sigma only matters for the Gaussian filter, so it is left out of the key otherwise.
    key = ("filter", CACHE.fingerprint(img), name, k, float(sigma) if name == "Gaussian" else None)

    def compute():
        if name == "Average":
            return filter_separable(img, average_kernel(k))
        if name == "Gaussian":
            return filter_separable(img, gaussian_kernel(k, sigma))
        return median_filter(img, k)

    return CACHE.get(key, compute)



# ======================================================================
# REQUIREMENT 3 - EDGE DETECTION
# Sobel, Prewitt, and Roberts: From Scratch
# Canny: OpenCV
# ======================================================================

import numpy as np
import cv2


def to_gray(img):
    """Convert a BGR image to grayscale using the luminance equation."""

    if img.ndim == 2:
        return img.astype(np.uint8) if img.dtype != np.uint8 else img

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input must be a grayscale or 3-channel BGR image.")

    b, g, r = (img[..., i].astype(np.float32) for i in range(3))

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def correlate2d(img, kernel):
    """
    Apply a 2D mask to a grayscale image from scratch.
    Border pixels are extended using edge padding.
    Supports 3x3 Sobel/Prewitt masks and 2x2 Roberts masks.
    """

    if img.ndim != 2:
        raise ValueError("correlate2d expects a grayscale image.")

    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array.")

    kh, kw = kernel.shape

    if kh == 0 or kw == 0:
        raise ValueError("Kernel dimensions must be positive.")

    top = (kh - 1) // 2
    bottom = kh - 1 - top
    left = (kw - 1) // 2
    right = kw - 1 - left

    padded = np.pad(
        img.astype(np.float32),
        ((top, bottom), (left, right)),
        mode="edge"
    )

    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float32)

    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + H, j:j + W]

    return out


# Define the gradient masks manually.
EDGE_MASKS = {
    "Sobel": (
        np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]     
        ], dtype=np.float32),

        np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
    ),

    "Prewitt": (
        np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float32),

        np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1]
        ], dtype=np.float32)
    ),

    "Roberts": (
        np.array([
            [1,  0],
            [0, -1]
        ], dtype=np.float32),

        np.array([
            [ 0, 1],
            [-1, 0]
        ], dtype=np.float32)
    )
}


def to_display(a):
    """Normalize a non-negative edge-strength map to the range 0-255."""

    a = np.abs(a).astype(np.float32)
    max_value = float(a.max())

    if max_value == 0:
        return np.zeros(a.shape, dtype=np.uint8)

    return np.rint(a * (255.0 / max_value)).astype(np.uint8)


def gradient_edges(gray, name, threshold=0):
    """
    Apply Sobel, Prewitt, or Roberts from scratch.

    Gx and Gy are calculated using the corresponding masks.
    Magnitude = sqrt(Gx^2 + Gy^2).

    A positive threshold creates a binary edge map using the
    normalized magnitude (0-255).
    """

    if gray.ndim != 2:
        raise ValueError("Input must be a grayscale image.")

    mx, my = EDGE_MASKS[name]

    gx = correlate2d(gray, mx)
    gy = correlate2d(gray, my)

    magnitude_raw = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = to_display(magnitude_raw)

    if threshold > 0:
        magnitude = np.where(
            magnitude >= threshold, 255, 0
        ).astype(np.uint8)

    return (
        to_display(gx),
        to_display(gy),
        magnitude
    )


def canny_edges(gray, low, high, aperture=3, accurate=False):
    """
    Apply Canny using OpenCV.

    low and high are hysteresis thresholds.
    aperture must be 3, 5, or 7.
    accurate selects the L2 gradient magnitude when True.
    """

    if gray.ndim != 2:
        raise ValueError("Canny expects a grayscale image.")

    if not (0 <= low <= high <= 255): #Hysteresis Thresholding
        raise ValueError("Require 0 <= low <= high <= 255.")

    if aperture not in (3, 5, 7):
        raise ValueError("Aperture must be 3, 5, or 7.")

    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return cv2.Canny(
        gray,
        low,
        high,
        apertureSize=aperture,
        L2gradient=accurate
    )


# ======================================================================
# REQUIREMENT 4 / 5 /8 ( HISTOGRAM, CDF, EQUALIZATION)
# ======================================================================


def histogram(channel):
    """
    Histogram of ONE channel (2D uint8 array): h[v] = how many pixels have intensity v.
    Returns 256 counts. (np.bincount does exactly: for every pixel, h[pixel] += 1)
    """
    return np.bincount(channel.ravel(), minlength=256)



def cdf(h):
    """Calculate the normalized cumulative distribution function."""

    total = np.sum(h)

    if total <= 0:
        raise ValueError("Histogram must contain at least one pixel.")

    return np.cumsum(h, dtype=np.float64) / total




def equalization_lut(h):
    """Build a 256-entry lookup table for histogram equalization."""

    if len(h) != 256 or np.any(h < 0):
        raise ValueError("Histogram must contain 256 non-negative counts.")

    total = h.sum()

    if total == 0:
        raise ValueError("Cannot equalize an empty image.")

    c = np.cumsum(h, dtype=np.float64) / total

    nonzero = np.flatnonzero(h)

    # Find the CDF value at the first intensity present in the image.
    first = nonzero[0]
    c_min = c[first]

    # A constant image cannot gain contrast through equalization.
    if c_min >= 1.0:
        return np.arange(256, dtype=np.uint8)

    lut = np.rint(
        (c - c_min) / (1.0 - c_min) * 255
    )

    return np.clip(lut, 0, 255).astype(np.uint8)



def equalize_channel(channel):
    """Equalize one gray channel: build the histogram, make the mapping, apply it to every pixel."""
    return equalization_lut(histogram(channel))[channel]           # LUT[pixel] = new pixel



def equalize(img, mode="Luminance"):
    """Apply histogram equalization using the selected mode."""

    # Grayscale images need only one equalization operation.
    if img.ndim == 2:
        return equalize_channel(img)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected a grayscale or 3-channel BGR image.")

    if mode == "Per channel":
        # Equalize each BGR channel independently.
        return np.dstack([
            equalize_channel(img[..., c])
            for c in range(3)
        ])

    elif mode == "Luminance":
        # Equalize only the brightness Y, then multiply B, G and R by Y_new / Y_old
        # so the colors (hue) stay the same.
        y = to_gray(img)
        ratio = equalize_channel(y).astype(np.float32) / np.maximum(y, 1)
        return np.clip(np.rint(img * ratio[..., None]), 0, 255).astype(np.uint8)

    else:
        raise ValueError("Mode must be 'Luminance' or 'Per channel'.")



def plot_histograms(img, width, height, title="", bg="#1c1a17", fg="#f3efe9", vline=None):
    """
    Draw histogram (solid) and CDF (dashed).
    Returns a BGR numpy image.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    dpi = 100
    fig = Figure(
        figsize=(width / dpi, height / dpi),
        dpi=dpi,
        facecolor=bg
    )
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111, facecolor=bg)
    ax2 = ax.twinx()

    # Limit the number of tick labels on the Y axes
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

    # Show large pixel counts in a compact format
    def format_count(value, pos):
        if value >= 1000:
            return f"{value / 1000:g}k"
        return f"{value:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(format_count))
    ax2.yaxis.set_major_formatter(
        FuncFormatter(lambda value, pos: f"{value:g}")
    )

    chans = [("Gray", img, "#e5e7eb")] if img.ndim == 2 else [
        ("B", img[..., 0], "#4da3ff"),
        ("G", img[..., 1], "#4ade80"),
        ("R", img[..., 2], "#f87171")
    ]

    x = np.arange(256)

    for name, ch, col in chans:
        h = histogram(ch)
        ax.plot(x, h, color=col, lw=1.2, label=name)
        ax.fill_between(x, h, color=col, alpha=0.15)
        ax2.plot(x, cdf(h), color=col, lw=1.2, ls="--")

    if vline is not None:
        ax.axvline(vline, color="#f97316", lw=1.8)
        ax.text(
            min(vline + 4, 200),
            ax.get_ylim()[1] * 0.92,
            f"T = {vline}",
            color="#f97316",
            fontsize=8
        )

    ax.set_xlim(0, 255)
    ax2.set_ylim(0, 1.02)

    ax.set_xlabel("Intensity (0-255)", color=fg, fontsize=8)
    ax.set_ylabel("Pixel count", color=fg, fontsize=8)
    ax2.set_ylabel("Cumulative (CDF)", color=fg, fontsize=8)
    ax.set_title(title, color=fg, fontsize=9)

    for a in (ax, ax2):
        a.tick_params(colors=fg, labelsize=7)
        for sp in a.spines.values():
            sp.set_color("#5a554e")

    ax.legend(
        loc="upper left",
        fontsize=7,
        facecolor=bg,
        labelcolor=fg,
        edgecolor="#5a554e"
    )

    # Reserve space for labels and prevent clipping
    fig.subplots_adjust(
        left=0.16,
        right=0.84,
        bottom=0.22,
        top=0.85
    )

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())

    return np.ascontiguousarray(rgba[..., 2::-1])

# ======================================================================
# REQUIREMENT 6 - NORMALIZATION
# ======================================================================


def normalize(img, method="Min-Max"):
    """
    Normalization (each channel on its own; a gray image has one channel).

    Min-Max : linear stretch of [min, max] to [0, 255]
                  new = (v - min) / (max - min) * 255
              The darkest pixel becomes 0 and the brightest 255.
    Z-score : subtract the mean, divide by the standard deviation  z = (v - mean) / std
              (z has mean 0 and std 1, so it is not a picture yet). To DISPLAY it we map
              z = -3 .. +3 to 0 .. 255:   new = (z + 3) / 6 * 255   (values outside are clipped)

    Both are LINEAR, so unlike equalization the SHAPE of the histogram is kept: it is only
    moved and stretched. A channel with a single value cannot be stretched and stays as it is.
    """
    f = img.astype(np.float32)
    axes = (0, 1)
    if method == "Min-Max":
        lo, hi = f.min(axis=axes, keepdims=True), f.max(axis=axes, keepdims=True)
        out = np.where(hi > lo, (f - lo) / np.maximum(hi - lo, 1e-9) * 255, f)
    else:
        mean, std = f.mean(axis=axes, keepdims=True), f.std(axis=axes, keepdims=True)
        out = np.where(std > 0, ((f - mean) / np.maximum(std, 1e-9) + 3) / 6 * 255, f)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ======================================================================
# REQUIREMENT 9 - FREQUENCY-DOMAIN FILTERS (low pass / high pass)
# ======================================================================


def frequency_mask(shape, kind="Low pass", shape_name="Gaussian", d0=30, order=2):
    """
    Build the filter H(u, v) (FROM SCRATCH) with the origin (0, 0) frequency in the CENTRE.
    D(u, v) = distance of the frequency (u, v) from the centre,  D0 = cut-off radius.

    LOW-pass masks (keep the low frequencies = smooth parts, remove the fine detail):
        Ideal       H = 1 if D <= D0 else 0                  (sharp cut, causes ringing)
        Gaussian    H = exp( -D^2 / (2 * D0^2) )             (smooth, no ringing)
        Butterworth H = 1 / (1 + (D / D0)^(2n))              (n = order, between the two)
    HIGH-pass mask (keep edges and fine detail, remove the smooth parts):
        H_high = 1 - H_low
    """
    # the same mask is needed again whenever the image or the view changes -> remember it
    key = ("mask", tuple(shape), kind, shape_name, float(d0),
           order if shape_name == "Butterworth" else None)      # order only matters here

    def compute():
        rows, cols = shape
        v, u = np.ogrid[:rows, :cols]
        d = np.sqrt((u - cols // 2) ** 2 + (v - rows // 2) ** 2)      # distance from the centre
        d0_ = max(float(d0), 1e-3)
        if shape_name == "Ideal":
            low = (d <= d0_).astype(np.float32)
        elif shape_name == "Gaussian":
            low = np.exp(-(d ** 2) / (2.0 * d0_ ** 2)).astype(np.float32)
        else:
            low = (1.0 / (1.0 + (d / d0_) ** (2 * order))).astype(np.float32)
        return low if kind == "Low pass" else 1.0 - low

    return CACHE.get(key, compute)


def image_spectra(img):
    """
    Steps 1 + 2 for EVERY channel of an image: centred FFT (1 spectrum for gray, 3 for BGR).
    This is the slowest part of frequency filtering and it depends only on the image, not on
    the mask, so it is computed ONCE per image and remembered (CACHE). Moving the cut-off
    slider or changing the mask shape then only costs "spectrum x mask" + one inverse FFT.
    """
    def compute():
        chans = [img] if img.ndim == 2 else [img[..., c] for c in range(3)]
        return tuple(np.fft.fftshift(np.fft.fft2(ch.astype(np.float32))) for ch in chans)

    return CACHE.get(("fft", CACHE.fingerprint(img)), compute)


def gray_spectrum(img):
    """Centred spectrum of the gray version of an image (used to draw the spectrum pictures)."""
    if img.ndim == 2:
        return image_spectra(img)[0]
    return CACHE.get(("fft-gray", CACHE.fingerprint(img)),
                     lambda: image_spectra(to_gray(img))[0])


def filter_channel_fft(channel, mask, spectrum=None):
    """
    The 4 steps of frequency-domain filtering for ONE gray channel:
        1) F = FFT2(f)                 (library: np.fft.fft2)
        2) shift F so that frequency (0, 0) is in the centre   (np.fft.fftshift)
        3) G = F * mask                (element-wise product, FROM SCRATCH)
        4) g = real( IFFT2(G) )        (shift back, then library: np.fft.ifft2)
    Returns (filtered float image, centred spectrum F, filtered spectrum G).
    `spectrum` = an already computed F (from image_spectra); steps 1-2 are then skipped.
    """
    f = (np.fft.fftshift(np.fft.fft2(channel.astype(np.float32)))
         if spectrum is None else spectrum)
    g = f * mask
    out = np.real(np.fft.ifft2(np.fft.ifftshift(g)))
    return out, f, g


def spectrum_image(spec):
    """Show a complex spectrum as a picture: log(1 + |F|) scaled to 0..255 (the log is needed
    because the DC term is millions of times bigger than the other frequencies)."""
    mag = np.log1p(np.abs(spec))
    return np.rint(mag / (mag.max() + 1e-9) * 255).astype(np.uint8)



def frequency_filter(img, kind="Low pass", shape_name="Gaussian", d0=30, order=2):
    """
    Apply a frequency-domain filter to a grayscale or color image.
    Returns:
        filtered image for display,
        spectrum before filtering,
        spectrum after filtering,
        filter mask.
    Same image + same settings -> the finished result is simply returned again (CACHE).
    """
    key = ("freq", CACHE.fingerprint(img), kind, shape_name, float(d0),
           order if shape_name == "Butterworth" else None)
    return CACHE.get(key, lambda: _frequency_filter(img, kind, shape_name, d0, order))


def _frequency_filter(img, kind, shape_name, d0, order):
    """The real work of frequency_filter (called only when the result is not remembered)."""

    mask = frequency_mask(img.shape[:2], kind, shape_name, d0, order)

    # Process each color channel independently (their spectra come from the cache).
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(3)]
    spectra = image_spectra(img)

    # Keep the raw floating-point results before clipping.
    raw_out = [filter_channel_fft(ch, mask, sp)[0] for ch, sp in zip(chans, spectra)]

    # Reconstruct the filtered image in its original shape.
    result = raw_out[0] if img.ndim == 2 else np.dstack(raw_out)

    # Prepare a display version of the filtered image.
    if kind == "High pass":
        display_result = result + 128.0
    else:
        display_result = result

    filtered = np.clip(
        np.rint(display_result), 0, 255
    ).astype(np.uint8)

    # Generate before/after spectrum images using the grayscale input (spectrum from the cache).
    f = gray_spectrum(img)
    g = f * mask

    return (
        filtered,
        spectrum_image(f),
        spectrum_image(g),
        np.rint(mask * 255).astype(np.uint8)
    )



# ======================================================================
# REQUIREMENT 10 - HYBRID IMAGES
# ======================================================================



def _hybrid_inputs(img_a, img_b):
    """Make A and B the same size and channel count (remembered, so the SAME two arrays come
    back every time and their spectra can be found in the cache)."""
    def compute():
        a, b = img_a, img_b
        if b.shape[:2] != a.shape[:2]:                       # resize image B to match image A
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        if a.ndim != b.ndim:                                 # gray if the channel counts differ
            a, b = to_gray(a), to_gray(b)
        return a, b

    return CACHE.get(("hybrid-in", CACHE.fingerprint(img_a), CACHE.fingerprint(img_b)), compute)


def hybrid_image(img_a, img_b, d0_low=15, d0_high=15):
    """
    Hybrid image = low frequencies of A + high frequencies of B.
    Keep floating-point filter responses during calculations to
    avoid losing details before creating the final hybrid image.
    The spectra of A and B are computed once (CACHE), so moving a cut-off slider only repeats
    "spectrum x mask" and the inverse FFT.
    """
    key = ("hybrid", CACHE.fingerprint(img_a), CACHE.fingerprint(img_b),
           float(d0_low), float(d0_high))
    return CACHE.get(key, lambda: _hybrid_image(img_a, img_b, d0_low, d0_high))


def _hybrid_image(img_a, img_b, d0_low, d0_high):
    """The real work of hybrid_image (called only when the result is not remembered)."""

    img_a, img_b = _hybrid_inputs(img_a, img_b)

    # Build the Gaussian low-pass and high-pass masks.
    mask_low = frequency_mask(
        img_a.shape[:2], "Low pass", "Gaussian", d0_low
    )
    mask_high = frequency_mask(
        img_b.shape[:2], "High pass", "Gaussian", d0_high
    )

    # Filter the images without converting the results to uint8.
    # filter_channel_fft works on ONE channel, so a color image is filtered channel by channel.
    chans_a = [img_a] if img_a.ndim == 2 else [img_a[..., c] for c in range(3)]
    chans_b = [img_b] if img_b.ndim == 2 else [img_b[..., c] for c in range(3)]
    spec_a, spec_b = image_spectra(img_a), image_spectra(img_b)      # remembered FFTs
    low_raw = [filter_channel_fft(ch, mask_low, sp)[0] for ch, sp in zip(chans_a, spec_a)]
    high_raw = [filter_channel_fft(ch, mask_high, sp)[0] for ch, sp in zip(chans_b, spec_b)]
    low_raw = low_raw[0] if img_a.ndim == 2 else np.dstack(low_raw)
    high_raw = high_raw[0] if img_b.ndim == 2 else np.dstack(high_raw)

    # Create display versions of the filtered images.
    low_display = np.clip(
        np.rint(low_raw), 0, 255
    ).astype(np.uint8)

    high_display = np.clip(
        np.rint(high_raw + 128.0), 0, 255
    ).astype(np.uint8)

    # Combine the low frequencies of A and high frequencies of B.
    hybrid_raw = low_raw + high_raw

    hybrid = np.clip(
        np.rint(hybrid_raw), 0, 255
    ).astype(np.uint8)

    return low_display, high_display, hybrid



# ======================================================================
# REQUIREMENT 11 - SHAPE DETECTION (lines, circles, ellipses) + OVERLAY
# ======================================================================
#   edges (Canny)  ->  Hough lines  /  Hough circles  /  ellipse fitting  ->  draw on the image
# Every detector works on a binary edge image (0 / 255) and returns plain Python numbers,
# so the result can be remembered by CACHE and drawn by draw_shapes().

LINE_COLOR = (0, 255, 255)       # Yellow
CIRCLE_COLOR = (255, 0, 255)     # Magenta
ELLIPSE_COLOR = (255, 255, 0)    # Cyan

def hough_lines(edges, min_len=60, max_gap=10, max_lines=40):
    """
    Straight lines by the Hough transform (FROM SCRATCH).
    Returns a list of line SEGMENTS (x0, y0, x1, y1).
    Same edge image + same settings -> the finished list is reused (CACHE).
    """
    key = ("hough-lines", CACHE.fingerprint(edges), int(min_len), int(max_gap), int(max_lines))
    return CACHE.get(key, lambda: _hough_lines(edges, int(min_len), int(max_gap), int(max_lines)))


def _hough_lines(edges, min_len, max_gap, max_lines):
    """
    A line is described by rho = x*cos(theta) + y*sin(theta)  (rho = distance from the origin,
    theta = direction of the normal). Every edge pixel votes for ALL lines that pass through it:
    one vote for each theta (0..179 degrees) in the cell (rho, theta) of the accumulator.
    A cell with many votes = many edge pixels lie on the same line.

      1) voting            acc[rho, theta] += 1        (np.bincount, one pass per theta)
      2) peaks             cells that are the maximum of their 9 x 9 neighbourhood and have
                           enough votes (theta wraps around: theta = 180 is theta = 0 with rho flipped)
      3) segments          for each peak (strongest first) take the edge pixels within 2 px of
                           the line, sort them along the line and cut it wherever there is a gap
                           bigger than max_gap. Pieces shorter than min_len are dropped.
                           Pixels already used by a stronger line are not used again.
    """
    H, W = edges.shape
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return []
    xf, yf = xs.astype(np.float32), ys.astype(np.float32)

    diag = int(np.ceil(np.hypot(H, W)))
    n_rho = 2 * diag + 1                                   # rho = -diag .. +diag
    thetas = np.deg2rad(np.arange(180))
    cos, sin = np.cos(thetas), np.sin(thetas)

    # 1) voting
    acc = np.zeros((n_rho, 180), np.int32)
    for t in range(180):
        rho = np.rint(xf * cos[t] + yf * sin[t]).astype(np.int32) + diag
        acc[:, t] = np.bincount(rho, minlength=n_rho)

    # 2) peaks (circular in theta: the columns on the other side come back with rho flipped)
    pad = 4
    ext = np.concatenate([acc[::-1, -pad:], acc, acc[::-1, :pad]], axis=1).astype(np.float32)
    need = max(2.0, 0.6 * min_len)
    peaks = (ext >= need) & (ext == cv2.dilate(ext, np.ones((9, 9), np.uint8)))
    rr, tt = np.nonzero(peaks[:, pad:-pad])
    order = np.argsort(-acc[rr, tt])[:300]                  # strongest first (at most 300)

    # 3) turn each peak into line segments
    lines = []
    used = np.zeros(len(xs), bool)
    for i in order:
        if len(lines) >= max_lines:
            break
        rho, c, s = float(rr[i] - diag), cos[tt[i]], sin[tt[i]]
        near = np.nonzero((np.abs(xf * c + yf * s - rho) <= 2.0) & ~used)[0]
        if len(near) < need:
            continue
        t = -xf[near] * s + yf[near] * c                    # position ALONG the line
        o = np.argsort(t)
        near, t = near[o], t[o]
        cuts = np.nonzero(np.diff(t) > max_gap)[0] + 1      # gaps -> separate segments
        for idx, tp in zip(np.split(near, cuts), np.split(t, cuts)):
            if tp[-1] - tp[0] < min_len:
                continue
            x0, y0 = rho * c - tp[0] * s, rho * s + tp[0] * c
            x1, y1 = rho * c - tp[-1] * s, rho * s + tp[-1] * c
            lines.append((int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))))
            used[idx] = True
            if len(lines) >= max_lines:
                break
    return lines


def hough_circles(edges, gray, r_min=15, r_max=100, sens=0.55, max_circles=15):
    """
    Circles by the Hough transform (FROM SCRATCH).
    Returns a list of circles (x, y, r) = centre and radius in pixels.
    sens = how much of the circle outline must be present (0.55 = 55 % of 2*pi*r edge pixels).
    """
    key = ("hough-circles", CACHE.fingerprint(edges), CACHE.fingerprint(gray),
           int(r_min), int(r_max), round(float(sens), 3), int(max_circles))
    return CACHE.get(key, lambda: _hough_circles(edges, gray, int(r_min), int(r_max),
                                                 float(sens), int(max_circles)))


def _hough_circles(edges, gray, r_min, r_max, sens, max_circles):
    """
    A circle is (x0, y0, r):  (x - x0)^2 + (y - y0)^2 = r^2.
    Trying every (x0, y0, r) for every edge pixel is slow, so we use the GRADIENT DIRECTION:
    at an edge pixel of a circle the gradient points along the radius, so the centre lies at
        (x, y) +- r * (gx, gy) / |(gx, gy)|          (+ or - depending on dark / bright inside)
    For each radius r every edge pixel votes for these two points; a centre that collects
    votes from (almost) the whole outline wins. The vote map is summed over a small window
    (3 x 3, or 5 x 5 for big radii) because the gradient direction is never perfectly exact.
    The score of a circle = votes / (2*pi*r)  (1.0 = complete outline).
    """
    H, W = edges.shape
    ys, xs = np.nonzero(edges)
    if len(xs) == 0 or r_max < r_min:
        return []

    mx, my = EDGE_MASKS["Sobel"]                             # gradient (our own correlation)
    gx = correlate2d(gray, mx)[ys, xs]
    gy = correlate2d(gray, my)[ys, xs]
    mag = np.hypot(gx, gy)
    keep = mag > 1e-6
    xs, ys = xs[keep].astype(np.float32), ys[keep].astype(np.float32)
    ux, uy = gx[keep] / mag[keep], gy[keep] / mag[keep]      # unit gradient direction

    cand = []
    k5 = np.ones((5, 5), np.uint8)
    for r in range(r_min, r_max + 1):
        cx = np.concatenate([xs + r * ux, xs - r * ux])
        cy = np.concatenate([ys + r * uy, ys - r * uy])
        cx, cy = np.rint(cx).astype(np.int32), np.rint(cy).astype(np.int32)
        ok = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)     # centres inside the picture only
        acc = np.bincount(cy[ok] * W + cx[ok], minlength=H * W).astype(np.float32).reshape(H, W)
        size = 3 if r < 40 else 5
        box = cv2.boxFilter(acc, -1, (size, size), normalize=False,
                            borderType=cv2.BORDER_CONSTANT)
        full = 2.0 * np.pi * r
        peak = (box >= sens * full) & (box == cv2.dilate(box, k5))
        for y, x in zip(*np.nonzero(peak)):
            cand.append((float(box[y, x]) / full, int(x), int(y), r))

    # keep the best candidates; drop near-duplicates (same centre AND about the same radius,
    # so concentric circles of different size are both kept)
    cand.sort(key=lambda c: -c[0])
    out = []
    for score, x, y, r in cand[:600]:
        if len(out) >= max_circles:
            break
        if any(np.hypot(x - ox, y - oy) < max(5, 0.3 * r) and abs(r - orr) < max(4, 0.2 * r)
               for ox, oy, orr in out):
            continue
        out.append((x, y, r))
    return out


def detect_ellipses(edges, min_pts=40, tol=0.06):
    """
    Ellipses from the edge contours.
    Returns a list of ellipses (cx, cy, major, minor, angle) - full axis lengths, angle in degrees.
    Same edge image + same settings -> the finished list is reused (CACHE).
    """
    key = ("ellipses", CACHE.fingerprint(edges), int(min_pts), round(float(tol), 3))
    return CACHE.get(key, lambda: _detect_ellipses(edges, int(min_pts), float(tol)))


def _detect_ellipses(edges, min_pts, tol):
    """
    1) cv2.findContours turns the edge image into chains of connected edge pixels.
    2) cv2.fitEllipse fits the best ellipse (least squares) to every chain.
    3) OUR validation (a chain that is not an ellipse can always be fitted by something):
         - radial error: for every contour point, D = sqrt((u/a)^2 + (v/b)^2) in the ellipse's own
           axes; D = 1 means the point lies exactly on the ellipse. mean |D - 1| must be <= tol.
         - coverage: the points must be spread around the outline (at least 7 of 12 angle
           sectors), so a short arc cannot produce a huge fake ellipse.
         - size / shape: axes >= 8 px, not a thin sliver, not a (near) circle - those are the
           job of the circle detector.
    4) near-identical ellipses (inner and outer border of one thick edge) are merged.
    """
    H, W = edges.shape
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    found = []
    for c in contours:
        if len(c) < max(5, min_pts):
            continue
        (cx, cy), (w, h), ang = cv2.fitEllipse(c)
        if w < h:                                            # make `w` the major axis
            w, h, ang = h, w, ang + 90.0
        ang %= 180.0
        if h < 8 or w > 2 * max(H, W) or h / w < 0.2 or h / w > 0.92:
            continue
        pts = c.reshape(-1, 2).astype(np.float32)
        t = np.deg2rad(ang)
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u = (dx * np.cos(t) + dy * np.sin(t)) / (w / 2.0)
        v = (-dx * np.sin(t) + dy * np.cos(t)) / (h / 2.0)
        err = float(np.mean(np.abs(np.hypot(u, v) - 1.0)))
        if err > tol:
            continue
        sector = ((np.arctan2(v, u) + np.pi) / (2 * np.pi) * 12).astype(int) % 12
        if len(np.unique(sector)) < 7:
            continue
        found.append((err, float(cx), float(cy), float(w), float(h), float(ang)))

    found.sort()                                             # best fit first
    out = []
    for err, cx, cy, w, h, ang in found:
        if any(abs(cx - ox) < 0.15 * w and abs(cy - oy) < 0.15 * w and abs(w - ow) < 0.15 * w
               and abs(h - oh) < 0.15 * w for ox, oy, ow, oh, _ in out):
            continue
        out.append((cx, cy, w, h, ang))
    return out


def detect_shapes(img, low=60, high=150, smooth=True,
                  do_lines=True, line_min=60, line_gap=10,
                  do_circles=True, circ_min=15, circ_max=100, circ_sens=0.55,
                  do_ellipses=True, ell_min=40, ell_tol=0.06):
    """
    The whole pipeline of requirement 11 for a gray or colour image:
        gray -> (Gaussian smoothing) -> Canny edges -> lines / circles / ellipses
    Returns a dict with 'edges' (binary image) and the lists 'lines', 'circles', 'ellipses'.
    """
    gray = to_gray(img)
    if smooth:                                               # less noise -> cleaner Canny edges
        gray = apply_filter(gray, "Gaussian", 5, 1.4)
    low = int(min(low, 254))
    high = int(min(max(high, low + 1), 255))
    edges = canny_edges(gray, low, high)

    res = {"edges": edges, "lines": [], "circles": [], "ellipses": []}
    if do_circles:
        res["circles"] = hough_circles(edges, gray, min(circ_min, circ_max),
                                       max(circ_min, circ_max), circ_sens)
    if do_ellipses:
        res["ellipses"] = detect_ellipses(edges, ell_min, ell_tol)
    if do_lines:
        # a big circle / ellipse contains many almost straight pieces that would be reported as
        # lines, so the edge pixels already explained by a circle or ellipse are removed first
        line_edges = edges
        if res["circles"] or res["ellipses"]:
            taken = np.zeros_like(edges)
            for x, y, r in res["circles"]:
                cv2.circle(taken, (x, y), r, 255, 9)
            for cx, cy, w, h, ang in res["ellipses"]:
                cv2.ellipse(taken, ((cx, cy), (w, h), ang), 255, 9)
            line_edges = np.where(taken > 0, 0, edges).astype(np.uint8)
        res["lines"] = hough_lines(line_edges, line_min, line_gap)
    return res


def draw_shapes(img, lines=(), circles=(), ellipses=(), thickness=2):
    """Superimpose the detected shapes on a copy of the image (always returned as BGR)."""
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for x0, y0, x1, y1 in lines:
        cv2.line(out, (x0, y0), (x1, y1), LINE_COLOR, thickness, cv2.LINE_AA)
    for x, y, r in circles:
        cv2.circle(out, (x, y), r, CIRCLE_COLOR, thickness, cv2.LINE_AA)
        cv2.circle(out, (x, y), 2, CIRCLE_COLOR, -1, cv2.LINE_AA)            # centre
    for cx, cy, w, h, ang in ellipses:
        cv2.ellipse(out, ((cx, cy), (w, h), ang), ELLIPSE_COLOR, thickness, cv2.LINE_AA)
        cv2.circle(out, (int(round(cx)), int(round(cy))), 2, ELLIPSE_COLOR, -1, cv2.LINE_AA)
    return out


# ======================================================================
# HELPERS - QUALITY METRIC
# ======================================================================


def psnr(a, b):
    """Peak Signal-to-Noise Ratio in dB (higher = closer to `a`). Identical -> infinity."""
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def fmt_db(v):
    return "∞" if np.isinf(v) else f"{v:.2f} dB"


# ======================================================================
# SELF-TEST (run this file directly; no window is opened)
# ======================================================================
if __name__ == "__main__":
    gray = np.zeros((120, 160), np.uint8)
    gray[40:80, 50:110] = 255                                   # white square on black
    img = cv2.merge([gray, gray // 2, gray // 3])               # a color test image
    for name, noisy in (("Uniform", uniform_noise(img.astype(np.float32), -40, 40)),
                        ("Gaussian", gaussian_noise(img.astype(np.float32), 0, 25)),
                        ("Salt & Pepper", salt_pepper_noise(img, 0.05, 0.5))):
        print(f"{name:14s} noisy PSNR {fmt_db(psnr(img, noisy)):>9s}", end="   ->  ")
        print("  ".join(f"{f} {fmt_db(psnr(img, apply_filter(noisy, f, 5, 1.5)))}"
                        for f in ("Average", "Median", "Gaussian")))
    g = to_gray(img)
    for op in ("Sobel", "Prewitt", "Roberts"):
        print(op, "edge pixels:", int((gradient_edges(g, op, 100)[2] > 0).sum()))
    print("Canny edge pixels:", int((canny_edges(g, 80, 160) > 0).sum()))
    low = np.clip(100 + 0.15 * (g.astype(np.float32) - 100) + np.random.randn(*g.shape) * 4, 0, 255).astype(np.uint8)
    eq = equalize(low)
    print("Equalize: range before", (int(low.min()), int(low.max())), "-> after", (int(eq.min()), int(eq.max())))
    for kind in ("Low pass", "High pass"):
        for shp in ("Ideal", "Gaussian", "Butterworth"):
            out = frequency_filter(img, kind, shp, 20)[0]
            print(f"Frequency {kind:9s} {shp:12s} -> output mean {out.mean():6.1f}, shape {out.shape}")
    print("Hybrid image shape:", hybrid_image(img, 255 - img, 10, 10)[2].shape)
    # requirement 11: a line, a circle and an ellipse on a plain background
    shapes = np.full((300, 400, 3), 40, np.uint8)
    cv2.line(shapes, (20, 30), (200, 40), (255, 255, 255), 2)
    cv2.circle(shapes, (90, 200), 45, (255, 255, 255), 2)
    cv2.ellipse(shapes, ((290, 200), (140, 70), 30), (255, 255, 255), 2)
    r = detect_shapes(shapes)
    print("Shapes found  lines:", len(r["lines"]), " circles:", r["circles"],
          " ellipses:", [tuple(round(v) for v in e) for e in r["ellipses"]])
    print(CACHE.summary())