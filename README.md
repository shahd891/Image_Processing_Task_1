<div align="center">

<img src="Pics/hero_banner.png" alt="Noise Lab" width="100%">

# Noise Lab

### An interactive lab for classic image processing

Add noise to any image, clean it with low-pass filters, find edges, study histograms, normalize,
filter in the frequency domain and build hybrid images - **live, with a slider for every parameter**.

![Python](https://img.shields.io/badge/Python-3-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![customtkinter](https://img.shields.io/badge/UI-customtkinter-f97316?style=for-the-badge)
![matplotlib](https://img.shields.io/badge/Plots-matplotlib-11557c?style=for-the-badge)

[**Quick start**](#quick-start) &nbsp;|&nbsp;
[**Interface**](#the-interface) &nbsp;|&nbsp;
[**Features**](#features) &nbsp;|&nbsp;
[**How it works**](#how-it-works) &nbsp;|&nbsp;
[**Technical notes**](#technical-notes) &nbsp;|&nbsp;
[**Code reference**](#code-reference)

<br>

<table>
  <tr>
    <td align="center" width="33%"><img src="Pics/main_interface_side_by_side.png" alt="Side by side" width="100%"><br><sub><b>Side by side</b></sub></td>
    <td align="center" width="33%"><img src="Pics/interactive_compare_view.png" alt="Compare view" width="100%"><br><sub><b>Compare with a draggable divider</b></sub></td>
    <td align="center" width="33%"><img src="Pics/hybrid_image_generation.png" alt="Hybrid image" width="100%"><br><sub><b>Hybrid images</b></sub></td>
  </tr>
</table>

</div>

---

## At a glance

| | |
|---|---|
| **Noise** | Uniform, Gaussian, Salt & Pepper |
| **Low-pass filters** | Average, Gaussian, Median (all written from scratch) |
| **Edge detection** | Sobel, Prewitt, Roberts (from scratch), Canny (OpenCV) |
| **Histogram tools** | Histogram, CDF, equalization (Luminance / Per channel / gray first) |
| **Normalization** | Min-Max and Z-score |
| **Frequency domain** | Low / High pass with Ideal, Gaussian, Butterworth masks, plus the spectrum before and after |
| **Hybrid images** | Low frequencies of image A + high frequencies of image B |
| **Views** | Side by side, Compare, Edges, Histogram, Normalize, Frequency, Hybrid |
| **Quality metric** | Live PSNR (dB) of the noisy and the filtered image |
| **Export** | Noisy, Filtered, Edges, Equalized, Normalized, Freq. filtered, Hybrid |
| **Modes** | RGB and Gray - one image is kept for each mode |

Most algorithms are implemented **from scratch with NumPy**, straight from their equations. The library is used only for: `cv2.randu` (Uniform noise), `cv2.Canny`, `np.fft` (FFT / inverse FFT), image loading and resizing, and matplotlib for drawing plots.

---

## Quick start

**1. Install**

```bash
pip install customtkinter opencv-python numpy pillow matplotlib
```

**2. Run the app** (keep `app_ui.py` and `main.py` in the same folder)

```bash
python app_ui.py
```

**3. Optional - run the processing self-test** (no window opens)

```bash
python main.py
```

It prints the PSNR of every noise type after each filter, the number of edge pixels found by every detector, the intensity range before and after equalization, and the output shape of all six frequency filters and of the hybrid image.

**4. Use it**

1. Press **Open RGB image** (or switch "Read as" to **Gray** and open a grayscale image).
2. Pick a **Noise type** and move its sliders.
3. Pick a **Low-pass filter** and compare the PSNR badges.
4. Open any other section of the sidebar (edges, histogram, normalization, frequency, hybrid) - the main area switches to the matching view automatically.
5. Use **Save as...** to export a result.

---

## The interface

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/app_initial_state.png" alt="First launch" width="100%"></td>
    <td width="44%" valign="middle"><b>First launch</b><br><br>Nothing is loaded yet, so the image cards ask you to open an image and the two PSNR badges show a dash.<br><br>Press <b>Open RGB image</b>, or switch <b>Read as</b> to <b>Gray</b> first and open a grayscale image.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Side by side - the default view</b><br><br><ul><li><b>Original | With noise | Filtered</b> cards.</li><li>The Filtered caption states the exact filter used, for example <i>Average 5x5</i>.</li><li>The two badges on the top left show the <b>PSNR</b> of the noisy and of the filtered image against the original.</li><li>The <b>view switch</b> sits under the badges and <b>Save as...</b> at the top right.</li><li>The <b>tip bar</b> at the bottom explains the selected filter and recommends one for the current noise type.</li></ul></td>
    <td width="56%" valign="middle"><img src="Pics/main_interface_side_by_side.png" alt="Side by side - the default view" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/ui_controls_sidebar.png" alt="The control sidebar" width="100%"></td>
    <td width="44%" valign="middle"><b>The control sidebar</b><br><br>All controls live in one scrollable sidebar, from top to bottom: image, mode, noise, low-pass filter, edge detection, histogram, normalization, frequency filter and hybrid image.<br><br>Only the controls that matter for your current choice are shown - for example <i>Sigma</i> and <i>Auto kernel size</i> appear only for the Gaussian filter, and <i>Butterworth order n</i> only for the Butterworth mask.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="60%" valign="middle"><b>Part 1 - image, noise, filter, edges</b><br><br>Title, <b>Open RGB image</b>, <b>Read as</b> (RGB / Gray), <b>Noise type</b> with its two sliders, <b>Low-pass filter</b> with <b>Kernel size</b>, and <b>Edge detection</b> with <b>Run it on</b> and the edge threshold.</td>
    <td width="40%" valign="middle"><img src="Pics/Sidebar%20part1.png" alt="Part 1 - image, noise, filter, edges" width="100%"></td>
  </tr>
</table>

### Sidebar, part by part

| Control | Range / options | Default |
|---------|-----------------|---------|
| Open image | PNG, JPG, JPEG, BMP, TIF, TIFF | - |
| Read as | RGB / Gray | RGB |
| Noise type | Uniform / Gaussian / Salt & Pepper | Gaussian |
| Uniform: Low / High | -100..0 / 0..100 | -40 / 40 |
| Gaussian: Mean / Sigma | -50..50 / 0..100 | 0 / 25 |
| Salt & Pepper: Amount / Salt share | 0..100 % / 0..100 % | 5 % / 50 % |
| Low-pass filter | Average / Median / Gaussian | Average |
| Kernel size | odd 3..15 | 5 |
| Sigma (Gaussian filter only) | 0.1..5.0 | 1.5 |
| Auto kernel size from sigma (Gaussian filter only) | on / off | off |
| Edge detection | Sobel / Prewitt / Roberts / Canny | Sobel |
| Run it on | Original / Noisy / Filtered | Filtered |
| Edge threshold (not Canny) | 0..255 (0 = off) | 0 |
| Canny low / high threshold | 0..255 | 80 / 160 |
| Canny aperture size | 3 / 5 / 7 | 3 |
| Accurate gradient (L2) | on / off | off |

<table>
  <tr>
    <td width="40%" valign="middle"><img src="Pics/Sidebar%20part2.png" alt="Part 2 - histogram and normalization" width="100%"></td>
    <td width="60%" valign="middle"><b>Part 2 - histogram and normalization</b><br><br>The <b>Histogram & equalization</b> controls (source image, <i>Per channel / Luminance</i>, <i>Convert to gray first</i>) and the <b>Normalization</b> controls (<i>Min-Max / Z-score</i> and source image).</td>
  </tr>
</table>

| Control | Range / options | Default |
|---------|-----------------|---------|
| Histogram source | Original / Noisy / Filtered | Original |
| Colour mode | Per channel / Luminance | Luminance |
| Convert to gray first | on / off | off |
| Normalization method | Min-Max / Z-score | Min-Max |
| Normalization source | Original / Noisy / Filtered | Original |

<table>
  <tr>
    <td width="60%" valign="middle"><b>Part 3 - frequency filter and hybrid image</b><br><br>The <b>Frequency domain filter</b> controls (source, <i>Low / High pass</i>, mask shape, cut-off D0) and the <b>Hybrid image</b> controls (second image button and the two cut-offs).</td>
    <td width="40%" valign="middle"><img src="Pics/Sidebar%20part3.png" alt="Part 3 - frequency filter and hybrid image" width="100%"></td>
  </tr>
</table>

| Control | Range / options | Default |
|---------|-----------------|---------|
| Frequency source | Original / Noisy / Filtered | Original |
| Pass | Low pass / High pass | Low pass |
| Mask | Ideal / Gaussian / Butterworth | Gaussian |
| Cut-off D0 | 1..200 | 30 |
| Butterworth order n (Butterworth only) | 1..10 | 2 |
| Open 2nd image (high frequencies) | image B | - |
| Low-pass cut-off (A) / High-pass cut-off (B) | 1..100 / 1..100 | 15 / 15 |

**Top bar:** the view switch (Side by side, Compare, Edges, Histogram, Normalize, Frequency, Hybrid) and **Save as...** (Noisy, Filtered, Edges, Equalized, Normalized, Freq. filtered, Hybrid).

---

## Features

### 1. Additive noise

| Noise | Implementation | Parameters (default) | How it works |
|-------|----------------|----------------------|--------------|
| **Uniform** | OpenCV `cv2.randu` | Low -100..0 (-40), High 0..100 (40) | Adds a random value in [Low, High) to every pixel of every channel |
| **Gaussian** | From scratch | Mean -50..50 (0), Sigma 0..100 (25) | Box-Muller transform: `z = sqrt(-2 ln u1) * cos(2 pi u2)`, noise = `mean + sigma * z` |
| **Salt & Pepper** | From scratch | Amount 0..100 % (5), Salt share 0..100 % (50) | One random number per pixel decides: pepper (0), salt (255) or unchanged |

Results are clipped to 0-255. For Salt & Pepper, `r < amount*(1-salt_share)` gives pepper and `r >= 1 - amount*salt_share` gives salt, and the same pixel position is changed in all colour channels. Noise is **only regenerated** when you change the noise type, a noise slider, the mode or the image - changing the filter never makes new noise, so filter comparisons are fair.

**Under the hood**

- Noise is added in floating point (`float32` image + noise), clipped to [0, 255] and stored as `uint8`; the fractional part is truncated, not rounded.
- **Gaussian (Box-Muller):** `u1 = 1 - rand()` lies in (0, 1], so `log(u1)` can never be `log(0)`. Only the cosine branch of the transform is used; the sine partner is not. One independent sample is drawn for every array element, **including every colour channel**, so the noise is colour noise, not grey noise.
- **Uniform:** `cv2.randu` fills a `float32` array with per-channel bounds `(low,)*n` / `(high,)*n`, again independent per channel. The noise has mean `(low + high) / 2` and standard deviation `(high - low) / sqrt(12)`; with the defaults (-40..40) that is about 23.1 grey levels.
- **Salt & Pepper:** the random mask has shape H x W (not H x W x 3), so a corrupted pixel becomes black or white in **all channels** - no coloured specks. Probabilities: pepper `amount * (1 - salt_share)`, salt `amount * salt_share`, so the expected corrupted fraction is `amount`.
- `main.py` never seeds `np.random`, so Gaussian and Salt & Pepper noise differ from run to run.

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/rgb_uniform_noise_average.png" alt="Uniform noise, RGB image" width="100%"></td>
    <td width="44%" valign="middle"><b>Uniform noise, RGB image</b><br><br><b>Noise:</b> Uniform (Low / High sliders).<br><b>Filter:</b> Average.<br><br>Uniform noise is spread evenly between Low and High, so every pixel moves by a random amount inside that range. The Average filter averages it away but also softens edges.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Gaussian noise, sigma 24</b><br><br><b>Noise:</b> Gaussian, sigma 24.<br><b>Filter:</b> Average.<br><br>Sigma is the noise strength. Compare the two PSNR badges: the higher the <i>Filtered</i> value is above the <i>Noisy</i> value, the more of the noise the filter removed.</td>
    <td width="56%" valign="middle"><img src="Pics/gaussian_noise_sigma24_average.png" alt="Gaussian noise, sigma 24" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/grayscale_gaussian_noise_average.png" alt="Gray mode" width="100%"></td>
    <td width="44%" valign="middle"><b>Gray mode</b><br><br><b>Read as:</b> Gray, with a grayscale image.<br><b>Noise:</b> Gaussian.<br><b>Filter:</b> Average.<br><br>Each mode keeps its own image, so you can switch between RGB and Gray without reopening files. Gray mode rejects colour images and RGB mode rejects grayscale ones.</td>
  </tr>
</table>

### 2. Low-pass filters (all from scratch)

| Filter | Equation / idea | Best for |
|--------|-----------------|----------|
| **Average** | Every weight equal: 1D weight `1/k` (2D weight `1/k^2`) | Uniform / Gaussian noise (blurs edges) |
| **Gaussian** | `G(x) = exp(-x^2 / (2 sigma^2))`, normalized to sum 1 | Gaussian noise, gentler on edges than Average |
| **Median** | Middle value of the sorted k x k neighbourhood (non-linear) | Salt & Pepper (extreme values are ignored) |

- Average and Gaussian kernels are **separable**, so they run as one horizontal pass plus one vertical pass (`2k` instead of `k*k` operations per pixel).
- The median stacks the `k*k` shifted copies of the image and uses `np.partition` to place the middle value, in chunks of 64 rows to save memory.
- Kernel size: odd, 3-15. Gaussian sigma: 0.1-5.0 (slider step 0.1). Borders use mirrored (`reflect`) padding.
- **Auto kernel size from sigma** (Gaussian only): `k = 2 * ceil(3 * sigma) + 1`, limited to 3-15, so the kernel always holds +-3 sigma (99.7 % of the bell curve). The kernel slider shows the chosen size and is locked while Auto is on.
- The tip bar recommends a filter for the selected noise: Uniform -> Average or Gaussian, Gaussian -> Gaussian or Average, Salt & Pepper -> Median.

**Under the hood**

- `convolve_1d` flips the kernel first (true convolution; Average and Gaussian kernels are symmetric, so the flip changes nothing), reflect-pads **only along the filtered axis** by `r = k // 2`, and then adds `k` shifted slices: `out += k1d[j] * shifted`. There is no per-pixel Python loop - the loop runs only `k` times over whole-image arrays.
- NumPy's `reflect` padding mirrors **without repeating** the border pixel (`... c b | a b c ...`), so the image does not get dark or bright borders.
- Accumulation is `float32`; the result is rounded with `np.rint`, clipped and stored as `uint8`.
- **Gaussian kernel:** weights `exp(-x^2 / (2 sigma^2))` for `x = -r..+r` are divided by their own sum, so the kernel gain at zero frequency is exactly 1 (brightness is preserved) even when the kernel cuts off the tails of the bell. A kernel that is too small for its sigma is truncated; this is what the *Auto kernel size* option prevents.
- **Median:** the `k*k` shifted views are stacked on a new last axis (shape `rows x W x 3 x k*k`), then `np.partition(stack, mid)[..., mid]` selects the middle value with `mid = k*k // 2` - an exact median because `k*k` is odd. Selection is cheaper than a full sort. Rows are processed 64 at a time: for an 800-pixel-wide colour image and `k = 15` one chunk holds `64 x 800 x 3 x 225` bytes, about 35 MB.
- The median is taken **per channel**, so on colour images it can output a colour combination that did not exist in the neighbourhood. Being non-linear, it cannot be made separable like the other two.
- `apply_filter` raises `ValueError` for an even / non-positive `k`, for `sigma <= 0` (Gaussian) and for an unknown name.

<table>
  <tr>
    <td width="44%" valign="middle"><b>Reading the PSNR badges</b><br><br><b>PSNR</b> = <code>10 log10(255^2 / MSE)</code> in dB; higher means closer to the original image.<br><br>The <i>Noisy</i> badge is the starting quality, the <i>Filtered</i> badge is the quality after the filter. An identical image shows an infinity sign.</td>
    <td width="56%" valign="middle"><img src="Pics/side_by_side_gaussian_noise_average_filter.png" alt="Reading the PSNR badges" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/salt_and_pepper_noise_average_filter.png" alt="Salt & Pepper + Average filter" width="100%"></td>
    <td width="44%" valign="middle"><b>Salt & Pepper + Average filter</b><br><br><b>Noise:</b> Salt & Pepper.<br><b>Filter:</b> Average.<br><br>Each black or white speck is averaged into its neighbours, so it gets smeared into a grey blotch instead of disappearing. Average is not the right tool for this noise.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Salt & Pepper + Gaussian filter</b><br><br><b>Noise:</b> Salt & Pepper.<br><b>Filter:</b> Gaussian.<br><br>A weighted mean has the same weakness: extreme pixels still pull the result towards black or white. Both linear filters blur the image while only partly hiding the specks.</td>
    <td width="56%" valign="middle"><img src="Pics/salt_and_pepper_gaussian_filter.png" alt="Salt & Pepper + Gaussian filter" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/salt_and_pepper_median_filter.png" alt="Salt & Pepper + Median filter" width="100%"></td>
    <td width="44%" valign="middle"><b>Salt & Pepper + Median filter</b><br><br><b>Noise:</b> Salt & Pepper.<br><b>Filter:</b> Median.<br><br>The extreme black and white values end up at the ends of the sorted neighbourhood and are ignored, so the median removes this noise while keeping edges fairly sharp. Use kernel 3 for light noise and a larger kernel for heavy noise (a larger kernel also erases fine detail).</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Compare view</b><br><br>One big image: <b>Noisy</b> on the left of a white divider line, <b>Filtered</b> on the right.<br><br>Drag the knob of the slider under the image to move the divider and inspect exactly what the filter changed.</td>
    <td width="56%" valign="middle"><img src="Pics/interactive_compare_view.png" alt="Compare view" width="100%"></td>
  </tr>
</table>

### 3. Edge detection

| Operator | Implementation | Masks |
|----------|----------------|-------|
| **Sobel** | From scratch | 3x3, centre row/column weighted 2 |
| **Prewitt** | From scratch | 3x3, all weights equal |
| **Roberts** | From scratch | 2x2 diagonal cross |
| **Canny** | OpenCV `cv2.Canny` | Hysteresis thresholds, aperture 3 / 5 / 7, optional L2 gradient |

- Sobel / Prewitt / Roberts compute `Gx` and `Gy` by correlating the gray image with the two masks (edge padding), then `magnitude = sqrt(Gx^2 + Gy^2)`. All three maps are scaled to 0-255 for display. The view shows **4 panels**: Gray input, X direction, Y direction, magnitude.
- **Edge threshold** (0-255, 0 = off) turns the magnitude into a binary map: 255 where the normalized magnitude >= threshold, else 0.
- Canny shows **2 panels**: Gray input and Canny edges. Low 0-255 (80), High 0-255 (160), aperture 3-7 (3), and the **Accurate gradient (L2)** switch. The UI keeps `high > low` automatically.
- "**Run it on**" chooses Original, Noisy or Filtered (default **Filtered**): edge detectors amplify noise, so denoise first.

**Under the hood**

- Every mask is a derivative along one direction multiplied by a smoothing across it: Sobel `[-1 0 1] x [1 2 1]`, Prewitt `[-1 0 1] x [1 1 1]`. Roberts is the pair of 2x2 diagonal differences `[[1, 0], [0, -1]]` and `[[0, 1], [-1, 0]]`.
- `correlate2d` is a **correlation** (the mask is not flipped) with **edge (replicate) padding**, so the image border produces no artificial edge. For an even-sized mask the padding is asymmetric, `top = (kh - 1) // 2` and `bottom = kh - 1 - top`: Roberts pads 0 before and 1 after, so each output pixel uses itself and its right / lower neighbours. The result is the sum of `kh * kw` shifted copies (9 for 3x3, 4 for 2x2).
- `Gx` and `Gy` stay signed `float32`; the magnitude `sqrt(Gx^2 + Gy^2)` is computed from these raw values. Only afterwards `to_display` takes `abs`, divides by that map's own maximum and rounds to 0-255. So the X and Y panels show `|Gx|` and `|Gy|` (the sign - dark-to-light or light-to-dark - is not shown), and each panel has its own brightness scale.
- The **edge threshold** is applied to the *normalized* magnitude (strongest edge = 255), so it is relative to the strongest edge of that image, not an absolute gradient value.
- **Canny (OpenCV):** Sobel gradient with the chosen aperture, non-maximum suppression (thin edges to 1 pixel), then hysteresis: pixels above `high` are kept, pixels between `low` and `high` are kept only if connected to a strong pixel. `cv2.Canny` does not blur the input. `L2gradient=True` uses `sqrt(Gx^2 + Gy^2)`, otherwise `|Gx| + |Gy|`. The wrapper checks gray input, `0 <= low <= high <= 255` and aperture in {3, 5, 7}.

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/edge_detection_sobel.png" alt="Sobel - four panels" width="100%"></td>
    <td width="44%" valign="middle"><b>Sobel - four panels</b><br><br><b>Gray input | X direction | Y direction | edges (magnitude)</b><br><br>The X mask finds vertical edges and the Y mask finds horizontal ones; the magnitude <code>sqrt(Gx^2 + Gy^2)</code> combines them. Sobel counts the centre row/column twice, so it smooths a little while detecting edges.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Prewitt</b><br><br>Same four panels as Sobel, but every mask weight is equal (no extra weight on the centre).<br><br>That makes it slightly more sensitive to noise than Sobel.</td>
    <td width="56%" valign="middle"><img src="Pics/edge_detection_prewitt.png" alt="Prewitt" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/edge_detection_roberts_noisy.png" alt="Roberts on the Noisy image" width="100%"></td>
    <td width="44%" valign="middle"><b>Roberts on the Noisy image</b><br><br><b>Run it on:</b> Noisy.<br><br>Roberts uses tiny 2x2 diagonal masks: sharp and fast, but the most sensitive to noise of the three, so the noise shows up strongly in the edge map.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Edges computed on the Noisy image</b><br><br><b>Run it on:</b> Noisy.<br><br>Every noisy pixel creates a strong local gradient, so the edge map is cluttered with false edges. Compare it with the next screenshot.</td>
    <td width="56%" valign="middle"><img src="Pics/edge_detection_on_noisy_image.png" alt="Edges computed on the Noisy image" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/sobel_edge_detection_on_filtered_image.png" alt="Sobel on the Filtered image" width="100%"></td>
    <td width="44%" valign="middle"><b>Sobel on the Filtered image</b><br><br><b>Run it on:</b> Filtered (the default).<br><br>The low-pass filter removes most of the noise first, so the real edges stand out. This is why <i>Filtered</i> is the default source for edge detection.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Canny on the Original image</b><br><br>Canny (OpenCV) shows <b>two panels</b>: the gray input and the thin Canny edges.<br><br>It finds gradients, thins the edges, then keeps strong edges (above the high threshold) plus weak edges connected to them (above the low threshold).</td>
    <td width="56%" valign="middle"><img src="Pics/edge_detection_canny_original.png" alt="Canny on the Original image" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/edge_detection_canny_filtered.png" alt="Canny on the Filtered image" width="100%"></td>
    <td width="44%" valign="middle"><b>Canny on the Filtered image</b><br><br><b>Run it on:</b> Filtered.<br><br>Canny does <b>not</b> blur the image itself, so run it on the Filtered image to avoid edges that come from noise.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Canny with custom thresholds</b><br><br>The Low and High sliders were moved away from their defaults (80 / 160).<br><br>Raise both thresholds for fewer, stronger edges; lower them to keep more detail. The app always keeps <code>high &gt; low</code>.</td>
    <td width="56%" valign="middle"><img src="Pics/canny_custom_thresholds.png" alt="Canny with custom thresholds" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/edge_detection_canny_l2_gradien.png" alt="Canny with the accurate (L2) gradient" width="100%"></td>
    <td width="44%" valign="middle"><b>Canny with the accurate (L2) gradient</b><br><br>The <b>Accurate gradient (L2)</b> switch is on.<br><br>OpenCV then measures gradient strength with <code>sqrt(Gx^2 + Gy^2)</code> instead of the cheaper <code>|Gx| + |Gy|</code>. <b>Aperture size</b> (3, 5 or 7) sets the Sobel kernel size used inside Canny.</td>
  </tr>
</table>

### 4 / 5 / 8. Histogram, CDF and equalization (from scratch)

- **Histogram**: `np.bincount` - how many pixels have each intensity 0-255.
- **CDF**: running total of the histogram divided by the pixel count (`np.cumsum`).
- **Equalization** maps every intensity with a 256-entry lookup table: `new = (CDF - CDF_min) / (1 - CDF_min) * 255`. A constant image is returned unchanged.
- Colour images have two modes:
  - **Luminance** (default): equalize only the brightness `Y = 0.299 R + 0.587 G + 0.114 B`, then multiply B, G and R by `Y_new / Y_old`, so the colours (hue) are kept.
  - **Per channel**: B, G and R are equalized independently (colours may shift).
- **Convert to gray first** switch turns the chosen image to gray before equalizing.
- The view is a 2 x 2 grid: Before and Equalized images on top, and below each one its **Histogram + CDF** plot (solid = histogram on the left axis, dashed = CDF on the right axis; B, G, R curves for colour images).

**Under the hood**

- `np.bincount(channel.ravel(), minlength=256)` gives the 256 counts; the CDF is `np.cumsum(h) / N` in `float64`.
- The lookup table is `lut[v] = rint((CDF[v] - CDF_min) / (1 - CDF_min) * 255)`, where `CDF_min` is the CDF at the **first intensity that is present**, so the darkest used level maps to 0 and the brightest to 255. It is applied with NumPy fancy indexing, `lut[channel]`, which looks up every pixel at once (cost proportional to the pixel count).
- A constant image (`CDF_min = 1`) gets the identity table, because equalization cannot create contrast from nothing.
- Equalization can only merge or spread existing levels (at most 256 output values), so the equalized CDF is approximately - not exactly - a straight line.
- **Luminance mode:** `y = to_gray(img)`, `ratio = lut[y] / max(y, 1)`, `out = clip(rint(img * ratio))`. B, G and R are multiplied by the same ratio, so their proportions (hue) stay the same; the `max(y, 1)` avoids a division by zero on black pixels. Very bright pixels can clip at 255, which shifts their colour slightly.
- **Per channel mode** builds three different tables, so the colour balance can change.
- `plot_histograms` draws with matplotlib's off-screen **Agg** backend (`Figure` + `FigureCanvasAgg`, 100 dpi, twin y-axes) and returns the canvas as a BGR array, so no extra window ever opens.

<table>
  <tr>
    <td width="44%" valign="middle"><b>Equalizing the Noisy image</b><br><br><b>Source:</b> Noisy.<br><br>Top row: <i>Before</i> and <i>Equalized</i> images. Bottom row: their histogram (solid) and CDF (dashed). After equalization the CDF becomes almost a straight line, meaning the intensities are spread over the full range.</td>
    <td width="56%" valign="middle"><img src="Pics/histogram_equalization_noisy.png" alt="Equalizing the Noisy image" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/histogram_equalization_filtered.png" alt="Equalizing the Filtered image" width="100%"></td>
    <td width="44%" valign="middle"><b>Equalizing the Filtered image</b><br><br><b>Source:</b> Filtered.<br><br>The same four panels for the denoised image. Filtering smooths the histogram, and equalization then stretches it to add contrast.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Luminance mode</b><br><br><b>Colour mode:</b> Luminance.<br><br>Only the brightness is equalized and B, G, R are scaled by the same ratio, so the <b>colours are kept</b> while the contrast improves.</td>
    <td width="56%" valign="middle"><img src="Pics/histogram_equalization_luminance.png" alt="Luminance mode" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/histogram_equalization_per_channel.png" alt="Per channel mode" width="100%"></td>
    <td width="44%" valign="middle"><b>Per channel mode</b><br><br><b>Colour mode:</b> Per channel.<br><br>B, G and R are equalized separately, each with its own lookup table. Contrast increases in every channel, but the <b>colours may shift</b>.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Gray image</b><br><br>A grayscale image (or a colour image with <b>Convert to gray first</b> on) needs only one equalization, so the plots show a single gray curve.</td>
    <td width="56%" valign="middle"><img src="Pics/histogram_equalization_grayscale.png" alt="Gray image" width="100%"></td>
  </tr>
</table>

### 6. Normalization (from scratch)

| Method | Equation | Notes |
|--------|----------|-------|
| **Min-Max** | `new = (v - min) / (max - min) * 255` | Darkest pixel -> 0, brightest -> 255 |
| **Z-score** | `z = (v - mean) / std`, shown as `(z + 3) / 6 * 255` | Mean becomes mid-gray; values beyond +-3 std are clipped |

Each channel is normalized separately. Both methods are **linear**, so the histogram keeps its shape (unlike equalization). A channel with a single value cannot be stretched and stays as it is. Because one extreme pixel already uses the full range, Min-Max barely changes a Salt & Pepper image.

**Under the hood**

- The image is converted to `float32` and the statistics are taken **per channel** over the whole image (`axis=(0, 1)`, `keepdims=True`), so B, G and R are scaled independently and the colour balance can change.
- Min-Max: `(f - lo) / max(hi - lo, 1e-9) * 255` where `hi > lo`, otherwise the channel is left unchanged. Z-score: `((f - mean) / max(std, 1e-9) + 3) / 6 * 255` where `std > 0`. The result is rounded with `np.rint` and clipped.
- Z-score maps the mean to 127.5 and +-3 standard deviations to 0 / 255.
- Both are affine maps `out = a*v + b` with `a > 0`: the order of the intensities and the shape of the histogram are preserved. Equalization, in contrast, is a non-linear map.

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/normalization_min_max_filtered.png" alt="Min-Max on the Filtered image" width="100%"></td>
    <td width="44%" valign="middle"><b>Min-Max on the Filtered image</b><br><br><b>Method:</b> Min-Max, <b>source:</b> Filtered.<br><br>The darkest pixel becomes 0 and the brightest 255. The histogram below is only moved and stretched - its shape stays the same.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Z-score on the Original image</b><br><br><b>Method:</b> Z-score, <b>source:</b> Original.<br><br>Values are shown as <code>(z + 3) / 6 * 255</code>, so the mean becomes mid-gray and +-3 standard deviations map to 0 and 255.</td>
    <td width="56%" valign="middle"><img src="Pics/normalization_z_score_original.png" alt="Z-score on the Original image" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/normalization_z_score_noisy.png" alt="Z-score on the Noisy image" width="100%"></td>
    <td width="44%" valign="middle"><b>Z-score on the Noisy image</b><br><br><b>Source:</b> Noisy.<br><br>Z-score is less sensitive to a single extreme pixel than Min-Max, but values beyond 3 standard deviations are clipped.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Z-score on the Filtered image</b><br><br><b>Source:</b> Filtered.<br><br>Compare the histogram of this panel with the Noisy and Original versions to see how the spread of the intensities changes.</td>
    <td width="56%" valign="middle"><img src="Pics/normalization_z_score_filtered.png" alt="Z-score on the Filtered image" width="100%"></td>
  </tr>
</table>

### 9. Frequency-domain filters

Pipeline for each channel: `FFT -> shift the centre -> spectrum x mask H -> inverse shift -> inverse FFT -> real part`. The FFT / inverse FFT come from `np.fft`; the masks and the multiplication are written from scratch.

| Mask | Low pass `H` | High pass |
|------|--------------|-----------|
| **Ideal** | `1 if D <= D0 else 0` (sharp cut, **ringing**) | `1 - H_low` |
| **Gaussian** | `exp(-D^2 / (2 D0^2))` (smooth, no ringing) | `1 - H_low` |
| **Butterworth** | `1 / (1 + (D/D0)^(2n))` (n = order, between the other two) | `1 - H_low` |

- `D` is the distance from the centre of the spectrum, `D0` the cut-off radius (1-200, default 30). Butterworth order `n`: 1-10 (default 2); the slider appears only for Butterworth.
- High-pass results are shown on a mid-gray background (+128) so negative values stay visible.
- The 2 x 2 view shows **Before**, **After**, and the log-magnitude spectrum before and after the mask (`log(1 + |F|)`, centre = low frequencies).

**Under the hood**

- **Mask geometry:** `np.ogrid` builds the distance `D` from the centre bin `(rows // 2, cols // 2)` - exactly where `np.fft.fftshift` puts frequency (0, 0) for both even and odd sizes. `D` is measured in frequency bins (cycles per image). `D0` is clamped to at least `1e-3` to avoid a division by zero.
- **What "cut-off" means differs per mask:**

  | Mask | Gain at `D = D0` | Gain at `D = 2 * D0` (Butterworth n = 2) |
  |------|------------------|------------------------------------------|
  | Ideal | 1 (it is 0 just beyond) | 0 |
  | Gaussian | `exp(-1/2)` = 0.607 | `exp(-2)` = 0.135 |
  | Butterworth | 0.5 for every n | `1 / (1 + 2^4)` = 0.059 |

- **Filtering:** `fft2` -> `fftshift` -> multiply by the mask -> `ifftshift` -> `ifft2` -> `np.real` (the leftover imaginary part is numerical noise). Everything stays in floating point; the result is rounded and clipped to `uint8` only at the end. Colour images are filtered channel by channel (B, G, R).
- Multiplying DFTs is a **circular** convolution and the code adds no padding, so the image is treated as periodic: detail can wrap around at the borders, most visibly with a small `D0`.
- **Ringing:** the Ideal mask's sharp edge corresponds to an oscillating (sinc-like) impulse response, which produces ripples around edges (Gibbs phenomenon). Gaussian has no ripples; Butterworth ripples grow with `n`.
- **High pass** is `1 - low`, which sets the zero-frequency term to 0. The output therefore has a mean near 0, which is why +128 is added for display.
- **Spectrum pictures:** `log(1 + |F|)` scaled by its maximum, because the DC term is orders of magnitude larger than the rest. The before / after spectra are computed from the **grayscale** version of the input, even for colour images. `frequency_filter` also returns the mask itself as a `uint8` image (`rint(mask * 255)`).
- **Cost:** an FFT costs `O(N log N)` per channel and does not depend on `D0`, while a spatial filter with a large kernel gets more expensive with its size.

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/frequency_domain_gaussian_lowpass.png" alt="Gaussian low pass" width="100%"></td>
    <td width="44%" valign="middle"><b>Gaussian low pass</b><br><br><b>Mask:</b> Gaussian, <b>pass:</b> Low pass.<br><br>Top: the image before and after. Bottom: the spectrum before and after multiplying by the mask - only the bright centre (low frequencies) survives. The result is a smooth blur with <b>no ringing</b>.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Gaussian low pass on the Noisy image</b><br><br><b>Source:</b> Noisy.<br><br>Noise lives mostly in the high frequencies, so removing them reduces the noise, at the price of blurring fine detail.</td>
    <td width="56%" valign="middle"><img src="Pics/frequency_domain_gaussian_lowpass_noisy.png" alt="Gaussian low pass on the Noisy image" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/frequency_domain_gaussian_lowpass_filtered.png" alt="Gaussian low pass on the Filtered image" width="100%"></td>
    <td width="44%" valign="middle"><b>Gaussian low pass on the Filtered image</b><br><br><b>Source:</b> Filtered.<br><br>Frequency filtering applied after the spatial filter. Use it to compare both approaches or to combine them.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Gaussian high pass</b><br><br><b>Mask:</b> Gaussian, <b>pass:</b> High pass (<code>1 - low pass</code>).<br><br>Only edges and fine detail are kept; the smooth parts disappear and the result is shown on a mid-gray background.</td>
    <td width="56%" valign="middle"><img src="Pics/frequency_domain_gaussian_highpass.png" alt="Gaussian high pass" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/frequency_domain_ideal_highpass.png" alt="Ideal high pass" width="100%"></td>
    <td width="44%" valign="middle"><b>Ideal high pass</b><br><br><b>Mask:</b> Ideal.<br><br>H is 1 outside a circle of radius D0 and 0 inside. The sharp cut creates <b>ringing</b> - ripples around edges.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>Butterworth high pass</b><br><br><b>Mask:</b> Butterworth, order n.<br><br><code>H = 1 / (1 + (D/D0)^(2n))</code> for the low pass, and high pass is its complement. n = 1 is very smooth like Gaussian; a large n approaches the Ideal filter.</td>
    <td width="56%" valign="middle"><img src="Pics/frequency_domain_butterworth_highpass.png" alt="Butterworth high pass" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/frequency_domain_butterworth_highpass_custom.png" alt="Butterworth high pass, custom D0 / order" width="100%"></td>
    <td width="44%" valign="middle"><b>Butterworth high pass, custom D0 / order</b><br><br>The <b>Cut-off D0</b> and <b>order n</b> sliders were changed from their defaults (30 and 2).<br><br>A smaller D0 keeps only the finest edges; a larger order makes the cut sharper.</td>
  </tr>
</table>

### 10. Hybrid images

`Hybrid = LowPass(A) + HighPass(B)`, both with **Gaussian** masks.

- **A** is the Original image of the current mode; **B** is a second image you open with **Open 2nd image (high frequencies)**. B is resized to A; if one image is colour and the other gray, both are converted to gray.
- Two sliders set the cut-offs: Low-pass cut-off (A) and High-pass cut-off (B), each 1-100, default 15.
- The view shows A low-passed, B high-passed, the **Hybrid**, and a **1/4-size copy** of the hybrid - roughly what it looks like from far away (you see A's smooth shape, while up close you see B's detail).

**Under the hood**

- Order of operations: B is resized to A's size with `cv2.INTER_AREA` (A decides the size); if one image is colour and the other gray, both become gray; A is filtered with a **Gaussian low pass** (`D0 = low cut-off`) and B with a **Gaussian high pass** (`D0 = high cut-off`).
- The filter outputs stay in floating point until the end: `hybrid = clip(rint(low_raw + high_raw))`. The high-pass part has a mean near 0, so adding it keeps A's overall brightness. The extra +128 exists only in the picture of "B high-passed".
- **Cut-off in pixels:** a Gaussian low pass with cut-off `D0` (in cycles per image) behaves like a spatial Gaussian blur with `sigma = W / (2 * pi * D0)` pixels. For an 800-pixel-wide image and `D0 = 15` that is about 8.5 px.
- `H_low + H_high = 1` for the same `D0`, so with equal cut-offs the low part and high part of the **same** image add up to that image again (perfect reconstruction). In a hybrid, the two cut-offs decide which detail sizes A and B each contribute.

<table>
  <tr>
    <td width="44%" valign="middle"><b>Building a hybrid image</b><br><br>Four panels: <b>A low-passed</b>, <b>B high-passed</b>, the <b>Hybrid</b>, and the hybrid at <b>1/4 size</b>.<br><br>Up close you see the sharp details of B; from far away (or in the small copy) you see the smooth shape of A.</td>
    <td width="56%" valign="middle"><img src="Pics/hybrid_image_generation.png" alt="Building a hybrid image" width="100%"></td>
  </tr>
</table>

<table>
  <tr>
    <td width="56%" valign="middle"><img src="Pics/hybrid_image_custom_cutoff.png" alt="Custom cut-offs" width="100%"></td>
    <td width="44%" valign="middle"><b>Custom cut-offs</b><br><br>Both cut-off sliders moved away from 15.<br><br>Use similar cut-offs for A and B and check the small copy to see which image dominates at a distance.</td>
  </tr>
</table>

<table>
  <tr>
    <td width="44%" valign="middle"><b>High cut-offs</b><br><br>Larger cut-offs keep more detail from both images.<br><br>The two pictures then blend less cleanly, so lower the cut-offs if the illusion gets weaker.</td>
    <td width="56%" valign="middle"><img src="Pics/hybrid_image_high_cutoff.png" alt="High cut-offs" width="100%"></td>
  </tr>
</table>

### Helpers

- **`to_gray`**: `gray = 0.299 R + 0.587 G + 0.114 B`.
- **PSNR**: `10 * log10(255^2 / MSE)` in dB; higher means closer to the original, identical images show an infinity sign.

### Interface extras

- **Read as RGB / Gray**: each mode keeps its own image. RGB mode rejects grayscale images and Gray mode rejects colour images (a colour image whose channels differ by less than 3 levels counts as gray).
- Loaded images are scaled so the longest side is at most **800 px** (`MAX_SIDE`); 16-bit images become 8-bit and an alpha channel is dropped.
- **Tip bar** under the images explains the active filter / operator / view in plain language.
- Sliders move **only when you drag the knob** (clicking the track does nothing), and heavy work is debounced so dragging stays smooth.
- **Save as...** exports the current result as PNG by default.

---

## How it works

```
 Open image ──► original ──► NOISE ──► noisy ──► LOW-PASS FILTER ──► filtered
                  │            ▲                      ▲                  │
                  │            │                      │                  │
                  │    noise slider moved       filter / kernel /        │
                  │    (new random noise)       sigma changed            ▼
                  │                             (same noisy image)   PSNR badges
                  │
                  └──► every view reads  original / noisy / filtered
                        │
        Side by side ◄──┤  Compare ◄──┤  Edges ◄──┤  Histogram ◄──┤
        Normalize ◄─────┤  Frequency ◄┤  Hybrid ◄─┘
```

- **`main.py` has no UI code.** Every function takes NumPy images (`uint8`, gray = H x W, colour = H x W x 3 in OpenCV's BGR order) and returns new images, so it can be tested alone.
- **`app_ui.py` has no image-processing math.** It builds the window, reads the sliders and calls `main.py`.
- Two-step processing: `apply()` makes new noise and then calls `apply_filter()`; `apply_filter()` re-filters the *current* noisy image.
- Every sidebar section that you touch switches the main area to its view (`show_edges`, `show_hist`, `show_norm`, `show_freq`, `show_hyb`).
- `render()` draws whichever view is active, sized to the current window.

---

## Technical notes

### Numeric types and rounding

| Stage | Working type | How the result returns to `uint8` |
|-------|--------------|------------------------------------|
| Input image | `uint8` (gray H x W, colour H x W x 3, BGR order) | - |
| Noise | `float32` image + noise | clip 0-255, **truncate** |
| Average / Gaussian filter | `float32` accumulators | `np.rint`, clip |
| Median filter | `uint8` values (selection only) | no arithmetic, no rounding |
| Sobel / Prewitt / Roberts | `float32` `Gx`, `Gy`, magnitude | scaled by the maximum, `np.rint` |
| Equalization | `uint8` lookup table | table lookup |
| Normalization | `float32` | `np.rint`, clip |
| Frequency filters / hybrid | floating-point FFT results | `np.rint`, clip (once, at the end) |
| PSNR | `float32` squared error | `float` in dB |

### Border handling

| Operation | Border rule | Why |
|-----------|-------------|-----|
| Average, Gaussian, Median | `reflect` (mirror) padding | no dark or bright frame around the image |
| Sobel, Prewitt, Roberts | `edge` (replicate) padding | the padded area is flat, so the border produces no false edge |
| Frequency filters | none - the FFT treats the image as periodic | consequence of the DFT (see the frequency section) |

### Cost per pixel

| Operation | Cost |
|-----------|------|
| Average / Gaussian, kernel `k` | `2k` multiply-adds (separable) instead of `k*k`; for `k = 15`: 30 instead of 225 |
| Median, kernel `k` | selection among `k*k` values (not separable) |
| Sobel / Prewitt | 9 multiply-adds per mask, 2 masks |
| Roberts | 4 multiply-adds per mask, 2 masks |
| Histogram equalization | one table lookup per pixel (after one pass to count) |
| Frequency filter | FFT + inverse FFT, `O(N log N)`, independent of `D0` |

### Design decisions

- **Filter the same noisy image.** Noise is generated once per noise change; changing the filter, kernel or sigma re-filters that image, so two filters are compared on identical noise.
- **Separable kernels** make large Average / Gaussian filters cheap without changing the result.
- **Reflect vs edge padding:** blur filters mirror the border; gradient masks replicate it, which avoids a strong false edge on the image frame.
- **Luminance equalization is the default** for colour images because it keeps the hue; Per channel is available when you want the independent behaviour.
- **Edge detection defaults to the Filtered image** because gradient operators amplify noise.
- **`main.py` and `app_ui.py` are separate** so all algorithms can be tested without a window (`python main.py`).

### Known limitations

- Frequency-domain filtering is circular (no padding), so detail can wrap around the borders.
- Normalization and Per channel equalization treat B, G and R independently and can shift colours; the median filter is also per channel.
- Edge detectors and Canny work on the gray version of the image.
- Gaussian and Salt & Pepper noise are not seeded in `main.py`, so results are not reproducible between runs.
- Images are limited to 800 px on the longest side (`MAX_SIDE`).

---

## Code reference

<details>
<summary><b>main.py - the image-processing functions</b> (click to expand)</summary>

<br>

| Function | What it does | Implementation |
|----------|--------------|----------------|
| `uniform_noise(img, low, high)` | Adds uniform noise in [low, high) to every channel, clips to 0-255 | OpenCV `cv2.randu` |
| `gaussian_noise(img, mean, sigma)` | Box-Muller Gaussian noise `N(mean, sigma^2)` on every value | From scratch |
| `salt_pepper_noise(img, amount, salt_share)` | Random mask per pixel: pepper 0 / salt 255 | From scratch |
| `pad_image(img, r)` | Mirrored border of `r` pixels | NumPy |
| `convolve_1d(img, k1d, axis)` | 1D convolution along an axis (flipped kernel, reflect padding) | From scratch |
| `filter_separable(img, k1d)` | Horizontal then vertical 1D pass = full 2D filter | From scratch |
| `average_kernel(k)` | `k` equal weights `1/k` | From scratch |
| `gaussian_kernel(k, sigma)` | 1D Gaussian weights, normalized to sum 1 | From scratch |
| `median_filter(img, k)` | Median of the k x k neighbourhood via `np.partition`, 64 rows at a time | From scratch |
| `apply_filter(img, name, k, sigma)` | Chooses Average / Gaussian / Median; raises `ValueError` for even `k` or `sigma <= 0` | From scratch |
| `to_gray(img)` | Luminance equation `0.299 R + 0.587 G + 0.114 B` | From scratch |
| `correlate2d(img, kernel)` | 2D mask correlation with edge padding, supports 3x3 and 2x2 masks | From scratch |
| `EDGE_MASKS` | Gx / Gy masks of Sobel, Prewitt, Roberts | Constants |
| `to_display(a)` | Scales `abs(a)` to 0-255 by its maximum | From scratch |
| `gradient_edges(gray, name, threshold)` | Returns (Gx, Gy, magnitude) images; optional binary threshold | From scratch |
| `canny_edges(gray, low, high, aperture, accurate)` | Canny; validates `0 <= low <= high <= 255` and aperture 3/5/7 | OpenCV `cv2.Canny` |
| `histogram(channel)` | 256 pixel counts | `np.bincount` |
| `cdf(h)` | Normalized cumulative distribution | `np.cumsum` |
| `equalization_lut(h)` | 256-entry equalization mapping | From scratch |
| `equalize_channel(channel)` | Histogram -> LUT -> apply to every pixel | From scratch |
| `equalize(img, mode)` | Gray, `"Per channel"` or `"Luminance"` (ratio method) | From scratch |
| `plot_histograms(img, w, h, title, ..., vline)` | Draws histogram + dashed CDF (gray or B/G/R) as a BGR image; optional vertical orange line | matplotlib (drawing only) |
| `normalize(img, method)` | `"Min-Max"` or `"Z-score"`, per channel | From scratch |
| `frequency_mask(shape, kind, shape_name, d0, order)` | Centred Ideal / Gaussian / Butterworth mask; high pass = `1 - low` | From scratch |
| `filter_channel_fft(channel, mask)` | FFT, shift, multiply by mask, inverse; returns (result, F, G) | `np.fft` + from scratch |
| `spectrum_image(spec)` | `log(1 + abs(F))` scaled to 0-255 | From scratch |
| `frequency_filter(img, kind, shape_name, d0, order)` | Filters every channel; returns (filtered, spectrum before, spectrum after, mask). High pass shown +128 | `np.fft` + from scratch |
| `hybrid_image(img_a, img_b, d0_low, d0_high)` | Returns (A low-passed, B high-passed, hybrid); Gaussian masks; B resized to A | `np.fft` + from scratch |
| `psnr(a, b)` / `fmt_db(v)` | PSNR in dB / text such as `27.31 dB` or an infinity sign | From scratch |

The `if __name__ == "__main__":` block is a **self-test**: it builds a white square on black, adds each noise type, prints PSNR after each filter, counts edge pixels for every detector, checks equalization range, runs all six frequency filters and a hybrid.

</details>

<details>
<summary><b>app_ui.py - the interface</b> (click to expand)</summary>

<br>

**Sections of the file**

1. **Settings and theme colours** - `MAX_SIDE = 800` and the palette:

   | Name | Colour | Used for |
   |------|--------|----------|
   | `BG` | `#12110f` | window background |
   | `PANEL` | `#1c1a17` | sidebar and image cards |
   | `CARD` | `#2a2622` | value chips, switches, badges |
   | `ACCENT` | `#f97316` | buttons, sliders, selected items |
   | `HOVER` | `#fb923c` | hovered accent |
   | `MUTED` | `#a39e96` | secondary text |
   | `TEXT` | `#f3efe9` | main text |

2. **Help texts** - `FILTER_INFO`, `EDGE_INFO`, `HIST_INFO`, `NORM_INFO`, `FREQ_INFO`, `FREQ_STEPS`, `HYBRID_INFO`, `BEST_FILTER`; shown in the tip bar.
3. **`knob_only(slider, lo, hi)`** - makes a slider move only while its knob is pressed and dragged, so clicking the track does not make it jump.
4. **`ParamSlider`** - title + value chip + slider. Options: `odd=True` returns only odd numbers (kernel sizes), `div=10` divides the value (sigma in steps of 0.1), `unit` adds a suffix such as `%`.
5. **`App`** - the main window.

**`App` methods**

| Group | Methods | Purpose |
|-------|---------|---------|
| Window | `fit_to_screen`, `schedule` | Never opens bigger than the screen (max 1360 x 780); `schedule` runs a function after a delay and cancels older calls (debouncing: noise 60 ms, filter 200 ms, edges / frequency / hybrid 80 ms, resize 80 ms) |
| Layout | `build_sidebar`, `build_stage`, `grid_view`, `card`, `badge`, `caption`, `style_seg` | Builds the scrollable sidebar, the 7 views, rounded image cards, PSNR badges |
| Show / hide | `set_view`, `show_params`, `show_fparams`, `show_eparams`, `show_fqparams`, `update_tip` | Show only the controls that matter for the current choice; update the tip text |
| Loading | `open_image`, `open_second_image` | File dialogs, 16-bit -> 8-bit, drop alpha, resize to `MAX_SIDE`, RGB / Gray check |
| Processing | `apply`, `apply_filter`, `pick` | Step 1: noise. Step 2: filter. `pick` returns Original / Noisy / Filtered |
| Drawing | `fit`, `show`, `render`, `grid_size` | Aspect-ratio resize, BGR -> RGB display, draw the active view |
| Edges | `show_edges`, `edge_panels`, `render_edges` | 4 panels for gradient operators, 2 for Canny |
| Histogram | `show_hist`, `hist_pair`, `render_hist` | Before / Equalized images + histogram and CDF plots |
| Normalize | `show_norm`, `norm_pair`, `render_norm` | Before / Normalized images + plots |
| Frequency | `show_freq`, `freq_results`, `render_freq` | Before, After and the two spectra |
| Hybrid | `show_hyb`, `hybrid_results`, `render_hyb` | A low, B high, hybrid, 1/4-size hybrid |
| Saving | `save` | Writes the chosen result with `cv2.imwrite` (default name e.g. `freq_filtered.png`) |

</details>

---

## Tips

| Situation | What to try |
|-----------|-------------|
| Salt & Pepper noise | **Median**, kernel 3 for light noise, larger for heavy noise |
| Gaussian or Uniform noise | **Gaussian** or **Average**; watch the PSNR badge go up |
| Noisy edge map | Set "Run it on" to **Filtered**, or raise the edge / Canny thresholds |
| Colours look wrong after equalization | Use **Luminance** instead of Per channel |
| Ringing around edges in the frequency view | Use **Gaussian** (or Butterworth with a small n) instead of Ideal |
| Hybrid image looks like only one picture | Use similar cut-offs for A and B; check the 1/4-size copy |
| Slow response with a big image | Lower `MAX_SIDE` at the top of `app_ui.py`; the Median filter is the heaviest |

---

## Project structure

```
Task1/
├── main.py          # all image-processing functions (no UI code)
├── app_ui.py        # customtkinter interface (no image-processing math)
├── README.md
└── Pics/            # screenshots used in this README
```

## Requirements

- Python 3 with Tkinter
- `customtkinter`, `opencv-python`, `numpy`, `pillow`
- `matplotlib` (histogram / CDF plots only)