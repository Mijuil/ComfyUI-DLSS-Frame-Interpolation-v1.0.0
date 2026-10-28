"""Bounded SDR luminance amplification and color grading after neural rendering.

This is output composition, not an additional neural inference pass. Keeping
it outside the worker protocol preserves compatibility with Windows workers.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class ColorGradeParams:
    """Post-DLSS color & light correction in display-referred space.

    luma_match auto-scales the rendered frame's exposure so its median
    luminance ratio versus the source is bounded - this compensates the
    darkening DLSS sometimes applies to skin. The remaining controls are a
    classic lift-free grade: exposure, contrast around mid-gray, saturation,
    and a smooth shadows/highlights tilt.
    """

    luma_match: float = 0.0   # 0..1, amount of median-luma matching to source
    exposure: float = 0.0     # -1..1 EV-ish, 0.0 = neutral
    contrast: float = 0.0     # -1..1, 0.0 = neutral
    saturation: float = 0.0   # -1..1, 0.0 = neutral
    shadows: float = 0.0      # -1..1, lift/crush dark tones
    highlights: float = 0.0   # -1..1, recover/burn bright tones

    @property
    def neutral(self) -> bool:
        return (
            self.luma_match == 0.0
            and self.exposure == 0.0
            and self.contrast == 0.0
            and self.saturation == 0.0
            and self.shadows == 0.0
            and self.highlights == 0.0
        )


def validate_color_grade(
    luma_match: float,
    exposure: float,
    contrast: float,
    saturation: float,
    shadows: float,
    highlights: float,
) -> ColorGradeParams:
    params = ColorGradeParams(
        luma_match=float(luma_match),
        exposure=float(exposure),
        contrast=float(contrast),
        saturation=float(saturation),
        shadows=float(shadows),
        highlights=float(highlights),
    )
    for name in ("luma_match", "exposure", "contrast", "saturation", "shadows", "highlights"):
        value = getattr(params, name)
        if not math.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError(f"Color grade {name} must be between -1 and 1.")
    return params


def color_grade_report(params: ColorGradeParams) -> dict:
    return {
        "luma_match": params.luma_match,
        "exposure": params.exposure,
        "contrast": params.contrast,
        "saturation": params.saturation,
        "shadows": params.shadows,
        "highlights": params.highlights,
        "applied": not params.neutral,
        "execution": "cpu" if not params.neutral else "bypass",
    }


def validate_detail_strength(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 1.0 <= value <= 2.0:
        raise ValueError("Output detail strength must be between 1 and 2.")
    return value


def composition_report(strength: float) -> dict:
    return {
        "output_detail_strength": strength,
        "applied": strength != 1.0,
        "method": "bounded-sdr-luminance-ratio",
        "execution": "cpu" if strength != 1.0 else "bypass",
        "maximum_luminance_ratio": 2.0,
    }


def compose_sdr(
    original: np.ndarray, rendered: np.ndarray, strength: float = 1.0
) -> np.ndarray:
    """Amplify the rendered luminance change without extrapolating RGB.

One is an exact bypass. Above one, raise the rendered/original luminance
ratio to the requested power, bound it to [1/2, 2], and scale rendered RGB
uniformly to preserve its hue before final 8-bit clipping. A small shared
luminance floor avoids unstable division in shadows. Work in row tiles to
bound temporary memory for 4K video; alpha is never part of composition.

Inputs are display-referred uint8 RGBA frames. A resized source provides a
reference when SR and NR have already produced a larger output; this is not
an isolated NR residual in that case.
    """
    strength = validate_detail_strength(strength)
    if strength == 1.0:
        return rendered
    for frame in (original, rendered):
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 4:
            raise ValueError("SDR composition requires uint8 RGBA frames.")
    height, width = rendered.shape[:2]
    if original.shape[:2] != (height, width):
        original = cv2.resize(
            original, (width, height), interpolation=cv2.INTER_LINEAR
        )
    result = rendered.copy()
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    floor = np.float32(1.0 / 512.0)
    for row in range(0, height, 64):
        end = min(row + 64, height)
        source = original[row:end, :, :3].astype(np.float32) / 255.0
        model = rendered[row:end, :, :3].astype(np.float32) / 255.0
        source_luma = source @ weights
        model_luma = model @ weights
        ratio = (model_luma + floor) / (source_luma + floor)
        bounded = np.clip(ratio ** strength, 0.5, 2.0)
        enhanced = model * (bounded / ratio)[..., None]
        # An empty model answer is not usable detail; retain the source.
        enhanced = np.where((model_luma <= 1e-5)[..., None], source, enhanced)
        result[row:end, :, :3] = np.rint(
            np.clip(enhanced, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return result


def _tilt_weights(length: int, width_px: int) -> np.ndarray:
    """Smooth 0..1..0 band used for shadows/highlights weighting."""
    half = max(1.0, width_px / 2.0)
    x = np.arange(length, dtype=np.float32) + 0.5
    return np.clip(1.0 - np.abs(x - length / 2.0) / half, 0.0, 1.0)


@dataclass(slots=True)
class EnhanceParams:
    """Post-DLSS detail & cleanup controls, all individually bypassable.

    gamma: midtone lift/burn in power space (0.5 = brighter mids, 2.0 = darker).
    sharpen: unsharp mask on the rendered frame; edge_only gates it to a
        Sobel-edge mask so flat areas (skin, sky) stay untouched.
    clarity: local mid-frequency contrast via a box-blur high-pass.
    skin_smoothing: edge-preserving smoothing that suppresses generation
        grain/wobble on skin without erasing pores (bilateral filter).
    deband: smooths 8-bit banding/blocking (H.264/H.265 staircases) by adding
        bounded low-amplitude noise along flat gradients.
    """

    gamma: float = 1.0        # 0.25..2.5, 1.0 = neutral
    sharpen: float = 0.0      # 0..2, 0 = off
    edge_only: bool = True    # sharpen only near strong edges
    clarity: float = 0.0      # -1..1, 0 = off
    skin_smoothing: float = 0.0  # 0..1, 0 = off
    deband: bool = False      # off by default; enable for compressed sources

    @property
    def neutral(self) -> bool:
        return (
            self.gamma == 1.0
            and self.sharpen == 0.0
            and self.clarity == 0.0
            and self.skin_smoothing == 0.0
            and not self.deband
        )


def validate_enhance(
    gamma: float,
    sharpen: float,
    clarity: float,
    skin_smoothing: float,
) -> EnhanceParams:
    params = EnhanceParams(
        gamma=float(gamma),
        sharpen=float(sharpen),
        clarity=float(clarity),
        skin_smoothing=float(skin_smoothing),
    )
    checks = (
        ("gamma", params.gamma, 0.25, 2.5),
        ("sharpen", params.sharpen, 0.0, 2.0),
        ("clarity", params.clarity, -1.0, 1.0),
        ("skin_smoothing", params.skin_smoothing, 0.0, 1.0),
    )
    for name, value, low, high in checks:
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"Enhance {name} must be between {low} and {high}."
            )
    return params


def enhance_report(params: EnhanceParams) -> dict:
    return {
        "gamma": params.gamma,
        "sharpen": params.sharpen,
        "edge_only": params.edge_only,
        "clarity": params.clarity,
        "skin_smoothing": params.skin_smoothing,
        "deband": params.deband,
        "applied": not params.neutral,
        "execution": "cpu" if not params.neutral else "bypass",
    }


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Midtone power adjustment; 1.0 is an exact bypass.

    Follows the image-editing convention: gamma < 1 brightens midtones
    (fixes DLSS dark skin), gamma > 1 darkens them.
    """
    if gamma == 1.0:
        return image
    if not math.isfinite(gamma) or not 0.25 <= gamma <= 2.5:
        raise ValueError("Gamma must be between 0.25 and 2.5.")
    height = image.shape[0]
    result = image.copy()
    lut = np.empty(256, dtype=np.uint8)
    lut[:] = np.rint(
        np.clip((np.arange(256, dtype=np.float32) / 255.0) ** gamma, 0.0, 1.0)
        * 255.0
    ).astype(np.uint8)
    for row in range(0, height, 256):
        end = min(row + 256, height)
        result[row:end, :, :3] = cv2.LUT(image[row:end, :, :3], lut)
    return result


def _sobel_edge_mask(gray: np.ndarray, threshold: float) -> np.ndarray:
    """Soft 0..1 mask that is 1 near strong luminance edges."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # Normalize against theoretical Sobel max (~4*255) and soften.
    normalized = np.clip(magnitude / (threshold * 255.0), 0.0, 1.0)
    return cv2.GaussianBlur(normalized, (0, 0), 1.0)


def apply_sharpen(
    rendered: np.ndarray,
    strength: float,
    edge_only: bool = True,
    edge_threshold: float = 0.10,
) -> np.ndarray:
    """Unsharp mask; edge_only confines it to a Sobel-edge mask."""
    if strength == 0.0:
        return rendered
    if not math.isfinite(strength) or not 0.0 <= strength <= 2.0:
        raise ValueError("Sharpen strength must be between 0 and 2.")
    height, width = rendered.shape[:2]
    result = rendered.copy()
    amount = np.float32(strength * 0.6)
    for row in range(0, height, 64):
        end = min(row + 64, height)
        block = rendered[row:end, :, :3].astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(block, (0, 0), 1.0)
        detail = block - blurred
        if edge_only:
            gray = cv2.cvtColor(
                rendered[row:end, :, :3], cv2.COLOR_RGB2GRAY
            ).astype(np.float32) / 255.0
            mask = _sobel_edge_mask(gray, edge_threshold)[..., None]
            detail = detail * mask
        result[row:end, :, :3] = np.rint(
            np.clip(block + detail * amount, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return result


def apply_clarity(image: np.ndarray, strength: float) -> np.ndarray:
    """Local mid-frequency contrast; negative softens, positive crisps."""
    if strength == 0.0:
        return image
    if not math.isfinite(strength) or not -1.0 <= strength <= 1.0:
        raise ValueError("Clarity must be between -1 and 1.")
    height = image.shape[0]
    result = image.copy()
    amount = np.float32(strength * 0.5)
    # Sigma scales with image width so clarity looks consistent across sizes.
    sigma = max(4.0, image.shape[1] / 240.0)
    for row in range(0, height, 64):
        end = min(row + 64, height)
        block = image[row:end, :, :3].astype(np.float32) / 255.0
        base = cv2.GaussianBlur(block, (0, 0), sigma)
        # Soft-light-ish blend of the high-pass back onto the base.
        low = base + (block - base) * (1.0 + amount)
        result[row:end, :, :3] = np.rint(
            np.clip(low, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return result


def apply_skin_smoothing(image: np.ndarray, strength: float) -> np.ndarray:
    """Bilateral smoothing that flattens grain while keeping edges/pores."""
    if strength == 0.0:
        return image
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("Skin smoothing must be between 0 and 1.")
    height, width = image.shape[:2]
    result = image.copy()
    # Diameter and sigmaColor scale with strength; 9px diameter stays texture-safe.
    diameter = 9
    sigma_color = np.float32(6.0 + 24.0 * strength)
    sigma_space = np.float32(4.0 + 8.0 * strength)
    for row in range(0, height, 64):
        end = min(row + 64, height)
        block = image[row:end, :, :3]
        smoothed = cv2.bilateralFilter(
            block, diameter, sigma_color, sigma_space, borderType=cv2.BORDER_REFLECT
        )
        blended = block.astype(np.float32) * (1.0 - strength * 0.65) + (
            smoothed.astype(np.float32) * strength * 0.65
        )
        result[row:end, :, :3] = np.rint(
            np.clip(blended, 0.0, 255.0)
        ).astype(np.uint8)
    return result


def apply_deband(image: np.ndarray) -> np.ndarray:
    """Break up 8-bit gradient banding with bounded low-amplitude noise."""
    height, width = image.shape[:2]
    result = image.copy()
    # Plain Gaussian blur would smear real detail; a tiny amplitude keeps it
    # sub-visible per frame but enough to dither staircases across gradients.
    noise_amplitude = 1.2
    rng = np.random.default_rng(0xD155)
    for row in range(0, height, 64):
        end = min(row + 64, height)
        block = image[row:end, :, :3].astype(np.float32)
        smoothed = cv2.GaussianBlur(block, (0, 0), 0.6)
        noise = rng.normal(
            0.0, noise_amplitude, smoothed.shape
        ).astype(np.float32)
        result[row:end, :, :3] = np.rint(
            np.clip(smoothed + noise, 0.0, 255.0)
        ).astype(np.uint8)
    return result


def apply_enhance(
    rendered: np.ndarray,
    params: EnhanceParams,
) -> np.ndarray:
    """Apply the full enhance chain in a fixed, artifact-safe order."""
    if params.neutral:
        return rendered
    result = rendered
    # Order matters: deband first (remove staircases), then gamma (tonal),
    # then skin smoothing (texture), then clarity and sharpen (detail last
    # so they re-introduce crispness that smoothing removed).
    if params.deband:
        result = apply_deband(result)
    if params.gamma != 1.0:
        result = apply_gamma(result, params.gamma)
    if params.skin_smoothing > 0.0:
        result = apply_skin_smoothing(result, params.skin_smoothing)
    if params.clarity != 0.0:
        result = apply_clarity(result, params.clarity)
    if params.sharpen > 0.0:
        result = apply_sharpen(
            result, params.sharpen, edge_only=params.edge_only
        )
    return result


class TemporalDeflicker:
    """Video-only global luma stabilization across a frame sequence.

    Tracks an exponential moving average of the frame's mean luminance and
    counteracts single-frame brightness spikes with a small bounded gain.
    The EMA follows gradual scene relighting within a few frames, while the
    gain ceiling keeps the correction from fighting real exposure changes.
    State resets on scene cuts so adjacent shots never bleed into each other.
    """

    _FOLLOW_RATE = 0.25   # EMA absorption speed for gradual light changes
    _MAX_GAIN = 0.06      # ±6% gain ceiling per frame

    def __init__(self, strength: float = 0.0):
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError("Temporal deflicker must be between 0 and 1.")
        self.strength = float(strength)
        self._ema_luma: float | None = None
        self._weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        self._floor = np.float32(1.0 / 512.0)

    def reset(self) -> None:
        self._ema_luma = None

    def apply(self, frame: np.ndarray, reset: bool = False) -> np.ndarray:
        if self.strength == 0.0:
            return frame
        if reset:
            self._ema_luma = None
        # Sparse-column sampling keeps the luma estimate cheap on 4K frames.
        step = max(1, frame.shape[1] // 512)
        sampled = frame[:, ::step, :3].astype(np.float32) / 255.0
        luma = float((sampled @ self._weights).mean())
        if self._ema_luma is None:
            self._ema_luma = luma
            return frame
        self._ema_luma += (luma - self._ema_luma) * self._FOLLOW_RATE
        deviation = (self._ema_luma - luma) / (luma + self._floor)
        bounded = max(-self._MAX_GAIN, min(self._MAX_GAIN, deviation))
        gain = np.float32(1.0 + bounded * self.strength)
        if abs(float(gain) - 1.0) < 1e-4:
            return frame
        height = frame.shape[0]
        result = frame.copy()
        for row in range(0, height, 64):
            end = min(row + 64, height)
            block = frame[row:end, :, :3].astype(np.float32) * gain
            result[row:end, :, :3] = np.rint(
                np.clip(block, 0.0, 255.0)
            ).astype(np.uint8)
        return result


def apply_color_grade(
    original: np.ndarray,
    rendered: np.ndarray,
    params: ColorGradeParams,
) -> np.ndarray:
    """Grade the rendered frame; luma_match uses the source as reference."""
    if params.neutral:
        return rendered
    height, width = rendered.shape[:2]
    # The original may be smaller than the rendered output (upscaling);
    # align it to the render size up front so row slices stay valid.
    if original.shape[:2] != (height, width):
        original = cv2.resize(
            original, (width, height), interpolation=cv2.INTER_LINEAR
        )
    result = rendered.copy()
    weights = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    floor = np.float32(1.0 / 512.0)

    gain = np.float32(2.0 ** params.exposure)
    contrast_k = np.float32(1.0 + 0.5 * params.contrast)
    sat = np.float32(1.0 + params.saturation)

    for row in range(0, height, 64):
        end = min(row + 64, height)
        block = rendered[row:end, :, :3].astype(np.float32) / 255.0
        luma = block @ weights

        # Luma match: bounded exposure correction toward the source median.
        if params.luma_match > 0.0:
            source = original[row:end, :, :3].astype(np.float32) / 255.0
            source_luma = source @ weights
            ratio = (source_luma.mean() + floor) / (luma.mean() + floor)
            bounded = np.clip(ratio, 0.5, 2.0) ** np.float32(params.luma_match)
            block = block * (gain * bounded)
        else:
            block = block * gain

        if params.contrast != 0.0:
            block = (block - 0.5) * contrast_k + 0.5

        if params.saturation != 0.0:
            block = luma[..., None] + (block - luma[..., None]) * sat

        if params.shadows != 0.0:
            # Band spans the full frame width (center-weighted, edges fall to
            # 0); shaped (1, width, 1) so it broadcasts over the row block.
            band = _tilt_weights(width, width)[None, :, None]
            shadow_mask = np.power(1.0 - luma, 2.0)[..., None]
            block = block + (params.shadows * 0.25) * shadow_mask * band

        if params.highlights != 0.0:
            band = _tilt_weights(width, width)[None, :, None]
            highlight_mask = np.power(luma, 2.0)[..., None]
            block = block + (params.highlights * 0.25) * highlight_mask * band

        result[row:end, :, :3] = np.rint(
            np.clip(block, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
    return result
