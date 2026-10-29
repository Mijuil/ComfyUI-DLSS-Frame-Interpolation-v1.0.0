"""Source analysis for automatic DLSS post-processing settings.

Computes cheap image statistics (sharpness, noise, exposure clipping,
contrast, compression blocking) from a single IMAGE frame or a uniform
sample of VIDEO frames, then maps them onto the "Quality & cleanup" and
"Color & light" widget values. Pure numpy/cv2/PyAV — no GPU work.
"""
from __future__ import annotations

import av
import cv2
import numpy as np

# Video frames sampled evenly across the file for statistics.
SAMPLE_FRAMES = 16

# Widgets the analyzer is allowed to suggest, with clamping ranges.
_WIDGET_RANGES = {
    "enable_enhance": (0, 1),
    "enable_deband": (0, 1),
    "gamma": (0.25, 2.5),
    "skin_smoothing": (0.0, 1.0),
    "clarity": (-1.0, 1.0),
    "sharpen": (0.0, 2.0),
    "enable_color": (0, 1),
    "luma_match": (0.0, 1.0),
    "exposure": (-1.0, 1.0),
    "contrast": (-1.0, 1.0),
    "saturation": (-1.0, 1.0),
    "shadows": (-1.0, 1.0),
    "highlights": (-1.0, 1.0),
}


def _clamp(name: str, value: float) -> float:
    low, high = _WIDGET_RANGES[name]
    return float(np.clip(value, low, high))


def _metrics_from_luma(luma: np.ndarray) -> dict:
    """Luma-plane metrics shared by image and video paths.

    luma: float32 in [0, 1], shape [h, w].
    """
    h, w = luma.shape
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        luma = cv2.resize(luma, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    hist = np.bincount(np.clip((luma * 255.0).astype(np.int32), 0, 255).ravel(), minlength=256)
    total = float(hist.sum())
    dark_clip = float(hist[:2].sum()) / total
    bright_clip = float(hist[254:].sum()) / total
    mean_luma = float(luma.mean())
    p5, p95 = (np.percentile(luma, q) for q in (5, 95))
    contrast = float(p95 - p5)
    # Laplacian variance on a luma-scale image: sharpness proxy.
    sharpness = float(cv2.Laplacian(luma, cv2.CV_32F).var())
    # Median absolute Laplacian response: noise / high-frequency energy.
    lap = cv2.Laplacian(luma, cv2.CV_32F)
    noise = float(np.median(np.abs(lap - np.median(lap))))
    # Blocking: horizontal gradient energy on 8px boundaries vs elsewhere.
    gx = np.abs(np.diff(luma, axis=1)).mean(axis=0)
    boundaries = gx[7::8]
    interior = np.delete(gx, np.s_[7::8])
    blocking = float(boundaries.mean() - interior.mean()) if interior.size else 0.0
    return {
        "mean_luma": mean_luma,
        "contrast_p5_p95": contrast,
        "sharpness_var": sharpness,
        "noise_mad": noise,
        "dark_clip": dark_clip,
        "bright_clip": bright_clip,
        "blocking": blocking,
    }


def _chroma_noise(image: np.ndarray) -> float:
    """Median-abs-Laplacian noise on the chroma planes (color grain).

    Luma-only noise metrics under-measure heavily colored sensor/compression
    grain, which is exactly the grain that survives upscaling.
    """
    ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
    h, w = ycc.shape[:2]
    scale = 512.0 / max(h, w)
    if scale < 1.0:
        ycc = cv2.resize(ycc, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    total = 0.0
    for plane in (ycc[..., 1], ycc[..., 2]):
        lap = cv2.Laplacian(plane, cv2.CV_32F)
        total += float(np.median(np.abs(lap - np.median(lap))))
    return total / 2.0


def analyze_image(image: np.ndarray) -> tuple[dict, dict]:
    """Analyze one RGB uint8 frame. Returns (metrics, suggestions)."""
    luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    metrics = _metrics_from_luma(luma)
    metrics["chroma_noise"] = _chroma_noise(image)
    return metrics, _suggest(metrics)


def analyze_video_frames(frames: list[np.ndarray]) -> tuple[dict, dict]:
    """Analyze uniformly sampled RGB uint8 frames. Returns (metrics, suggestions)."""
    luma_frames = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        for frame in frames
    ]
    stacked = {
        key: float(np.mean([_metrics_from_luma(l)[key] for l in luma_frames]))
        for key in ("mean_luma", "contrast_p5_p95", "sharpness_var", "noise_mad", "dark_clip", "bright_clip", "blocking")
    }
    stacked["chroma_noise"] = float(np.mean([_chroma_noise(frame) for frame in frames]))
    # Flicker: mean absolute luma delta between consecutive samples.
    deltas = [
        float(np.abs(b.mean() - a.mean()))
        for a, b in zip(luma_frames, luma_frames[1:])
    ]
    stacked["flicker"] = float(np.mean(deltas)) if deltas else 0.0
    return stacked, _suggest(stacked)


def sample_video_frames(path: str, count: int = SAMPLE_FRAMES) -> list[np.ndarray]:
    """Decode up to `count` frames spread evenly across the video."""
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        total = stream.frames or 0
        frames: list[np.ndarray] = []
        if total and total > count:
            step = total / count
            wanted = {min(int(i * step), total - 1) for i in range(count)}
            index = -1
            for frame in container.decode(stream):
                index += 1
                if index in wanted:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    if len(frames) == count:
                        break
        else:
            for frame in container.decode(stream):
                frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= count:
                    break
        if not frames:
            raise RuntimeError("The input video contains no decodable frames to analyze.")
        return frames
    finally:
        container.close()


def _suggest(m: dict) -> dict:
    """Map metrics onto widget values. Keeps a reasoning note per widget."""
    notes: dict[str, str] = {}
    suggestions: dict[str, float] = {}

    def set_widget(name: str, value: float, note: str) -> None:
        suggestions[name] = _clamp(name, value)
        notes[name] = note

    noise = m["noise_mad"]
    chroma = m.get("chroma_noise", 0.0)
    sharpness = m["sharpness_var"]
    mean_luma = m["mean_luma"]
    contrast = m["contrast_p5_p95"]
    bright_clip = m["bright_clip"]
    dark_clip = m["dark_clip"]
    blocking = m["blocking"]

    # --- Quality & cleanup -------------------------------------------------
    # Noise: median abs Laplacian above ~0.012 indicates visible grain.
    # Chroma grain counts at half weight - colored grain survives upscaling
    # and reads as "dirty" skin even when luma looks fine.
    total_noise = noise + 0.5 * chroma
    if total_noise > 0.012:
        set_widget("skin_smoothing", min(0.5, 0.2 + (total_noise - 0.012) * 8.0),
                   f"noise {noise:.3f} + chroma {chroma:.3f} above 0.012")
    # Sharpness: low Laplacian variance means soft source.
    if sharpness < 0.0015:
        set_widget("clarity", min(0.4, 0.2 + (0.0015 - sharpness) * 100.0),
                   f"sharpness {sharpness:.4f} below 0.0015")
        set_widget("sharpen", 0.3, "soft source")
    # Overbright: clipped highlights or very bright mean -> darken via gamma.
    if bright_clip > 0.01 or mean_luma > 0.62:
        set_widget("gamma", 1.0 + min(0.5, max(bright_clip * 3.0, (mean_luma - 0.62) * 1.2)),
                   f"bright clip {bright_clip:.3f}, mean luma {mean_luma:.2f}")
        set_widget("highlights", -0.3, "recovery for clipped highlights")
    # Underexposed: crushed blacks or dark mean -> lift exposure & shadows.
    if dark_clip > 0.02 or mean_luma < 0.30:
        set_widget("exposure", min(0.5, (0.30 - mean_luma) * 1.8),
                   f"mean luma {mean_luma:.2f} below 0.30")
        set_widget("shadows", 0.2, "lift for crushed shadows")
    # Flat / hazy: narrow p5-p95 span -> add contrast.
    if contrast < 0.55:
        set_widget("contrast", 0.15, f"contrast span {contrast:.2f} below 0.55")
    # Compression blocking -> deband.
    if blocking > 0.004:
        set_widget("enable_deband", 1, f"8px blocking {blocking:.4f}")

    # --- Section toggles ----------------------------------------------------
    enhance_keys = ("gamma", "skin_smoothing", "clarity", "sharpen")
    color_keys = ("exposure", "contrast", "shadows", "highlights")
    if any(k in suggestions for k in enhance_keys):
        set_widget("enable_enhance", 1, "cleanup suggestions present")
    if any(k in suggestions for k in color_keys):
        set_widget("enable_color", 1, "color suggestions present")

    return {"values": suggestions, "reasons": notes, "metrics": m}
