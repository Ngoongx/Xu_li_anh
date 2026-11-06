"""HSV color object detector for still images and webcam frames.

Input and output images use OpenCV's BGR uint8 convention.  This module has no
GUI dependency; see README.md for the interface intended for integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ColorRange:
    """One or more inclusive HSV ranges (OpenCV H=0..179, S/V=0..255)."""

    name: str
    ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]
    bgr: tuple[int, int, int]


# Initial values from the report. Red straddles the ends of OpenCV's Hue scale.
DEFAULT_COLORS: dict[str, ColorRange] = {
    "red": ColorRange("Đỏ", (((0, 100, 80), (10, 255, 255)), ((170, 100, 80), (179, 255, 255))), (0, 0, 255)),
    "orange": ColorRange("Cam", (((11, 100, 80), (22, 255, 255)),), (0, 140, 255)),
    "yellow": ColorRange("Vàng", (((23, 100, 80), (34, 255, 255)),), (0, 220, 255)),
    "green": ColorRange("Xanh lá", (((35, 100, 80), (85, 255, 255)),), (0, 200, 0)),
    "blue": ColorRange("Xanh dương", (((86, 100, 80), (125, 255, 255)),), (255, 100, 0)),
    "purple": ColorRange("Tím", (((126, 100, 80), (169, 255, 255)),), (200, 0, 180)),
}


@dataclass(frozen=True)
class DetectorConfig:
    min_area: float = 500.0
    gaussian_kernel: int = 5
    morphology_kernel: int = 5
    open_iterations: int = 1
    close_iterations: int = 1
    enabled_colors: tuple[str, ...] | None = None
    # Optional per-color override. Values are ColorRange instances.
    colors: Mapping[str, ColorRange] | None = None


def _odd_kernel(value: int, label: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{label} phải lớn hơn hoặc bằng 1")
    return value if value % 2 else value + 1


def _validate_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray dạng BGR")
    if image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh phải có kích thước H×W×3 và không được rỗng")
    if image.dtype != np.uint8:
        raise TypeError("Ảnh BGR phải có kiểu uint8")
    return image


def load_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR; raise a useful error for a missing/invalid file."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return image


def open_camera(index: int = 0, width: int | None = 640, height: int | None = 480) -> cv2.VideoCapture:
    """Open a webcam or raise RuntimeError. Caller must release() it."""
    cap = cv2.VideoCapture(index)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không mở được webcam số {index}")
    return cap


def create_mask(hsv: np.ndarray, color: ColorRange) -> np.ndarray:
    """Create a combined binary mask for a configured color."""
    if hsv.ndim != 3 or hsv.shape[2] != 3 or hsv.dtype != np.uint8:
        raise ValueError("HSV phải là mảng uint8 có dạng H×W×3")
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color.ranges:
        lo = np.asarray(lower, dtype=np.uint8)
        hi = np.asarray(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def _clean_mask(mask: np.ndarray, kernel_size: int, open_iterations: int, close_iterations: int) -> np.ndarray:
    size = _odd_kernel(kernel_size, "morphology_kernel")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    clean = mask
    if open_iterations > 0:
        clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=int(open_iterations))
    if close_iterations > 0:
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=int(close_iterations))
    return clean


def process_frame(
    frame_bgr: np.ndarray,
    config: DetectorConfig | None = None,
    *,
    return_masks: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect configured colors in one BGR frame.

    Returns ``(annotated_bgr, stats)``. ``stats`` contains per-color counts,
    total, and a list of object records with color, bbox, center, and area.
    Set ``return_masks=True`` to also receive cleaned per-color masks in stats.
    """
    frame = _validate_bgr(frame_bgr)
    cfg = config or DetectorConfig()
    if cfg.min_area < 0:
        raise ValueError("min_area не может âm")
    blur_k = _odd_kernel(cfg.gaussian_kernel, "gaussian_kernel")
    colors = cfg.colors or DEFAULT_COLORS
    selected: Iterable[str] = cfg.enabled_colors if cfg.enabled_colors is not None else colors.keys()
    unknown = set(selected) - set(colors)
    if unknown:
        raise ValueError(f"Màu chưa cấu hình: {', '.join(sorted(unknown))}")

    blurred = cv2.GaussianBlur(frame, (blur_k, blur_k), 0) if blur_k > 1 else frame
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    annotated = frame.copy()
    counts = {key: 0 for key in selected}
    objects: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}

    for key in selected:
        color = colors[key]
        mask = _clean_mask(create_mask(hsv, color), cfg.morphology_kernel,
                           cfg.open_iterations, cfg.close_iterations)
        if return_masks:
            masks[key] = mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < cfg.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"]:
                cx = int(round(moments["m10"] / moments["m00"]))
                cy = int(round(moments["m01"] / moments["m00"]))
            else:
                cx, cy = x + w // 2, y + h // 2
            counts[key] += 1
            record = {"index": len(objects) + 1, "color": key, "label": color.name,
                      "bbox": (int(x), int(y), int(w), int(h)),
                      "center": (cx, cy), "area": round(area, 2)}
            objects.append(record)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color.bgr, 2)
            cv2.circle(annotated, (cx, cy), 4, color.bgr, -1)
            cv2.putText(annotated, f"{color.name} #{counts[key]}", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, color.bgr, 2, cv2.LINE_AA)

    summary = "  ".join(f"{colors[k].name}: {v}" for k, v in counts.items())
    cv2.putText(annotated, f"Tong: {len(objects)}  {summary}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2, cv2.LINE_AA)

    stats: dict[str, Any] = {"counts": counts, "total": len(objects), "objects": objects}
    if return_masks:
        stats["masks"] = masks
    return annotated, stats


def process_image_path(path: str | Path, config: DetectorConfig | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Convenience API for Người 3: pass a path, receive output image + stats."""
    return process_frame(load_image(path), config)


def save_result(path: str | Path, image_bgr: np.ndarray) -> None:
    """Save an output image and report write errors."""
    _validate_bgr(image_bgr)
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Không lưu được ảnh kết quả: {path}")


def open_config(path: str | Path) -> DetectorConfig:
    """Load HSV thresholds and processing parameters from a JSON config."""
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    colors: dict[str, ColorRange] = {}
    for key, spec in data.get("colors", {}).items():
        ranges = spec.get("ranges", [spec.get("range")])
        ranges = [r for r in ranges if r is not None]
        colors[key] = ColorRange(
            spec["name"], tuple((tuple(r[0]), tuple(r[1])) for r in ranges),
            tuple(spec.get("bgr", [0, 255, 0])))
    defaults = data.get("processing", {})
    return DetectorConfig(
        min_area=float(defaults.get("min_area", 500)),
        gaussian_kernel=int(defaults.get("gaussian_kernel", 5)),
        morphology_kernel=int(defaults.get("morphology_kernel", 5)),
        open_iterations=int(defaults.get("open_iterations", 1)),
        close_iterations=int(defaults.get("close_iterations", 1)),
        enabled_colors=tuple(data.get("enabled_colors", colors.keys())), colors=colors)


def stats_to_rows(stats: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return object records ready for a GUI table or CSV DictWriter."""
    return [dict(obj) for obj in stats.get("objects", [])]


__all__ = ["ColorRange", "DetectorConfig", "DEFAULT_COLORS", "load_image",
           "open_camera", "create_mask", "process_frame", "process_image_path",
           "save_result", "open_config", "stats_to_rows"]
