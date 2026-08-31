"""
backend/models/detector.py
===========================
Real-time object detection with uncertainty-penalised confidence scores.

Pipeline
--------
1. Accept a reconstructed RGB numpy array  (H, W, 3)  uint8 in [0, 255]
   and a 2-D aleatoric uncertainty heatmap (H, W)     float32 in [0, 1]
   produced by ``run_mc_inference`` in uncertainty.py.

2. Run YOLOv8 inference (ultralytics) on the RGB array to obtain real
   bounding boxes (x1, y1, x2, y2), class labels, and raw confidence scores.

3. For every detected box crop the corresponding spatial slice from the
   uncertainty heatmap, compute its mean uncertainty μ_unc, then apply:

       Adjusted Confidence = Raw Confidence × (1.0 − μ_unc)

4. Return a list of ``DetectionResult`` dataclass instances — one per box —
   each carrying the original and adjusted metrics plus the mean uncertainty.

Edge cases handled
------------------
- Zero detections          → returns empty list immediately.
- Zero-area bounding box   → μ_unc set to 0.0 (no uncertainty penalty).
- Box coordinates outside  → clamped to [0, H/W] before slicing.
  image boundaries
- GPU tensors              → YOLO results are moved to CPU; the uncertainty
                             map is accepted as a plain numpy array (CPU).
- Confidence ≤ 0           → adjusted score is also ≤ 0 (floor at 0.0).
- ultralytics not installed → ImportError raised at import time with a
                             helpful install hint.

Dependencies
------------
    pip install ultralytics torch numpy

YOLOv8 model weights are downloaded automatically on first use by the
ultralytics library (e.g. yolov8n.pt → ~/.config/Ultralytics/).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
    _YOLO_IMPORT_EXC: BaseException = RuntimeError("ok")
except BaseException as exc:  # pragma: no cover
    # BaseException required: ultralytics triggers a KeyboardInterrupt (not
    # ImportError) on Python 3.13/3.14 due to pandas Cython breakage.
    _YOLO_AVAILABLE = False
    _YOLO_IMPORT_EXC = exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Structured output for a single detected object.

    Attributes
    ----------
    class_id : int
        YOLO class index.
    class_name : str
        Human-readable class label (e.g. ``"car"``, ``"building"``).
    bbox : tuple[int, int, int, int]
        Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates,
        clamped to the image boundary.
    raw_confidence : float
        Raw detection confidence in ``[0, 1]`` from YOLOv8.
    mean_uncertainty : float
        Mean aleatoric uncertainty in ``[0, 1]`` over the bounding-box region
        of the 2-D uncertainty heatmap.
    adjusted_confidence : float
        ``raw_confidence × (1.0 − mean_uncertainty)``, floored at ``0.0``.
    """

    class_id: int
    class_name: str
    bbox: tuple          # (x1, y1, x2, y2) ints
    raw_confidence: float
    mean_uncertainty: float
    adjusted_confidence: float

    def to_dict(self) -> Dict:
        """Serialise to a plain JSON-compatible dict."""
        x1, y1, x2, y2 = self.bbox
        return {
            "class_id":           self.class_id,
            "class_name":         self.class_name,
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "raw_confidence":     round(self.raw_confidence,     6),
            "mean_uncertainty":   round(self.mean_uncertainty,   6),
            "adjusted_confidence": round(self.adjusted_confidence, 6),
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class UncertaintyAwareDetector:
    """YOLOv8-based spatial detector with MC-Dropout uncertainty penalisation.

    Parameters
    ----------
    model_path:
        Path to a YOLOv8 weights file or a model name resolvable by
        ultralytics (e.g. ``"yolov8n.pt"``, ``"yolov8s.pt"``).
        Weights are downloaded automatically on first use.
    device:
        PyTorch device string: ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
        Defaults to ``"cpu"`` for portability; pass ``"cuda"`` when a
        GPU is available.
    conf_threshold:
        Minimum raw YOLO confidence to keep a detection.  Default ``0.25``.
    iou_threshold:
        IoU threshold for non-maximum suppression.  Default ``0.45``.

    Example
    -------
    >>> detector = UncertaintyAwareDetector("yolov8n.pt", device="cpu")
    >>> results = detector.detect(rgb_array, uncertainty_map)
    >>> for r in results:
    ...     print(r.to_dict())
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "yolov8n.pt",
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics (YOLOv8) could not be imported. "
                f"Original error: {_YOLO_IMPORT_EXC}. "
                "Install with: pip install ultralytics"
            )
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        logger.info("Loading YOLO model from %s on device=%s", model_path, device)
        self._model = YOLO(str(model_path))
        # Move model to requested device; ultralytics handles CPU/CUDA internally
        self._model.to(device)

    # ------------------------------------------------------------------
    def detect(
        self,
        rgb_image: np.ndarray,
        uncertainty_map: np.ndarray,
        verbose: bool = False,
    ) -> List[DetectionResult]:
        """Run detection and return uncertainty-penalised results.

        Parameters
        ----------
        rgb_image:
            Reconstructed RGB array, shape ``(H, W, 3)``, dtype ``uint8``
            with values in ``[0, 255]``.  This is the direct output of
            converting ``MCInferenceResult.mean_rgb`` via
            :func:`_tensor_to_uint8_rgb` (see helper below).

        uncertainty_map:
            2-D aleatoric uncertainty heatmap, shape ``(H, W)``,
            dtype ``float32``, values in ``[0, 1]``.  Typically
            ``MCInferenceResult.uncertainty_map[0, 0].numpy()``.

        verbose:
            If ``True``, pass ``verbose=True`` to YOLO inference (prints
            per-frame timing).  Default ``False``.

        Returns
        -------
        list[DetectionResult]
            One entry per retained detection, sorted by adjusted confidence
            descending.  Empty list when no objects are detected.

        Raises
        ------
        ValueError
            If ``rgb_image`` is not (H, W, 3) uint8, or ``uncertainty_map``
            is not (H, W) float32/float64, or their spatial dimensions differ.
        """
        _validate_inputs(rgb_image, uncertainty_map)

        H, W = rgb_image.shape[:2]

        # ── 1. YOLOv8 inference ───────────────────────────────────────────────
        # ultralytics accepts numpy uint8 HWC arrays directly.
        yolo_results = self._model.predict(
            source=rgb_image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=verbose,
        )

        # yolo_results is a list with one Results object per image.
        # We always pass a single frame so index [0].
        frame_result = yolo_results[0]

        boxes_tensor = frame_result.boxes  # ultralytics Boxes object
        if boxes_tensor is None or len(boxes_tensor) == 0:
            logger.debug("No detections in frame.")
            return []

        # ── 2. Extract raw data from YOLO result (move to CPU / numpy) ────────
        # .xyxy  → (N, 4) float32 pixel coords [x1, y1, x2, y2]
        # .conf  → (N,)   float32 confidence
        # .cls   → (N,)   float32 class id
        xyxy  = boxes_tensor.xyxy.cpu().numpy().astype(np.float32)   # (N, 4)
        confs = boxes_tensor.conf.cpu().numpy().astype(np.float32)   # (N,)
        cls   = boxes_tensor.cls.cpu().numpy().astype(np.int32)      # (N,)
        names: Dict[int, str] = frame_result.names                   # {id: label}

        # ── 3. Per-box uncertainty extraction and score adjustment ────────────
        detections: List[DetectionResult] = []

        for i in range(len(xyxy)):
            raw_conf  = float(confs[i])
            class_id  = int(cls[i])
            class_name = names.get(class_id, f"class_{class_id}")

            # Clamp box coordinates to valid image bounds
            x1 = int(np.clip(np.floor(xyxy[i, 0]), 0, W - 1))
            y1 = int(np.clip(np.floor(xyxy[i, 1]), 0, H - 1))
            x2 = int(np.clip(np.ceil( xyxy[i, 2]), 0, W))
            y2 = int(np.clip(np.ceil( xyxy[i, 3]), 0, H))

            # ── μ_unc: mean uncertainty over the bounding-box region ──────────
            mean_unc = _region_mean_uncertainty(uncertainty_map, x1, y1, x2, y2)

            # ── Adjusted confidence ───────────────────────────────────────────
            adj_conf = float(np.clip(raw_conf * (1.0 - mean_unc), 0.0, 1.0))

            detections.append(DetectionResult(
                class_id=class_id,
                class_name=class_name,
                bbox=(x1, y1, x2, y2),
                raw_confidence=raw_conf,
                mean_uncertainty=float(mean_unc),
                adjusted_confidence=adj_conf,
            ))

        # Sort by adjusted confidence descending — most reliable detections first
        detections.sort(key=lambda d: d.adjusted_confidence, reverse=True)

        logger.debug(
            "detect(): %d raw boxes → %d results (conf≥%.2f)",
            len(xyxy), len(detections), self.conf_threshold,
        )
        return detections


# ---------------------------------------------------------------------------
# Public helper — convert MCInferenceResult.mean_rgb to uint8 HWC array
# ---------------------------------------------------------------------------

def tensor_to_uint8_rgb(mean_rgb_tensor) -> np.ndarray:
    """Convert ``MCInferenceResult.mean_rgb`` (single batch item) to uint8 HWC.

    Parameters
    ----------
    mean_rgb_tensor:
        A ``torch.Tensor`` of shape ``(3, H, W)`` or ``(1, 3, H, W)`` with
        values in ``[−1, 1]`` (Tanh output of UNetGenerator).

    Returns
    -------
    numpy.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``, values in ``[0, 255]``.
        Ready to pass directly to :meth:`UncertaintyAwareDetector.detect`.
    """
    import torch  # local import — torch is optional at module-import time

    t = mean_rgb_tensor
    if t.ndim == 4:
        if t.shape[0] != 1:
            raise ValueError(
                f"Batch size must be 1 when converting to uint8. Got shape {t.shape}."
            )
        t = t[0]  # (3, H, W)
    if t.ndim != 3 or t.shape[0] != 3:
        raise ValueError(
            f"Expected (3, H, W) or (1, 3, H, W) tensor. Got shape {mean_rgb_tensor.shape}."
        )
    # [-1, 1] → [0, 255]
    arr = t.detach().cpu().float()
    arr = ((arr + 1.0) * 127.5).clamp(0, 255).byte()
    return arr.permute(1, 2, 0).numpy()  # (H, W, 3) uint8


def uncertainty_map_to_numpy(uncertainty_map_tensor) -> np.ndarray:
    """Extract a single-item 2-D uncertainty heatmap from an ``MCInferenceResult``.

    Parameters
    ----------
    uncertainty_map_tensor:
        ``MCInferenceResult.uncertainty_map`` — shape ``(B, 1, H, W)`` or
        ``(1, H, W)`` or ``(H, W)``, float32 in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Shape ``(H, W)``, dtype ``float32``.
    """
    import torch  # local import

    t = uncertainty_map_tensor
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]   # (1, H, W)
        if t.ndim == 3:
            t = t[0]   # (H, W)
        return t.numpy()
    # Already numpy
    arr = np.asarray(t, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    return arr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _region_mean_uncertainty(
    uncertainty_map: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> float:
    """Return the mean uncertainty value inside a bounding-box slice.

    Returns ``0.0`` for zero-area boxes or empty slices (no penalty applied).
    """
    if x2 <= x1 or y2 <= y1:
        # Zero-area box — no spatial extent to measure uncertainty over
        return 0.0

    region = uncertainty_map[y1:y2, x1:x2]

    if region.size == 0:
        return 0.0

    return float(np.mean(region))


def _validate_inputs(
    rgb_image: np.ndarray,
    uncertainty_map: np.ndarray,
) -> None:
    """Raise ValueError for incompatible or malformed inputs."""
    if not isinstance(rgb_image, np.ndarray):
        raise ValueError(
            f"rgb_image must be a numpy.ndarray. Got {type(rgb_image)}."
        )
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(
            f"rgb_image must be shape (H, W, 3). Got {rgb_image.shape}."
        )
    if rgb_image.dtype != np.uint8:
        raise ValueError(
            f"rgb_image must be dtype uint8. Got {rgb_image.dtype}. "
            "Use tensor_to_uint8_rgb() to convert MCInferenceResult.mean_rgb."
        )
    if not isinstance(uncertainty_map, np.ndarray):
        raise ValueError(
            f"uncertainty_map must be a numpy.ndarray. Got {type(uncertainty_map)}."
        )
    if uncertainty_map.ndim != 2:
        raise ValueError(
            f"uncertainty_map must be 2-D (H, W). Got ndim={uncertainty_map.ndim}. "
            "Use uncertainty_map_to_numpy() to convert MCInferenceResult.uncertainty_map."
        )
    img_h, img_w = rgb_image.shape[:2]
    map_h, map_w = uncertainty_map.shape
    if img_h != map_h or img_w != map_w:
        raise ValueError(
            f"Spatial dimensions of rgb_image ({img_h}×{img_w}) and "
            f"uncertainty_map ({map_h}×{map_w}) must match."
        )
