"""
backend/app.py
===============
Production-grade FastAPI microservice — Chitra.ai thermal analysis pipeline.

Orchestration order (per request)
----------------------------------
  1. Parse & validate uploaded file (GeoTIFF / PNG / JPEG)
  2. Decode bytes → single-channel numpy array (H × W)
  3. normalize_thermal()      → float32 tensor  (1, 1, H, W) in [0, 1]
  4. run_mc_inference()       → mean_rgb (1,3,H,W), uncertainty_map (1,1,H,W)
  5. tensor_to_uint8_rgb()    → (H, W, 3) uint8 for YOLO
  6. uncertainty_map_to_numpy()→ (H, W) float32
  7. detector.detect()        → list[DetectionResult]
  8. build_agent().generate_intelligence_briefing() → BriefingResult
  9. Encode RGB + heatmap + bbox-overlay → Base64 PNG strings
 10. Return structured JSON response

Startup lifespan
-----------------
UNetGenerator and UncertaintyAwareDetector are constructed once on startup
and stored in ``app.state``.  They are reused across all requests — no
per-request model loading.

Error handling
--------------
  • Corrupt / unreadable image  → HTTP 422 with detail
  • Image too small for U-Net   → HTTP 422 (H or W < 16)
  • Multi-channel image with no
    single-band extraction path  → HTTP 422
  • CUDA OOM                    → HTTP 503, CUDA cache cleared
  • Any other pipeline error    → HTTP 500, full traceback logged

Environment variables
---------------------
  WATSONX_APIKEY        — IBM watsonx.ai API key (optional; fallback used if absent)
  WATSONX_PROJECT_ID    — IBM watsonx.ai project ID (optional)
  WATSONX_URL           — watsonx.ai endpoint (optional; has default)
  YOLO_MODEL            — YOLOv8 weights name/path (default: yolov8n.pt)
  UNET_WEIGHTS          — Path to trained UNet .pt checkpoint (optional;
                          uses random-init weights when absent so the
                          service still starts for integration testing)
  PORT                  — uvicorn port (default: 8000, used by __main__ only)
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import os
import traceback
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from backend.models.detector import (
    UncertaintyAwareDetector,
    tensor_to_uint8_rgb,
    uncertainty_map_to_numpy,
)
from backend.models.generator import UNetGenerator
from backend.models.uncertainty import run_mc_inference
from backend.rag.agent import build_agent
from backend.utils.preprocessing import normalize_thermal

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

_YOLO_MODEL    = os.getenv("YOLO_MODEL",   "yolov8n.pt")
_UNET_WEIGHTS  = os.getenv("UNET_WEIGHTS", "")        # empty → random init
_PORT          = int(os.getenv("PORT", "8000"))

# Spatial minimum required by the U-Net (4 stride-2 downsamples)
_MIN_SPATIAL   = 16


# ---------------------------------------------------------------------------
# Device selection (module-level, resolved once)
# ---------------------------------------------------------------------------

def _select_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA available — using GPU.")
        return "cuda"
    logger.info("CUDA not available — using CPU.")
    return "cpu"


_DEVICE = _select_device()


# ---------------------------------------------------------------------------
# Startup / shutdown lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models into app.state once; clean up on shutdown."""
    logger.info("=== Chitra.ai startup: loading models on %s ===", _DEVICE)

    # ── UNetGenerator ─────────────────────────────────────────────────────────
    generator = UNetGenerator(base_ch=16, p_drop=0.2)
    if _UNET_WEIGHTS and os.path.isfile(_UNET_WEIGHTS):
        logger.info("Loading UNet weights from %s", _UNET_WEIGHTS)
        state = torch.load(_UNET_WEIGHTS, map_location=_DEVICE)
        generator.load_state_dict(state)
    else:
        logger.warning(
            "UNET_WEIGHTS not set or file not found — using random-init weights. "
            "Set UNET_WEIGHTS=/path/to/checkpoint.pt for production use."
        )
    generator = generator.to(_DEVICE)
    generator.eval()   # BN uses running stats; MC-Dropout stays ON unconditionally
    app.state.generator = generator

    # ── YOLOv8 detector ───────────────────────────────────────────────────────
    detector = UncertaintyAwareDetector(
        model_path=_YOLO_MODEL,
        device=_DEVICE,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )
    app.state.detector = detector

    # ── RAG / Granite agent ───────────────────────────────────────────────────
# Disabled on Render free tier to avoid 512 MB startup OOM.
app.state.agent = None

    logger.info("=== Startup complete — service ready ===")
    yield

    # ── Shutdown cleanup ──────────────────────────────────────────────────────
    logger.info("Shutting down — releasing model memory.")
    del app.state.generator
    del app.state.detector
    del app.state.agent
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chitra.ai Thermal Intelligence API",
    description=(
        "Real-time thermal-to-RGB reconstruction, MC-Dropout uncertainty "
        "estimation, YOLOv8 object detection, and IBM Granite RAG briefing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok", "device": _DEVICE}


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/analyze-thermal",
    summary="Full thermal analysis pipeline",
    tags=["analysis"],
)
async def analyze_thermal(
    file:       UploadFile = File(..., description="Single-channel thermal image (GeoTIFF, PNG, JPEG)"),
    mc_passes: int = Form(default=2, ge=2, le=10,  description="Monte Carlo forward passes"),
    latitude:   Optional[float] = Form(default=None, description="Scene centre latitude (WGS-84)"),
    longitude:  Optional[float] = Form(default=None, description="Scene centre longitude (WGS-84)"),
) -> JSONResponse:
    """
    Process a single-channel thermal image through the full Chitra.ai pipeline:

    1. Decode uploaded bytes → normalised tensor
    2. Run MC-Dropout inference → mean RGB + uncertainty map
    3. YOLOv8 detection → uncertainty-penalised bounding boxes
    4. IBM Granite RAG briefing
    5. Return Base64-encoded images + structured JSON
    """

    # ── Retrieve cached models from app state ─────────────────────────────────
    generator: UNetGenerator               = app.state.generator
    detector:  UncertaintyAwareDetector    = app.state.detector
    agent                                  = app.state.agent

    raw_bytes: bytes = await file.read()

    try:
        # ══════════════════════════════════════════════════════════════════════
        # STEP 1 — Decode uploaded file → single-channel numpy array (H, W)
        # ══════════════════════════════════════════════════════════════════════
        thermal_arr = _decode_thermal_image(raw_bytes, file.filename or "")

        H, W = thermal_arr.shape
        if H < _MIN_SPATIAL or W < _MIN_SPATIAL:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Image too small: ({H}, {W}). "
                    f"Minimum spatial size is {_MIN_SPATIAL}×{_MIN_SPATIAL} "
                    "pixels (required by the U-Net architecture)."
                ),
            )

        # Pad to nearest multiple of 16 so stride-2 downs are exact
        thermal_arr, (pad_h, pad_w) = _pad_to_multiple(thermal_arr, multiple=16)
        H_pad, W_pad = thermal_arr.shape

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2 — Normalize → tensor (1, 1, H, W) on target device
        # ══════════════════════════════════════════════════════════════════════
        input_tensor = normalize_thermal(thermal_arr)           # (1, H, W)
        input_tensor = input_tensor.unsqueeze(0).to(_DEVICE)    # (1, 1, H, W)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3 — MC-Dropout inference
        # ══════════════════════════════════════════════════════════════════════
        mc_result = run_mc_inference(generator, input_tensor, passes=mc_passes)

        # Strip padding from tensors before downstream processing
        mean_rgb_t  = mc_result.mean_rgb[   :, :, :H_pad - pad_h or H_pad, :W_pad - pad_w or W_pad]
        unc_map_t   = mc_result.uncertainty_map[:, :, :H_pad - pad_h or H_pad, :W_pad - pad_w or W_pad]

        # Scalar uncertainty metrics (over the unpadded region)
        unc_np_full  = uncertainty_map_to_numpy(unc_map_t)        # (H, W) float32
        mean_unc     = float(np.mean(unc_np_full))
        max_unc      = float(np.max(unc_np_full))

        # ══════════════════════════════════════════════════════════════════════
        # STEP 4 — Convert tensors to numpy for YOLO
        # ══════════════════════════════════════════════════════════════════════
        rgb_uint8  = tensor_to_uint8_rgb(mean_rgb_t)    # (H, W, 3) uint8
        unc_np     = unc_np_full                        # (H, W) float32 [0,1]

        # ══════════════════════════════════════════════════════════════════════
        # STEP 5 — YOLOv8 detection + uncertainty penalisation
        # ══════════════════════════════════════════════════════════════════════
        detections = detector.detect(rgb_uint8, unc_np, verbose=False)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 6 — IBM Granite RAG briefing
        # ══════════════════════════════════════════════════════════════════════
        coords_str: Optional[str] = None
        if latitude is not None and longitude is not None:
            coords_str = f"{latitude:.5f}°N {longitude:.5f}°E"

        analysis_metrics = {
            "resolution_h":     int(H),          # original (unpadded) dimensions
            "resolution_w":     int(W),
            "mean_uncertainty": mean_unc,
            "max_uncertainty":  max_unc,
            "mc_passes":        mc_passes,
        }

        if agent is not None:
    briefing_result = agent.generate_intelligence_briefing(
        analysis_metrics=analysis_metrics,
        detections=detections,
        coordinates=coords_str,
    )

    briefing_text = briefing_result.briefing_text
    agent_meta = {
        "model_id": briefing_result.model_id,
        "used_fallback": briefing_result.used_fallback,
        "retrieved_context_ids": briefing_result.retrieved_context_ids,
    }
else:
    briefing_text = (
        f"Thermal analysis completed. "
        f"Detected {len(detections)} object(s). "
        f"Mean uncertainty: {mean_unc:.3f}. "
        f"Maximum uncertainty: {max_unc:.3f}."
    )
    agent_meta = {
        "model_id": "lightweight-mode",
        "used_fallback": True,
        "retrieved_context_ids": [],
    }
        )

        # ══════════════════════════════════════════════════════════════════════
        # STEP 7 — Encode images as Base64 PNG strings
        # ══════════════════════════════════════════════════════════════════════
        rgb_b64      = _array_to_base64_png(rgb_uint8)
        heatmap_b64  = _heatmap_to_base64_png(unc_np)
        overlay_b64  = _bbox_overlay_to_base64_png(rgb_uint8, detections)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 8 — Build structured JSON response
        # ══════════════════════════════════════════════════════════════════════
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "metrics": {
                    "mean_uncertainty":   round(mean_unc, 6),
                    "max_uncertainty":    round(max_unc,  6),
                    "mc_passes_executed": mc_passes,
                    "device":             _DEVICE,
                },
                "detections": [d.to_dict() for d in detections],
                "images": {
                    "reconstructed_rgb":  rgb_b64,
                    "uncertainty_heatmap": heatmap_b64,
                    "bbox_overlay":        overlay_b64,
                },
                "agent_briefing": briefing_text,
                "agent_meta": agent_meta,
                },
            },
        )

    except HTTPException:
        raise  # re-raise FastAPI validation errors as-is

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        logger.error("CUDA OOM during analyze-thermal", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GPU out-of-memory. Try reducing mc_passes or uploading a "
                "smaller image. CUDA cache has been cleared."
            ),
        )

    except (UnidentifiedImageError, ValueError, OSError) as exc:
        logger.warning("Image decode / validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not process the uploaded file: {exc}",
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled pipeline error:\n%s", traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal pipeline error: {type(exc).__name__}: {exc}",
        )

    finally:
        # Always attempt tensor + CUDA memory cleanup after each request
        _cleanup_tensors()


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def _decode_thermal_image(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Decode uploaded bytes into a single-channel uint8/uint16 numpy array (H, W).

    Supported formats:
      - GeoTIFF (.tif / .tiff) — read via rasterio; band 1 extracted.
      - PNG / JPEG             — read via Pillow; converted to grayscale.

    Returns
    -------
    np.ndarray
        Shape (H, W).  Dtype preserved from source (uint8 or uint16).

    Raises
    ------
    HTTPException (422)
        For unreadable, corrupt, or multi-channel images that cannot be
        reduced to a single channel.
    """
    fname_lower = filename.lower()
    is_tiff = fname_lower.endswith((".tif", ".tiff"))

    if is_tiff:
        return _decode_tiff(raw_bytes, filename)

    return _decode_pil_image(raw_bytes, filename)


def _decode_tiff(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Decode a GeoTIFF via rasterio, extracting band 1."""
    try:
        import rasterio  # type: ignore
        from rasterio.io import MemoryFile  # type: ignore
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="rasterio is required to process GeoTIFF files. "
                   "Install it with: pip install rasterio",
        ) from exc

    try:
        with MemoryFile(raw_bytes) as mem:
            with mem.open() as src:
                if src.count < 1:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"GeoTIFF '{filename}' has no bands.",
                    )
                arr = src.read(1)   # (H, W) — band 1
                logger.info(
                    "GeoTIFF decoded: shape=%s dtype=%s crs=%s",
                    arr.shape, arr.dtype, src.crs,
                )
                return arr
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to read GeoTIFF '{filename}': {exc}",
        ) from exc


def _decode_pil_image(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Decode PNG/JPEG via Pillow, converting to grayscale."""
    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Cannot identify image format for '{filename}': {exc}",
        ) from exc

    # Convert multi-channel images to grayscale luminance
    if img.mode not in ("L", "I", "I;16"):
        logger.info(
            "Image mode '%s' is not single-channel — converting to grayscale 'L'.",
            img.mode,
        )
        img = img.convert("L")

    arr = np.array(img)

    if arr.ndim != 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"After grayscale conversion image still has ndim={arr.ndim}. "
                "Please upload a single-channel thermal image."
            ),
        )

    logger.info("PIL image decoded: shape=%s dtype=%s", arr.shape, arr.dtype)
    return arr


def _pad_to_multiple(
    arr: np.ndarray, multiple: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Right/bottom-pad *arr* so (H, W) are divisible by *multiple*.

    Returns the padded array and (pad_h, pad_w) so callers can un-pad later.
    Padding uses edge-replication to avoid introducing artificial boundaries.
    """
    H, W = arr.shape
    pad_h = (-H) % multiple
    pad_w = (-W) % multiple
    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0)
    arr_padded = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
    return arr_padded, (pad_h, pad_w)


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _array_to_base64_png(rgb_uint8: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 array as a data-URI Base64 PNG string."""
    img = Image.fromarray(rgb_uint8, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _heatmap_to_base64_png(unc_map: np.ndarray) -> str:
    """Encode a (H, W) float32 uncertainty map as a coloured Base64 PNG.

    Applies a perceptually-ordered viridis-like colormap by mapping
    [0, 1] uncertainty through a simple R/G/B gradient:
        0.0 → cool blue  (#0d0887)
        0.5 → warm purple (#cc4778)
        1.0 → bright yellow (#f0f921)
    This is computed without matplotlib to avoid an optional dependency.
    """
    unc_clipped = np.clip(unc_map, 0.0, 1.0)

    # Three-stop gradient: blue → purple → yellow
    r = np.where(
        unc_clipped < 0.5,
        13  + (204 - 13)  * (unc_clipped / 0.5),
        204 + (240 - 204) * ((unc_clipped - 0.5) / 0.5),
    )
    g = np.where(
        unc_clipped < 0.5,
        8   + (71  - 8)   * (unc_clipped / 0.5),
        71  + (249 - 71)  * ((unc_clipped - 0.5) / 0.5),
    )
    b = np.where(
        unc_clipped < 0.5,
        135 + (120 - 135) * (unc_clipped / 0.5),
        120 + (33  - 120) * ((unc_clipped - 0.5) / 0.5),
    )

    heatmap = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return _array_to_base64_png(heatmap)


def _bbox_overlay_to_base64_png(
    rgb_uint8: np.ndarray,
    detections: List,
) -> str:
    """Draw bounding boxes and labels onto rgb_uint8; return Base64 PNG.

    Colour-codes each box by adjusted confidence:
        adj ≥ 0.6  → green  (reliable)
        adj ≥ 0.3  → orange (verify)
        adj <  0.3 → red    (unreliable)

    Uses only Pillow — no cv2 dependency.
    """
    img = Image.fromarray(rgb_uint8, mode="RGB")

    if not detections:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    from PIL import ImageDraw, ImageFont  # type: ignore

    draw = ImageDraw.Draw(img)

    # Attempt to load a basic font; fall back to default if unavailable
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, rgb_uint8.shape[0] // 40))
    except (IOError, OSError):
        font = ImageFont.load_default()

    for det in detections:
        adj = det.adjusted_confidence
        colour = (
            (0, 200, 60)   if adj >= 0.6 else
            (255, 160, 0)  if adj >= 0.3 else
            (220, 30, 30)
        )

        x1, y1, x2, y2 = det.bbox
        lw = max(2, rgb_uint8.shape[0] // 256)   # line width scales with image size

        # Draw box with thick border (simulate by drawing nested rectangles)
        for offset in range(lw):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=colour,
            )

        # Label: "class_name raw→adj"
        label = f"{det.class_name} {det.raw_confidence:.2f}→{det.adjusted_confidence:.2f}"
        text_y = max(0, y1 - lw - 14)
        draw.text((x1 + 2, text_y), label, fill=colour, font=font)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Memory cleanup helper
# ---------------------------------------------------------------------------

def _cleanup_tensors() -> None:
    """Best-effort CUDA / garbage collection after each request."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host="0.0.0.0",
        port=_PORT,
        reload=False,
        log_level="info",
    )
