"""
backend/app.py
==============
Production-grade FastAPI microservice — Chitra.ai thermal analysis pipeline.
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
from backend.utils.preprocessing import normalize_thermal


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Your deployed frontend can be supplied through FRONTEND_URL.
# Example:
# FRONTEND_URL=https://your-project.vercel.app
_FRONTEND_URL = os.getenv("FRONTEND_URL", "").strip().rstrip("/")

_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if _FRONTEND_URL:
    _ALLOWED_ORIGINS.append(_FRONTEND_URL)

_YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
_UNET_WEIGHTS = os.getenv("UNET_WEIGHTS", "")
_PORT = int(os.getenv("PORT", "8000"))

_MIN_SPATIAL = 16


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _select_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA available — using GPU.")
        return "cuda"

    logger.info("CUDA not available — using CPU.")
    return "cpu"


_DEVICE = _select_device()


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once during startup and clean them up on shutdown."""

    logger.info(
        "=== Chitra.ai startup: loading models on %s ===",
        _DEVICE,
    )

    # ------------------------------------------------------------------
    # UNet Generator
    # ------------------------------------------------------------------

    generator = UNetGenerator(
        base_ch=16,
        p_drop=0.2,
    )

    if _UNET_WEIGHTS and os.path.isfile(_UNET_WEIGHTS):
        logger.info(
            "Loading UNet weights from %s",
            _UNET_WEIGHTS,
        )

        state = torch.load(
            _UNET_WEIGHTS,
            map_location=_DEVICE,
        )

        # Support checkpoints stored either directly as state_dict
        # or inside a "state_dict" key.
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        generator.load_state_dict(state)

        logger.info("UNet weights loaded successfully.")

    else:
        logger.warning(
            "UNET_WEIGHTS not set or file not found — "
            "using random-init weights."
        )

    generator = generator.to(_DEVICE)
    generator.eval()

    app.state.generator = generator

    # ------------------------------------------------------------------
    # YOLO detector
    # ------------------------------------------------------------------

    logger.info(
        "Loading YOLO detector: %s",
        _YOLO_MODEL,
    )

    detector = UncertaintyAwareDetector(
        model_path=_YOLO_MODEL,
        device=_DEVICE,
        conf_threshold=0.25,
        iou_threshold=0.45,
    )

    app.state.detector = detector

    # ------------------------------------------------------------------
    # RAG / Granite agent
    # ------------------------------------------------------------------

    # Disabled on Render free tier to reduce memory usage.
    app.state.agent = None

    logger.info(
        "=== Startup complete — service ready ==="
    )

    yield

    # ------------------------------------------------------------------
    # Shutdown cleanup
    # ------------------------------------------------------------------

    logger.info(
        "Shutting down — releasing model memory."
    )

    if hasattr(app.state, "generator"):
        del app.state.generator

    if hasattr(app.state, "detector"):
        del app.state.detector

    if hasattr(app.state, "agent"):
        del app.state.agent

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Chitra.ai Thermal Intelligence API",
    description=(
        "Real-time thermal-to-RGB reconstruction, MC-Dropout "
        "uncertainty estimation, YOLOv8 object detection, "
        "and IBM Granite RAG briefing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    # Explicit origins
    allow_origins=_ALLOWED_ORIGINS,

    # IMPORTANT:
    # Allows Vercel preview deployments such as:
    # https://chitra-ai-xxxxx.vercel.app
    # https://chitra-ai-git-main-xxxxx.vercel.app
    allow_origin_regex=(
        r"https://.*\.vercel\.app"
    ),

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["ops"],
)
async def health() -> dict:
    return {
        "status": "ok",
        "device": _DEVICE,
        "service": "chitra.ai",
    }


# ---------------------------------------------------------------------------
# Root endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["ops"],
)
async def root() -> dict:
    return {
        "status": "ok",
        "service": "Chitra.ai Thermal Intelligence API",
        "version": "1.0.0",
        "device": _DEVICE,
        "docs": "/docs",
    }


# ---------------------------------------------------------------------------
# Main thermal analysis endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/analyze-thermal",
    summary="Full thermal analysis pipeline",
    tags=["analysis"],
)
async def analyze_thermal(
    file: UploadFile = File(
        ...,
        description=(
            "Single-channel thermal image "
            "(GeoTIFF, PNG, JPEG)"
        ),
    ),

    mc_passes: int = Form(
        default=2,
        ge=2,
        le=10,
        description="Monte Carlo forward passes",
    ),

    latitude: Optional[float] = Form(
        default=None,
        description="Scene centre latitude (WGS-84)",
    ),

    longitude: Optional[float] = Form(
        default=None,
        description="Scene centre longitude (WGS-84)",
    ),
) -> JSONResponse:

    # ------------------------------------------------------------------
    # Retrieve cached models
    # ------------------------------------------------------------------

    generator: UNetGenerator = app.state.generator

    detector: UncertaintyAwareDetector = (
        app.state.detector
    )

    agent = app.state.agent

    raw_bytes: bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    try:

        # ==============================================================
        # STEP 1 — Decode uploaded image
        # ==============================================================

        logger.info(
            "Processing uploaded file: %s",
            file.filename,
        )

        thermal_arr = _decode_thermal_image(
            raw_bytes,
            file.filename or "",
        )

        H, W = thermal_arr.shape

        logger.info(
            "Thermal image size: %sx%s",
            W,
            H,
        )

        if H < _MIN_SPATIAL or W < _MIN_SPATIAL:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Image too small: ({H}, {W}). "
                    f"Minimum spatial size is "
                    f"{_MIN_SPATIAL}×{_MIN_SPATIAL} pixels."
                ),
            )

        # ==============================================================
        # STEP 2 — Pad image
        # ==============================================================

        thermal_arr, (pad_h, pad_w) = (
            _pad_to_multiple(
                thermal_arr,
                multiple=16,
            )
        )

        H_pad, W_pad = thermal_arr.shape

        logger.info(
            "Padded image size: %sx%s",
            W_pad,
            H_pad,
        )

        # ==============================================================
        # STEP 3 — Normalize thermal image
        # ==============================================================

        input_tensor = normalize_thermal(
            thermal_arr
        )

        input_tensor = (
            input_tensor
            .unsqueeze(0)
            .to(_DEVICE)
        )

        # ==============================================================
        # STEP 4 — MC-Dropout inference
        # ==============================================================

        logger.info(
            "Running MC inference with %s passes...",
            mc_passes,
        )

        mc_result = run_mc_inference(
            generator,
            input_tensor,
            passes=mc_passes,
        )

        # ==============================================================
        # Remove padding safely
        # ==============================================================

        end_h = (
            H_pad - pad_h
            if pad_h
            else H_pad
        )

        end_w = (
            W_pad - pad_w
            if pad_w
            else W_pad
        )

        mean_rgb_t = mc_result.mean_rgb[
            :,
            :,
            :end_h,
            :end_w,
        ]

        unc_map_t = mc_result.uncertainty_map[
            :,
            :,
            :end_h,
            :end_w,
        ]

        # ==============================================================
        # STEP 5 — Uncertainty metrics
        # ==============================================================

        unc_np_full = uncertainty_map_to_numpy(
            unc_map_t
        )

        mean_unc = float(
            np.mean(unc_np_full)
        )

        max_unc = float(
            np.max(unc_np_full)
        )

        # ==============================================================
        # STEP 6 — Convert tensors to numpy
        # ==============================================================

        rgb_uint8 = tensor_to_uint8_rgb(
            mean_rgb_t
        )

        unc_np = unc_np_full

        # ==============================================================
        # STEP 7 — YOLO detection
        # ==============================================================

        logger.info(
            "Running YOLO detection..."
        )

        detections = detector.detect(
            rgb_uint8,
            unc_np,
            verbose=False,
        )

        logger.info(
            "YOLO detected %s object(s).",
            len(detections),
        )

        # ==============================================================
        # STEP 8 — Coordinates
        # ==============================================================

        coords_str: Optional[str] = None

        if (
            latitude is not None
            and longitude is not None
        ):
            lat_direction = (
                "N" if latitude >= 0 else "S"
            )

            lon_direction = (
                "E" if longitude >= 0 else "W"
            )

            coords_str = (
                f"{abs(latitude):.5f}°{lat_direction} "
                f"{abs(longitude):.5f}°{lon_direction}"
            )

        # ==============================================================
        # Analysis metrics
        # ==============================================================

        analysis_metrics = {
            "resolution_h": int(H),
            "resolution_w": int(W),
            "mean_uncertainty": mean_unc,
            "max_uncertainty": max_unc,
            "mc_passes": mc_passes,
        }

        # ==============================================================
        # STEP 9 — Intelligence briefing
        # ==============================================================

        if agent is not None:

            briefing_result = (
                agent.generate_intelligence_briefing(
                    analysis_metrics=analysis_metrics,
                    detections=detections,
                    coordinates=coords_str,
                )
            )

            briefing_text = (
                briefing_result.briefing_text
            )

            agent_meta = {
                "model_id": (
                    briefing_result.model_id
                ),
                "used_fallback": (
                    briefing_result.used_fallback
                ),
                "retrieved_context_ids": (
                    briefing_result.retrieved_context_ids
                ),
            }

        else:

            # Lightweight Render mode
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

        # ==============================================================
        # STEP 10 — Encode images
        # ==============================================================

        logger.info(
            "Encoding output images..."
        )

        rgb_b64 = _array_to_base64_png(
            rgb_uint8
        )

        heatmap_b64 = (
            _heatmap_to_base64_png(
                unc_np
            )
        )

        overlay_b64 = (
            _bbox_overlay_to_base64_png(
                rgb_uint8,
                detections,
            )
        )

        # ==============================================================
        # STEP 11 — Return response
        # ==============================================================

        logger.info(
            "=== Thermal analysis completed successfully ==="
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",

                "metrics": {
                    "mean_uncertainty": round(
                        mean_unc,
                        6,
                    ),

                    "max_uncertainty": round(
                        max_unc,
                        6,
                    ),

                    "mc_passes_executed": (
                        mc_passes
                    ),

                    "device": _DEVICE,

                    "resolution_h": int(H),
                    "resolution_w": int(W),
                },

                "detections": [
                    d.to_dict()
                    for d in detections
                ],

                "images": {
                    "reconstructed_rgb": rgb_b64,
                    "uncertainty_heatmap": (
                        heatmap_b64
                    ),
                    "bbox_overlay": overlay_b64,
                },

                "agent_briefing": briefing_text,

                "agent_meta": agent_meta,
            },
        )

    # ------------------------------------------------------------------
    # HTTP errors
    # ------------------------------------------------------------------

    except HTTPException:
        raise

    # ------------------------------------------------------------------
    # CUDA OOM
    # ------------------------------------------------------------------

    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()
        gc.collect()

        logger.error(
            "CUDA OOM during analyze-thermal",
            exc_info=True,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "GPU out-of-memory. "
                "Try reducing mc_passes or "
                "uploading a smaller image."
            ),
        )

    # ------------------------------------------------------------------
    # Image / validation errors
    # ------------------------------------------------------------------

    except (
        UnidentifiedImageError,
        ValueError,
        OSError,
    ) as exc:

        logger.warning(
            "Image decode / validation error: %s",
            exc,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                f"Could not process the uploaded "
                f"file: {exc}"
            ),
        )

    # ------------------------------------------------------------------
    # Unexpected errors
    # ------------------------------------------------------------------

    except Exception as exc:

        logger.error(
            "Unhandled pipeline error:\n%s",
            traceback.format_exc(),
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                f"Internal pipeline error: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    finally:

        _cleanup_tensors()


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def _decode_thermal_image(
    raw_bytes: bytes,
    filename: str,
) -> np.ndarray:

    """Decode uploaded bytes into a single-channel array."""

    fname_lower = filename.lower()

    is_tiff = fname_lower.endswith(
        (".tif", ".tiff")
    )

    if is_tiff:
        return _decode_tiff(
            raw_bytes,
            filename,
        )

    return _decode_pil_image(
        raw_bytes,
        filename,
    )


# ---------------------------------------------------------------------------
# GeoTIFF decoder
# ---------------------------------------------------------------------------

def _decode_tiff(
    raw_bytes: bytes,
    filename: str,
) -> np.ndarray:

    """Decode GeoTIFF using rasterio."""

    try:

        import rasterio
        from rasterio.io import MemoryFile

    except ImportError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                "rasterio is required to process "
                "GeoTIFF files."
            ),
        ) from exc

    try:

        with MemoryFile(raw_bytes) as mem:

            with mem.open() as src:

                if src.count < 1:

                    raise HTTPException(
                        status_code=(
                            status.HTTP_422_UNPROCESSABLE_ENTITY
                        ),
                        detail=(
                            f"GeoTIFF '{filename}' "
                            "has no bands."
                        ),
                    )

                arr = src.read(1)

                logger.info(
                    "GeoTIFF decoded: shape=%s "
                    "dtype=%s crs=%s",
                    arr.shape,
                    arr.dtype,
                    src.crs,
                )

                return arr

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                f"Failed to read GeoTIFF "
                f"'{filename}': {exc}"
            ),
        ) from exc


# ---------------------------------------------------------------------------
# PIL decoder
# ---------------------------------------------------------------------------

def _decode_pil_image(
    raw_bytes: bytes,
    filename: str,
) -> np.ndarray:

    """Decode PNG/JPEG using Pillow."""

    try:

        img = Image.open(
            io.BytesIO(raw_bytes)
        )

        # Force actual decoding now so corrupted/truncated
        # images fail here rather than later.
        img.load()

    except UnidentifiedImageError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                f"Cannot identify image format "
                f"for '{filename}': {exc}"
            ),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                f"Cannot read image "
                f"'{filename}': {exc}"
            ),
        ) from exc

    # Convert multi-channel images to grayscale.
    if img.mode not in (
        "L",
        "I",
        "I;16",
    ):

        logger.info(
            "Image mode '%s' is not single-channel "
            "— converting to grayscale.",
            img.mode,
        )

        img = img.convert("L")

    arr = np.array(img)

    if arr.ndim != 2:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=(
                f"After grayscale conversion image "
                f"still has ndim={arr.ndim}."
            ),
        )

    logger.info(
        "PIL image decoded: shape=%s dtype=%s",
        arr.shape,
        arr.dtype,
    )

    return arr


# ---------------------------------------------------------------------------
# Padding helper
# ---------------------------------------------------------------------------

def _pad_to_multiple(
    arr: np.ndarray,
    multiple: int,
) -> tuple[np.ndarray, tuple[int, int]]:

    """Pad image so height and width are divisible by multiple."""

    H, W = arr.shape

    pad_h = (-H) % multiple
    pad_w = (-W) % multiple

    if pad_h == 0 and pad_w == 0:
        return arr, (0, 0)

    arr_padded = np.pad(
        arr,
        (
            (0, pad_h),
            (0, pad_w),
        ),
        mode="edge",
    )

    return arr_padded, (pad_h, pad_w)


# ---------------------------------------------------------------------------
# Base64 PNG helper
# ---------------------------------------------------------------------------

def _array_to_base64_png(
    rgb_uint8: np.ndarray,
) -> str:

    """Encode RGB numpy array as Base64 PNG."""

    img = Image.fromarray(
        rgb_uint8,
        mode="RGB",
    )

    buf = io.BytesIO()

    img.save(
        buf,
        format="PNG",
        optimize=False,
    )

    b64 = base64.b64encode(
        buf.getvalue()
    ).decode("ascii")

    return (
        "data:image/png;base64,"
        + b64
    )


# ---------------------------------------------------------------------------
# Heatmap helper
# ---------------------------------------------------------------------------

def _heatmap_to_base64_png(
    unc_map: np.ndarray,
) -> str:

    """Encode uncertainty map as coloured PNG."""

    unc_clipped = np.clip(
        unc_map,
        0.0,
        1.0,
    )

    r = np.where(
        unc_clipped < 0.5,
        13
        + (204 - 13)
        * (unc_clipped / 0.5),

        204
        + (240 - 204)
        * ((unc_clipped - 0.5) / 0.5),
    )

    g = np.where(
        unc_clipped < 0.5,
        8
        + (71 - 8)
        * (unc_clipped / 0.5),

        71
        + (249 - 71)
        * ((unc_clipped - 0.5) / 0.5),
    )

    b = np.where(
        unc_clipped < 0.5,
        135
        + (120 - 135)
        * (unc_clipped / 0.5),

        120
        + (33 - 120)
        * ((unc_clipped - 0.5) / 0.5),
    )

    heatmap = np.stack(
        [r, g, b],
        axis=-1,
    ).astype(np.uint8)

    return _array_to_base64_png(
        heatmap
    )


# ---------------------------------------------------------------------------
# Bounding-box overlay helper
# ---------------------------------------------------------------------------

def _bbox_overlay_to_base64_png(
    rgb_uint8: np.ndarray,
    detections: List,
) -> str:

    """Draw detection bounding boxes and labels."""

    img = Image.fromarray(
        rgb_uint8,
        mode="RGB",
    )

    if not detections:

        buf = io.BytesIO()

        img.save(
            buf,
            format="PNG",
        )

        b64 = base64.b64encode(
            buf.getvalue()
        ).decode("ascii")

        return (
            "data:image/png;base64,"
            + b64
        )

    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(img)

    try:

        font = ImageFont.truetype(
            "arial.ttf",
            size=max(
                12,
                rgb_uint8.shape[0] // 40,
            ),
        )

    except (IOError, OSError):

        font = ImageFont.load_default()

    for det in detections:

        adj = float(
            det.adjusted_confidence
        )

        if adj >= 0.6:

            colour = (
                0,
                200,
                60,
            )

        elif adj >= 0.3:

            colour = (
                255,
                160,
                0,
            )

        else:

            colour = (
                220,
                30,
                30,
            )

        x1, y1, x2, y2 = det.bbox

        lw = max(
            2,
            rgb_uint8.shape[0] // 256,
        )

        # Draw thick bounding box.
        for offset in range(lw):

            draw.rectangle(
                [
                    x1 - offset,
                    y1 - offset,
                    x2 + offset,
                    y2 + offset,
                ],
                outline=colour,
            )

        label = (
            f"{det.class_name} "
            f"{det.raw_confidence:.2f}"
            f"→"
            f"{det.adjusted_confidence:.2f}"
        )

        text_y = max(
            0,
            y1 - lw - 14,
        )

        draw.text(
            (
                x1 + 2,
                text_y,
            ),
            label,
            fill=colour,
            font=font,
        )

    buf = io.BytesIO()

    img.save(
        buf,
        format="PNG",
    )

    b64 = base64.b64encode(
        buf.getvalue()
    ).decode("ascii")

    return (
        "data:image/png;base64,"
        + b64
    )


# ---------------------------------------------------------------------------
# Memory cleanup
# ---------------------------------------------------------------------------

def _cleanup_tensors() -> None:

    """Best-effort memory cleanup after each request."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    gc.collect()


# ---------------------------------------------------------------------------
# Dev server
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
