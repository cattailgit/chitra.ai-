"""
backend/utils/preprocessing.py
================================
Preprocessing utilities for Landsat thermal imagery.

normalize_thermal
-----------------
Converts a single-channel 8-bit or 16-bit thermal raster (Landsat 8 ST_B10)
into a float32 torch.Tensor scaled to [0, 1].

Supported input types
~~~~~~~~~~~~~~~~~~~~~
  • numpy.ndarray  — shape (H, W) or (1, H, W), dtype uint8 or uint16
  • torch.Tensor   — same shapes and dtypes

Normalisation strategy
~~~~~~~~~~~~~~~~~~~~~~
  8-bit  : divide by 255
  16-bit : divide by 65535

An optional (valid_min, valid_max) window can be supplied to perform
min-max normalisation over the *physical* value range instead of the
full bit-depth range.  Values outside the window are clamped before
scaling.  This is useful when the raw DN values are known to span only a
sub-range (e.g. Landsat Collection 2 ST_B10 DN range 1–65455).

The output is always a float32 torch.Tensor with shape (1, H, W).
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_thermal(
    image: ArrayLike,
    valid_range: Optional[Tuple[float, float]] = None,
) -> torch.Tensor:
    """Normalise a single-channel thermal image to a float32 tensor in [0, 1].

    Parameters
    ----------
    image:
        Single-channel thermal band as a numpy array or torch.Tensor.
        Accepted shapes: (H, W) or (1, H, W).
        Accepted dtypes: uint8, uint16, int16, float32, float64.
        Pixels equal to 0 are treated as fill/nodata and are preserved as 0
        in the output (they are excluded from min-max when *valid_range* is
        derived automatically — see *valid_range* note below).

    valid_range:
        Optional (min, max) tuple that defines the physical range to map onto
        [0, 1].  Values outside this range are clamped before scaling.

        If ``None`` (default) the full bit-depth range is used:
          • uint8  → (0, 255)
          • uint16 or int16 → (0, 65535)
          • float32/float64 → assumes values are already in [0, 1] and returns
            the tensor clamped + cast to float32.

        Pass an explicit range to perform windowed normalisation, e.g.:
            valid_range=(7500, 65455)  # typical ST_B10 DN window

    Returns
    -------
    torch.Tensor
        Shape (1, H, W), dtype float32, values in [0, 1].

    Raises
    ------
    ValueError
        If the image has more than one channel, an unsupported dtype, an
        invalid *valid_range*, or a non-2D/3D shape.

    Examples
    --------
    >>> import numpy as np
    >>> img_8bit = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
    >>> t = normalize_thermal(img_8bit)
    >>> t.shape, t.dtype, t.min().item() >= 0, t.max().item() <= 1
    (torch.Size([1, 512, 512]), torch.float32, True, True)

    >>> img_16bit = np.random.randint(7500, 65455, (512, 512), dtype=np.uint16)
    >>> t = normalize_thermal(img_16bit, valid_range=(7500, 65455))
    >>> t.shape, t.min().item() >= 0, t.max().item() <= 1
    (torch.Size([1, 512, 512]), True, True)
    """
    # ── 1. Convert to numpy float32 ──────────────────────────────────────────
    arr = _to_numpy_float32(image)

    # ── 2. Validate shape → ensure (H, W) ────────────────────────────────────
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(
                f"Expected a single-channel image, got shape {arr.shape}."
            )
        arr = arr[0]  # (H, W)
    elif arr.ndim != 2:
        raise ValueError(
            f"Image must be 2-D (H, W) or 3-D (1, H, W). Got ndim={arr.ndim}."
        )

    # ── 3. Determine normalisation range ─────────────────────────────────────
    lo, hi = _resolve_range(image, valid_range)

    if lo >= hi:
        raise ValueError(
            f"valid_range lo must be strictly less than hi. Got ({lo}, {hi})."
        )

    # ── 4. Clamp → scale to [0, 1] ───────────────────────────────────────────
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo)  # float32 in [0, 1]

    # ── 5. Add channel dimension and wrap in tensor ───────────────────────────
    tensor = torch.from_numpy(arr[np.newaxis, :, :])  # (1, H, W)
    return tensor  # float32, [0, 1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy_float32(image: ArrayLike) -> np.ndarray:
    """Convert *image* to a float32 numpy array without copying if possible."""
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(
            f"image must be a numpy.ndarray or torch.Tensor. Got {type(image)}."
        )

    # Guard against unsupported dtypes early
    _assert_supported_dtype(arr.dtype)

    return arr.astype(np.float32, copy=False)


_SUPPORTED_DTYPES = {
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("int16"),
    np.dtype("float32"),
    np.dtype("float64"),
}


def _assert_supported_dtype(dtype: np.dtype) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype {dtype}. "
            f"Supported: {[str(d) for d in _SUPPORTED_DTYPES]}."
        )


def _resolve_range(
    original: ArrayLike,
    valid_range: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """Return the (lo, hi) normalisation range."""
    if valid_range is not None:
        return float(valid_range[0]), float(valid_range[1])

    # Infer from original dtype
    if isinstance(original, torch.Tensor):
        dtype = np.dtype(str(original.dtype).replace("torch.", ""))
    else:
        dtype = np.array(original).dtype

    if dtype == np.dtype("uint8"):
        return 0.0, 255.0
    if dtype in {np.dtype("uint16"), np.dtype("int16")}:
        return 0.0, 65535.0
    if dtype in {np.dtype("float32"), np.dtype("float64")}:
        # Assume already in [0, 1]; clamp only
        return 0.0, 1.0

    # Fallback (shouldn't reach here after _assert_supported_dtype)
    return 0.0, 1.0
