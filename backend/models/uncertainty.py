"""
backend/models/uncertainty.py
==============================
Monte Carlo Dropout inference for UNetGenerator.

run_mc_inference
----------------
Runs a single input tensor through the model for N stochastic forward passes
(MC-Dropout is always active — see _MCDropout2d in generator.py), then
computes:

  • samples          — raw per-pass RGB predictions, shape (N, B, 3, H, W)
  • mean_rgb         — pixel-wise mean over passes,   shape (B, 3, H, W)
  • uncertainty_map  — pixel-wise variance collapsed to a 2-D spatial map,
                       normalised to [0, 1],           shape (B, 1, H, W)

Uncertainty derivation
~~~~~~~~~~~~~~~~~~~~~~
For each pixel position (h, w) the variance is computed across N passes and
averaged over the 3 RGB channels:

    var_map[b, h, w] = mean_over_c( Var_over_n( samples[:, b, c, h, w] ) )

This channel-averaged variance is a proxy for predictive uncertainty: high
variance indicates regions where the model is uncertain about the colour
reconstruction — typically cloud edges, mixed land-cover boundaries, or
thermally ambiguous areas.

The raw variance map is then min-max normalised per batch item to [0, 1] so
it can be used directly as a 2-D Aleatoric Uncertainty Map overlay.

Notes
~~~~~
- The function does *not* call model.eval() or model.train(); callers control
  mode externally.  MC-Dropout is unconditionally active in UNetGenerator
  regardless of training mode.
- All computation stays on whichever device the model and input_tensor reside.
- torch.no_grad() is used to suppress gradient tracking during all passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MCInferenceResult:
    """Typed container returned by :func:`run_mc_inference`.

    Attributes
    ----------
    samples : torch.Tensor
        All N per-pass predictions. Shape ``(N, B, 3, H, W)``, range ``[−1, 1]``.
    mean_rgb : torch.Tensor
        Pixel-wise mean over N passes. Shape ``(B, 3, H, W)``, range ``[−1, 1]``.
    uncertainty_map : torch.Tensor
        Channel-averaged variance, min-max normalised to ``[0, 1]`` per batch
        item. Shape ``(B, 1, H, W)``.
    raw_variance : torch.Tensor
        Channel-averaged variance *before* normalisation. Shape ``(B, 1, H, W)``.
        Useful for thresholding or logging in physical units.
    """

    samples: torch.Tensor
    mean_rgb: torch.Tensor
    uncertainty_map: torch.Tensor
    raw_variance: torch.Tensor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_mc_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    passes: int = 10,
) -> MCInferenceResult:
    """Run Monte Carlo Dropout inference over *passes* stochastic forward passes.

    Parameters
    ----------
    model:
        A :class:`~backend.models.generator.UNetGenerator` instance (or any
        ``nn.Module`` whose MC-Dropout layers are always active).  The model
        is not moved to a different device inside this function.

    input_tensor:
        Normalised single-channel thermal input. Shape ``(B, 1, H, W)``,
        dtype ``float32``, values in ``[0, 1]``.  Must already reside on the
        same device as *model*.

    passes:
        Number of stochastic forward passes ``N``.  Minimum 2 (variance is
        undefined for a single sample).  Default ``10``.

    Returns
    -------
    MCInferenceResult
        A dataclass with fields ``samples``, ``mean_rgb``, ``uncertainty_map``,
        and ``raw_variance``.  See :class:`MCInferenceResult` for shapes.

    Raises
    ------
    ValueError
        If *passes* < 2 or *input_tensor* does not have exactly 4 dimensions.

    Examples
    --------
    >>> model = UNetGenerator()
    >>> x = normalize_thermal(landsat_band)          # (1, 1, 256, 256)
    >>> result = run_mc_inference(model, x, passes=20)
    >>> result.mean_rgb.shape
    torch.Size([1, 3, 256, 256])
    >>> result.uncertainty_map.shape
    torch.Size([1, 1, 256, 256])
    >>> result.uncertainty_map.min(), result.uncertainty_map.max()
    (tensor(0.), tensor(1.))
    """
    if passes < 2:
        raise ValueError(
            f"passes must be >= 2 to compute a meaningful variance map. Got {passes}."
        )
    if input_tensor.ndim != 4:
        raise ValueError(
            f"input_tensor must be 4-D (B, 1, H, W). Got ndim={input_tensor.ndim}."
        )

    # ── 1. Collect N stochastic forward passes ────────────────────────────────
    # Shape of each pass output: (B, 3, H, W)  values in [-1, 1]
    collected: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(passes):
            pred = model(input_tensor)   # (B, 3, H, W)
            collected.append(pred)

    # Stack along a new leading dimension → (N, B, 3, H, W)
    samples = torch.stack(collected, dim=0)

    # ── 2. Pixel-wise mean over N passes → (B, 3, H, W) ─────────────────────
    mean_rgb = samples.mean(dim=0)

    # ── 3. Pixel-wise variance over N passes → (N, B, 3, H, W) → (B, 3, H, W)
    # Use Bessel-corrected variance (unbiased=True, default) so the estimate
    # is unbiased when passes is small.
    var_per_channel = samples.var(dim=0, unbiased=True)   # (B, 3, H, W)

    # Collapse channels: average variance across R, G, B → (B, 1, H, W)
    # This gives a single spatial uncertainty scalar per pixel.
    raw_variance = var_per_channel.mean(dim=1, keepdim=True)  # (B, 1, H, W)

    # ── 4. Min-max normalise per batch item to [0, 1] ─────────────────────────
    # Normalise independently for each image in the batch so the map spans
    # [0, 1] regardless of the absolute variance magnitude.
    B = raw_variance.shape[0]
    flat = raw_variance.view(B, -1)                          # (B, H*W)
    v_min = flat.min(dim=1).values.view(B, 1, 1, 1)         # (B, 1, 1, 1)
    v_max = flat.max(dim=1).values.view(B, 1, 1, 1)         # (B, 1, 1, 1)

    # Guard against degenerate case where all pixels have identical variance
    denom = (v_max - v_min).clamp(min=1e-8)
    uncertainty_map = (raw_variance - v_min) / denom         # (B, 1, H, W) in [0, 1]

    return MCInferenceResult(
        samples=samples,
        mean_rgb=mean_rgb,
        uncertainty_map=uncertainty_map,
        raw_variance=raw_variance,
    )
