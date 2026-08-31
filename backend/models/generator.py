"""
backend/models/generator.py
============================
UNetGenerator
-------------
Maps a single-channel Thermal input (Landsat 8 ST_B10) to a 3-channel RGB
reconstruction using a symmetric U-Net with skip connections.

Architecture
------------
  Encoder  : 4 down-sampling blocks (Conv → BN → LeakyReLU, stride-2)
  Bottleneck: 2 residual convolution blocks + Dropout2d(p=0.2)
  Decoder  : 4 up-sampling blocks (ConvTranspose → BN → ReLU, skip concat)
              each decoder block has a Dropout2d(p=0.2) layer
  Head     : 1×1 Conv → Tanh  (output mapped to [−1, 1], caller re-scales)

Monte Carlo Dropout
-------------------
Dropout2d layers use F.dropout2d with training=True unconditionally so that
stochastic channel dropout is active both during training *and* at inference,
enabling MC-Dropout uncertainty estimation without any model modification at
call time.  Set mc_passes=1 to recover deterministic behaviour externally by
averaging a single forward pass.

Input  : (B, 1, H, W)  — normalised thermal in [0, 1]
Output : (B, 3, H, W)  — reconstructed RGB in [−1, 1]

H and W must be divisible by 16 (4 stride-2 downs).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _MCDropout2d(nn.Module):
    """Channel-wise spatial dropout that stays active at inference time.

    Wraps F.dropout2d with ``training=True`` hard-wired so that MC-Dropout
    uncertainty estimation works without toggling model.train() at call time.
    """

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1). Got {p}.")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # training=True is intentional: keeps dropout active during eval()
        return F.dropout2d(x, p=self.p, training=True, inplace=False)

    def extra_repr(self) -> str:
        return f"p={self.p}, mc_always_on=True"


class _EncoderBlock(nn.Module):
    """Conv(stride-2) → BN → LeakyReLU  (no dropout in encoder)."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _BottleneckBlock(nn.Module):
    """3×3 Conv → BN → ReLU → MCDropout2d (used in bottleneck residual pair)."""

    def __init__(self, channels: int, p_drop: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            _MCDropout2d(p=p_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + x  # residual


class _DecoderBlock(nn.Module):
    """ConvTranspose(stride-2) → BN → ReLU → MCDropout2d.

    The in_ch is the number of channels of the upsampled feature map *before*
    skip concatenation.  After concatenation with the skip tensor (skip_ch
    channels) the decoder convolution receives in_ch + skip_ch channels.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, in_ch, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            _MCDropout2d(p=p_drop),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Guard against off-by-one from odd spatial sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ---------------------------------------------------------------------------
# UNetGenerator
# ---------------------------------------------------------------------------

class UNetGenerator(nn.Module):
    """U-Net generator: 1-channel thermal → 3-channel RGB reconstruction.

    Parameters
    ----------
    base_ch:
        Channel width at the first encoder stage (doubles each stage).
        Default 64 gives channel sequence 64 → 128 → 256 → 512 (bottleneck).
    p_drop:
        Spatial dropout probability for MCDropout2d layers in the bottleneck
        and all decoder blocks.  Default 0.2.

    Example
    -------
    >>> model = UNetGenerator()
    >>> x = torch.randn(2, 1, 256, 256)
    >>> out = model(x)          # shape: (2, 3, 256, 256)
    >>> out.shape
    torch.Size([2, 3, 256, 256])

    MC-Dropout inference (e.g. T=20 stochastic passes):
    >>> model.eval()            # BN uses running stats; dropout stays ON
    >>> samples = torch.stack([model(x) for _ in range(20)])  # (20, 2, 3, 256, 256)
    >>> mean  = samples.mean(0)
    >>> uncertainty = samples.var(0)
    """

    def __init__(self, base_ch: int = 64, p_drop: float = 0.2) -> None:
        super().__init__()

        ch = base_ch  # 64

        # ── Encoder ──────────────────────────────────────────────────────────
        # First encoder block has no BN (input is single-channel normalised map)
        self.enc1 = _EncoderBlock(1,      ch,     use_bn=False)   # → ch   H/2
        self.enc2 = _EncoderBlock(ch,     ch * 2)                  # → 2ch  H/4
        self.enc3 = _EncoderBlock(ch * 2, ch * 4)                  # → 4ch  H/8
        self.enc4 = _EncoderBlock(ch * 4, ch * 8)                  # → 8ch  H/16

        # ── Bottleneck (two residual blocks with MC-Dropout) ──────────────────
        self.bottleneck = nn.Sequential(
            _BottleneckBlock(ch * 8, p_drop=p_drop),
            _BottleneckBlock(ch * 8, p_drop=p_drop),
        )

        # ── Decoder (skip_ch mirrors the matching encoder output channels) ───
        self.dec4 = _DecoderBlock(ch * 8, ch * 8, ch * 4, p_drop=p_drop)  # concat enc4
        self.dec3 = _DecoderBlock(ch * 4, ch * 4, ch * 2, p_drop=p_drop)  # concat enc3
        self.dec2 = _DecoderBlock(ch * 2, ch * 2, ch,     p_drop=p_drop)  # concat enc2
        self.dec1 = _DecoderBlock(ch,     ch,     ch,     p_drop=p_drop)  # concat enc1

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(ch, 3, kernel_size=1, bias=True),
            nn.Tanh(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        """Kaiming-normal init for Conv/ConvTranspose; constant for BN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 1, H, W).  Normalised thermal values in [0, 1].

        Returns
        -------
        torch.Tensor
            Shape (B, 3, H, W).  Reconstructed RGB in [−1, 1].
        """
        # Encoder — store activations for skip connections
        s1 = self.enc1(x)   # (B, ch,   H/2,  W/2)
        s2 = self.enc2(s1)  # (B, 2ch,  H/4,  W/4)
        s3 = self.enc3(s2)  # (B, 4ch,  H/8,  W/8)
        s4 = self.enc4(s3)  # (B, 8ch,  H/16, W/16)

        # Bottleneck
        b = self.bottleneck(s4)  # (B, 8ch,  H/16, W/16)

        # Decoder with skip connections
        d4 = self.dec4(b,  s4)  # (B, 4ch,  H/8,  W/8)
        d3 = self.dec3(d4, s3)  # (B, 2ch,  H/4,  W/4)
        d2 = self.dec2(d3, s2)  # (B, ch,   H/2,  W/2)
        d1 = self.dec1(d2, s1)  # (B, ch,   H,    W)

        return self.head(d1)    # (B, 3,    H,    W)  in [-1, 1]
