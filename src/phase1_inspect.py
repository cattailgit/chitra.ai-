"""
Phase 1 — Landsat Scene Inspection
====================================
Reads every .TIF/.tif/.TIFF file found under data/raw/ and reports:
  • CRS and affine transform
  • Pixel resolution (metres)
  • Raster shape (bands × rows × cols)
  • Data types per band
  • Per-band statistics (min / max / mean / nodata fraction)
  • Spatial consistency across all bands in the scene
  • Cloud-mask (QA_PIXEL) flag statistics if the band is present

Usage
-----
    python src/phase1_inspect.py

Outputs a summary to stdout and saves outputs/phase1_report.txt.
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "phase1_report.txt"

# ---------------------------------------------------------------------------
# QA_PIXEL bit definitions (Landsat Collection 2 Level-2)
# ---------------------------------------------------------------------------
QA_BITS: Dict[str, int] = {
    "fill":           0b0000_0000_0000_0001,  # bit 0
    "dilated_cloud":  0b0000_0000_0000_0010,  # bit 1
    "cirrus":         0b0000_0000_0000_0100,  # bit 2
    "cloud":          0b0000_0000_0000_1000,  # bit 3
    "cloud_shadow":   0b0000_0000_0001_0000,  # bit 4
    "snow":           0b0000_0000_0010_0000,  # bit 5
    "clear":          0b0000_0000_0100_0000,  # bit 6
    "water":          0b0000_0000_1000_0000,  # bit 7
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_tifs(directory: Path) -> List[Path]:
    """Return all TIF files under *directory*, sorted by name."""
    tifs = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in {".tif", ".tiff"} and p.is_file()
    )
    return tifs


def pixel_resolution(transform: rasterio.transform.Affine) -> Tuple[float, float]:
    """Return (x_res, y_res) in the native CRS units (usually metres)."""
    return abs(transform.a), abs(transform.e)


def band_stats(data: np.ndarray, nodata) -> dict:
    """Compute statistics for a 2-D array, masking nodata."""
    mask = np.zeros(data.shape, dtype=bool)
    if nodata is not None:
        if np.isnan(nodata):
            mask = np.isnan(data)
        else:
            mask = data == nodata
    valid = data[~mask]
    total_px = data.size
    nodata_frac = mask.sum() / total_px if total_px > 0 else float("nan")
    if valid.size == 0:
        return {"min": None, "max": None, "mean": None, "nodata_frac": nodata_frac}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "nodata_frac": float(nodata_frac),
    }


def qa_flag_stats(qa: np.ndarray) -> str:
    """Return a formatted string of QA_PIXEL flag percentages."""
    total = qa.size
    lines = []
    for name, mask in QA_BITS.items():
        hit = int(np.count_nonzero(qa & mask))
        lines.append(f"      {name:<18s}: {hit:>10,d} px  ({100*hit/total:6.2f} %)")
    return "\n".join(lines)


def inspect_file(path: Path) -> str:
    """Return a detailed report string for a single TIF file."""
    lines: List[str] = []
    sep = "─" * 70

    lines.append(sep)
    lines.append(f"FILE : {path.relative_to(ROOT)}")
    lines.append(sep)

    with rasterio.open(path) as src:
        crs: CRS = src.crs
        transform = src.transform
        x_res, y_res = pixel_resolution(transform)
        nodata = src.nodata

        lines.append(f"  CRS          : {crs.to_string() if crs else 'undefined'}")
        lines.append(f"  EPSG         : {crs.to_epsg() if crs else 'n/a'}")
        lines.append(f"  Resolution   : {x_res:.4f} × {y_res:.4f} (native CRS units)")
        lines.append(f"  Dimensions   : {src.count} band(s) × {src.height} rows × {src.width} cols")
        lines.append(f"  Dtype(s)     : {', '.join(set(src.dtypes))}")
        lines.append(f"  Nodata value : {nodata}")
        lines.append(f"  Transform    :")
        for row in str(transform).splitlines():
            lines.append(f"      {row}")

        # Per-band statistics
        is_qa = "QA_PIXEL" in path.stem.upper()
        lines.append(f"  Band statistics:")
        for b in range(1, src.count + 1):
            data = src.read(b)
            stats = band_stats(data, nodata)
            lines.append(
                f"    Band {b:>2d}: min={stats['min']}, max={stats['max']}, "
                f"mean={stats['mean']:.4f}" if stats["mean"] is not None else
                f"    Band {b:>2d}: all nodata"
            )
            lines.append(f"           nodata fraction = {stats['nodata_frac']:.4f}")

            if is_qa:
                lines.append(f"    QA_PIXEL flag breakdown (band {b}):")
                lines.append(qa_flag_stats(data))

    return "\n".join(lines)


def spatial_consistency_check(tifs: List[Path]) -> str:
    """Report whether all TIFs share the same CRS, resolution, and extent."""
    lines = ["", "SPATIAL CONSISTENCY CHECK", "─" * 70]
    if not tifs:
        lines.append("  No files to compare.")
        return "\n".join(lines)

    records = []
    for p in tifs:
        with rasterio.open(p) as src:
            records.append({
                "file": p.name,
                "crs": src.crs.to_epsg() if src.crs else None,
                "res": pixel_resolution(src.transform),
                "bounds": src.bounds,
                "shape": (src.height, src.width),
            })

    ref = records[0]
    all_ok = True
    for rec in records[1:]:
        issues = []
        if rec["crs"] != ref["crs"]:
            issues.append(f"CRS mismatch ({rec['crs']} vs {ref['crs']})")
        if rec["res"] != ref["res"]:
            issues.append(f"resolution mismatch ({rec['res']} vs {ref['res']})")
        if rec["shape"] != ref["shape"]:
            issues.append(f"shape mismatch ({rec['shape']} vs {ref['shape']})")
        if rec["bounds"] != ref["bounds"]:
            issues.append(f"bounds mismatch")
        if issues:
            lines.append(f"  ✗ {rec['file']}: {'; '.join(issues)}")
            all_ok = False

    if all_ok:
        lines.append(f"  ✓ All {len(tifs)} file(s) share the same CRS, resolution, shape, and bounds.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tifs = find_tifs(RAW_DIR)

    header = textwrap.dedent(f"""\
        ╔══════════════════════════════════════════════════════════════════════╗
        ║            Chitra.ai — Phase 1: Landsat Scene Inspection            ║
        ╚══════════════════════════════════════════════════════════════════════╝
        Raw directory : {RAW_DIR}
        Files found   : {len(tifs)}
    """)

    print(header)
    report_parts = [header]

    if not tifs:
        msg = (
            "  ⚠  No .TIF files found in data/raw/.\n"
            "     Copy your Landsat 9 scene folder into data/raw/ and re-run."
        )
        print(msg)
        report_parts.append(msg)
        REPORT_PATH.write_text("\n".join(report_parts))
        sys.exit(0)

    for tif in tifs:
        file_report = inspect_file(tif)
        print(file_report)
        report_parts.append(file_report)

    consistency = spatial_consistency_check(tifs)
    print(consistency)
    report_parts.append(consistency)

    full_report = "\n".join(report_parts)
    REPORT_PATH.write_text(full_report, encoding="utf-8")
    print(f"\n✓ Report saved to {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
