"use client";

/**
 * frontend/components/ImageSlider.tsx
 * =====================================
 * Interactive split-view comparison slider for the Chitra.ai dashboard.
 *
 * Props (all Base64 strings come directly from POST /api/v1/analyze-thermal):
 *
 *   thermalSrc          — raw thermal input image (data-URI or URL)
 *   reconstructedRgbSrc — Base64 PNG: MCInferenceResult.mean_rgb
 *   uncertaintyHeatmapSrc — Base64 PNG: normalised aleatoric uncertainty map
 *   bboxOverlaySrc      — Base64 PNG: RGB with bounding-box annotations
 *   detections          — array of DetectionResult.to_dict() objects
 *   imageWidth / imageHeight — original image pixel dimensions (for bbox mapping)
 *
 * Features
 * ---------
 *  • Horizontal drag-divider split view: thermal (left) ↔ RGB (right)
 *  • Uncertainty heatmap alpha-blend toggle + opacity slider (0–100 %)
 *  • Bounding box overlay toggle (drawn on canvas in real time)
 *  • Hover inspection: pixel (X, Y) coordinate + bbox hit-test tooltip
 *  • Responsive canvas scaling — maintains aspect ratio at all breakpoints
 *  • Touch support for the drag divider
 */

import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

// ---------------------------------------------------------------------------
// Types — mirror backend DetectionResult.to_dict()
// ---------------------------------------------------------------------------

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Detection {
  class_id: number;
  class_name: string;
  bbox: BBox;
  raw_confidence: number;
  mean_uncertainty: number;
  adjusted_confidence: number;
}

export interface ImageSliderProps {
  /** Raw thermal / IR input image — left panel */
  thermalSrc: string;
  /** Base64 PNG reconstructed RGB — right panel base layer */
  reconstructedRgbSrc: string;
  /** Base64 PNG uncertainty heatmap — blended on top of RGB */
  uncertaintyHeatmapSrc: string;
  /** Base64 PNG RGB with server-rendered bbox annotations */
  bboxOverlaySrc: string;
  /** Detections from backend — used for live hover hit-testing */
  detections: Detection[];
  /** Original image width in pixels (backend coordinate space) */
  imageWidth: number;
  /** Original image height in pixels (backend coordinate space) */
  imageHeight: number;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DIVIDER_WIDTH_PX = 3;
const HANDLE_RADIUS = 18;
const MIN_SPLIT_FRACTION = 0.02;
const MAX_SPLIT_FRACTION = 0.98;

/** Colour by adjusted_confidence tier — matches backend bbox overlay logic */
function detectionColour(adj: number): string {
  if (adj >= 0.6) return "#00c83c";   // green  — reliable
  if (adj >= 0.3) return "#ffa020";   // orange — verify
  return "#dc1e1e";                   // red    — unreliable
}

// ---------------------------------------------------------------------------
// Hook — stable image loading
// ---------------------------------------------------------------------------

function useImage(src: string): HTMLImageElement | null {
  const [img, setImg] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!src) { setImg(null); return; }
    const el = new Image();
    el.onload  = () => setImg(el);
    el.onerror = () => setImg(null);
    el.src = src;
  }, [src]);

  return img;
}

// ---------------------------------------------------------------------------
// Hook — container width for responsive scaling
// ---------------------------------------------------------------------------

function useContainerWidth(ref: React.RefObject<HTMLDivElement | null>): number {
  const [width, setWidth] = useState(0);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) setWidth(entry.contentRect.width);
    });
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, [ref]);

  return width;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function ImageSlider({
  thermalSrc,
  reconstructedRgbSrc,
  uncertaintyHeatmapSrc,
  bboxOverlaySrc,
  detections,
  imageWidth,
  imageHeight,
}: ImageSliderProps) {
  // ── Loaded images ──────────────────────────────────────────────────────────
  const thermalImg    = useImage(thermalSrc);
  const rgbImg        = useImage(reconstructedRgbSrc);
  const heatmapImg    = useImage(uncertaintyHeatmapSrc);
  const bboxImg       = useImage(bboxOverlaySrc);

  // ── Responsive layout ─────────────────────────────────────────────────────
  const containerRef  = useRef<HTMLDivElement>(null);
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const containerW    = useContainerWidth(containerRef);

  const aspectRatio   = imageHeight > 0 ? imageHeight / imageWidth : 1;
  const canvasW       = containerW > 0 ? containerW : imageWidth;
  const canvasH       = Math.round(canvasW * aspectRatio);

  // Scale factor: canvas pixels → image pixels (for bbox hit-testing)
  const scaleX = canvasW   > 0 ? imageWidth  / canvasW   : 1;
  const scaleY = canvasH   > 0 ? imageHeight / canvasH   : 1;

  // ── Slider state ──────────────────────────────────────────────────────────
  const [splitFraction, setSplitFraction] = useState(0.5);
  const isDragging = useRef(false);

  // ── Overlay toggles ───────────────────────────────────────────────────────
  const [showHeatmap,  setShowHeatmap]  = useState(false);
  const [heatmapAlpha, setHeatmapAlpha] = useState(50);   // 0–100
  const [showBboxes,   setShowBboxes]   = useState(true);

  // ── Hover inspection ──────────────────────────────────────────────────────
  const [hoverPos,  setHoverPos]  = useState<{ x: number; y: number } | null>(null);
  const [hoveredDet, setHoveredDet] = useState<Detection | null>(null);

  // ---------------------------------------------------------------------------
  // Bbox hit-test (in image-pixel space)
  // ---------------------------------------------------------------------------
  const hitTest = useCallback(
    (imgX: number, imgY: number): Detection | null => {
      for (const det of detections) {
        const { x1, y1, x2, y2 } = det.bbox;
        if (imgX >= x1 && imgX <= x2 && imgY >= y1 && imgY <= y2) return det;
      }
      return null;
    },
    [detections]
  );

  // ---------------------------------------------------------------------------
  // Render loop — redraws the canvas whenever any state changes
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const split = Math.round(canvasW * splitFraction);

    ctx.clearRect(0, 0, canvasW, canvasH);

    // ── Left panel: raw thermal ──────────────────────────────────────────────
    if (thermalImg) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, split, canvasH);
      ctx.clip();
      ctx.drawImage(thermalImg, 0, 0, canvasW, canvasH);
      ctx.restore();
    } else {
      ctx.save();
      ctx.fillStyle = "#1a1a2e";
      ctx.fillRect(0, 0, split, canvasH);
      ctx.fillStyle = "#4a5568";
      ctx.font = `${Math.max(12, canvasW / 40)}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText("Thermal Input", split / 2, canvasH / 2);
      ctx.restore();
    }

    // ── Right panel: reconstructed RGB ──────────────────────────────────────
    if (rgbImg) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(split, 0, canvasW - split, canvasH);
      ctx.clip();
      ctx.drawImage(rgbImg, 0, 0, canvasW, canvasH);

      // Uncertainty heatmap blend
      if (showHeatmap && heatmapImg) {
        ctx.globalAlpha = heatmapAlpha / 100;
        ctx.drawImage(heatmapImg, 0, 0, canvasW, canvasH);
        ctx.globalAlpha = 1;
      }

      // Bounding-box overlay (server-rendered image OR live canvas redraw)
      if (showBboxes) {
        if (bboxImg) {
          // Prefer the server-rendered overlay for label accuracy
          ctx.globalAlpha = 0.85;
          ctx.drawImage(bboxImg, 0, 0, canvasW, canvasH);
          ctx.globalAlpha = 1;
        } else {
          // Fallback: draw bboxes directly on canvas in image-scaled coordinates
          _drawBboxes(ctx, detections, canvasW, canvasH, scaleX, scaleY);
        }
      }
      ctx.restore();
    } else {
      ctx.save();
      ctx.fillStyle = "#0d1117";
      ctx.fillRect(split, 0, canvasW - split, canvasH);
      ctx.fillStyle = "#4a5568";
      ctx.font = `${Math.max(12, canvasW / 40)}px system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText("Reconstructed RGB", split + (canvasW - split) / 2, canvasH / 2);
      ctx.restore();
    }

    // ── Divider line ─────────────────────────────────────────────────────────
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth   = DIVIDER_WIDTH_PX;
    ctx.beginPath();
    ctx.moveTo(split, 0);
    ctx.lineTo(split, canvasH);
    ctx.stroke();

    // Handle circle
    const hy = canvasH / 2;
    ctx.fillStyle   = "rgba(255,255,255,0.95)";
    ctx.shadowColor = "rgba(0,0,0,0.4)";
    ctx.shadowBlur  = 8;
    ctx.beginPath();
    ctx.arc(split, hy, HANDLE_RADIUS, 0, Math.PI * 2);
    ctx.fill();

    // Arrows on handle
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#374151";
    ctx.lineWidth   = 2;
    ctx.lineCap     = "round";
    const arrowOffset = 6;
    // Left arrow
    ctx.beginPath();
    ctx.moveTo(split - arrowOffset + 3, hy - 5);
    ctx.lineTo(split - arrowOffset - 2, hy);
    ctx.lineTo(split - arrowOffset + 3, hy + 5);
    ctx.stroke();
    // Right arrow
    ctx.beginPath();
    ctx.moveTo(split + arrowOffset - 3, hy - 5);
    ctx.lineTo(split + arrowOffset + 2, hy);
    ctx.lineTo(split + arrowOffset - 3, hy + 5);
    ctx.stroke();
    ctx.restore();

    // ── Panel labels ─────────────────────────────────────────────────────────
    _drawLabel(ctx, "THERMAL INPUT",     14,         14, "#60a5fa");
    _drawLabel(ctx, "RECONSTRUCTED RGB", split + 14, 14, "#4ade80");

  }, [
    canvasW, canvasH, splitFraction,
    thermalImg, rgbImg, heatmapImg, bboxImg,
    showHeatmap, heatmapAlpha, showBboxes,
    detections, scaleX, scaleY,
  ]);

  // ---------------------------------------------------------------------------
  // Drag handlers — mouse + touch
  // ---------------------------------------------------------------------------
  const computeSplit = useCallback(
    (clientX: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect  = canvas.getBoundingClientRect();
      const rawF  = (clientX - rect.left) / rect.width;
      const clamped = Math.min(MAX_SPLIT_FRACTION, Math.max(MIN_SPLIT_FRACTION, rawF));
      setSplitFraction(clamped);
    },
    []
  );

  const onMouseDown  = useCallback((e: React.MouseEvent) => {
    isDragging.current = true;
    computeSplit(e.clientX);
  }, [computeSplit]);

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    isDragging.current = true;
    computeSplit(e.touches[0].clientX);
  }, [computeSplit]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!isDragging.current) return;
      computeSplit(e.clientX);
    };
    const onTouchMove = (e: TouchEvent) => {
      if (!isDragging.current) return;
      computeSplit(e.touches[0].clientX);
    };
    const onUp = () => { isDragging.current = false; };

    window.addEventListener("mousemove",  onMove);
    window.addEventListener("mouseup",   onUp);
    window.addEventListener("touchmove",  onTouchMove, { passive: true });
    window.addEventListener("touchend",  onUp);
    return () => {
      window.removeEventListener("mousemove",  onMove);
      window.removeEventListener("mouseup",   onUp);
      window.removeEventListener("touchmove",  onTouchMove);
      window.removeEventListener("touchend",  onUp);
    };
  }, [computeSplit]);

  // ---------------------------------------------------------------------------
  // Mouse hover inspection
  // ---------------------------------------------------------------------------
  const onMouseMove = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect  = canvas.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;

      // Canvas coords → image pixel coords
      const imgX = Math.round(cx * scaleX);
      const imgY = Math.round(cy * scaleY);
      setHoverPos({ x: imgX, y: imgY });
      setHoveredDet(hitTest(imgX, imgY));
    },
    [scaleX, scaleY, hitTest]
  );

  const onMouseLeave = useCallback(() => {
    setHoverPos(null);
    setHoveredDet(null);
  }, []);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col gap-3 w-full select-none">

      {/* ── Canvas ────────────────────────────────────────────────────── */}
      <div
        ref={containerRef}
        className="relative w-full rounded-xl overflow-hidden border border-gray-700 bg-gray-950 shadow-2xl"
        style={{ cursor: isDragging.current ? "col-resize" : "crosshair" }}
      >
        <canvas
          ref={canvasRef}
          width={canvasW}
          height={canvasH}
          className="block w-full touch-none"
          style={{ imageRendering: "crisp-edges" }}
          onMouseDown={onMouseDown}
          onTouchStart={onTouchStart}
          onMouseMove={onMouseMove}
          onMouseLeave={onMouseLeave}
        />

        {/* ── Hover tooltip ───────────────────────────────────────────── */}
        {hoverPos && (
          <HoverTooltip
            x={hoverPos.x}
            y={hoverPos.y}
            detection={hoveredDet}
            canvasW={canvasW}
            canvasH={canvasH}
            imageWidth={imageWidth}
            imageHeight={imageHeight}
          />
        )}
      </div>

      {/* ── Controls bar ──────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-gray-700 bg-gray-900/80 px-5 py-3 text-sm backdrop-blur-sm">

        {/* Split position readout */}
        <div className="flex items-center gap-2 text-gray-400 shrink-0">
          <span className="font-mono text-xs text-blue-400">
            ◀ {Math.round(splitFraction * 100)} %
          </span>
          <span className="text-gray-600">|</span>
          <span className="font-mono text-xs text-green-400">
            {Math.round((1 - splitFraction) * 100)} % ▶
          </span>
        </div>

        {/* Divider ─── */}
        <div className="hidden sm:block w-px h-5 bg-gray-700" />

        {/* Uncertainty heatmap toggle */}
        <label className="flex items-center gap-2 cursor-pointer shrink-0">
          <ToggleSwitch checked={showHeatmap} onChange={setShowHeatmap} />
          <span className={showHeatmap ? "text-purple-300" : "text-gray-400"}>
            Uncertainty Heatmap
          </span>
        </label>

        {showHeatmap && (
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-gray-500 text-xs">0 %</span>
            <input
              type="range"
              min={0}
              max={100}
              value={heatmapAlpha}
              onChange={(e) => setHeatmapAlpha(Number(e.target.value))}
              className="w-28 accent-purple-500"
              aria-label="Heatmap opacity"
            />
            <span className="text-gray-400 text-xs font-mono w-8">
              {heatmapAlpha} %
            </span>
          </div>
        )}

        {/* Divider ─── */}
        <div className="hidden sm:block w-px h-5 bg-gray-700" />

        {/* Bounding box toggle */}
        <label className="flex items-center gap-2 cursor-pointer shrink-0">
          <ToggleSwitch checked={showBboxes} onChange={setShowBboxes} />
          <span className={showBboxes ? "text-green-300" : "text-gray-400"}>
            Detection Boxes
          </span>
          {detections.length > 0 && (
            <span className="rounded-full bg-gray-700 px-2 py-0.5 text-xs text-gray-300">
              {detections.length}
            </span>
          )}
        </label>

        {/* Push detection confidence legend to the right */}
        <div className="ml-auto flex items-center gap-3 text-xs text-gray-500 shrink-0">
          <LegendDot colour="#00c83c" label="≥ 0.6" />
          <LegendDot colour="#ffa020" label="≥ 0.3" />
          <LegendDot colour="#dc1e1e" label="< 0.3" />
        </div>
      </div>

      {/* ── Detection list (optional compact view) ────────────────────── */}
      {detections.length > 0 && showBboxes && (
        <DetectionList detections={detections} hoveredDet={hoveredDet} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface HoverTooltipProps {
  x: number;
  y: number;
  detection: Detection | null;
  canvasW: number;
  canvasH: number;
  imageWidth: number;
  imageHeight: number;
}

function HoverTooltip({
  x, y, detection, canvasW, canvasH, imageWidth, imageHeight,
}: HoverTooltipProps) {
  // Position tooltip in canvas-relative %, keeping it on-screen
  const pctX = (x / imageWidth) * 100;
  const pctY = (y / imageHeight) * 100;
  const flipX = pctX > 65;
  const flipY = pctY > 70;

  return (
    <div
      className="pointer-events-none absolute z-20 rounded-lg border border-gray-600 bg-gray-900/95 px-3 py-2 text-xs shadow-xl backdrop-blur-sm"
      style={{
        left: flipX ? "auto" : `calc(${pctX}% + 10px)`,
        right: flipX ? `calc(${100 - pctX}% + 10px)` : "auto",
        top:  flipY ? "auto" : `calc(${pctY}% + 10px)`,
        bottom: flipY ? `calc(${100 - pctY}% + 10px)` : "auto",
        minWidth: "160px",
      }}
    >
      <div className="font-mono text-gray-300 mb-1">
        X: <span className="text-blue-300">{x}</span>
        {"  "}
        Y: <span className="text-blue-300">{y}</span>
      </div>
      {detection ? (
        <div className="space-y-0.5 border-t border-gray-700 pt-1">
          <div className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: detectionColour(detection.adjusted_confidence) }}
            />
            <span className="font-semibold capitalize text-white">
              {detection.class_name}
            </span>
          </div>
          <div className="text-gray-400">
            Raw conf:{" "}
            <span className="text-yellow-300 font-mono">
              {detection.raw_confidence.toFixed(3)}
            </span>
          </div>
          <div className="text-gray-400">
            Uncertainty:{" "}
            <span className="text-orange-300 font-mono">
              {detection.mean_uncertainty.toFixed(3)}
            </span>
          </div>
          <div className="text-gray-400">
            Adj. conf:{" "}
            <span
              className="font-mono font-semibold"
              style={{ color: detectionColour(detection.adjusted_confidence) }}
            >
              {detection.adjusted_confidence.toFixed(3)}
            </span>
          </div>
          <div className="text-gray-500 text-[10px] mt-1">
            bbox ({detection.bbox.x1},{detection.bbox.y1})→(
            {detection.bbox.x2},{detection.bbox.y2})
          </div>
        </div>
      ) : (
        <div className="text-gray-600 text-[10px] border-t border-gray-800 pt-1">
          No detection at cursor
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

interface DetectionListProps {
  detections: Detection[];
  hoveredDet: Detection | null;
}

function DetectionList({ detections, hoveredDet }: DetectionListProps) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-900/60 px-4 py-3">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
        Detections — {detections.length} object{detections.length !== 1 ? "s" : ""}
      </p>
      <div className="flex flex-wrap gap-2">
        {detections.map((det, i) => {
          const colour = detectionColour(det.adjusted_confidence);
          const isHovered = hoveredDet?.class_id === det.class_id
            && hoveredDet?.bbox.x1 === det.bbox.x1;
          return (
            <div
              key={i}
              className={
                "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs transition-colors " +
                (isHovered
                  ? "border-white/30 bg-gray-700"
                  : "border-gray-700 bg-gray-800/50")
              }
            >
              <span
                className="h-2 w-2 shrink-0 rounded-full"
                style={{ background: colour }}
              />
              <span className="capitalize text-gray-200 font-medium">
                {det.class_name}
              </span>
              <span className="text-gray-500">
                {det.raw_confidence.toFixed(2)}
                <span className="text-gray-600 mx-1">→</span>
                <span style={{ color: colour }}>{det.adjusted_confidence.toFixed(2)}</span>
              </span>
              {det.mean_uncertainty > 0.4 && (
                <span className="text-orange-400 text-[10px]" title="High uncertainty region">
                  ⚠
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (v: boolean) => void;
}

function ToggleSwitch({ checked, onChange }: ToggleSwitchProps) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={
        "relative inline-flex h-5 w-9 items-center rounded-full transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 " +
        (checked ? "bg-blue-600" : "bg-gray-600")
      }
    >
      <span
        className={
          "inline-block h-3.5 w-3.5 transform rounded-full bg-white shadow transition-transform " +
          (checked ? "translate-x-4" : "translate-x-1")
        }
      />
    </button>
  );
}

// ---------------------------------------------------------------------------

interface LegendDotProps {
  colour: string;
  label: string;
}

function LegendDot({ colour, label }: LegendDotProps) {
  return (
    <span className="flex items-center gap-1">
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: colour }}
      />
      <span>{label}</span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Canvas drawing helpers (used when server bbox overlay is unavailable)
// ---------------------------------------------------------------------------

function _drawBboxes(
  ctx: CanvasRenderingContext2D,
  detections: Detection[],
  canvasW: number,
  canvasH: number,
  scaleX: number,
  scaleY: number
): void {
  const fontSize = Math.max(11, canvasW / 55);
  ctx.font = `${fontSize}px system-ui, sans-serif`;
  ctx.lineWidth = Math.max(1.5, canvasW / 400);

  for (const det of detections) {
    const colour = detectionColour(det.adjusted_confidence);
    const cx1 = det.bbox.x1 / scaleX;
    const cy1 = det.bbox.y1 / scaleY;
    const cx2 = det.bbox.x2 / scaleX;
    const cy2 = det.bbox.y2 / scaleY;

    ctx.strokeStyle = colour;
    ctx.strokeRect(cx1, cy1, cx2 - cx1, cy2 - cy1);

    // Label background
    const label = `${det.class_name} ${det.raw_confidence.toFixed(2)}→${det.adjusted_confidence.toFixed(2)}`;
    const metrics = ctx.measureText(label);
    const lh = fontSize + 4;
    const ly = Math.max(0, cy1 - lh);
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.fillRect(cx1, ly, metrics.width + 6, lh);
    ctx.fillStyle = colour;
    ctx.fillText(label, cx1 + 3, ly + fontSize);
  }
}

function _drawLabel(
  ctx: CanvasRenderingContext2D,
  text: string,
  x: number,
  y: number,
  colour: string
): void {
  const fontSize = Math.max(10, ctx.canvas.width / 70);
  ctx.save();
  ctx.font = `${fontSize}px system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.fillRect(x - 4, y - 2, w, fontSize + 6);
  ctx.fillStyle = colour;
  ctx.fillText(text, x, y + fontSize - 1);
  ctx.restore();
}
