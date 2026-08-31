"use client";

/**
 * frontend/app/page.tsx
 * ======================
 * Chitra.ai — Primary Operational Dashboard
 *
 * Layout (3-column, dark mode)
 * ─────────────────────────────────────────
 *  ┌──────────────────────────────────────────────────────────────┐
 *  │  TOP NAV — Chitra.ai | Satellite IR Reconstruction & Intelligence Engine │
 *  ├─────────────────┬──────────────────────────┬─────────────────┤
 *  │  CONTROL        │   WORKSPACE CANVAS        │  INTELLIGENCE   │
 *  │  SIDEBAR        │   ImageSlider + overlays  │  CO-PILOT       │
 *  │                 │                           │                 │
 *  │  • Dropzone     │   (empty state / result)  │  • Telemetry    │
 *  │  • MC passes    │                           │  • Granite badge│
 *  │  • Conf thresh  │                           │  • Briefing     │
 *  │  • Lat / Lon    │                           │  • Detections   │
 *  │  • Run button   │                           │    table        │
 *  └─────────────────┴──────────────────────────┴─────────────────┘
 *
 * API: POST http://localhost:8000/api/v1/analyze-thermal (multipart/form-data)
 */

import React, {
  useCallback,
  useRef,
  useState,
  DragEvent,
  ChangeEvent,
} from "react";
import ImageSlider, { Detection } from "../components/ImageSlider";

// ---------------------------------------------------------------------------
// Types — mirror backend JSON response
// ---------------------------------------------------------------------------

interface AnalysisMetrics {
  mean_uncertainty: number;
  max_uncertainty: number;
  mc_passes_executed: number;
  device: string;
}

interface AnalysisImages {
  reconstructed_rgb: string;
  uncertainty_heatmap: string;
  bbox_overlay: string;
}

interface AgentMeta {
  model_id: string;
  used_fallback: boolean;
  retrieved_context_ids: string[];
}

interface AnalysisResponse {
  status: string;
  metrics: AnalysisMetrics;
  detections: Detection[];
  images: AnalysisImages;
  agent_briefing: string;
  agent_meta: AgentMeta;
}

type PipelineState = "idle" | "loading" | "success" | "error";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const API_URL = "http://localhost:8000/api/v1/analyze-thermal";
const ACCEPTED_TYPES = ["image/png", "image/jpeg", "image/tiff", "image/tif"];
const ACCEPTED_EXT   = ".png,.jpg,.jpeg,.tif,.tiff";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatUncertainty(v: number): string {
  return (v * 100).toFixed(2) + " %";
}

function uncertaintyColour(v: number): string {
  if (v < 0.2) return "text-green-400";
  if (v < 0.4) return "text-yellow-300";
  if (v < 0.6) return "text-orange-400";
  return "text-red-400";
}

function adjConfColour(v: number): string {
  if (v >= 0.6) return "text-green-400";
  if (v >= 0.3) return "text-yellow-300";
  return "text-red-400";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

// ── Telemetry badge ──────────────────────────────────────────────────────────

interface TelemetryBadgeProps {
  label: string;
  value: string;
  valueClass?: string;
  icon: React.ReactNode;
}
function TelemetryBadge({ label, value, valueClass = "text-white", icon }: TelemetryBadgeProps) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-gray-700 bg-gray-800/60 px-3 py-2.5">
      <div className="shrink-0 text-gray-400">{icon}</div>
      <div className="min-w-0">
        <p className="text-[10px] uppercase tracking-wider text-gray-500">{label}</p>
        <p className={`font-mono text-sm font-semibold truncate ${valueClass}`}>{value}</p>
      </div>
    </div>
  );
}

// ── Slider input ─────────────────────────────────────────────────────────────

interface LabelledSliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  unit?: string;
  onChange: (v: number) => void;
  accentClass?: string;
}
function LabelledSlider({
  label, value, min, max, step = 1, unit = "", onChange, accentClass = "accent-blue-500",
}: LabelledSliderProps) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className="font-mono text-blue-300">{value}{unit}</span>
      </div>
      <input
        type="range"
        min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={`w-full h-1.5 rounded-full appearance-none bg-gray-700 cursor-pointer ${accentClass}`}
      />
      <div className="flex justify-between text-[10px] text-gray-600">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

// ── Section heading ──────────────────────────────────────────────────────────

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-widest text-gray-500 mb-3">
      <span className="h-px flex-1 bg-gray-700/80" />
      {children}
      <span className="h-px flex-1 bg-gray-700/80" />
    </h2>
  );
}

// ── Spinner ──────────────────────────────────────────────────────────────────

function Spinner({ size = 16 }: { size?: number }) {
  return (
    <svg
      className="animate-spin"
      width={size} height={size}
      viewBox="0 0 24 24" fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}

// ── Briefing text renderer ────────────────────────────────────────────────────
// Renders the structured plain-text briefing from Granite with section headings highlighted.

function BriefingRenderer({ text }: { text: string }) {
  const SECTION_HEADINGS = [
    "VISUAL RECONSTRUCTION INTEGRITY",
    "TACTICAL OBJECT ASSESSMENT",
    "OPERATIONAL GUIDANCE",
  ];

  const lines = text.split("\n");

  return (
    <div className="space-y-1 font-mono text-xs leading-relaxed text-gray-300">
      {lines.map((line, i) => {
        const trimmed = line.trim();

        // Separator lines
        if (/^[─━═\-]{4,}$/.test(trimmed)) {
          return <hr key={i} className="border-gray-700 my-1" />;
        }

        // Section headings
        const isHeading = SECTION_HEADINGS.some((h) =>
          trimmed.toUpperCase().includes(h)
        );
        if (isHeading) {
          return (
            <p key={i} className="mt-3 text-blue-300 font-bold tracking-wide text-[11px] uppercase">
              {trimmed}
            </p>
          );
        }

        // Dash-prefixed lines (bullets)
        if (trimmed.startsWith("-") || trimmed.startsWith("•")) {
          return (
            <p key={i} className="pl-3 text-gray-300">
              {trimmed}
            </p>
          );
        }

        // WARNING lines
        if (trimmed.toUpperCase().startsWith("WARNING:")) {
          return (
            <p key={i} className="text-orange-300 font-semibold">
              {trimmed}
            </p>
          );
        }

        // NOTE lines
        if (trimmed.toUpperCase().startsWith("NOTE:")) {
          return (
            <p key={i} className="text-yellow-400/80 italic">
              {trimmed}
            </p>
          );
        }

        // Empty lines
        if (!trimmed) return <div key={i} className="h-1" />;

        return <p key={i}>{trimmed}</p>;
      })}
    </div>
  );
}

// ── Detection table row ───────────────────────────────────────────────────────

function DetectionRow({ det, index }: { det: Detection; index: number }) {
  const rowUnc = det.mean_uncertainty;
  return (
    <tr className="border-b border-gray-800 hover:bg-gray-800/40 transition-colors">
      <td className="py-1.5 pl-3 pr-2 text-gray-500 font-mono text-xs">{index + 1}</td>
      <td className="py-1.5 px-2">
        <span className="capitalize text-gray-200 font-medium text-xs">{det.class_name}</span>
      </td>
      <td className="py-1.5 px-2 font-mono text-xs text-yellow-300 text-right">
        {det.raw_confidence.toFixed(3)}
      </td>
      <td className={`py-1.5 px-2 font-mono text-xs text-right ${uncertaintyColour(rowUnc)}`}>
        {formatUncertainty(rowUnc)}
      </td>
      <td className={`py-1.5 pr-3 font-mono text-xs text-right font-semibold ${adjConfColour(det.adjusted_confidence)}`}>
        {det.adjusted_confidence.toFixed(3)}
        {rowUnc > 0.4 && (
          <span className="ml-1 text-orange-400" title="High-uncertainty region">⚠</span>
        )}
      </td>
    </tr>
  );
}

// ── Upload dropzone ──────────────────────────────────────────────────────────

interface DropzoneProps {
  file: File | null;
  onFile: (f: File) => void;
  disabled?: boolean;
}

function Dropzone({ file, onFile, disabled = false }: DropzoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const dropped = e.dataTransfer.files[0];
      if (dropped) onFile(dropped);
    },
    [disabled, onFile]
  );

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const picked = e.target.files?.[0];
      if (picked) onFile(picked);
    },
    [onFile]
  );

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload thermal image"
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => e.key === "Enter" && !disabled && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      className={[
        "relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed",
        "px-4 py-6 text-center transition-colors cursor-pointer",
        disabled
          ? "border-gray-700 bg-gray-800/20 cursor-not-allowed opacity-50"
          : dragOver
          ? "border-blue-400 bg-blue-500/10"
          : file
          ? "border-green-600 bg-green-900/10"
          : "border-gray-600 bg-gray-800/30 hover:border-gray-500 hover:bg-gray-800/50",
      ].join(" ")}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_EXT}
        className="sr-only"
        onChange={handleChange}
        disabled={disabled}
      />
      {file ? (
        <>
          <IconCheckCircle className="mb-2 h-7 w-7 text-green-400" />
          <p className="text-xs font-medium text-green-300 truncate max-w-full px-2">{file.name}</p>
          <p className="text-[10px] text-gray-500 mt-0.5">
            {(file.size / 1024).toFixed(1)} KB — click to replace
          </p>
        </>
      ) : (
        <>
          <IconUpload className="mb-2 h-7 w-7 text-gray-500" />
          <p className="text-xs text-gray-400">
            Drop thermal image here
          </p>
          <p className="text-[10px] text-gray-600 mt-1">PNG · JPEG · GeoTIFF</p>
        </>
      )}
    </div>
  );
}

// ── Inline SVG icons (no external icon lib dependency) ───────────────────────

function IconUpload({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 16V4m0 0L8 8m4-4 4 4" />
      <path d="M20 16v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2" />
    </svg>
  );
}

function IconCheckCircle({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <path d="M8 12l3 3 5-5" />
    </svg>
  );
}

function IconCpu({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <rect x="7" y="7" width="10" height="10" rx="1" />
      <path d="M9 1v2M15 1v2M9 21v2M15 21v2M1 9h2M1 15h2M21 9h2M21 15h2" />
    </svg>
  );
}

function IconWave({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12c1.5-4 3-6 4-6s2.5 4 4 6 2.5 4 4 4 2.5-2 4-6" />
    </svg>
  );
}

function IconTarget({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <circle cx="12" cy="12" r="6" />
      <circle cx="12" cy="12" r="2" />
    </svg>
  );
}

function IconTimer({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="13" r="8" />
      <path d="M12 9v4l3 3" />
      <path d="M9 2h6" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function DashboardPage() {
  // ── Form state ─────────────────────────────────────────────────────────────
  const [file,      setFile]      = useState<File | null>(null);
  const [mcPasses,  setMcPasses]  = useState(10);
  const [confThresh, setConfThresh] = useState(25);   // stored as 0–100
  const [latitude,  setLatitude]  = useState("");
  const [longitude, setLongitude] = useState("");

  // ── Pipeline state ─────────────────────────────────────────────────────────
  const [pipelineState, setPipelineState] = useState<PipelineState>("idle");
  const [result,        setResult]        = useState<AnalysisResponse | null>(null);
  const [errorMsg,      setErrorMsg]      = useState<string>("");
  const [execMs,        setExecMs]        = useState<number | null>(null);

  // Derived: object-URL for the thermal preview on the left panel
  const [thermalPreviewUrl, setThermalPreviewUrl] = useState<string>("");

  const handleFileSelected = useCallback((f: File) => {
    setFile(f);
    setThermalPreviewUrl(URL.createObjectURL(f));
    // Reset previous results when a new file is picked
    setResult(null);
    setErrorMsg("");
    setPipelineState("idle");
  }, []);

  // ── Submit handler ─────────────────────────────────────────────────────────
  const handleRun = useCallback(async () => {
    if (!file) return;

    setPipelineState("loading");
    setErrorMsg("");
    setResult(null);
    const t0 = performance.now();

    try {
      const form = new FormData();
      form.append("file",      file);
      form.append("mc_passes", String(mcPasses));
      if (latitude.trim())  form.append("latitude",  latitude.trim());
      if (longitude.trim()) form.append("longitude", longitude.trim());

      const resp = await fetch(API_URL, { method: "POST", body: form });
      const elapsed = performance.now() - t0;
      setExecMs(elapsed);

      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`;
        try { detail = (await resp.json()).detail ?? detail; } catch {}
        throw new Error(detail);
      }

      const data: AnalysisResponse = await resp.json();
      if (data.status !== "success") throw new Error("Backend returned non-success status.");

      setResult(data);
      setPipelineState("success");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      setPipelineState("error");
    }
  }, [file, mcPasses, latitude, longitude]);

  const isLoading = pipelineState === "loading";

  // ── Derived display values ─────────────────────────────────────────────────
  const filteredDetections = result
    ? result.detections.filter(
        (d) => d.adjusted_confidence >= confThresh / 100
      )
    : [];

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-gray-950 text-gray-100">

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* TOP NAVIGATION BAR                                                  */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <header className="flex h-12 shrink-0 items-center justify-between border-b border-gray-800 bg-gray-950/95 px-5 backdrop-blur-sm z-30">
        <div className="flex items-center gap-3">
          {/* Logo mark */}
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-blue-600/20 border border-blue-500/40">
            <svg viewBox="0 0 20 20" fill="none" className="h-4 w-4">
              <circle cx="10" cy="10" r="7" stroke="#60a5fa" strokeWidth="1.5" />
              <circle cx="10" cy="10" r="3" fill="#60a5fa" opacity="0.6" />
              <path d="M10 3v2M10 15v2M3 10h2M15 10h2" stroke="#60a5fa" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-sm font-bold tracking-tight text-white">Chitra.ai</span>
            <span className="hidden sm:inline text-gray-600">|</span>
            <span className="hidden sm:inline text-xs text-gray-400">
              Satellite IR Reconstruction &amp; Intelligence Engine
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs">
          {result && (
            <span className={[
              "rounded-full px-2.5 py-0.5 font-medium",
              result.agent_meta.used_fallback
                ? "bg-yellow-900/50 text-yellow-300 border border-yellow-700/50"
                : "bg-blue-900/50 text-blue-300 border border-blue-700/50",
            ].join(" ")}>
              {result.agent_meta.used_fallback ? "Fallback Agent" : "IBM Granite 3.0 Live"}
            </span>
          )}
          <span className="rounded-full bg-gray-800 px-2.5 py-0.5 text-gray-400 border border-gray-700">
            v1.0
          </span>
        </div>
      </header>

      {/* ═══════════════════════════════════════════════════════════════════ */}
      {/* THREE-COLUMN BODY                                                    */}
      {/* ═══════════════════════════════════════════════════════════════════ */}
      <div className="flex flex-1 min-h-0 overflow-hidden">

        {/* ─────────────────────────────────────────────────────────────── */}
        {/* LEFT: CONTROL SIDEBAR                                            */}
        {/* ─────────────────────────────────────────────────────────────── */}
        <aside className="flex w-64 shrink-0 flex-col gap-4 overflow-y-auto border-r border-gray-800 bg-gray-900/60 p-4 backdrop-blur-sm">

          <SectionHeading>Image Input</SectionHeading>
          <Dropzone file={file} onFile={handleFileSelected} disabled={isLoading} />

          <SectionHeading>Parameters</SectionHeading>

          <LabelledSlider
            label="Monte Carlo Passes"
            value={mcPasses}
            min={5} max={30} step={1}
            onChange={setMcPasses}
            accentClass="accent-blue-500"
          />

          <LabelledSlider
            label="Confidence Threshold"
            value={confThresh}
            min={0} max={100} step={5}
            unit=" %"
            onChange={setConfThresh}
            accentClass="accent-green-500"
          />

          <SectionHeading>Geospatial Context</SectionHeading>

          <div className="space-y-2">
            <div className="space-y-1">
              <label className="text-xs text-gray-400">Latitude</label>
              <input
                type="number"
                step="0.00001"
                placeholder="e.g. 37.77452"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                disabled={isLoading}
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-xs text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
              />
            </div>
            <div className="space-y-1">
              <label className="text-xs text-gray-400">Longitude</label>
              <input
                type="number"
                step="0.00001"
                placeholder="e.g. -122.41941"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                disabled={isLoading}
                className="w-full rounded-lg border border-gray-700 bg-gray-800 px-3 py-2 text-xs text-gray-200 placeholder-gray-600 focus:border-blue-500 focus:outline-none disabled:opacity-50"
              />
            </div>
          </div>

          {/* ── Run button ──────────────────────────────────────────────── */}
          <div className="mt-auto pt-2">
            <button
              onClick={handleRun}
              disabled={!file || isLoading}
              className={[
                "flex w-full items-center justify-center gap-2 rounded-xl px-4 py-3",
                "text-sm font-semibold transition-all duration-200",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500",
                !file || isLoading
                  ? "cursor-not-allowed bg-gray-800 text-gray-500 border border-gray-700"
                  : "bg-blue-600 text-white hover:bg-blue-500 active:scale-[0.98] shadow-lg shadow-blue-900/30",
              ].join(" ")}
            >
              {isLoading ? (
                <>
                  <Spinner size={16} />
                  <span>Analysing…</span>
                </>
              ) : (
                <>
                  <IconTarget className="h-4 w-4" />
                  <span>Run Reconstruction &amp; Analysis</span>
                </>
              )}
            </button>

            {/* Progress hint while loading */}
            {isLoading && (
              <p className="mt-2 text-center text-[10px] text-gray-500 animate-pulse">
                Running {mcPasses} MC-Dropout passes…
              </p>
            )}
          </div>
        </aside>

        {/* ─────────────────────────────────────────────────────────────── */}
        {/* CENTER: WORKSPACE CANVAS                                         */}
        {/* ─────────────────────────────────────────────────────────────── */}
        <main className="flex flex-1 min-w-0 flex-col overflow-y-auto p-4 gap-4">

          {/* Error banner */}
          {pipelineState === "error" && errorMsg && (
            <div className="rounded-xl border border-red-700/60 bg-red-900/20 px-4 py-3 text-sm text-red-300">
              <span className="font-semibold">Pipeline error: </span>{errorMsg}
            </div>
          )}

          {/* Empty state */}
          {pipelineState === "idle" && !result && (
            <div className="flex flex-1 flex-col items-center justify-center gap-4 text-center">
              <div className="rounded-2xl border border-gray-800 bg-gray-900/40 px-10 py-12">
                <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-gray-700 bg-gray-800/60">
                  <svg viewBox="0 0 40 40" fill="none" className="h-8 w-8 text-gray-600">
                    <circle cx="20" cy="20" r="16" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 3" />
                    <circle cx="20" cy="20" r="8"  stroke="currentColor" strokeWidth="1.5" />
                    <circle cx="20" cy="20" r="2"  fill="currentColor" opacity="0.4" />
                  </svg>
                </div>
                <h3 className="text-base font-semibold text-gray-300 mb-1">
                  No scene loaded
                </h3>
                <p className="text-xs text-gray-500 max-w-xs">
                  Upload a single-channel Landsat thermal band (ST_B10) in the sidebar
                  and click <strong className="text-gray-400">Run Reconstruction &amp; Analysis</strong>.
                </p>
              </div>
            </div>
          )}

          {/* Loading skeleton */}
          {isLoading && (
            <div className="space-y-3">
              <div className="h-8 w-48 animate-pulse rounded-lg bg-gray-800" />
              <div className="aspect-video w-full animate-pulse rounded-xl bg-gray-800" />
              <div className="h-4 w-full animate-pulse rounded bg-gray-800" />
            </div>
          )}

          {/* ── Main ImageSlider ────────────────────────────────────────── */}
          {result && !isLoading && (
            <>
              <div className="flex items-center justify-between">
                <h1 className="text-sm font-semibold text-gray-200">
                  Scene Analysis
                  {execMs !== null && (
                    <span className="ml-3 text-xs font-normal text-gray-500">
                      completed in {(execMs / 1000).toFixed(2)} s
                    </span>
                  )}
                </h1>
                <span className="rounded-full border border-green-700/50 bg-green-900/20 px-2.5 py-0.5 text-xs text-green-400">
                  ● {result.metrics.mc_passes_executed} passes
                </span>
              </div>

              <ImageSlider
                thermalSrc={thermalPreviewUrl}
                reconstructedRgbSrc={result.images.reconstructed_rgb}
                uncertaintyHeatmapSrc={result.images.uncertainty_heatmap}
                bboxOverlaySrc={result.images.bbox_overlay}
                detections={filteredDetections}
                imageWidth={256}
                imageHeight={256}
              />
            </>
          )}
        </main>

        {/* ─────────────────────────────────────────────────────────────── */}
        {/* RIGHT: INTELLIGENCE CO-PILOT PANEL                               */}
        {/* ─────────────────────────────────────────────────────────────── */}
        <aside className="flex w-80 shrink-0 flex-col overflow-y-auto border-l border-gray-800 bg-gray-900/60 backdrop-blur-sm">

          {/* ── Telemetry badges ──────────────────────────────────────── */}
          <div className="border-b border-gray-800 p-4 space-y-3">
            <SectionHeading>Pipeline Telemetry</SectionHeading>

            {result ? (
              <div className="grid grid-cols-2 gap-2">
                <TelemetryBadge
                  icon={<IconCpu className="h-4 w-4" />}
                  label="Device"
                  value={result.metrics.device.toUpperCase()}
                  valueClass={result.metrics.device === "cuda" ? "text-green-400" : "text-blue-300"}
                />
                <TelemetryBadge
                  icon={<IconTimer className="h-4 w-4" />}
                  label="Exec Time"
                  value={execMs !== null ? `${(execMs / 1000).toFixed(2)} s` : "—"}
                />
                <TelemetryBadge
                  icon={<IconWave className="h-4 w-4" />}
                  label="Mean Unc."
                  value={formatUncertainty(result.metrics.mean_uncertainty)}
                  valueClass={uncertaintyColour(result.metrics.mean_uncertainty)}
                />
                <TelemetryBadge
                  icon={<IconWave className="h-4 w-4" />}
                  label="Max Unc."
                  value={formatUncertainty(result.metrics.max_uncertainty)}
                  valueClass={uncertaintyColour(result.metrics.max_uncertainty)}
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="h-14 animate-pulse rounded-lg bg-gray-800" />
                ))}
              </div>
            )}
          </div>

          {/* ── Agent / model badge ───────────────────────────────────── */}
          {result && (
            <div className="border-b border-gray-800 px-4 py-3">
              <div className="flex items-center gap-2">
                <div className={[
                  "h-2 w-2 rounded-full",
                  result.agent_meta.used_fallback ? "bg-yellow-400" : "bg-blue-400 animate-pulse",
                ].join(" ")} />
                <span className="text-xs text-gray-400">
                  {result.agent_meta.used_fallback
                    ? "GraniteFallbackAgent (offline)"
                    : "IBM Granite 3.0 — Live inference"}
                </span>
              </div>
              <p className="mt-1 text-[10px] text-gray-600 font-mono truncate">
                {result.agent_meta.model_id}
              </p>
            </div>
          )}

          {/* ── IBM Granite briefing ──────────────────────────────────── */}
          <div className="flex flex-col flex-1 min-h-0 p-4 gap-3">
            <SectionHeading>Intelligence Briefing</SectionHeading>

            {result ? (
              <div className="flex-1 overflow-y-auto rounded-lg border border-gray-800 bg-gray-950/60 p-3 max-h-72">
                <BriefingRenderer text={result.agent_briefing} />
              </div>
            ) : isLoading ? (
              <div className="space-y-2">
                {[100, 90, 95, 70, 85].map((w, i) => (
                  <div
                    key={i}
                    className="h-3 animate-pulse rounded bg-gray-800"
                    style={{ width: `${w}%` }}
                  />
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-gray-800 bg-gray-900/40 px-4 py-6 text-center">
                <p className="text-xs text-gray-600">
                  Briefing will appear here after analysis.
                </p>
              </div>
            )}

            {/* ── Detection table ─────────────────────────────────────── */}
            <SectionHeading>Detections</SectionHeading>

            {result && result.detections.length > 0 ? (
              <div className="overflow-x-auto rounded-lg border border-gray-800">
                <table className="w-full min-w-full border-collapse text-left">
                  <thead>
                    <tr className="border-b border-gray-700 bg-gray-800/60">
                      <th className="py-2 pl-3 pr-2 text-[10px] uppercase tracking-wider text-gray-500">#</th>
                      <th className="py-2 px-2 text-[10px] uppercase tracking-wider text-gray-500">Class</th>
                      <th className="py-2 px-2 text-right text-[10px] uppercase tracking-wider text-gray-500">Raw</th>
                      <th className="py-2 px-2 text-right text-[10px] uppercase tracking-wider text-gray-500">BBox Unc.</th>
                      <th className="py-2 pr-3 text-right text-[10px] uppercase tracking-wider text-gray-500">Adj. Conf</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDetections.map((det, i) => (
                      <DetectionRow key={i} det={det} index={i} />
                    ))}
                  </tbody>
                </table>
                {filteredDetections.length < result.detections.length && (
                  <p className="px-3 py-1.5 text-[10px] text-gray-600 border-t border-gray-800">
                    {result.detections.length - filteredDetections.length} detection(s) hidden by confidence threshold ({confThresh} %)
                  </p>
                )}
              </div>
            ) : result && result.detections.length === 0 ? (
              <div className="rounded-lg border border-gray-800 bg-gray-900/40 px-4 py-5 text-center">
                <p className="text-xs text-gray-500">No objects detected in this scene.</p>
              </div>
            ) : isLoading ? (
              <div className="space-y-1.5">
                {[...Array(4)].map((_, i) => (
                  <div key={i} className="h-8 animate-pulse rounded bg-gray-800" />
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-gray-800 bg-gray-900/40 px-4 py-5 text-center">
                <p className="text-xs text-gray-600">Detection results will appear after analysis.</p>
              </div>
            )}

            {/* ── Retrieved RAG context IDs ────────────────────────────── */}
            {result && result.agent_meta.retrieved_context_ids.length > 0 && (
              <div className="mt-1">
                <p className="mb-1.5 text-[10px] uppercase tracking-wider text-gray-600">
                  RAG Context Retrieved
                </p>
                <div className="flex flex-wrap gap-1">
                  {result.agent_meta.retrieved_context_ids.map((id) => (
                    <span
                      key={id}
                      className="rounded border border-gray-700 bg-gray-800/60 px-1.5 py-0.5 text-[10px] font-mono text-gray-500"
                    >
                      {id}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
