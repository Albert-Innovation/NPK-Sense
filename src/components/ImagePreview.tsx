"use client";
import React, { memo } from "react";
import { Upload, Eye, Loader2, Crosshair, RefreshCw, Play } from "lucide-react";

// ─── Types ────────────────────────────────────────────────────────────────────

export type CalibrationStep = "idle" | "calibrating" | "done";
export type ActivePickMode  = "n" | "p" | "k" | "filler";

interface Point { x: number; y: number }

interface ImagePreviewProps {
  loading:             boolean;
  processedImage:      string | null;
  currentDisplayImage: string | null;
  onToggleStart:       () => void;
  onToggleEnd:         () => void;
  showCompare?:        boolean;
  // Multi-point calibration
  calibrationStep?:    CalibrationStep;
  activePickMode?:     ActivePickMode;
  refNPoints?:         Point[];
  refPPoints?:         Point[];
  refKPoints?:         Point[];
  refFillerPoints?:    Point[];
  onCalibrationClick?: (point: Point) => void;
  onSetPickMode?:      (mode: ActivePickMode) => void;
  onStartCalibration?: () => void;
  onRunCalibration?:   () => void;
  onRecalibrate?:      () => void;
  onUndoLastPoint?:      () => void;
  onClearAllPoints?:     () => void;
  onRemovePoint?:        (mode: ActivePickMode, index: number) => void;
  onCancelCalibration?:  () => void;
  autoRunLabel?:         string;
}

// ─── Style constants ──────────────────────────────────────────────────────────
// Extracting long class strings keeps JSX readable and ensures consistent styles.

const TOGGLE_BASE =
  "flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold " +
  "transition-all select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-white/40";

const TOGGLE_INACTIVE =
  "bg-white/10 text-slate-400 hover:bg-white/20 hover:text-white";

const TOGGLE_N_ACTIVE =
  "bg-slate-200 text-slate-800 ring-2 ring-slate-400 ring-offset-1 ring-offset-black";

const TOGGLE_P_ACTIVE =
  "bg-emerald-100 text-emerald-800 ring-2 ring-emerald-400 ring-offset-1 ring-offset-black";

const TOGGLE_K_ACTIVE =
  "bg-rose-100 text-rose-800 ring-2 ring-rose-400 ring-offset-1 ring-offset-black";

const TOGGLE_F_ACTIVE =
  "bg-amber-100 text-amber-800 ring-2 ring-amber-300 ring-offset-1 ring-offset-black";

const COUNT_BADGE_BASE =
  "px-1.5 py-0.5 rounded-full text-[10px] font-black min-w-[18px] text-center";

// Status badge config keyed by calibration step — avoids three separate conditional blocks.
const STATUS_BADGE_CONFIG: Record<
  CalibrationStep,
  { bg: string; label: (total: number) => string }
> = {
  idle:        { bg: "bg-green-500/90",  label: ()      => "Analysis Complete"           },
  calibrating: { bg: "bg-purple-600/90", label: ()      => "Calibrating…"                },
  done:        { bg: "bg-purple-600/90", label: (total) => `Calibrated (${total} pts)`   },
};

// ─── RefDot ───────────────────────────────────────────────────────────────────
// Memoized so that adding a new point doesn't re-render all existing dots.

const DOT_STYLES: Record<string, { bg: string; ring: string; text: string; label: string }> = {
  n:      { bg: "bg-slate-200",   ring: "ring-slate-400",   text: "text-slate-700",   label: "N" },
  p:      { bg: "bg-emerald-400", ring: "ring-emerald-300", text: "text-emerald-900", label: "P" },
  k:      { bg: "bg-rose-400",    ring: "ring-rose-300",    text: "text-white",       label: "K" },
  filler: { bg: "bg-amber-400",   ring: "ring-amber-300",   text: "text-amber-900",   label: "F" },
};

const RefDot = memo(function RefDot({
  point, index, variant, showDelete, onDelete,
}: {
  point: Point;
  index: number;
  variant: "n" | "p" | "k" | "filler";
  showDelete: boolean;
  onDelete: () => void;
}) {
  const style = DOT_STYLES[variant] ?? DOT_STYLES.n;
  return (
    <div
      style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
      className={[
        "absolute -translate-x-1/2 -translate-y-1/2 z-20",
        "w-5 h-5 rounded-full border-2 border-white shadow-lg ring-2",
        showDelete ? "pointer-events-auto" : "pointer-events-none",
        style.bg, style.ring,
      ].join(" ")}
      title={`${style.label} ref ${index + 1}`}
    >
      <span className={["absolute inset-0 flex items-center justify-center text-[7px] font-black", style.text].join(" ")}>
        {style.label}
      </span>
      {showDelete && (
        <button
          onClick={e => { e.stopPr opagation(); onDelete(); }}
          style={{ top: -5, right: -5 }}
          className="absolute z-30 pointer-events-auto w-3.5 h-3.5 rounded-full bg-white border border-slate-400 flex items-center justify-center text-[8px] font-black text-slate-600 leading-none cursor-pointer hover:bg-red-100 hover:border-red-400 hover:text-red-600 shadow"
          title="Remove point"
        >
          ×
        </button>
      )}
    </div>
  );
});

// ─── ImagePreview ─────────────────────────────────────────────────────────────

export default function ImagePreview({
  loading,
  processedImage,
  currentDisplayImage,
  onToggleStart,
  onToggleEnd,
  showCompare      = true,
  calibrationStep  = "idle",
  activePickMode   = "n",
  refNPoints       = [],
  refPPoints       = [],
  refKPoints       = [],
  refFillerPoints  = [],
  onCalibrationClick,
  onSetPickMode,
  onStartCalibration,
  onRunCalibration,
  onRecalibrate,
  onUndoLastPoint,
  onClearAllPoints,
  onRemovePoint,
  onCancelCalibration,
  autoRunLabel,
}: ImagePreviewProps) {
  const isCalibrating  = calibrationStep === "calibrating";
  const totalRefPoints = refNPoints.length + refPPoints.length + refKPoints.length + refFillerPoints.length;
  const canRunAnalysis = totalRefPoints >= 1;
  const badge          = STATUS_BADGE_CONFIG[calibrationStep];

  // Capture normalized click coordinates relative to the rendered image element.
  const handleImageClick = (e: React.MouseEvent<HTMLImageElement>) => {
    if (!isCalibrating || !onCalibrationClick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(1, (e.clientX - rect.left)  / rect.width));
    const y = Math.max(0, Math.min(1, (e.clientY - rect.top)   / rect.height));
    onCalibrationClick({ x, y });
  };

  return (
    /*
     * flex-col layout: image area (flex-1) + calibration bar (flex-shrink-0).
     * The calibration bar is NOT absolutely positioned — it lives in the normal
     * flow so it naturally pushes the image area up without any overlap.
     */
    <div className="relative flex flex-col flex-1 bg-slate-900 rounded-3xl overflow-hidden shadow-2xl shadow-slate-300 min-h-[500px] border-4 border-white">

      {/* ── Loading overlay ───────────────────────────────────────────────── */}
      {loading && (
        <div className="absolute inset-0 bg-white/90 backdrop-blur-md z-30 flex flex-col items-center justify-center gap-2">
          <Loader2 className="animate-spin text-blue-600" size={48} />
          <span className="font-bold text-lg text-slate-700 tracking-tight">Processing Physics Model…</span>
          <span className="text-sm text-slate-400">Applying shape correction &amp; density</span>
        </div>
      )}

      {/* ── Absolute overlays (badges + floating buttons) ─────────────────── */}
      {processedImage && (
        <>
          {/* Status badge — top left, single block driven by config map */}
          <div
            className={[
              "absolute top-5 left-5 z-20",
              badge.bg,
              "backdrop-blur-md text-white px-3 py-1.5 rounded-full text-[10px]",
              "font-bold flex items-center gap-1.5 shadow-lg",
            ].join(" ")}
          >
            <span className="w-2 h-2 bg-white rounded-full animate-pulse flex-shrink-0" />
            {badge.label(totalRefPoints)}
          </div>

          {/* Hold-to-compare — top right, hidden while calibrating to reduce clutter */}
          {showCompare && !isCalibrating && (
            <button
              onMouseDown={onToggleStart}
              onMouseUp={onToggleEnd}
              onMouseLeave={onToggleEnd}
              onTouchStart={onToggleStart}
              onTouchEnd={onToggleEnd}
              className={[
                "absolute top-5 right-5 z-20",
                "bg-black/60 hover:bg-black/80 active:scale-95",
                "backdrop-blur-md text-white pl-3 pr-4 py-2 rounded-full",
                "text-xs font-bold border border-white/10 transition-all",
                "flex items-center gap-2 select-none shadow-lg",
              ].join(" ")}
            >
              <Eye size={14} className="text-blue-400" />
              Hold to compare
            </button>
          )}

          {/* Calibrate button — bottom-center, idle state only */}
          {calibrationStep === "idle" && (
            <button
              onClick={onStartCalibration}
              className={[
                "absolute bottom-5 left-1/2 -translate-x-1/2 z-20",
                "bg-purple-600/90 hover:bg-purple-500 active:scale-95",
                "backdrop-blur-md text-white px-5 py-2 rounded-full",
                "text-xs font-bold border border-purple-400/30",
                "transition-all flex items-center gap-2 shadow-lg whitespace-nowrap",
              ].join(" ")}
            >
              <Crosshair size={14} />
              Calibrate Colors
            </button>
          )}

          {/* Recalibrate button — bottom-center, done state only */}
          {calibrationStep === "done" && (
            <button
              onClick={onRecalibrate}
              className={[
                "absolute bottom-5 left-1/2 -translate-x-1/2 z-20",
                "bg-black/60 hover:bg-black/80 active:scale-95",
                "backdrop-blur-md text-white px-5 py-2 rounded-full",
                "text-xs font-bold border border-white/10",
                "transition-all flex items-center gap-2 shadow-lg whitespace-nowrap",
              ].join(" ")}
            >
              <RefreshCw size={14} />
              Recalibrate
            </button>
          )}
        </>
      )}

      {/* ── Image area ────────────────────────────────────────────────────── */}
      {/*
       * flex-1 + min-h-0 lets this section shrink correctly when the
       * calibration bar is rendered below it in the flex-col parent.
       */}
      <div className="flex-1 flex items-center justify-center p-5 min-h-0">
        {currentDisplayImage ? (
          /*
           * inline-block wrapper: matches the rendered image dimensions exactly,
           * so percentage-based dot positions (left/top) are relative to the
           * image itself — not the surrounding flex container.
           */
          <div className="relative inline-block leading-[0]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={currentDisplayImage}
              alt="Analysis result"
              draggable={false}
              onClick={handleImageClick}
              className={[
                "relative z-0 max-w-full max-h-[700px] object-contain shadow-2xl rounded-lg select-none",
                isCalibrating ? "cursor-crosshair" : "",
              ].join(" ")}
            />

            {/* N reference dots */}
            {refNPoints.map((pt, i) => (
              <RefDot key={`n-${i}`} point={pt} index={i} variant="n"
                showDelete={isCalibrating} onDelete={() => onRemovePoint?.("n", i)} />
            ))}

            {/* P reference dots */}
            {refPPoints.map((pt, i) => (
              <RefDot key={`p-${i}`} point={pt} index={i} variant="p"
                showDelete={isCalibrating} onDelete={() => onRemovePoint?.("p", i)} />
            ))}

            {/* K reference dots */}
            {refKPoints.map((pt, i) => (
              <RefDot key={`k-${i}`} point={pt} index={i} variant="k"
                showDelete={isCalibrating} onDelete={() => onRemovePoint?.("k", i)} />
            ))}

            {/* Filler reference dots */}
            {refFillerPoints.map((pt, i) => (
              <RefDot key={`f-${i}`} point={pt} index={i} variant="filler"
                showDelete={isCalibrating} onDelete={() => onRemovePoint?.("filler", i)} />
            ))}
          </div>
        ) : (
          /* Empty state */
          <div className="text-center space-y-3 px-4">
            <div className="w-20 h-20 bg-slate-800 rounded-full flex items-center justify-center mx-auto">
              <Upload size={32} className="text-slate-500" />
            </div>
            <h3 className="text-slate-400 font-medium text-lg">Upload an image to start analysis</h3>
            <p className="text-slate-600 text-sm max-w-xs mx-auto">
              Supports JPEG, PNG. Optimized for high-resolution fertilizer images.
            </p>
          </div>
        )}
      </div>

      {/* ── Calibration control bar ───────────────────────────────────────── */}
      {/*
       * In-flow (not absolute) — sits naturally at the bottom of the flex-col
       * parent and pushes the image area up. This prevents any overlap.
       *
       * Layout: [toggle group | flex-1 spacer w/ hint | run button]
       * The flex-1 spacer always exists, keeping Run Analysis anchored right.
       * On mobile the hint text inside the spacer is simply hidden.
       */}
      {isCalibrating && (
        <div className="flex-shrink-0 flex flex-col gap-2 px-4 py-3 bg-black/85 backdrop-blur-sm border-t border-white/5">

          {/* Row 1: 4-button toggle group + Run Analysis */}
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 flex-shrink-0 flex-wrap">

              {/* N mode toggle */}
              <button
                onClick={() => onSetPickMode?.("n")}
                className={`${TOGGLE_BASE} ${activePickMode === "n" ? TOGGLE_N_ACTIVE : TOGGLE_INACTIVE}`}
              >
                <span className="w-2.5 h-2.5 rounded-full bg-slate-300 flex-shrink-0" />
                {`N (${refNPoints.length})`}
              </button>

              {/* P mode toggle */}
              <button
                onClick={() => onSetPickMode?.("p")}
                className={`${TOGGLE_BASE} ${activePickMode === "p" ? TOGGLE_P_ACTIVE : TOGGLE_INACTIVE}`}
              >
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 flex-shrink-0" />
                {`P (${refPPoints.length})`}
              </button>

              {/* K mode toggle */}
              <button
                onClick={() => onSetPickMode?.("k")}
                className={`${TOGGLE_BASE} ${activePickMode === "k" ? TOGGLE_K_ACTIVE : TOGGLE_INACTIVE}`}
              >
                <span className="w-2.5 h-2.5 rounded-full bg-rose-400 flex-shrink-0" />
                {`K (${refKPoints.length})`}
              </button>

              {/* Filler mode toggle */}
              <button
                onClick={() => onSetPickMode?.("filler")}
                className={`${TOGGLE_BASE} ${activePickMode === "filler" ? TOGGLE_F_ACTIVE : TOGGLE_INACTIVE}`}
              >
                <span className="w-2.5 h-2.5 rounded-full bg-amber-400 flex-shrink-0" />
                {`Filler (${refFillerPoints.length})`}
              </button>
            </div>

            <div className="flex-1 min-w-0" />

            {/* Run Analysis */}
            <button
              onClick={onRunCalibration}
              disabled={!canRunAnalysis}
              title={canRunAnalysis ? "Run calibrated analysis" : "Select at least 1 reference point first"}
              className={[
                "flex-shrink-0 flex items-center gap-1.5 px-4 py-1.5 rounded-full",
                "text-xs font-bold transition-all select-none",
                canRunAnalysis
                  ? "bg-purple-600 hover:bg-purple-500 active:scale-95 text-white shadow-md shadow-purple-900/40"
                  : "bg-white/10 text-slate-500 opacity-50 cursor-not-allowed",
              ].join(" ")}
            >
              <Play size={11} />
              Run Analysis
            </button>
          </div>

          {/* Row 2: Undo / Clear All / hint */}
          <div className="flex items-center gap-2">
            <button
              onClick={onUndoLastPoint}
              disabled={
                activePickMode === "n" ? refNPoints.length === 0 :
                activePickMode === "p" ? refPPoints.length === 0 :
                activePickMode === "k" ? refKPoints.length === 0 :
                refFillerPoints.length === 0
              }
              className={[
                "text-[10px] px-2.5 py-1 rounded-full border transition-all select-none",
                "border-white/20 text-slate-400 hover:text-white hover:border-white/40",
                "disabled:opacity-30 disabled:cursor-not-allowed",
              ].join(" ")}
            >
              Undo last
            </button>
            <button
              onClick={onClearAllPoints}
              disabled={totalRefPoints === 0}
              className={[
                "text-[10px] px-2.5 py-1 rounded-full border transition-all select-none",
                "border-white/20 text-slate-400 hover:text-red-400 hover:border-red-400/40",
                "disabled:opacity-30 disabled:cursor-not-allowed",
              ].join(" ")}
            >
              Clear all
            </button>
            {processedImage && onCancelCalibration && (
              <button
                onClick={onCancelCalibration}
                className={[
                  "text-[10px] px-2.5 py-1 rounded-full border transition-all select-none",
                  "border-white/20 text-slate-400 hover:text-amber-400 hover:border-amber-400/40",
                ].join(" ")}
              >
                Cancel
              </button>
            )}
            {autoRunLabel ? (
              <span className="hidden sm:flex items-center gap-1.5 text-[10px] text-purple-300 font-medium ml-1">
                <span className="inline-block w-2.5 h-2.5 border border-purple-400 border-t-transparent rounded-full animate-spin" />
                {autoRunLabel}
              </span>
            ) : (
              <span className="hidden sm:block text-slate-600 text-[10px] truncate ml-1">
                {activePickMode === "n" ? "← click white N prills" :
                 activePickMode === "p" ? "← click green P granules" :
                 activePickMode === "k" ? "← click red K granules" :
                 "← click tan Filler particles"}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
