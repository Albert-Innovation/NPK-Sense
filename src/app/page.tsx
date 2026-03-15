"use client";

import React, { useState, useEffect, useMemo, useCallback, Suspense } from "react";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { useSearchParams } from "next/navigation";
import { CheckCircle2, Zap, Microscope, Calculator as CalcIcon, Camera, RotateCcw, CheckCheck } from "lucide-react";
import ControlPanel from "@/components/ControlPanel";
import ImagePreview, { CalibrationStep, ActivePickMode } from "@/components/ImagePreview";
import StatCard from "@/components/StatCard";
import PerspectiveCropper from "@/components/PerspectiveCropper";

ChartJS.register(ArcElement, Tooltip, Legend);

// ─── API endpoints ────────────────────────────────────────────────────────────

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://thadzy-npksense.hf.space";
const API_URL = `${BASE_URL}/analyze_interactive`;
const HEALTH_URL = `${BASE_URL}/health`;

// ─── Constants ────────────────────────────────────────────────────────────────

// Minimum number of scans required before averaging.
// 3 scans gives enough statistical spread to reduce sampling variance
// without making the workflow too tedious for the user.
const REQUIRED_SCANS = 3;

// ─── Types ────────────────────────────────────────────────────────────────────

type Point = { x: number; y: number };
type BackendStatus = "unknown" | "warming" | "ready" | "error";
type MassScores = { N: number; P: number; K: number; Filler: number };

// Stores the result of one completed scan so we can average across scans later.
interface ScanResult {
  scanIndex: number;
  massScores: MassScores;
  previewImage: string;
}

// ─── Helper: average mass scores across multiple scans ───────────────────────

/**
 * Averages N, P, K, Filler scores across all completed scan results.
 * This reduces sampling variance caused by random pellet distribution
 * differences between photos of the same mixture.
 */
function averageMassScores(results: ScanResult[]): MassScores {
  if (results.length === 0) return { N: 0, P: 0, K: 0, Filler: 0 };
  const sum = results.reduce(
    (acc, r) => ({
      N: acc.N + r.massScores.N,
      P: acc.P + r.massScores.P,
      K: acc.K + r.massScores.K,
      Filler: acc.Filler + r.massScores.Filler,
    }),
    { N: 0, P: 0, K: 0, Filler: 0 }
  );
  const n = results.length;
  return { N: sum.N / n, P: sum.P / n, K: sum.K / n, Filler: sum.Filler / n };
}

// ─── DashboardContent ─────────────────────────────────────────────────────────

function DashboardContent() {
  const searchParams = useSearchParams();

  // Image & upload state
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [originalImage, setOriginalImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [croppedRawImage, setCroppedRawImage] = useState<string | null>(null);
  const [currentDisplayImage, setCurrentDisplayImage] = useState<string | null>(null);

  // Perspective-crop state
  const [isCropping, setIsCropping] = useState(false);
  const [lastCropPoints, setLastCropPoints] = useState<Point[] | null>(null);

  // Backend health
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("unknown");

  // Fertilizer targets & measurements
  const [totalWeight, setTotalWeight] = useState(100);
  const [targets, setTargets] = useState({ N: 15, P: 15, K: 15, Filler: 55 });

  // Multi-scan state
  const [scanResults, setScanResults] = useState<ScanResult[]>([]);
  const [massScores, setMassScores] = useState<MassScores>({ N: 0, P: 0, K: 0, Filler: 0 });
  const [scanningComplete, setScanningComplete] = useState(false);

  // Per-scan UI calibration state — what the user is currently clicking
  const [calibrationStep, setCalibrationStep] = useState<CalibrationStep>("idle");
  const [activePickMode, setActivePickMode] = useState<ActivePickMode>("n");
  const [refNPoints, setRefNPoints] = useState<Point[]>([]);
  const [refPPoints, setRefPPoints] = useState<Point[]>([]);
  const [refKPoints, setRefKPoints] = useState<Point[]>([]);
  const [refFillerPoints, setRefFillerPoints] = useState<Point[]>([]);

  // Session-level calibration: locked in after the FIRST scan's calibration run.
  // Every subsequent scan reuses these points automatically so the classification
  // boundary is identical across all scans in the session.
  // Only cleared when the user explicitly starts a new session.
  const [sessionRefNPoints, setSessionRefNPoints] = useState<Point[]>([]);
  const [sessionRefPPoints, setSessionRefPPoints] = useState<Point[]>([]);
  const [sessionRefKPoints, setSessionRefKPoints] = useState<Point[]>([]);
  const [sessionRefFillerPoints, setSessionRefFillerPoints] = useState<Point[]>([]);

  // ── Backend health polling ──────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    const poll = async (attempts = 0) => {
      if (cancelled) return;
      setBackendStatus("warming");
      try {
        const r = await fetch(HEALTH_URL);
        if (cancelled) return;
        if (r.ok) { setBackendStatus("ready"); return; }
        if (r.status === 503 && attempts < 20) setTimeout(() => poll(attempts + 1), 3000);
        else setBackendStatus("error");
      } catch {
        if (!cancelled && attempts < 20) setTimeout(() => poll(attempts + 1), 3000);
        else if (!cancelled) setBackendStatus("error");
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []);

  // ── URL params → pre-fill targets ──────────────────────────────────────────

  useEffect(() => {
    const n = parseFloat(searchParams.get("n") || "0");
    const p = parseFloat(searchParams.get("p") || "0");
    const k = parseFloat(searchParams.get("k") || "0");
    const w = parseFloat(searchParams.get("weight") || "100");
    if (searchParams.get("n") || searchParams.get("p") || searchParams.get("k")) {
      setTargets({ N: n, P: p, K: k, Filler: Math.max(0, 100 - (n + p + k)) });
      setTotalWeight(w);
      setTimeout(scrollToAnalyzer, 500);
    }
  }, [searchParams]);

  // ── Derived weights & chart data ───────────────────────────────────────────

  const { finalWeights, pieChartData } = useMemo(() => {
    const total = Object.values(massScores).reduce((a, b) => a + b, 0);
    const factor = total > 0 ? totalWeight / total : 0;
    const fw = {
      N: massScores.N * factor,
      P: massScores.P * factor,
      K: massScores.K * factor,
      Filler: massScores.Filler * factor,
    };
    return {
      finalWeights: fw,
      pieChartData: {
        labels: ["N", "Filler", "P", "K"],
        datasets: [{
          data: [fw.N, fw.Filler, fw.P, fw.K],
          backgroundColor: ["#94a3b8", "#facc15", "#10b981", "#ef4444"],
          borderWidth: 0,
        }],
      },
    };
  }, [massScores, totalWeight]);

  // ── Handlers ───────────────────────────────────────────────────────────────

  const scrollToAnalyzer = () => {
    document.getElementById("analyzer-section")?.scrollIntoView({ behavior: "smooth" });
  };

  const retryBackend = useCallback(() => {
    setBackendStatus("warming");
    fetch(HEALTH_URL)
      .then(r => setBackendStatus(r.ok ? "ready" : "error"))
      .catch(() => setBackendStatus("error"));
  }, []);

  const handleTargetChange = useCallback((key: string, value: number) => {
    const newVal = isNaN(value) ? 0 : value;
    setTargets(prev => {
      if (key === "Filler") return { ...prev, Filler: newVal };
      const updated = { ...prev, [key]: newVal };
      const nutrientSum = (key === "N" ? newVal : prev.N)
        + (key === "P" ? newVal : prev.P)
        + (key === "K" ? newVal : prev.K);
      return { ...updated, Filler: Math.max(0, 100 - nutrientSum) };
    });
  }, []);

  const resetCalibration = () => {
    setRefNPoints([]);
    setRefPPoints([]);
    setRefKPoints([]);
    setRefFillerPoints([]);
    setActivePickMode("n");
  };

  // Resets the entire scan session so the user can start a new set of 3 scans.
  // Also clears session-level calibration so the next session calibrates fresh.
  const handleResetSession = useCallback(() => {
    setScanResults([]);
    setMassScores({ N: 0, P: 0, K: 0, Filler: 0 });
    setScanningComplete(false);
    setFile(null);
    setProcessedImage(null);
    setCroppedRawImage(null);
    setCurrentDisplayImage(null);
    setLastCropPoints(null);
    setCalibrationStep("idle");
    resetCalibration();
    setSessionRefNPoints([]);
    setSessionRefPPoints([]);
    setSessionRefKPoints([]);
    setSessionRefFillerPoints([]);
  }, []);

  // ── File upload ────────────────────────────────────────────────────────────

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (!selectedFile) return;

    // Auto-reset if a full session is already complete
    if (scanningComplete) handleResetSession();

    setFile(selectedFile);
    setProcessedImage(null);
    setCroppedRawImage(null);
    setCurrentDisplayImage(null);
    setLastCropPoints(null);
    setCalibrationStep("idle");
    resetCalibration();

    const reader = new FileReader();
    reader.onload = (ev) => {
      const imgUrl = ev.target?.result as string | undefined;
      if (!imgUrl) return;
      setOriginalImage(imgUrl);
      setIsCropping(true);
    };
    reader.readAsDataURL(selectedFile);
    e.target.value = "";
  };

  const handleCropConfirm = (points: Point[]) => {
    setIsCropping(false);
    setLastCropPoints(points);
    if (file) {
      if (sessionRefNPoints.length > 0) {
        // Session calibration already exists from scan 1.
        // Skip crop_only mode and go straight to full analysis using the
        // saved reference points — no user interaction needed.
        analyzeImage(file, points, sessionRefNPoints, sessionRefPPoints, sessionRefKPoints, sessionRefFillerPoints, false);
        setCalibrationStep("done");
      } else {
        // First scan: request crop_only so the user can calibrate.
        analyzeImage(file, points, [], [], [], [], true);
      }
      scrollToAnalyzer();
    }
  };

  // ── Core analysis call ─────────────────────────────────────────────────────

  const analyzeImage = async (
    selectedFile: File,
    points: Point[] | null = null,
    refN: Point[] = [],
    refP: Point[] = [],
    refK: Point[] = [],
    refFiller: Point[] = [],
    enterCalibration = false,
  ) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    const cropPoints = points ?? lastCropPoints;
    if (cropPoints) formData.append("points", JSON.stringify(cropPoints));
    if (refN.length > 0) formData.append("ref_n_points", JSON.stringify(refN));
    if (refP.length > 0) formData.append("ref_p_points", JSON.stringify(refP));
    if (refK.length > 0) formData.append("ref_k_points", JSON.stringify(refK));
    if (refFiller.length > 0) formData.append("ref_filler_points", JSON.stringify(refFiller));

    formData.append("mode", enterCalibration ? "crop_only" : "analyze");

    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });

      if (res.status === 503) {
        setBackendStatus("warming");
        alert("The AI backend is waking up. Please wait ~30 seconds and try again.");
        return;
      }
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        throw new Error(`Server error ${res.status}: ${errText}`);
      }

      const data = await res.json();

      if (enterCalibration) {
        const rawCrop = data.raw_cropped_b64
          ? `data:image/jpeg;base64,${data.raw_cropped_b64}`
          : null;
        if (rawCrop) {
          setCroppedRawImage(rawCrop);
          setCurrentDisplayImage(rawCrop);
          resetCalibration();
          setCalibrationStep("calibrating");
        }
      } else {
        const procImg = `data:image/jpeg;base64,${data.image_b64}`;
        const rawCrop = data.raw_cropped_b64
          ? `data:image/jpeg;base64,${data.raw_cropped_b64}`
          : null;

        setProcessedImage(procImg);
        if (rawCrop) setCroppedRawImage(rawCrop);
        setCurrentDisplayImage(procImg);

        if (data.areas) {
          // Use functional update to get the latest scanResults in this closure
          setScanResults(prev => {
            const newResult: ScanResult = {
              scanIndex: prev.length,
              massScores: data.areas as MassScores,
              previewImage: procImg,
            };
            const updated = [...prev, newResult];

            // Compute and apply the running average immediately
            const averaged = averageMassScores(updated);
            setMassScores(averaged);

            if (updated.length >= REQUIRED_SCANS) {
              setScanningComplete(true);
            }
            return updated;
          });
        }
      }

      setBackendStatus("ready");
    } catch (err) {
      console.error("analyzeImage error:", err);
      alert(`Analysis failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  // ── Calibration handlers ───────────────────────────────────────────────────

  const handleStartCalibration = () => {
    if (croppedRawImage) setCurrentDisplayImage(croppedRawImage);
    resetCalibration();
    setCalibrationStep("calibrating");
  };

  const handleCalibrationClick = (point: Point) => {
    if (calibrationStep !== "calibrating") return;
    if (activePickMode === "n")          setRefNPoints(prev => [...prev, point]);
    else if (activePickMode === "p")     setRefPPoints(prev => [...prev, point]);
    else if (activePickMode === "k")     setRefKPoints(prev => [...prev, point]);
    else                                 setRefFillerPoints(prev => [...prev, point]);
  };

  const handleRunCalibration = () => {
    const totalPoints = refNPoints.length + refPPoints.length + refKPoints.length + refFillerPoints.length;
    if (!file || totalPoints < 1) return;
    setCalibrationStep("done");

    // Lock these picks as the session reference on the very first scan.
    // All subsequent scans will reuse them automatically, keeping the
    // classification boundary identical across all scans.
    if (sessionRefNPoints.length === 0 && totalPoints > 0) {
      setSessionRefNPoints(refNPoints);
      setSessionRefPPoints(refPPoints);
      setSessionRefKPoints(refKPoints);
      setSessionRefFillerPoints(refFillerPoints);
    }

    analyzeImage(file, lastCropPoints, refNPoints, refPPoints, refKPoints, refFillerPoints);
  };

  const handleRecalibrate = () => {
    setActivePickMode("n");
    setCalibrationStep("calibrating");
    if (croppedRawImage) setCurrentDisplayImage(croppedRawImage);
  };

  const handleUndoLastPoint = () => {
    if (activePickMode === "n")      setRefNPoints(prev => prev.slice(0, -1));
    else if (activePickMode === "p") setRefPPoints(prev => prev.slice(0, -1));
    else if (activePickMode === "k") setRefKPoints(prev => prev.slice(0, -1));
    else                             setRefFillerPoints(prev => prev.slice(0, -1));
  };

  const handleClearAllPoints = () => {
    setRefNPoints([]); setRefPPoints([]);
    setRefKPoints([]); setRefFillerPoints([]);
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  const scansRemaining = Math.max(0, REQUIRED_SCANS - scanResults.length);

  return (
    <div className="bg-white font-sans selection:bg-blue-100">

      {isCropping && originalImage && (
        <PerspectiveCropper
          imageSrc={originalImage}
          onConfirm={handleCropConfirm}
          onCancel={() => { setIsCropping(false); setFile(null); }}
        />
      )}

      {/* Hero */}
      <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-4 overflow-hidden py-20">
        <div className="absolute inset-0 w-full h-full pointer-events-none">
          <div className="absolute inset-0 bg-white" />
          <div className="absolute -top-[10%] -right-[10%] w-[70vw] h-[70vw] rounded-full bg-gradient-to-b from-cyan-100 via-blue-200 to-transparent opacity-70 blur-[80px]" />
          <div className="absolute top-[0%] -left-[10%] w-[60vw] h-[60vw] rounded-full bg-gradient-to-r from-indigo-100 via-purple-100 to-transparent opacity-70 blur-[100px]" />
          <div className="absolute -bottom-[20%] left-[20%] w-[60vw] h-[60vw] rounded-full bg-blue-50 opacity-80 blur-[120px]" />
        </div>

        <div className="relative z-10 text-center max-w-5xl mx-auto space-y-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white border border-blue-100 shadow-sm text-sm font-semibold text-blue-700 mb-4">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-500 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-600" />
            </span>
            AI-Powered Fertilizer Analysis 2.0
          </div>

          {backendStatus === "warming" && (
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700 text-xs font-medium mb-2">
              <span className="animate-spin inline-block w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full" />
              AI backend waking up... (may take ~30s)
            </div>
          )}
          {backendStatus === "error" && (
            <div onClick={retryBackend} className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-50 border border-red-200 text-red-600 text-xs font-medium mb-2 cursor-pointer hover:bg-red-100 transition-colors">
              Backend unreachable - click to retry
            </div>
          )}
          {backendStatus === "ready" && (
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium mb-2">
              Backend ready
            </div>
          )}

          <h1 className="text-5xl md:text-7xl font-black text-slate-900 tracking-tight leading-tight">
            Precision Farming <br />
            Starts with <span className="text-blue-600">Perfect NPK.</span>
          </h1>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <button onClick={scrollToAnalyzer} className="px-8 py-4 bg-blue-600 hover:bg-blue-700 active:scale-95 text-white font-bold rounded-2xl shadow-xl flex items-center justify-center gap-2 text-lg transition-all">
              <Microscope size={24} /> Start Analyzing
            </button>
          </div>

          <div className="pt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
            <FeatureCard icon={<Zap className="w-6 h-6" />} title="Instant AI Analysis" desc="Detects N, P, K particles in milliseconds." />
            <FeatureCard icon={<CheckCircle2 className="w-6 h-6 text-emerald-500" />} title="Physics Engine" desc="Calculates weight based on volume and density." />
            <FeatureCard icon={<CalcIcon className="w-6 h-6 text-purple-500" />} title="Reverse Recipe" desc="Reverse engineering your mix recipe." />
          </div>
        </div>
      </section>

      {/* Dashboard */}
      <div id="analyzer-section" className="min-h-screen py-20 bg-white border-t border-slate-100 relative z-20">
        <div className="max-w-7xl mx-auto px-4 lg:px-8">
          <div className="mb-12 text-center">
            <h2 className="text-3xl font-black text-slate-900">Analysis Dashboard</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div className="lg:col-span-4 space-y-6">
              <ControlPanel
                file={file}
                totalWeight={totalWeight}
                targets={targets}
                pieChartData={pieChartData}
                onFileUpload={handleFileUpload}
                onWeightChange={setTotalWeight}
                onTargetChange={handleTargetChange}
              />
            </div>

            <div className="lg:col-span-8 space-y-6 flex flex-col">

              {/* Multi-scan progress */}
              <ScanProgressBar
                scanResults={scanResults}
                requiredScans={REQUIRED_SCANS}
                scanningComplete={scanningComplete}
                loading={loading}
                onReset={handleResetSession}
              />

              {/* Cancel & Start Over — visible during an active (incomplete) multi-scan session */}
              {scanResults.length >= 1 && !scanningComplete && (
                <button
                  onClick={handleResetSession}
                  className="w-full px-4 py-2.5 rounded-2xl border-2 border-red-300 text-red-600 text-sm font-semibold hover:bg-red-50 active:scale-[0.99] transition-all"
                >
                  Cancel &amp; Start Over
                </button>
              )}

              <ImagePreview
                loading={loading}
                processedImage={processedImage}
                currentDisplayImage={currentDisplayImage}
                onToggleStart={() => { if (croppedRawImage) setCurrentDisplayImage(croppedRawImage); }}
                onToggleEnd={() => { if (processedImage) setCurrentDisplayImage(processedImage); }}
                calibrationStep={calibrationStep}
                activePickMode={activePickMode}
                refNPoints={refNPoints}
                refPPoints={refPPoints}
                refKPoints={refKPoints}
                refFillerPoints={refFillerPoints}
                onCalibrationClick={handleCalibrationClick}
                onSetPickMode={setActivePickMode}
                onStartCalibration={handleStartCalibration}
                onRunCalibration={handleRunCalibration}
                onRecalibrate={handleRecalibrate}
                onUndoLastPoint={handleUndoLastPoint}
                onClearAllPoints={handleClearAllPoints}
              />

              {/* Session calibration locked indicator — shown from scan 2 onward */}
              {sessionRefNPoints.length > 0 && !scanningComplete && scanResults.length > 0 && (
                <div className="flex items-center gap-3 px-5 py-3 bg-violet-50 border border-violet-200 rounded-2xl text-xs text-violet-700 font-medium">
                  <span className="shrink-0 w-2 h-2 rounded-full bg-violet-400" />
                  <span>
                    Using calibration from scan 1 ({[
                      sessionRefNPoints.length > 0 && `${sessionRefNPoints.length}N`,
                      sessionRefPPoints.length > 0 && `${sessionRefPPoints.length}P`,
                      sessionRefKPoints.length > 0 && `${sessionRefKPoints.length}K`,
                      sessionRefFillerPoints.length > 0 && `${sessionRefFillerPoints.length}F`,
                    ].filter(Boolean).join(" + ")} pts)
                    — boundary is locked for consistent classification.
                  </span>
                  <button
                    onClick={() => { setSessionRefNPoints([]); setSessionRefPPoints([]); setSessionRefKPoints([]); setSessionRefFillerPoints([]); handleRecalibrate(); }}
                    className="ml-auto text-xs text-violet-500 hover:text-violet-800 underline underline-offset-2 transition-colors whitespace-nowrap"
                  >
                    Override
                  </button>
                </div>
              )}

              {/* Next scan prompt */}
              {calibrationStep === "done" && !scanningComplete && (
                <div className="flex items-center gap-3 px-5 py-4 bg-blue-50 border border-blue-200 rounded-2xl text-sm text-blue-700 font-medium">
                  <Camera size={18} className="shrink-0" />
                  <span>
                    Scan {scanResults.length} of {REQUIRED_SCANS} recorded.
                    {scansRemaining > 0
                      ? ` Shake the sample and upload ${scansRemaining} more photo${scansRemaining > 1 ? "s" : ""} to improve accuracy.`
                      : " Processing final average..."}
                  </span>
                </div>
              )}

              {/* Completion banner */}
              {scanningComplete && (
                <div className="flex items-center gap-3 px-5 py-4 bg-emerald-50 border border-emerald-200 rounded-2xl text-sm text-emerald-700 font-semibold">
                  <CheckCheck size={18} className="shrink-0" />
                  <span>All {REQUIRED_SCANS} scans averaged. Results are now statistically stable.</span>
                  <button
                    onClick={handleResetSession}
                    className="ml-auto flex items-center gap-1.5 text-xs font-semibold text-emerald-600 hover:text-emerald-800 transition-colors"
                  >
                    <RotateCcw size={13} /> New session
                  </button>
                </div>
              )}

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="N (Urea)" subLabel="46-0-0" value={finalWeights.N} total={totalWeight} target={targets.N} color="text-slate-600" barColor="bg-slate-400" />
                <StatCard label="P (DAP)" subLabel="18-46-0" value={finalWeights.P} total={totalWeight} target={targets.P} color="text-emerald-600" barColor="bg-emerald-500" />
                <StatCard label="K (Potash)" subLabel="0-0-60" value={finalWeights.K} total={totalWeight} target={targets.K} color="text-rose-600" barColor="bg-rose-500" />
                <StatCard label="Filler" subLabel="Inert" value={finalWeights.Filler} total={totalWeight} target={targets.Filler} color="text-amber-600" barColor="bg-amber-400" />
              </div>

              {/* Per-scan breakdown */}
              {scanResults.length > 0 && (
                <ScanBreakdownTable scanResults={scanResults} massScores={massScores} />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── ScanProgressBar ──────────────────────────────────────────────────────────

interface ScanProgressBarProps {
  scanResults: ScanResult[];
  requiredScans: number;
  scanningComplete: boolean;
  loading: boolean;
  onReset: () => void;
}

function ScanProgressBar({ scanResults, requiredScans, scanningComplete, loading, onReset }: ScanProgressBarProps) {
  return (
    <div className="bg-white border border-slate-200 rounded-2xl px-5 py-4 shadow-sm">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold text-slate-700">Multi-Scan Progress</span>
          <span className="text-xs text-slate-400">({requiredScans} scans required)</span>
        </div>
        {scanResults.length > 0 && !loading && (
          <button onClick={onReset} className="flex items-center gap-1 text-xs text-slate-400 hover:text-red-500 transition-colors">
            <RotateCcw size={11} /> Reset
          </button>
        )}
      </div>

      <div className="flex items-center gap-2">
        {Array.from({ length: requiredScans }).map((_, i) => {
          const done = i < scanResults.length;
          const active = i === scanResults.length && loading;
          return (
            <React.Fragment key={i}>
              <div className={`
                flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold border-2 transition-all
                ${done ? "bg-emerald-500 border-emerald-500 text-white"
                  : active ? "bg-blue-50 border-blue-400 text-blue-600 animate-pulse"
                    : "bg-slate-50 border-slate-200 text-slate-400"}
              `}>
                {done ? <CheckCheck size={13} /> : i + 1}
              </div>
              {i < requiredScans - 1 && (
                <div className={`flex-1 h-0.5 rounded-full transition-all ${i < scanResults.length ? "bg-emerald-300" : "bg-slate-100"}`} />
              )}
            </React.Fragment>
          );
        })}
        <span className="ml-3 text-xs font-medium text-slate-500">
          {scanningComplete
            ? "Averaged"
            : loading
              ? `Processing scan ${scanResults.length + 1}...`
              : scanResults.length === 0
                ? "Upload first photo to begin"
                : `${scanResults.length}/${requiredScans} — shake sample, upload next photo`}
        </span>
      </div>
    </div>
  );
}

// ─── ScanBreakdownTable ───────────────────────────────────────────────────────

interface ScanBreakdownTableProps {
  scanResults: ScanResult[];
  massScores: MassScores;
}

function ScanBreakdownTable({ scanResults, massScores }: ScanBreakdownTableProps) {
  const toPercent = (scores: MassScores) => {
    const total = scores.N + scores.P + scores.K + scores.Filler;
    if (total === 0) return { N: 0, P: 0, K: 0, Filler: 0 };
    return {
      N: (scores.N / total) * 100,
      P: (scores.P / total) * 100,
      K: (scores.K / total) * 100,
      Filler: (scores.Filler / total) * 100,
    };
  };

  const avgPct = toPercent(massScores);

  return (
    <div className="bg-white border border-slate-100 rounded-2xl overflow-hidden shadow-sm">
      <div className="px-5 py-3 border-b border-slate-100 bg-slate-50">
        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">Per-Scan Breakdown</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-slate-400 font-semibold uppercase tracking-wider border-b border-slate-100">
              <td className="px-5 py-2">Scan</td>
              <td className="px-4 py-2 text-right text-slate-500">N%</td>
              <td className="px-4 py-2 text-right text-emerald-600">P%</td>
              <td className="px-4 py-2 text-right text-rose-500">K%</td>
              <td className="px-4 py-2 text-right text-amber-500">Filler%</td>
            </tr>
          </thead>
          <tbody>
            {scanResults.map((r, idx) => {
              const pct = toPercent(r.massScores);
              return (
                <tr key={idx} className="border-b border-slate-50 hover:bg-slate-50 transition-colors">
                  <td className="px-5 py-2.5 text-slate-600 font-medium">Scan {idx + 1}</td>
                  <td className="px-4 py-2.5 text-right text-slate-600">{pct.N.toFixed(1)}</td>
                  <td className="px-4 py-2.5 text-right text-emerald-600">{pct.P.toFixed(1)}</td>
                  <td className="px-4 py-2.5 text-right text-rose-500">{pct.K.toFixed(1)}</td>
                  <td className="px-4 py-2.5 text-right text-amber-500">{pct.Filler.toFixed(1)}</td>
                </tr>
              );
            })}
            <tr className="bg-blue-50 font-bold">
              <td className="px-5 py-2.5 text-blue-700">Average</td>
              <td className="px-4 py-2.5 text-right text-slate-700">{avgPct.N.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-right text-emerald-700">{avgPct.P.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-right text-rose-600">{avgPct.K.toFixed(1)}</td>
              <td className="px-4 py-2.5 text-right text-amber-600">{avgPct.Filler.toFixed(1)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ─── FeatureCard ──────────────────────────────────────────────────────────────

interface FeatureCardProps { icon: React.ReactNode; title: string; desc: string }

function FeatureCard({ icon, title, desc }: FeatureCardProps) {
  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm text-left">
      <div className="mb-4 bg-blue-50 w-12 h-12 rounded-xl flex items-center justify-center text-blue-600">{icon}</div>
      <h3 className="font-bold text-slate-800 text-lg mb-3">{title}</h3>
      <p className="text-slate-500 text-sm">{desc}</p>
    </div>
  );
}

// ─── Root export ──────────────────────────────────────────────────────────────

export default function NPKSenseDashboard() {
  return (
    <Suspense fallback={
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    }>
      <DashboardContent />
    </Suspense>
  );
}