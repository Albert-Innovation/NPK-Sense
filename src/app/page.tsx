"use client";

import React, { useState, useEffect, useMemo, useCallback, Suspense } from "react";
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { useSearchParams } from "next/navigation";
import { CheckCircle2, Zap, Microscope, Calculator as CalcIcon } from "lucide-react";
import ControlPanel from "@/components/ControlPanel";
import ImagePreview, { CalibrationStep, ActivePickMode } from "@/components/ImagePreview";
import StatCard from "@/components/StatCard";
import PerspectiveCropper from "@/components/PerspectiveCropper";

ChartJS.register(ArcElement, Tooltip, Legend);

// ─── API endpoints ────────────────────────────────────────────────────────────

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "https://thadzy-npksense.hf.space";
const API_URL = `${BASE_URL}/analyze_interactive`;
const HEALTH_URL = `${BASE_URL}/health`;

// ─── Types ────────────────────────────────────────────────────────────────────

type Point = { x: number; y: number };
type BackendStatus = "unknown" | "warming" | "ready" | "error";

// แทรกฟังก์ชันนี้ไว้ด้านบนสุด (ใต้บรรทัด import หรือก่อนเริ่ม function DashboardContent)
/**
 * Resizes an image file on the client-side using HTML5 Canvas.
 * @param {File} file - Original image file.
 * @param {number} maxWidth - Maximum width allowed.
 * @returns {Promise<File>} Resized file.
 */
const resizeImageClientSide = (file: File, maxWidth: number = 1024): Promise<File> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target?.result as string;
      img.onload = () => {
        const canvas = document.createElement('canvas');
        let newWidth = img.width;
        let newHeight = img.height;

        if (img.width > maxWidth) {
          newWidth = maxWidth;
          newHeight = Math.round((img.height * maxWidth) / img.width);
        }

        canvas.width = newWidth;
        canvas.height = newHeight;
        const ctx = canvas.getContext('2d');
        if (!ctx) return reject(new Error("Canvas failed"));

        ctx.drawImage(img, 0, 0, newWidth, newHeight);
        canvas.toBlob((blob) => {
          if (blob) resolve(new File([blob], file.name, { type: 'image/jpeg' }));
          else reject(new Error("Blob failed"));
        }, 'image/jpeg', 0.8);
      };
      img.onerror = reject;
    };
    reader.onerror = reject;
  });
};

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
  const [massScores, setMassScores] = useState({ N: 0, P: 0, K: 0, Filler: 0 });

  // Multi-point calibration state
  const [calibrationStep, setCalibrationStep] = useState<CalibrationStep>("idle");
  const [activePickMode, setActivePickMode] = useState<ActivePickMode>("n");
  const [refNPoints, setRefNPoints] = useState<Point[]>([]);
  const [refFillerPoints, setRefFillerPoints] = useState<Point[]>([]);

  // ── Backend health polling ──────────────────────────────────────────────────
  // HF Spaces free tier sleeps after inactivity and returns 503 while waking up.

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

  // ── URL params → pre-fill targets (from calculator deep-link) ──────────────

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
  // Memoized so they don't re-compute on unrelated state changes.

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

  // When N/P/K targets change, auto-compute Filler so the total stays at 100%.
  // When Filler is edited directly, accept it as a manual override.
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
    setRefFillerPoints([]);
    setActivePickMode("n");
  };

  // ── File upload ────────────────────────────────────────────────────────────

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedOriginalFile = e.target.files?.[0];
    if (!selectedOriginalFile) return;

    try {
      // ทำการย่อขนาดภาพก่อนทำงานอื่น
      const selectedFile = await resizeImageClientSide(selectedOriginalFile, 1024);

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
        setIsCropping(true); // เปิดหน้าต่างลากมุมทันที
      };
      // อ่านไฟล์ที่ถูกย่อขนาดแล้ว
      reader.readAsDataURL(selectedFile);
      e.target.value = "";
    } catch (error) {
      console.error("Error resizing image:", error);
      alert("Image processing failed.");
    }
  };

  const handleCropConfirm = (points: Point[]) => {
    setIsCropping(false);
    setLastCropPoints(points);
    if (file) {
      // Pass enterCalibration=true so the user is forced to calibrate
      // before any annotated result is displayed.
      analyzeImage(file, points, [], [], true);
      scrollToAnalyzer();
    }
  };

  // ── Core analysis call ─────────────────────────────────────────────────────

  /**
   * Triggers the backend analysis or image cropping process.
   * Handles both 'crop_only' early exits and full YOLO inference responses.
   * * @param {File} selectedFile - The uploaded image file.
   * @param {Point[] | null} points - Coordinates for the perspective transform (warping).
   * @param {Point[]} refN - Selected LAB reference points for Nitrogen.
   * @param {Point[]} refFiller - Selected LAB reference points for Filler.
   * @param {boolean} enterCalibration - If true, requests only the cropped image from backend (bypasses YOLO).
   */
  const analyzeImage = async (
    selectedFile: File,
    points: Point[] | null = null,
    refN: Point[] = [],
    refFiller: Point[] = [],
    enterCalibration = false,
  ) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    const cropPoints = points ?? lastCropPoints;
    if (cropPoints) formData.append("points", JSON.stringify(cropPoints));
    if (refN.length > 0) formData.append("ref_n_points", JSON.stringify(refN));
    if (refFiller.length > 0) formData.append("ref_filler_points", JSON.stringify(refFiller));

    // 1. กำหนด Mode การทำงานส่งไปให้ Backend
    if (enterCalibration) {
      formData.append("mode", "crop_only");
    } else {
      formData.append("mode", "analyze");
    }

    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });

      if (res.status === 503) {
        setBackendStatus("warming");
        alert("The AI backend is waking up (cold start). Please wait ~30 seconds and try again.");
        return;
      }
      if (!res.ok) {
        const errText = await res.text().catch(() => res.statusText);
        throw new Error(`Server error ${res.status}: ${errText}`);
      }

      const data = await res.json();

      // 2. แยกการจัดการข้อมูลตอบกลับ (Response Handling) ตาม Mode
      if (enterCalibration) {
        // กรณี Crop อย่างเดียว Backend จะส่งมาแค่ raw_cropped_b64
        const rawCrop = data.raw_cropped_b64
          ? `data:image/jpeg;base64,${data.raw_cropped_b64}`
          : null;

        if (rawCrop) {
          setCroppedRawImage(rawCrop);
          setCurrentDisplayImage(rawCrop); // แสดงภาพดิบทันที
          resetCalibration();
          setCalibrationStep("calibrating"); // เปิดแถบเครื่องมือจิ้มสี
        }
      } else {
        // กรณี Analyze เต็มรูปแบบ Backend จะส่งมาครบทั้งภาพที่ประมวลผลแล้วและค่าสัดส่วน
        const procImg = `data:image/jpeg;base64,${data.image_b64}`;
        const rawCrop = data.raw_cropped_b64
          ? `data:image/jpeg;base64,${data.raw_cropped_b64}`
          : null;

        setProcessedImage(procImg);
        if (rawCrop) setCroppedRawImage(rawCrop);

        setCurrentDisplayImage(procImg); // แสดงภาพที่มี Mask สีต่างๆ
        if (data.areas) {
          setMassScores(data.areas); // อัปเดตตัวเลขเปอร์เซ็นต์
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
    // Switch to the raw (un-annotated) image so the user sees true pellet colours.
    if (croppedRawImage) setCurrentDisplayImage(croppedRawImage);
    resetCalibration();
    setCalibrationStep("calibrating");
  };

  const handleCalibrationClick = (point: Point) => {
    if (calibrationStep !== "calibrating") return;
    if (activePickMode === "n") setRefNPoints(prev => [...prev, point]);
    else setRefFillerPoints(prev => [...prev, point]);
  };

  const handleRunCalibration = () => {
    if (!file || refNPoints.length < 1 || refFillerPoints.length < 1) return;
    setCalibrationStep("done");
    analyzeImage(file, lastCropPoints, refNPoints, refFillerPoints);
  };

  const handleRecalibrate = () => {
    // Re-enter picking mode without wiping existing points (user may want to add more).
    setActivePickMode("n");
    setCalibrationStep("calibrating");
    if (croppedRawImage) setCurrentDisplayImage(croppedRawImage);
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="bg-white font-sans selection:bg-blue-100">

      {isCropping && originalImage && (
        <PerspectiveCropper
          imageSrc={originalImage}
          onConfirm={handleCropConfirm}
          onCancel={() => { setIsCropping(false); setFile(null); }}
        />
      )}

      {/* ── Hero section ──────────────────────────────────────────────────── */}
      <section className="relative min-h-[90vh] flex flex-col items-center justify-center px-4 overflow-hidden py-20">

        {/* Background gradient blobs */}
        <div className="absolute inset-0 w-full h-full pointer-events-none">
          <div className="absolute inset-0 bg-white" />
          <div className="absolute -top-[10%] -right-[10%] w-[70vw] h-[70vw] rounded-full bg-gradient-to-b from-cyan-100 via-blue-200 to-transparent opacity-70 blur-[80px]" />
          <div className="absolute top-[0%] -left-[10%] w-[60vw] h-[60vw] rounded-full bg-gradient-to-r from-indigo-100 via-purple-100 to-transparent opacity-70 blur-[100px]" />
          <div className="absolute -bottom-[20%] left-[20%] w-[60vw] h-[60vw] rounded-full bg-blue-50 opacity-80 blur-[120px]" />
        </div>

        <div className="relative z-10 text-center max-w-5xl mx-auto space-y-10">

          {/* Live indicator pill */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white border border-blue-100 shadow-sm text-sm font-semibold text-blue-700 mb-4">
            <span className="flex h-2 w-2 relative">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-500 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-600" />
            </span>
            AI-Powered Fertilizer Analysis 2.0
          </div>

          {/* Backend status badge */}
          {backendStatus === "warming" && (
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 border border-amber-200 text-amber-700 text-xs font-medium mb-2">
              <span className="animate-spin inline-block w-3 h-3 border-2 border-amber-400 border-t-transparent rounded-full" />
              AI backend waking up… (may take ~30s)
            </div>
          )}
          {backendStatus === "error" && (
            <div
              onClick={retryBackend}
              className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-red-50 border border-red-200 text-red-600 text-xs font-medium mb-2 cursor-pointer hover:bg-red-100 transition-colors"
            >
              ⚠ Backend unreachable — click to retry
            </div>
          )}
          {backendStatus === "ready" && (
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700 text-xs font-medium mb-2">
              ✓ Backend ready
            </div>
          )}

          <h1 className="text-5xl md:text-7xl font-black text-slate-900 tracking-tight leading-tight">
            Precision Farming <br />
            Starts with <span className="text-blue-600">Perfect NPK.</span>
          </h1>

          <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
            <button
              onClick={scrollToAnalyzer}
              className="px-8 py-4 bg-blue-600 hover:bg-blue-700 active:scale-95 text-white font-bold rounded-2xl shadow-xl flex items-center justify-center gap-2 text-lg transition-all"
            >
              <Microscope size={24} /> Start Analyzing
            </button>
          </div>

          <div className="pt-16 grid grid-cols-1 md:grid-cols-3 gap-6">
            <FeatureCard icon={<Zap className="w-6 h-6" />} title="Instant AI Analysis" desc="Detects N, P, K particles in milliseconds." />
            <FeatureCard icon={<CheckCircle2 className="w-6 h-6 text-emerald-500" />} title="Physics Engine" desc="Calculates weight based on volume & density." />
            <FeatureCard icon={<CalcIcon className="w-6 h-6 text-purple-500" />} title="Reverse Recipe" desc="Reverse engineering your mix recipe." />
          </div>
        </div>
      </section>

      {/* ── Analysis dashboard ─────────────────────────────────────────────── */}
      <div id="analyzer-section" className="min-h-screen py-20 bg-white border-t border-slate-100 relative z-20">
        <div className="max-w-7xl mx-auto px-4 lg:px-8">

          <div className="mb-12 text-center">
            <h2 className="text-3xl font-black text-slate-900">Analysis Dashboard</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

            {/* Left column: controls */}
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

            {/* Right column: image preview + stat cards */}
            <div className="lg:col-span-8 space-y-6 flex flex-col">
              <ImagePreview
                loading={loading}
                processedImage={processedImage}
                currentDisplayImage={currentDisplayImage}
                onToggleStart={() => { if (croppedRawImage) setCurrentDisplayImage(croppedRawImage); }}
                onToggleEnd={() => { if (processedImage) setCurrentDisplayImage(processedImage); }}
                calibrationStep={calibrationStep}
                activePickMode={activePickMode}
                refNPoints={refNPoints}
                refFillerPoints={refFillerPoints}
                onCalibrationClick={handleCalibrationClick}
                onSetPickMode={setActivePickMode}
                onStartCalibration={handleStartCalibration}
                onRunCalibration={handleRunCalibration}
                onRecalibrate={handleRecalibrate}
              />

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="N (Urea)" subLabel="46-0-0" value={finalWeights.N} total={totalWeight} target={targets.N} color="text-slate-600" barColor="bg-slate-400" />
                <StatCard label="P (DAP)" subLabel="18-46-0" value={finalWeights.P} total={totalWeight} target={targets.P} color="text-emerald-600" barColor="bg-emerald-500" />
                <StatCard label="K (Potash)" subLabel="0-0-60" value={finalWeights.K} total={totalWeight} target={targets.K} color="text-rose-600" barColor="bg-rose-500" />
                <StatCard label="Filler" subLabel="Inert" value={finalWeights.Filler} total={totalWeight} target={targets.Filler} color="text-amber-600" barColor="bg-amber-400" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── FeatureCard ──────────────────────────────────────────────────────────────

interface FeatureCardProps { icon: React.ReactNode; title: string; desc: string }

function FeatureCard({ icon, title, desc }: FeatureCardProps) {
  return (
    <div className="bg-white p-6 rounded-2xl border border-slate-100 shadow-sm text-left">
      <div className="mb-4 bg-blue-50 w-12 h-12 rounded-xl flex items-center justify-center text-blue-600">
        {icon}
      </div>
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
