"""
NPK-Sense Backend API
Description: Edge-AI system for fertilizer nutrient analysis using YOLO11m-seg
and Multi-Point Calibration (K-Nearest Neighbors).
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import json
import math

# ==============================================================================
# Application Setup & Configuration
# ==============================================================================

app = FastAPI(
    title="NPK-Sense API",
    description="Backend API for fertilizer analysis",
    version="1.0.4"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL CONFIGURATION ---
# Two models are available:
#   v2 (default) — npksensev2.pt  — YOLO11m-seg, faster (~30s), balanced accuracy
#   x            — npksenseX.pt   — YOLO11x-seg, slower (~60s), higher accuracy
#
# The active model is selected per-request via the model_name form parameter.
# If the X model fails to load (e.g. file not present), v2 is used as fallback.

MODEL_V2_PATH = "npksensev2.pt"
MODEL_X_PATH  = "npksenseX.pt"

model_v2 = None
model_x  = None

try:
    model_v2 = YOLO(MODEL_V2_PATH)
    print(f"Loaded model v2: {MODEL_V2_PATH}")
except Exception as e:
    print(f"Critical Error: Failed to load v2 model from {MODEL_V2_PATH}. Details: {e}")

try:
    model_x = YOLO(MODEL_X_PATH)
    print(f"Loaded model X: {MODEL_X_PATH}")
except Exception as e:
    print(f"Warning: Failed to load X model from {MODEL_X_PATH}. Details: {e}")

MODELS = {"v2": model_v2, "x": model_x}

# Class ID mapping from YOLO training
CLASS_ID_MAP = {0: 'K', 1: 'N', 2: 'P'}

# Chemical Composition Factors (N-P-K percentages)
NUTRIENT_FACTORS = {
    'N':      {'N': 0.46, 'P': 0.00, 'K': 0.00},  # Urea 46-0-0
    'P':      {'N': 0.18, 'P': 0.46, 'K': 0.00},  # DAP 18-46-0
    'K':      {'N': 0.00, 'P': 0.00, 'K': 0.60},  # MOP 0-0-60
    'Filler': {'N': 0.00, 'P': 0.00, 'K': 0.00},  # Inert material
}

# Physical properties for area-to-mass estimation.
# density: g/cm3
# shape_factor: 1.0 = perfect sphere, <1.0 = flatter/irregular pellet
#
# HOW THESE VALUES WERE DERIVED:
# Calibrated over 4 rounds using a real 15-15-15 fertilizer test image.
#
# Back-calculation method:
#   relative_mass ∝ shape_factor^3 × density
#   correction = (target_ratio / actual_ratio)^(1/3)
#   where ratio = nutrient_X / nutrient_N
#
# Round 1: N=17.0, P=9.9,  K=9.4,  Filler=63.7  → sf P=0.84, K=0.73
# Round 2: N=12.2, P=13.4, K=13.2, Filler=61.2  → sf P=0.81, K=0.71, Filler=0.72
# Round 3: N=13.7, P=14.1, K=14.0, Filler=58.2  → no change (within tolerance)
# Round 4 (3-scan avg): N=15.0, P=13.9, K=13.8, Filler=57.3
#          → sf P=0.83, K=0.73, Filler=0.71
#
# Re-calibrate if camera angle, pellet brand, or image resolution changes.
MATERIAL_PROPS = {
    'N':      {'density': 1.33, 'shape_factor': 1.00},
    'P':      {'density': 1.61, 'shape_factor': 0.83},
    'K':      {'density': 1.98, 'shape_factor': 0.73},
    'Filler': {'density': 2.40, 'shape_factor': 0.71},
}

# ==============================================================================
# YOLO Prior Threshold for 4-Class KNN
# ==============================================================================
#
# When 4-class calibration is active, every pellet is re-classified by KNN
# in LAB color space. However YOLO distinguishes P and K better than color
# alone (it also uses shape and texture). This threshold protects YOLO's
# P/K classification unless the color evidence is strongly different.
#
# How it works:
#   - If YOLO says a pellet is P or K, compute:
#       dist_yolo = LAB distance to the YOLO-assigned class reference
#       dist_best = LAB distance to the KNN-best class reference
#   - If KNN wants to change the label but (dist_yolo - dist_best) < threshold,
#     keep the YOLO label.
#   - If (dist_yolo - dist_best) >= threshold, the color evidence is strong
#     enough to override YOLO.
#
# Tuning guide:
#   Higher value  → trust YOLO more for P/K, KNN rarely overrides
#   Lower value   → trust KNN more, may fix color-based misclassification
#   0             → pure KNN (old behavior, no YOLO prior)
#
# 15.0 LAB units is roughly the color difference between a slightly off-white
# and a cream pellet — meaningful but not overwhelming.
YOLO_TRUST_THRESHOLD = 15.0

# ==============================================================================
# Helper Functions
# ==============================================================================

def bgr_to_base64(img: np.ndarray) -> str:
    """Converts an OpenCV BGR image to a base64-encoded JPEG string."""
    _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')


def resize_for_response(img: np.ndarray, max_dim: int = 800) -> np.ndarray:
    """Resizes an image while maintaining aspect ratio to reduce network payload."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def four_point_transform(image: np.ndarray, pts: list) -> np.ndarray:
    """Applies a perspective transform to obtain a top-down view of an image region."""
    rect = np.array(pts, dtype="float32")
    (tl, tr, br, bl) = rect

    widthA  = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB  = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA  = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB  = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# ==============================================================================
# Lighting Normalization
# ==============================================================================

def normalize_lighting(img: np.ndarray) -> np.ndarray:
    """
    Normalizes uneven lighting using CLAHE on the L channel only.

    Why:
        Camera photos often have vignetting or uneven lighting. The same pellet
        color looks different at different positions in the frame, which confuses
        LAB-based classification.

    How:
        Convert to LAB, apply CLAHE only on L (lightness), leave A and B (color)
        untouched, convert back. YOLO still sees the original image — normalization
        is only used for LAB color sampling.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_normalized = cv2.merge([clahe.apply(l), a, b])
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

# ==============================================================================
# Mass Estimation
# ==============================================================================

def estimate_relative_mass(area_2d: float, props: dict) -> float:
    """
    Estimates relative pellet mass from its 2D detected area.

    Why not pow(area_2d, 1.5):
        area_2d is in pixel^2. Raising to 1.5 gives pixel^3 only for a perfect
        sphere at 1:1 pixel scale, which is never true. shape_factor applied
        after volume estimation behaves non-linearly and is hard to calibrate.

    Correct approach:
        1. r = sqrt(area / pi)          — radius from 2D cross-section
        2. r_eff = r * shape_factor     — flatten/round the pellet
        3. V = (4/3) * pi * r_eff^3    — sphere volume
        4. mass = V * density           — relative mass (consistent for ratios)
    """
    if area_2d <= 0:
        return 0.0
    radius_px        = math.sqrt(area_2d / math.pi)
    effective_radius = radius_px * props['shape_factor']
    volume           = (4.0 / 3.0) * math.pi * (effective_radius ** 3)
    return volume * props['density']

# ==============================================================================
# Otsu Auto Classification (N vs Filler fallback)
# ==============================================================================

def _classify_by_otsu(class1_data: list) -> dict:
    """
    Classifies class-1 pellets (N or Filler) using Otsu threshold on L channel.

    Why better than hardcoded L > 140:
        The 140 threshold was tuned for 15-15-15 lighting. For low-N formulas
        (e.g. 10-4-24), many Filler pellets exceed 140 and get misclassified as N.
        Otsu finds the optimal split point from the actual L distribution per image.
        Implemented in NumPy — no extra library needed.
    """
    labels = {}
    if not class1_data:
        return labels
    if len(class1_data) == 1:
        labels[class1_data[0]["index"]] = "N"
        return labels

    l_values = np.array([d["lab"][0] for d in class1_data], dtype=np.float32)

    if np.ptp(l_values) < 1.0:
        threshold = 140.0
    else:
        sorted_l    = np.sort(np.unique(l_values))
        best_thresh = 140.0
        best_var    = float("inf")
        for t in sorted_l[:-1]:
            group_n      = l_values[l_values >  t]
            group_filler = l_values[l_values <= t]
            if len(group_n) == 0 or len(group_filler) == 0:
                continue
            w_n        = len(group_n)      / len(l_values)
            w_filler   = len(group_filler) / len(l_values)
            within_var = w_n * np.var(group_n) + w_filler * np.var(group_filler)
            if within_var < best_var:
                best_var    = within_var
                best_thresh = float(t)
        threshold = best_thresh

    for d in class1_data:
        labels[d["index"]] = "N" if d["lab"][0] > threshold else "Filler"
    return labels

# ==============================================================================
# Calibration Helpers
# ==============================================================================

def find_clicked_pellet_lab(clicked_norm: dict, contour_data: list, img_shape: tuple):
    """
    Returns the mean LAB of the pellet that contains the clicked point.
    Falls back to the nearest contour centroid if the click misses all contours.
    """
    h, w = img_shape[:2]
    cx = int(clicked_norm['x'] * w)
    cy = int(clicked_norm['y'] * h)

    for d in contour_data:
        cnt_f = d['cnt'].astype(np.float32)
        if cv2.pointPolygonTest(cnt_f, (float(cx), float(cy)), False) >= 0:
            return d['lab']

    min_dist    = float('inf')
    closest_lab = None
    for d in contour_data:
        M = cv2.moments(d['cnt'])
        if M['m00'] > 0:
            mcx  = M['m10'] / M['m00']
            mcy  = M['m01'] / M['m00']
            dist = math.sqrt((cx - mcx) ** 2 + (cy - mcy) ** 2)
            if dist < min_dist:
                min_dist    = dist
                closest_lab = d['lab']
    return closest_lab


def find_clicked_pellets_lab(clicked_norms: list, contour_data: list, img_shape: tuple) -> list:
    """Batch version of find_clicked_pellet_lab."""
    return [
        lab for p in clicked_norms
        if (lab := find_clicked_pellet_lab(p, contour_data, img_shape)) is not None
    ]

# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/")
async def root():
    return {"message": "NPK-Sense API is up and running.", "version": "1.0.4"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/model_info")
async def model_info():
    """Returns info for all loaded models."""
    result = {}
    if model_v2:
        result["v2"] = {
            "path": MODEL_V2_PATH,
            "names": model_v2.names,
            "nc": model_v2.model.nc,
            "description": "YOLO11m-seg — Standard model, faster (~30s)",
        }
    if model_x:
        result["x"] = {
            "path": MODEL_X_PATH,
            "names": model_x.names,
            "nc": model_x.model.nc,
            "description": "YOLO11x-seg — Advanced model, more accurate (~60s)",
        }
    return result


@app.get("/debug_last")
async def debug_last():
    """Returns class distribution from the most recent inference for diagnostics."""
    return getattr(app, "_last_debug", {"message": "No inference run yet"})


@app.post("/analyze_interactive")
async def analyze_interactive(
    file:              UploadFile = File(...),
    points:            str = Form(None),
    ref_n_points:      str = Form(None),
    ref_filler_points: str = Form(None),
    ref_p_points:      str = Form(None),
    ref_k_points:      str = Form(None),
    mode:              str = Form("analyze"),
    model_name:        str = Form("v2"),
):
    """
    Main analysis endpoint.
    mode='crop_only' — perspective-warp and return the cropped image only.
    mode='analyze'   — full YOLO inference + classification + mass estimation.
    model_name='v2'  — use Standard model (default).
    model_name='x'   — use Advanced model.
    """
    try:
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # ── 1. Perspective Transform ───────────────────────────────────────────
        raw_cropped = None
        if points:
            try:
                pts_norm  = json.loads(points)
                h, w      = img.shape[:2]
                pts_pixel = [[p['x'] * w, p['y'] * h] for p in pts_norm]
                img         = four_point_transform(img, pts_pixel)
                raw_cropped = img.copy()
            except Exception as e:
                print(f"Warp Error: {e}")

        # ── Early exit: crop only ──────────────────────────────────────────────
        if mode == "crop_only" and raw_cropped is not None:
            resized = resize_for_response(raw_cropped, max_dim=800)
            return JSONResponse({
                "raw_cropped_b64": bgr_to_base64(resized),
                "message": "Crop successful. Awaiting calibration.",
            })

        # ── 2. Lighting Normalization ──────────────────────────────────────────
        # Done AFTER warping so we only normalize the region of interest.
        # YOLO uses the original image; LAB extraction uses the normalized one.
        img_normalized = normalize_lighting(img)

        # ── 3. YOLO Inference ──────────────────────────────────────────────────
        # Select the requested model; fall back to v2 if unavailable.
        selected_model = MODELS.get(model_name) or model_v2
        if selected_model is None:
            return JSONResponse(status_code=503, content={"error": "No model available."})

        results = selected_model.predict(
            img, verbose=False, max_det=3000, conf=0.15, iou=0.6, imgsz=1024
        )
        lab_img = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2LAB)

        mass_scores = {'N': 0.0, 'P': 0.0, 'K': 0.0, 'Filler': 0.0}
        dark_bg     = cv2.addWeighted(img, 0.4, np.zeros_like(img), 0.6, 0)
        thick_lines = np.zeros_like(img)
        thin_lines  = np.zeros_like(img)
        method      = "otsu_auto"

        if results[0].masks is not None:
            masks_xy    = results[0].masks.xy
            classes_ids = results[0].boxes.cls.cpu().numpy()

            # ── PASS 1: Feature Extraction ────────────────────────────────────
            # Split detections into:
            #   class1_data    — YOLO class 1 (N or Filler, ambiguous)
            #   other_contours — YOLO class 0 (K) and class 2 (P)
            class1_data    = []
            other_contours = []

            for i, polygon in enumerate(masks_xy):
                if len(polygon) < 3:
                    continue
                cls_id  = int(classes_ids[i])
                cnt     = np.array(polygon, dtype=np.int32)
                area_2d = cv2.contourArea(cnt)

                mask       = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.drawContours(mask, [cnt], -1, 255, -1)
                kernel     = np.ones((3, 3), np.uint8)
                mask_inner = cv2.erode(mask, kernel, iterations=1)
                if cv2.countNonZero(mask_inner) == 0:
                    mask_inner = mask
                mean_lab = cv2.mean(lab_img, mask=mask_inner)
                lab_val  = [mean_lab[0], mean_lab[1], mean_lab[2]]

                if cls_id == 1:
                    class1_data.append({
                        'index': i, 'cnt': cnt, 'area': area_2d, 'lab': lab_val,
                    })
                else:
                    other_contours.append({
                        'index': i, 'cls_id': cls_id, 'cnt': cnt,
                        'area': area_2d, 'lab': lab_val,
                    })

            # ── PASS 1.5: Build unified contour list with YOLO class labels ───
            all_contour_data = []

            for d in class1_data:
                all_contour_data.append({
                    'index': d['index'], 'cnt': d['cnt'], 'area': d['area'],
                    'lab': d['lab'],
                    'yolo_class': 'N_or_Filler',  # ambiguous — will be resolved
                })

            for d in other_contours:
                yolo_cls = 'K' if d['cls_id'] == 0 else 'P'
                all_contour_data.append({
                    'index': d['index'], 'cnt': d['cnt'], 'area': d['area'],
                    'lab': d['lab'],
                    'yolo_class': yolo_cls,  # YOLO is confident here
                })

            # ── PASS 2: Classification ────────────────────────────────────────
            all_labels = {}

            ref_n_list = json.loads(ref_n_points)       if ref_n_points       else []
            ref_f_list = json.loads(ref_filler_points)  if ref_filler_points  else []
            ref_p_list = json.loads(ref_p_points)       if ref_p_points       else []
            ref_k_list = json.loads(ref_k_points)       if ref_k_points       else []

            labs_n = find_clicked_pellets_lab(ref_n_list, all_contour_data, img.shape) if ref_n_list else []
            labs_f = find_clicked_pellets_lab(ref_f_list, all_contour_data, img.shape) if ref_f_list else []
            labs_p = find_clicked_pellets_lab(ref_p_list, all_contour_data, img.shape) if ref_p_list else []
            labs_k = find_clicked_pellets_lab(ref_k_list, all_contour_data, img.shape) if ref_k_list else []

            non_empty_classes = sum(1 for labs in [labs_n, labs_f, labs_p, labs_k] if labs)

            if non_empty_classes >= 2:
                # ── 4-Class KNN with YOLO Prior ───────────────────────────────
                #
                # Problem with pure KNN:
                #   KNN uses LAB color only. P and K often have similar colors
                #   (both are granular, off-white to pale pink). YOLO uses shape
                #   and texture too, so it's more reliable for P vs K.
                #   Pure KNN reclassifying everything causes P/K accuracy to DROP
                #   compared to letting YOLO decide.
                #
                # Solution — YOLO prior for P and K:
                #   If YOLO says a pellet is P or K AND the KNN color evidence
                #   for changing it is not strong enough (< YOLO_TRUST_THRESHOLD),
                #   keep the YOLO label. Only override YOLO when the color
                #   difference is unambiguously large.
                #
                #   N and Filler are both YOLO class 1 (ambiguous), so KNN is
                #   always used to separate them.
                arrs = {}
                if labs_n: arrs['N']      = [np.array(l, dtype=np.float32) for l in labs_n]
                if labs_p: arrs['P']      = [np.array(l, dtype=np.float32) for l in labs_p]
                if labs_k: arrs['K']      = [np.array(l, dtype=np.float32) for l in labs_k]
                if labs_f: arrs['Filler'] = [np.array(l, dtype=np.float32) for l in labs_f]

                for d in all_contour_data:
                    lab = np.array(d['lab'], dtype=np.float32)

                    # Compute LAB distance to every available reference class
                    distances = {
                        cls_name: min(float(np.linalg.norm(lab - a)) for a in ref_list)
                        for cls_name, ref_list in arrs.items()
                    }
                    best_knn = min(distances, key=distances.get)
                    yolo_cls = d.get('yolo_class', '')

                    if yolo_cls in ('P', 'K') and yolo_cls in distances:
                        # YOLO has a confident class (P or K).
                        # Only let KNN override if color evidence is strong.
                        dist_yolo = distances[yolo_cls]
                        dist_best = distances[best_knn]
                        if best_knn != yolo_cls and (dist_yolo - dist_best) < YOLO_TRUST_THRESHOLD:
                            # Color difference is small — trust YOLO
                            all_labels[d['index']] = yolo_cls
                        else:
                            # Color difference is large enough — trust KNN
                            all_labels[d['index']] = best_knn
                    else:
                        # N_or_Filler: YOLO cannot distinguish, use KNN
                        all_labels[d['index']] = best_knn

                method = (
                    f"4class_yolo_prior("
                    f"n={len(arrs.get('N', []))}, "
                    f"p={len(arrs.get('P', []))}, "
                    f"k={len(arrs.get('K', []))}, "
                    f"f={len(arrs.get('Filler', []))}, "
                    f"threshold={YOLO_TRUST_THRESHOLD}, "
                    f"model={model_name})"
                )

            else:
                # ── Fallback: N/Filler KNN or Otsu + YOLO for P/K ────────────
                class1_labels = {}

                if ref_n_list and ref_f_list and class1_data:
                    labs_n_fb = find_clicked_pellets_lab(ref_n_list, class1_data, img.shape)
                    labs_f_fb = find_clicked_pellets_lab(ref_f_list, class1_data, img.shape)
                    if labs_n_fb and labs_f_fb:
                        arrs_n = [np.array(l, dtype=np.float32) for l in labs_n_fb]
                        arrs_f = [np.array(l, dtype=np.float32) for l in labs_f_fb]
                        for d in class1_data:
                            lab    = np.array(d['lab'], dtype=np.float32)
                            dist_n = min(float(np.linalg.norm(lab - a)) for a in arrs_n)
                            dist_f = min(float(np.linalg.norm(lab - a)) for a in arrs_f)
                            class1_labels[d['index']] = 'N' if dist_n <= dist_f else 'Filler'
                        method = f"knn_nf(n={len(labs_n_fb)},f={len(labs_f_fb)})+yolo_pk+model={model_name}"
                    else:
                        class1_labels = _classify_by_otsu(class1_data)
                        method = f"otsu_auto(cal_missed)+yolo_pk+model={model_name}"
                else:
                    class1_labels = _classify_by_otsu(class1_data)
                    method = f"otsu_auto+yolo_pk+model={model_name}"

                for d in class1_data:
                    all_labels[d['index']] = class1_labels.get(d['index'], 'N')
                for d in other_contours:
                    all_labels[d['index']] = CLASS_ID_MAP.get(d['cls_id'], 'Unknown')

            # ── PASS 3: Mass Calculation & Visualization ──────────────────────
            COLOR_MAP = {
                'N':      (200, 200, 200),
                'P':      (50,  255, 50),
                'K':      (50,  50,  255),
                'Filler': (0,   255, 255),
            }

            for d in all_contour_data:
                final_name = all_labels.get(d['index'], 'N')
                cnt        = d['cnt']
                area_2d    = d['area']

                color    = COLOR_MAP.get(final_name, (200, 200, 200))
                props    = MATERIAL_PROPS.get(final_name, {'density': 1.0, 'shape_factor': 1.0})
                rel_mass = estimate_relative_mass(area_2d, props)
                factors  = NUTRIENT_FACTORS.get(final_name, {'N': 0, 'P': 0, 'K': 0})

                mass_scores['N'] += rel_mass * factors['N']
                mass_scores['P'] += rel_mass * factors['P']
                mass_scores['K'] += rel_mass * factors['K']
                total_nutrient    = factors['N'] + factors['P'] + factors['K']
                mass_scores['Filler'] += rel_mass * (1.0 - total_nutrient)

                cv2.drawContours(thick_lines, [cnt], -1, color, 3)
                contrast = (0, 0, 0) if final_name == 'N' else (255, 255, 255)
                cv2.drawContours(thin_lines,  [cnt], -1, contrast, 1)

        # ── Blend visualization ────────────────────────────────────────────────
        final_vis = cv2.add(dark_bg, thick_lines)
        mask_thin = cv2.cvtColor(thin_lines, cv2.COLOR_BGR2GRAY) > 0
        final_vis[mask_thin] = thin_lines[mask_thin]

        response = {
            "image_b64": bgr_to_base64(resize_for_response(final_vis, max_dim=800)),
            "areas":     mass_scores,
            "method":    method,
        }

        if raw_cropped is not None:
            response["raw_cropped_b64"] = bgr_to_base64(
                resize_for_response(raw_cropped, max_dim=800)
            )

        # ── Debug snapshot ────────────────────────────────────────────────────
        if results[0].masks is not None:
            app._last_debug = {
                "model_used":       model_name,
                "total_detections": len(masks_xy),
                "class1_N_Filler":  len(class1_data),
                "other_K_P":        len(other_contours),
                "cls_breakdown": {
                    "K(0)": int(sum(1 for c in classes_ids if int(c) == 0)),
                    "N(1)": int(sum(1 for c in classes_ids if int(c) == 1)),
                    "P(2)": int(sum(1 for c in classes_ids if int(c) == 2)),
                },
            }
        else:
            app._last_debug = {"model_used": model_name, "total_detections": 0}

        return JSONResponse(response)

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)