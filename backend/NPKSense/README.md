---
title: NPKSense
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# NPKSense Backend API

> FastAPI backend for the NPKSense fertilizer composition analyzer.
> Performs perspective correction, YOLO11 instance segmentation, KNN color calibration, and physics-based mass estimation.

**Version:** 1.0.4

---

## Table of Contents

1. [Overview](#overview)
2. [Tech Stack](#tech-stack)
3. [File Structure](#file-structure)
4. [Getting Started](#getting-started)
5. [API Reference](#api-reference)
6. [Processing Pipeline](#processing-pipeline)
7. [Algorithms In Detail](#algorithms-in-detail)
8. [Configuration & Constants](#configuration--constants)
9. [YOLO Model Files](#yolo-model-files)
10. [Docker & Deployment](#docker--deployment)
11. [Known Limitations & TODOs](#known-limitations--todos)

---

## Overview

The NPKSense backend receives a fertilizer sample image from the frontend, processes it through a multi-stage computer vision pipeline, and returns:

- The **mass percentage** of each class: Nitrogen (N), Phosphorus (P), Potassium (K), and Filler
- A **base64-encoded annotated image** showing the detected and classified pellets

The pipeline uses two techniques in tandem:
- **YOLO11m-seg / YOLO11x-seg** — instance segmentation to detect individual pellets and classify K and P (shape-based)
- **KNN color classifier** — trained on user-clicked reference pellets during calibration to distinguish N from Filler (color-based)

This hybrid approach compensates for the fact that P and K have visually distinctive shapes that YOLO handles well, while N and Filler are shape-similar but differ in color.

---

## Tech Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI |
| ASGI server | Uvicorn |
| Computer vision | OpenCV (opencv-python-headless) |
| ML inference | Ultralytics YOLO11 |
| Numerical computing | NumPy |
| Runtime | Python 3.12 |
| Containerization | Docker (python:3.12-slim) |

---

## File Structure

```
backend/NPKSense/
├── main.py           # Entire backend — FastAPI app, all endpoints, all algorithms
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container definition for HuggingFace Spaces deployment
├── npksensev2.pt     # YOLO11m-seg weights (model v2 — default)
└── npksenseX.pt      # YOLO11x-seg weights (model x — higher accuracy)
```

All logic lives in `main.py`. There are no sub-modules or separate files.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Both YOLO weight files (`npksensev2.pt`, `npksenseX.pt`) placed in the same directory as `main.py`
- Optional: CUDA-compatible GPU for faster inference (CPU works but is slow)

### Install & Run

```bash
cd backend/NPKSense

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
# or equivalently:
uvicorn main:app --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Connect Frontend

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `npksense-web/.env.local`, then run `npm run dev`.

---

## API Reference

### `GET /`
Basic status check.

**Response:**
```json
{
  "message": "NPK-Sense API is up and running.",
  "version": "1.0.4"
}
```

---

### `GET /health`
Liveness probe. Used by the frontend to show the connection status badge.

**Response:**
```json
{ "status": "ok" }
```

---

### `GET /model_info`
Returns metadata about the available models and their class definitions.

**Response:**
```json
{
  "v2": {
    "name": "YOLO11m-seg",
    "classes": 3,
    "description": "Balanced speed and accuracy"
  },
  "x": {
    "name": "YOLO11x-seg",
    "classes": 3,
    "description": "Higher accuracy, slower inference"
  }
}
```

---

### `GET /debug_last`
Returns debug statistics from the most recent inference call. Useful for diagnosing low-accuracy results without re-uploading.

**Response:**
```json
{
  "model_used": "v2",
  "total_detections": 312,
  "class1_N_Filler": 198,
  "other_K_P": 114,
  "cls_breakdown": { "K": 57, "N": 145, "P": 57, "Filler": 53 }
}
```

---

### `POST /analyze_interactive`

Main analysis endpoint. Accepts `multipart/form-data`.

**Request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File (JPEG/PNG) | Yes | Raw fertilizer sample image |
| `mode` | string | Yes | `"crop_only"` or `"analyze"` |
| `points` | JSON string | Yes | 4 corner coordinates (normalized 0–1): `[{"x": 0.1, "y": 0.1}, ...]` — order: top-left, top-right, bottom-right, bottom-left |
| `model_name` | string | No | `"v2"` (default) or `"x"` |
| `ref_n_points` | JSON string | No | Reference clicks for Nitrogen: `[{"x": 0.2, "y": 0.3}, ...]` |
| `ref_p_points` | JSON string | No | Reference clicks for Phosphorus |
| `ref_k_points` | JSON string | No | Reference clicks for Potassium |
| `ref_filler_points` | JSON string | No | Reference clicks for Filler |

**Notes on `mode`:**
- `crop_only`: Applies perspective transform and returns the cropped image only. YOLO is not run. Use this after the user sets crop points, before calibration.
- `analyze`: Runs the full pipeline. Returns annotated image + mass percentages.

**Response:**

```json
{
  "image_b64": "<base64-encoded JPEG — annotated with colored pellet outlines>",
  "raw_cropped_b64": "<base64-encoded JPEG — perspective-corrected crop, no annotations>",
  "areas": {
    "N": 28.3,
    "P": 14.1,
    "K": 21.5,
    "Filler": 36.1
  },
  "method": "4class_yolo_prior(n=4, p=2, k=2, f=3, threshold=15.0, model=v2)"
}
```

- `areas` values sum to 100.0 (mass percentages)
- `raw_cropped_b64` is always returned; `image_b64` is only present in `analyze` mode
- `method` is a debug string describing which classification strategy was used

**Example curl — crop only:**
```bash
curl -X POST http://localhost:8000/analyze_interactive \
  -F "file=@photo.jpg" \
  -F "mode=crop_only" \
  -F 'points=[{"x":0.1,"y":0.1},{"x":0.9,"y":0.1},{"x":0.9,"y":0.9},{"x":0.1,"y":0.9}]' \
  -F "model_name=v2"
```

**Example curl — full analysis:**
```bash
curl -X POST http://localhost:8000/analyze_interactive \
  -F "file=@photo.jpg" \
  -F "mode=analyze" \
  -F 'points=[{"x":0.1,"y":0.1},{"x":0.9,"y":0.1},{"x":0.9,"y":0.9},{"x":0.1,"y":0.9}]' \
  -F 'ref_n_points=[{"x":0.2,"y":0.3},{"x":0.5,"y":0.5}]' \
  -F 'ref_p_points=[{"x":0.7,"y":0.2}]' \
  -F 'ref_k_points=[{"x":0.3,"y":0.7}]' \
  -F 'ref_filler_points=[{"x":0.6,"y":0.6}]' \
  -F "model_name=v2"
```

---

## Processing Pipeline

The following steps execute inside a single call to `POST /analyze_interactive`:

```
1. Decode uploaded image (cv2.imdecode)
        │
        ▼
2. Perspective Transform
   Convert normalized 4-point coords → pixel coords
   cv2.getPerspectiveTransform + cv2.warpPerspective
   → top-down orthogonal view of sample plate
        │
        ├─── (mode=crop_only) → return raw_cropped_b64 and exit
        │
        ▼
3. Lighting Normalization (for color sampling only)
   BGR → LAB color space
   Apply CLAHE on L channel only (preserves hue)
   → used only for KNN color sampling, NOT for YOLO input
        │
        ▼
4. YOLO Inference
   Input: original (pre-CLAHE) image
   Model: YOLO11m-seg or YOLO11x-seg
   Params: conf=0.15, iou=0.6, imgsz=1024, max_det=3000
   Classes: {0: K, 1: N/Filler (ambiguous), 2: P}
   Output: polygon masks + class IDs per pellet
        │
        ▼
5. Feature Extraction (per pellet)
   - Extract polygon contour from mask
   - Compute 2D area (cv2.contourArea)
   - Sample mean LAB value from CLAHE-normalized image
     (eroded inner mask to avoid boundary noise)
   - Split into two groups:
       class1_data  → YOLO class 1 (N or Filler)
       other_contours → YOLO class 0 (K) and class 2 (P)
        │
        ▼
6. Classification
   (see "Classification Decision Tree" below)
        │
        ▼
7. Mass Estimation (per pellet)
   area_2d → radius → effective_radius → volume → relative_mass
   Apply MATERIAL_PROPS (density, shape_factor) per class
   Apply NUTRIENT_FACTORS (nutrient yield per class)
   Accumulate into mass_scores {N, P, K, Filler}
        │
        ▼
8. Normalize scores → percentages (sum = 100%)
        │
        ▼
9. Visualization
   Darken original image (40% opacity overlay)
   Draw colored contours per class:
     N → gray (200, 200, 200)
     P → green (50, 255, 50)
     K → red (50, 50, 255)
     Filler → cyan (0, 255, 255)
   Encode as base64 JPEG (quality=80)
        │
        ▼
10. Return JSON response
```

---

## Algorithms In Detail

### Perspective Transform

Uses OpenCV's homography-based perspective warp — a true 3D projective transform, not the affine approximation a browser canvas provides. This removes lens perspective distortion and produces a normalized top-down view of the sample, which improves both YOLO detection accuracy and area estimation.

```python
src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
dst_pts = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (W, H))
```

---

### CLAHE Lighting Normalization

Lighting is normalized **only for the color sampling step**, not for YOLO inference. This prevents the model from seeing artificially equalized images that differ from its training distribution.

```python
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lab[:, :, 0] = clahe.apply(lab[:, :, 0])   # equalize L only
```

LAB is used because it is a perceptually uniform color space — equal numeric distances correspond to equal perceived color differences. This makes Euclidean distance a meaningful similarity metric for pellet color.

---

### Classification Decision Tree

After YOLO inference, pellets are re-classified using this logic:

```
Count non-empty ref_* classes provided by user
        │
        ├─── ≥ 2 classes → 4-Class KNN with YOLO Prior (best accuracy)
        │
        ├─── ref_n AND ref_filler only → KNN for N/Filler + YOLO for P/K
        │
        └─── < 2 classes → Otsu auto-threshold for N/Filler + YOLO for P/K
```

#### 4-Class KNN with YOLO Prior

For each detected pellet:

```python
# Compute Euclidean distance to each class's reference samples (in LAB space)
dist = {
    'N':      min(||lab_pellet - lab_n_i|| for each ref_n sample),
    'P':      min(||lab_pellet - lab_p_i|| for each ref_p sample),
    'K':      min(||lab_pellet - lab_k_i|| for each ref_k sample),
    'Filler': min(||lab_pellet - lab_f_i|| for each ref_f sample),
}

best_knn = argmin(dist)   # nearest neighbor in color space

# YOLO Prior: P and K pellets look similar in color but have distinctive shapes.
# If YOLO confidently identified the pellet as P or K, trust YOLO
# unless the color evidence strongly disagrees.
if yolo_class in ('P', 'K'):
    confidence_gap = dist[yolo_class] - dist[best_knn]
    if confidence_gap < YOLO_TRUST_THRESHOLD:   # 15.0 LAB units
        label = yolo_class   # color difference too small, trust YOLO shape decision
    else:
        label = best_knn     # large color difference, override with color
else:
    label = best_knn         # N/Filler: always use KNN
```

`YOLO_TRUST_THRESHOLD = 15.0` corresponds roughly to the perceptual difference between cream-white and off-white pellets.

#### Otsu Auto-Threshold (fallback)

When the user provides no calibration points, the backend uses Otsu's method on the L (lightness) channel to split YOLO class-1 pellets into N and Filler:

```python
# For each candidate threshold T in unique_L_values:
#   w_n     = fraction of pellets with L > T
#   w_filler = fraction of pellets with L <= T
#   score    = w_n * var(L > T) + w_filler * var(L <= T)
# Choose T that minimizes score (minimum within-class variance)
```

Fallback: if all L values are nearly identical (variance < 1.0), threshold defaults to `L > 140.0`.

---

### Mass Estimation

**Problem:** A 2D image area does not directly give mass. Pellets are 3D spheroids of varying density and shape.

**Solution:** Assume each pellet is a sphere, apply per-class correction factors:

```python
MATERIAL_PROPS = {
    'N':      {'density': 1.33, 'shape_factor': 1.00},  # Urea — nearly perfect sphere
    'P':      {'density': 1.61, 'shape_factor': 0.83},  # DAP — slightly flatter
    'K':      {'density': 1.98, 'shape_factor': 0.73},  # MOP — irregular crystal
    'Filler': {'density': 2.40, 'shape_factor': 0.71},  # Inert — dense, flat
}

# Per pellet:
r        = sqrt(area_2d / π)
r_eff    = r * shape_factor
volume   = (4/3) * π * r_eff³
mass_rel = volume * density
```

These values were calibrated experimentally using 15-15-15 fertilizer samples across 4 rounds of testing and averaging.

**Nutrient yield per material:**

```python
NUTRIENT_FACTORS = {
    # Material  → contributes to: N,    P,    K
    'N':        {'N': 0.46, 'P': 0.00, 'K': 0.00},  # Urea 46-0-0
    'P':        {'N': 0.18, 'P': 0.46, 'K': 0.00},  # DAP  18-46-0
    'K':        {'N': 0.00, 'P': 0.00, 'K': 0.60},  # MOP  0-0-60
    'Filler':   {'N': 0.00, 'P': 0.00, 'K': 0.00},  # Inert
}
```

Note: DAP contributes to both N and P scores, so detected P pellets increment both nutrient pools.

---

## Configuration & Constants

All configuration is hardcoded in `main.py`. There is no `.env` file for the backend.

| Constant | Value | Description |
|---|---|---|
| `MODEL_V2_PATH` | `"npksensev2.pt"` | Path to YOLO11m-seg weights |
| `MODEL_X_PATH` | `"npksenseX.pt"` | Path to YOLO11x-seg weights |
| `CLASS_ID_MAP` | `{0: 'K', 1: 'N', 2: 'P'}` | YOLO output class index to name |
| `YOLO_TRUST_THRESHOLD` | `15.0` | LAB distance gap to prefer YOLO over KNN for P/K |
| `YOLO conf` | `0.15` | Detection confidence threshold |
| `YOLO iou` | `0.6` | NMS IoU threshold |
| `YOLO imgsz` | `1024` | Input image size for inference |
| `YOLO max_det` | `3000` | Max detections per image |
| `CLAHE clipLimit` | `2.0` | Contrast limit for CLAHE |
| `CLAHE tileGridSize` | `(8, 8)` | Grid size for CLAHE |
| `JPEG quality` | `80` | Response image compression |
| Response max dim | `800 px` | Annotated image is resized to max 800px before encoding |

**CORS:** All origins allowed (`*`). Must be locked down before public production deployment.

---

## YOLO Model Files

| File | Architecture | Speed | Accuracy | Use Case |
|---|---|---|---|---|
| `npksensev2.pt` | YOLO11m-seg | ~30s/scan | Good | Default — everyday use |
| `npksenseX.pt` | YOLO11x-seg | ~60s/scan | Better | When accuracy matters more than speed |

Both models detect 3 classes: `K` (class 0), `N` (class 1), `P` (class 2).

> **Important:** These files are large binary files. They are NOT tracked by git in this repository. You must obtain them from the original developer and place them in `backend/NPKSense/` before running the server.

If a model file is missing, the server will still start but will log a warning, and requests specifying that model variant will fall back to the available one.

---

## Docker & Deployment

### Dockerfile Summary

```dockerfile
FROM python:3.12-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user (required for HuggingFace Spaces)
RUN useradd -m -u 1000 user
USER user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . .

# HuggingFace Spaces uses port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Build & Run Locally with Docker

```bash
cd backend/NPKSense

docker build -t npksense-backend .
docker run -p 7860:7860 npksense-backend
# API available at http://localhost:7860
```

### HuggingFace Spaces Deployment

This repo is configured for HuggingFace Spaces (Docker SDK, port 7860). To deploy:

1. Create a new Space on HuggingFace with **Docker** as the SDK
2. Push this directory (including the `.pt` model files — or download them at container startup)
3. The Space will build the Docker image and serve the API at `https://<username>-<space-name>.hf.space`
4. Update `NEXT_PUBLIC_API_URL` in the frontend environment to point to the new Space URL

> **Note on model files and HF Spaces:** The `.pt` files are large (~50–200 MB each). Committing them to the Space repo is the simplest approach. Alternatively, download them from a storage URL inside the Dockerfile using `RUN wget ...`.

---

## Known Limitations & TODOs

| Area | Issue | Priority |
|---|---|---|
| **Authentication** | No API key or auth — any client can hit the endpoint | High |
| **CORS** | `allow_origins=["*"]` — must be restricted to frontend domain in production | High |
| **Hardcoded config** | All constants (thresholds, YOLO params) are hardcoded — should be configurable via env vars | Medium |
| **Model paths** | Model file paths are hardcoded relative paths — will break if the working directory changes | Medium |
| **No input validation** | `points` and `ref_*_points` JSON is parsed without schema validation — malformed input raises unhandled exceptions | Medium |
| **Single-threaded inference** | YOLO inference blocks the event loop — concurrent requests will queue | Medium |
| **No rate limiting** | Unlimited requests per client — could be abused or overload the server | Low |
| **GPU support** | Works on CPU; no explicit CUDA setup in Dockerfile — add `nvidia/cuda` base image for GPU inference | Low |
| **Filler in NUTRIENT_FACTORS** | Filler contributes 0 to all nutrient scores, but in practice some fillers may contain trace nutrients | Low |

---

## Handoff Notes

- All logic is in a single file: `main.py` (635 lines). There are no sub-modules.
- The most complex function is the classification block inside `analyze_interactive` — read the comments carefully before modifying the KNN/YOLO prior interaction.
- `MATERIAL_PROPS` and `NUTRIENT_FACTORS` were tuned experimentally. Changing them requires re-validation against physical samples.
- `YOLO_TRUST_THRESHOLD = 15.0` is a critical tuning parameter. Increasing it makes YOLO more dominant; decreasing it makes color more dominant for P/K classification.
- The server loads both model files at startup. If startup is slow, it is because YOLO is initializing on first load.
- The backend is stateless except for `_last_debug_info` (a module-level variable used by `/debug_last`). This is not thread-safe under concurrent load.
