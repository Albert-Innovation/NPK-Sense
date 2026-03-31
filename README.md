# NPKSense Web — Frontend Application

> AI-powered fertilizer composition analyzer. Upload an image of a fertilizer sample, calibrate interactively, and receive accurate mass percentage breakdowns of N, P, K, and filler materials.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Pages & Features](#pages--features)
6. [Components](#components)
7. [Getting Started](#getting-started)
8. [Environment Variables](#environment-variables)
9. [Available Scripts](#available-scripts)
10. [Backend API Reference](#backend-api-reference)
11. [Analysis Workflow](#analysis-workflow)
12. [Deployment](#deployment)
13. [Known Limitations & TODOs](#known-limitations--todos)

---

## Project Overview

NPKSense is an Edge-AI system for fertilizer quality control. Given a photograph of a fertilizer sample spread on a flat surface, the system:

1. Lets the user crop the region of interest using a 4-point perspective transform
2. Walks the user through an interactive calibration step (clicking reference pellets to train a color classifier)
3. Runs YOLO11m-seg instance segmentation + KNN color classification against the cropped image
4. Averages results across 3 scans (user shakes sample between scans to reduce spatial bias)
5. Returns the mass percentage of each nutrient class (N, P, K, Filler) alongside annotated images

The `/calculator` page allows users to input target nutrient percentages and total batch weight, then calculate the exact ingredient amounts needed using standard fertilizer materials (Urea, DAP, MOP, Filler).

This repository contains only the **frontend** (Next.js). The backend lives in `../npksense-api/` and must be running separately.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User's Browser                           │
│                                                              │
│  ┌─────────────────────┐     ┌──────────────────────────┐   │
│  │  / (Analyzer)        │     │  /calculator             │   │
│  │  - Upload image      │     │  - Input target N/P/K%   │   │
│  │  - Perspective crop  │     │  - Calculate ingredient  │   │
│  │  - Calibrate KNN     │     │    weights               │   │
│  │  - 3-scan average    │     │  - Redirect to /         │   │
│  │  - View results      │     │    with URL params       │   │
│  └──────────┬──────────┘     └──────────────────────────┘   │
│             │                                                │
│             │  POST /analyze_interactive (multipart/form)   │
│             ▼                                                │
└─────────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              npksense-api (FastAPI + Python)                 │
│                                                              │
│  1. Perspective warp (OpenCV)                               │
│  2. LAB color normalization                                 │
│  3. YOLO11m-seg inference                                   │
│  4. KNN classification with calibration points             │
│  5. Mass estimation via contour area                        │
│  6. Return JSON (percentages + base64 images)               │
└─────────────────────────────────────────────────────────────┘
```

The frontend and backend are **fully decoupled**. The frontend can be hosted on Vercel while the backend runs on Hugging Face Spaces or any Python server.

---

## Tech Stack

| Category | Technology | Version |
|---|---|---|
| Framework | Next.js (App Router) | 16.0.10 |
| UI Library | React | 19.2.1 |
| Language | TypeScript | 5 |
| Styling | Tailwind CSS | 4 |
| Charts | Chart.js + react-chartjs-2 | latest |
| Icons | lucide-react | latest |
| Linter | ESLint | 9 |

No external UI component library is used. All components are hand-crafted with Tailwind CSS utilities.

---

## Project Structure

```
npksense-web/
├── public/
│   └── NPKSense.png              # App logo
│
├── src/
│   ├── app/
│   │   ├── layout.tsx            # Root layout — wraps all pages with <Navbar>
│   │   ├── globals.css           # Tailwind base + CSS variables
│   │   ├── page.tsx              # Main analyzer page (~830 lines)
│   │   └── calculator/
│   │       └── page.tsx          # Recipe calculator page (~245 lines)
│   │
│   └── components/
│       ├── Navbar.tsx            # Top navigation bar
│       ├── ControlPanel.tsx      # Upload, weight, target recipe inputs
│       ├── ImagePreview.tsx      # Result display + calibration UI
│       ├── PerspectiveCropper.tsx# Interactive 4-point crop selector
│       └── StatCard.tsx          # Per-nutrient result card with progress bar
│
├── .env.local                    # API URL (not committed — create manually)
├── next.config.ts                # Minimal Next.js config
├── tailwind.config.ts            # Custom color palette (npk-gray, npk-gold, etc.)
├── tsconfig.json                 # TypeScript config (@/* alias → ./src/*)
└── package.json
```

### Path Alias

TypeScript is configured with `@/*` pointing to `./src/*`, so imports look like:

```typescript
import { StatCard } from "@/components/StatCard";
```

---

## Pages & Features

### `/` — Analyzer Dashboard

The core feature of the application. A multi-step workflow:

#### Step 1: Upload & Configure
- Upload a JPG/PNG image of a fertilizer sample spread on a flat surface
- Set the **total sample weight** (default: 100 g)
- Optionally set **target recipe** percentages (N%, P%, K%, Filler%) for deviation tracking
- Select the model variant (v2 or x)

#### Step 2: Perspective Crop
- The `PerspectiveCropper` component renders the image with 4 draggable corner handles
- User drags handles to the edges of the fertilizer sample to isolate it
- Crop coordinates are sent to the backend (the server performs the actual warp, not the browser)

#### Step 3: Calibration
- The backend returns the cropped image with initial YOLO detection results
- User clicks on reference pellets (3–7 clicks per class) to train the KNN classifier:
  - Nitrogen pellets (typically white/translucent)
  - Filler pellets
  - Phosphorus and Potassium are handled by YOLO shape priors and do not require calibration clicks
- Auto-mode is available; manual mode gives better accuracy for unusual samples

#### Step 4: Multi-Scan Analysis
- System requires **3 scans** by default
- Between scans, the user shakes the sample to redistribute pellets and reduce spatial bias
- Results from all 3 scans are averaged to reduce variance
- The UI shows a progress indicator and scan counter

#### Step 5: Results
- Doughnut chart showing NPK distribution
- Individual `StatCard` for each nutrient showing:
  - Measured percentage
  - Calculated weight (g)
  - Deviation from target (highlighted red if ±2% or more)
- Annotated before/after image comparison

#### Backend Health Indicator
- On mount, the page polls `GET /health` and shows a connection status badge
- If the backend is unreachable, analysis is blocked and an error is displayed

---

### `/calculator` — Recipe Calculator

A utility page for formulating fertilizer batches.

**Inputs:**
- Target N%, P%, K%
- Total batch weight (g or kg)

**Calculation:**
Uses the standard NPK matrix for common materials:

| Material | N% | P% | K% |
|---|---|---|---|
| Urea | 46 | 0 | 0 |
| DAP | 18 | 46 | 0 |
| MOP (Muriate of Potash) | 0 | 0 | 60 |
| Filler | 0 | 0 | 0 |

Solves the linear system to find the required weight of each ingredient. Validates that the target percentages are achievable (alerts if over 100% or if nutrient targets exceed what materials can provide).

**Integration with Analyzer:**
After calculation, the page redirects to `/` with URL query parameters pre-filling the target recipe fields, so the user can immediately verify the batch composition against the analyzer.

---

## Components

### `Navbar.tsx`
Top navigation bar with links to `/` (Analyzer) and `/calculator` (Calculator). Shows the NPKSense logo. Active route is highlighted.

### `ControlPanel.tsx`
Left-side panel on the Analyzer page. Contains:
- File upload input (image)
- Total weight input
- Target recipe percentage inputs (N, P, K, Filler)
- Model selector toggle
- Analyze / Reset buttons

### `ImagePreview.tsx`
Right-side panel on the Analyzer page. Handles two modes:
- **Calibration mode**: Overlays a click-capture layer on the image; collects calibration point coordinates and class labels
- **Result mode**: Displays the annotated image returned by the backend side-by-side with the original

Also renders the doughnut chart (Chart.js) showing the final NPK distribution.

### `PerspectiveCropper.tsx`
Standalone interactive component for 4-point perspective selection.
- Renders the image inside a `<div>` with absolutely positioned draggable handles
- Uses `onMouseDown` / `onTouchStart` events for drag tracking
- Outputs `{ topLeft, topRight, bottomLeft, bottomRight }` pixel coordinates normalized to the image's natural resolution

### `StatCard.tsx`
Displays the result for a single nutrient class:
- Nutrient name and color-coded icon
- Measured % (large text)
- Weight in grams
- Progress bar
- Target deviation badge (green if within ±2%, red if outside)

---

## Getting Started

### Prerequisites

- Node.js 18+ (LTS recommended)
- npm 9+ or equivalent
- The backend API must be running (see Backend Setup below)

### Frontend Setup

```bash
# 1. Install dependencies
npm install

# 2. Copy and configure environment variables
cp .env.example .env.local
# Edit .env.local — set NEXT_PUBLIC_API_URL to your backend URL

# 3. Start development server
npm run dev
# App available at http://localhost:3000
```

### Backend Setup

The backend is a separate Python project located at `../npksense-api/`.

```bash
cd ../npksense-api

# Install Python dependencies
pip install fastapi uvicorn ultralytics opencv-python numpy

# Place the YOLO weights file in the same directory
# Expected filename: npksensev2.pt
# Obtain this file from the project owner — it is NOT committed to the repo

# Start the API server
python main.py
# Runs on http://localhost:8000
```

> **Important:** The YOLO weights file (`npksensev2.pt`) is not committed to the repository. You must obtain it separately from the original developer before the backend will function.

---

## Environment Variables

Create a `.env.local` file in the root of `npksense-web/`:

```env
# URL of the npksense-api backend

# Local development:
NEXT_PUBLIC_API_URL=http://localhost:8000

# Production (Hugging Face Spaces example):
# NEXT_PUBLIC_API_URL=https://your-username-npksense.hf.space
```

The `NEXT_PUBLIC_` prefix is required by Next.js to expose the variable to the browser at build time.

---

## Available Scripts

```bash
npm run dev       # Start development server with hot reload (http://localhost:3000)
npm run build     # Create optimized production build
npm run start     # Serve the production build (requires build first)
npm run lint      # Run ESLint across the project
```

---

## Backend API Reference

All requests target `NEXT_PUBLIC_API_URL`.

### `GET /health`
Liveness probe. Returns `{ "status": "ok" }`. Used by the frontend health indicator.

### `GET /model_info`
Returns YOLO class names and model configuration metadata.

```json
{
  "classes": ["N", "P", "K"],
  "model_path": "npksensev2.pt"
}
```

### `POST /analyze_interactive`
Main analysis endpoint. Accepts `multipart/form-data`.

**Request fields:**

| Field | Type | Description |
|---|---|---|
| `image` | File | JPG or PNG image of the sample |
| `mode` | string | `"crop_only"` or `"analyze"` |
| `crop_points` | JSON string | `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` — corner coordinates |
| `calibration_points` | JSON string | List of `{"x": n, "y": n, "label": "N"\|"Filler"}` objects |
| `total_weight` | float | Total sample weight in grams |
| `model_variant` | string | `"v2"` or `"x"` |

**Response (`analyze` mode):**

```json
{
  "results": {
    "N":      { "percentage": 28.3, "weight_g": 28.3 },
    "P":      { "percentage": 14.1, "weight_g": 14.1 },
    "K":      { "percentage": 21.5, "weight_g": 21.5 },
    "Filler": { "percentage": 36.1, "weight_g": 36.1 }
  },
  "annotated_image": "<base64-encoded PNG>",
  "cropped_image":   "<base64-encoded PNG>"
}
```

### `GET /debug_last`
Returns statistics from the most recent inference call. Useful for diagnosing low-accuracy results without re-uploading.

---

## Analysis Workflow

The following diagram shows the complete data flow for a single scan:

```
User uploads image
        │
        ▼
PerspectiveCropper (client)
  → user drags 4 corner handles to isolate sample
  → outputs pixel coordinates (natural image resolution)
        │
        ▼
POST /analyze_interactive  (mode: crop_only)
  → server: perspective warp → LAB normalization → YOLO inference
  → returns: cropped image + initial detections
        │
        ▼
ImagePreview — Calibration Mode (client)
  → user clicks N pellets (3–7 points)
  → user clicks Filler pellets (3–7 points)
  → stores calibration_points list in component state
        │
        ▼
POST /analyze_interactive  (mode: analyze)
  → server: KNN trained on calibration_points
  → YOLO detections re-labeled via KNN
  → mass estimation via contour area per class
  → returns: mass percentages + annotated images
        │
        ▼
StatCards + Doughnut Chart (client)
  → displays N/P/K/Filler percentages
  → compares against target recipe
  → appends to scan buffer (needs 3 total)
        │
        ▼
After 3 scans → average results are displayed as final output
```

---

## Deployment

### Frontend — Vercel (Recommended)

1. Push `npksense-web` to a GitHub repository
2. Import the repository in [vercel.com](https://vercel.com)
3. Set environment variable: `NEXT_PUBLIC_API_URL` → your backend URL
4. Deploy — Vercel handles `npm run build` automatically

### Backend — Hugging Face Spaces

The backend is pre-configured for Hugging Face Spaces (port 7860, Docker-based). See `../npksense-api/` for the Dockerfile and Space metadata. After deployment, copy the Space URL into the Vercel environment variable.

### Self-Hosted

Any Node.js-compatible host (Railway, Render, VPS + PM2) works:

```bash
npm run build
npm run start   # serves on port 3000 by default
```

Set `NEXT_PUBLIC_API_URL` in the host's environment variable panel before building.

---

## Known Limitations & TODOs

| Area | Issue | Priority |
|---|---|---|
| **Authentication** | No auth — API is open to anyone who knows the URL | High |
| **CORS** | Backend allows all origins (`*`) — should be locked to the frontend domain in production | High |
| **Model weights** | `npksensev2.pt` is not version-controlled; must be shared out-of-band | Medium |
| **Scan count** | 3-scan requirement is hardcoded; should be a configurable setting | Low |
| **Mobile UX** | PerspectiveCropper touch handling works but is not optimized for small screens | Medium |
| **Error messages** | Backend errors surface as raw FastAPI JSON; frontend shows generic error text | Low |
| **Calibration persistence** | Calibration points reset on page refresh; localStorage could help repeat users | Low |

---

## Handoff Notes

- **YOLO weights** (`npksensev2.pt`) — must be obtained from the original developer; without this file the backend will not start
- **Core state machine** — all multi-step workflow logic lives in `src/app/page.tsx`; this is the most complex file in the project
- **All ML logic** is in `../npksense-api/main.py` — the frontend is purely a UI layer
- **Color theme** for N/P/K classes is defined in `tailwind.config.ts` under the `npk` key
- The project uses **Next.js App Router** (not Pages Router) — layouts, loading states, and metadata APIs differ from the older pattern
- There is no global state library; state is managed entirely with React `useState` / `useCallback` hooks
