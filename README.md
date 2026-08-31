# Chitra.ai

### End-to-End Satellite Thermal IR Reconstruction, Probabilistic Uncertainty Estimation & Agentic Intelligence Platform

> **IBM AI Builders Challenge — Space Exploration Theme**
> Transforming monochrome Landsat thermal infrared imagery into spatially-aware, uncertainty-quantified RGB reconstructions with real-time object detection and IBM Granite-powered operational briefings.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture & Pipeline](#2-system-architecture--pipeline)
3. [Core Innovations](#3-core-innovations--differentiating-features)
4. [Tech Stack](#4-tech-stack)
5. [Repository Structure](#5-repository-structure)
6. [Setup & Installation](#6-setup--installation)
7. [Environment Variables](#7-environment-variables)
8. [Running the Application](#8-running-the-application)
9. [API Reference](#9-api-reference)
10. [IBM Bob — Development Tool](#10-ibm-bob--development-tool)
11. [Challenge Alignment](#11-ibm-ai-builders-challenge-alignment)

---

## 1. Problem Statement

Satellite thermal infrared imagery — specifically the **Landsat 8/9 Band 10 (ST_B10)** Surface Temperature product — captures radiant heat signatures of Earth's surface at 30-metre resolution. This data is invaluable for urban heat island mapping, wildfire detection, agricultural stress analysis, and tactical situational awareness.

**The limitations of raw thermal data:**

- **Monochromatic**: Single-channel 16-bit DN values carry no visual colour context. Human analysts cannot intuitively parse spatial features from a greyscale heat map.
- **Ambiguous class boundaries**: Mixed pixels at 30 m resolution blend urban, vegetated, and water surface types, making object-level feature extraction unreliable.
- **No uncertainty signal**: Standard image colourisation pipelines (paletting, histogram equalisation) produce visually plausible outputs but carry no confidence signal — the analyst has no way to know which reconstructed pixels are reliable and which are speculative.
- **Static analysis**: Traditional pipelines produce images; they do not reason about what they see or flag ambiguous zones for operator review.

**What Chitra.ai solves:**

Chitra.ai replaces static thermal visualisation with a **probabilistic, reasoning-aware pipeline**. A U-Net generator with Monte Carlo Dropout translates thermal inputs into RGB reconstructions *and simultaneously quantifies how uncertain it is about each pixel*. That uncertainty is then propagated into object detection confidence scores and surfaced as an IBM Granite-generated operational briefing — giving analysts not just an image, but a calibrated, explainable intelligence product.

---

## 2. System Architecture & Pipeline

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CHITRA.AI  END-TO-END PIPELINE                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  INPUT LAYER                                                             │
  │                                                                          │
  │  Landsat 8/9 ST_B10 GeoTIFF  ──or──  PNG / JPEG thermal image           │
  │  Single channel · 16-bit DN (uint16) · 30 m spatial resolution          │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  RADIOMETRIC PREPROCESSING   backend/utils/preprocessing.py             │
  │                                                                          │
  │  • GeoTIFF decoded via rasterio (band 1 extraction)                     │
  │  • PNG/JPEG decoded via Pillow (grayscale conversion)                   │
  │  • DN windowed normalisation → float32 tensor in [0, 1]                 │
  │  • Edge-padding to nearest multiple of 16 (U-Net stride requirement)    │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │  (1, 1, H, W) float32
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  PYTORCH U-NET GENERATOR   backend/models/generator.py                  │
  │                                                                          │
  │  Encoder  ──►  enc1 (64ch H/2)  ──►  enc2 (128ch H/4)                  │
  │                enc3 (256ch H/8) ──►  enc4 (512ch H/16)                  │
  │                                     ▼                                   │
  │  Bottleneck ─► 2× [Conv→BN→ReLU→MCDropout2d(p=0.2)]   ← residual      │
  │                                     ▼                                   │
  │  Decoder  ──►  dec4 + skip(enc4) + MCDropout2d                         │
  │                dec3 + skip(enc3) + MCDropout2d                         │
  │                dec2 + skip(enc2) + MCDropout2d                         │
  │                dec1 + skip(enc1) + MCDropout2d                         │
  │                     ▼                                                   │
  │  Head: Conv(1×1) → Tanh  ──►  (1, 3, H, W)  RGB in [−1, 1]            │
  │                                                                          │
  │  MCDropout2d: F.dropout2d(training=True) — always ON at inference       │
  └──────────────────────────────────┬──────────────────────────────────────┘
                                     │  N stochastic forward passes
                                     ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  MONTE CARLO INFERENCE   backend/models/uncertainty.py                  │
  │                                                                          │
  │  samples  (N, B, 3, H, W)  ←──  N passes through U-Net                 │
  │                                                                          │
  │  mean_rgb       = samples.mean(dim=0)           (B, 3, H, W)            │
  │  var_per_channel = samples.var(dim=0, unbiased=True)                    │
  │  raw_variance   = var_per_channel.mean(dim=1, keepdim=True)             │
  │                                                                          │
  │  2D Aleatoric Uncertainty Map                                            │
  │  uncertainty_map = min-max normalise(raw_variance)  → [0, 1]           │
  └────────────┬─────────────────────────────┬────────────────────────────--┘
               │ mean_rgb (B,3,H,W)           │ uncertainty_map (B,1,H,W)
               ▼                             ▼
  ┌────────────────────────┐     ┌────────────────────────────────────────── ┐
  │  YOLOv8 DETECTION      │     │  2D UNCERTAINTY HEATMAP                  │
  │  backend/models/       │     │                                           │
  │  detector.py           │     │  Viridis-like colormap (NumPy, no mpl)   │
  │                        │     │  blue(0) → purple(0.5) → yellow(1.0)     │
  │  RGB uint8 → YOLO      │◄────┤                                           │
  │  Boxes (x1,y1,x2,y2)  │     │  Blended onto RGB in ImageSlider          │
  │  Class labels          │     └───────────────────────────────────────────┘
  │  Raw confidence        │
  │         ▼              │
  │  Per-box μ_unc crop    │
  │  from uncertainty_map  │
  │         ▼              │
  │  Adjusted Confidence   │
  │  = raw × (1 − μ_unc)  │
  └──────────┬─────────────┘
             │ list[DetectionResult]
             ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  AGENTIC RAG + IBM GRANITE   backend/rag/agent.py                       │
  │                                                                          │
  │  ChromaDB ──► HuggingFace sentence-transformers embeddings              │
  │  12 expert satellite domain documents (knowledge_base.py)               │
  │  Top-4 retrieved passages ──► prompt context                            │
  │                                                                          │
  │  Prompt structure:                                                       │
  │    [SYSTEM ROLE]  ──  [RAG CONTEXT]  ──  [METRICS TABLE]  ──            │
  │    [INSTRUCTION: produce 3-section briefing]                             │
  │                                                                          │
  │  WatsonxLLM (ibm/granite-3-8b-instruct)  ──or──  GraniteFallbackAgent  │
  │                                                                          │
  │  Output: BriefingResult                                                  │
  │    • VISUAL RECONSTRUCTION INTEGRITY                                     │
  │    • TACTICAL OBJECT ASSESSMENT                                          │
  │    • OPERATIONAL GUIDANCE                                                │
  └──────────────────────────────┬───────────────────────────────────────── ┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  FASTAPI MICROSERVICE   backend/app.py                                  │
  │                                                                          │
  │  POST /api/v1/analyze-thermal  (multipart/form-data)                    │
  │    file, mc_passes, latitude, longitude                                  │
  │                                                                          │
  │  Response JSON:                                                          │
  │    status · metrics · detections · images (Base64 PNG) · agent_briefing │
  └──────────────────────────────┬───────────────────────────────────────── ┘
                                 │ JSON over HTTP
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  NEXT.JS VISUAL DASHBOARD   frontend/app/page.tsx                       │
  │                                                                          │
  │  ┌─────────────────┬──────────────────────────┬───────────────────────┐ │
  │  │ Control Sidebar │  ImageSlider Canvas       │ Intelligence Co-Pilot │ │
  │  │                 │                           │                       │ │
  │  │ • Dropzone      │  Thermal ◄──────► RGB     │ • Telemetry badges    │ │
  │  │ • MC passes     │  drag-divider split       │ • Granite badge       │ │
  │  │ • Conf thresh   │                           │ • Briefing text       │ │
  │  │ • Lat / Lon     │  + Heatmap overlay        │ • Detection table     │ │
  │  │ • Run button    │  + BBox overlay           │ • RAG context chips   │ │
  │  └─────────────────┴──────────────────────────┴───────────────────────┘ │
  │                           frontend/components/ImageSlider.tsx            │
  └──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Innovations & Differentiating Features

### 3.1 Aleatoric Uncertainty Quantification — beyond colourisation

Standard thermal-to-RGB colourisation maps DN values to a fixed colour palette. Every output pixel looks equally confident. Chitra.ai fundamentally differs:

| Dimension | Standard Colourisation | Chitra.ai |
|---|---|---|
| Output | Fixed colour per DN value | Stochastic ensemble of N RGB predictions |
| Confidence signal | None | Per-pixel variance map normalised to [0, 1] |
| Analyst guidance | "This is the image" | "These zones are reliable / speculative" |
| Object detection | Full raw confidence | Uncertainty-penalised adjusted confidence |
| Reasoning | None | IBM Granite briefing with RAG context |

### 3.2 MC-Dropout always-on inference

The `_MCDropout2d` module hard-wires `F.dropout2d(training=True)` so dropout remains stochastic even when `model.eval()` is called. This is the key implementation detail that enables MC-Dropout uncertainty estimation at inference time without any model surgery or separate evaluation mode.

### 3.3 Uncertainty-penalised object detection

For every YOLOv8 bounding box, the mean aleatoric uncertainty is extracted from the corresponding spatial slice of the 2D uncertainty map:

```
Adjusted Confidence = Raw Confidence × (1.0 − μ_unc)
```

A box covering a cloud-shadow-contaminated region (high uncertainty) is automatically discounted — the analyst sees this reflected numerically and visually, colour-coded green/orange/red in the overlay.

### 3.4 ChromaDB RAG grounding for IBM Granite

The briefing agent retrieves the 4 most relevant passages from a 12-document satellite domain knowledge base before calling IBM Granite. This prevents hallucinated class labels, false confidence statistics, and incorrect land-cover descriptions. Every briefing is anchored in factual sensor specifications, emissivity physics, and operational interpretation thresholds.

### 3.5 Structured three-section operational briefing

IBM Granite is explicitly instructed to produce:
1. **Visual Reconstruction Integrity** — quantitative fidelity assessment
2. **Tactical Object Assessment** — per-detection reliability analysis
3. **Operational Guidance** — explicit warnings about high-uncertainty spatial zones

This structure mirrors real geospatial intelligence (GEOINT) product formatting.

---

## 4. Tech Stack

| Category | Technology | Role |
|---|---|---|
| **Deep Learning** | PyTorch ≥ 2.0 | U-Net generator, MC-Dropout inference |
| **Architecture** | U-Net with skip connections | Thermal → RGB reconstruction |
| **Uncertainty** | Monte Carlo Dropout | Aleatoric uncertainty estimation |
| **Object Detection** | YOLOv8 (ultralytics ≥ 8.0) | Real-time bounding box detection |
| **Preprocessing** | rasterio ≥ 1.3, NumPy ≥ 1.24 | GeoTIFF decode, DN normalisation |
| **Image I/O** | Pillow ≥ 10.0 | PNG/JPEG decode, Base64 encoding |
| **LLM Reasoning** | IBM Granite 3 8B Instruct | Operational intelligence briefing |
| **LLM Framework** | LangChain ≥ 0.2 + langchain-ibm ≥ 0.1 | LLM orchestration, prompt chaining |
| **LLM Platform** | IBM watsonx.ai | Granite model hosting & API |
| **Vector Store** | ChromaDB ≥ 0.5 | Local in-process RAG retrieval |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Knowledge base embedding |
| **API Backend** | FastAPI ≥ 0.110 + Uvicorn ≥ 0.29 | Async microservice, CORS, file upload |
| **Frontend** | Next.js 14 (App Router) + React 18 | Dashboard SPA |
| **Styling** | Tailwind CSS | Dark-mode responsive UI |
| **Visualisation** | HTML5 Canvas API | Split-view slider, heatmap overlay, bbox rendering |

---

## 5. Repository Structure

```
chitra.ai/
│
├── backend/                          ← Python FastAPI microservice
│   ├── app.py                        ← FastAPI app, startup lifespan, /api/v1/analyze-thermal
│   ├── models/
│   │   ├── generator.py              ← UNetGenerator (MC-Dropout)
│   │   ├── uncertainty.py            ← run_mc_inference(), MCInferenceResult
│   │   └── detector.py               ← UncertaintyAwareDetector (YOLOv8)
│   ├── rag/
│   │   ├── agent.py                  ← SatelliteIntelligenceAgent, GraniteFallbackAgent
│   │   └── knowledge_base.py         ← 12 satellite domain knowledge documents
│   └── utils/
│       └── preprocessing.py          ← normalize_thermal()
│
├── frontend/                         ← Next.js App Router SPA
│   ├── app/
│   │   └── page.tsx                  ← DashboardPage (full operational dashboard)
│   └── components/
│       └── ImageSlider.tsx           ← Split-view canvas slider component
│
├── src/                              ← Phase 1 Landsat scene inspection scripts
│   └── phase1_inspect.py
│
├── data/
│   └── raw/                          ← Place Landsat scene folder here
│
├── outputs/                          ← Generated reports and figures
│
├── docs/
│   └── ibm_bob_usage.md              ← IBM Bob development tool documentation
│
├── requirements.txt                  ← Python dependencies
└── README.md                         ← This file
```

---

## 6. Setup & Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.10 | 3.11 recommended |
| Node.js | ≥ 18.0 | For Next.js frontend |
| npm | ≥ 9.0 | Bundled with Node.js |
| CUDA (optional) | ≥ 11.8 | For GPU acceleration; CPU fallback is automatic |
| GDAL / rasterio | ≥ 1.3 | Required for GeoTIFF input |

### Python backend

```bash
# 1. Clone the repository
git clone https://github.com/your-org/chitra-ai.git
cd chitra-ai

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (GPU) Install CUDA-enabled PyTorch — skip if CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Frontend

```bash
# 5. Navigate to the frontend directory
cd frontend

# 6. Install Node.js dependencies
npm install

# 7. Return to project root
cd ..
```

---

## 7. Environment Variables

Create a `.env` file in the project root **or** export variables in your shell:

```bash
# Required for live IBM Granite inference via watsonx.ai
# Leave unset to use the GraniteFallbackAgent (local development mode)
export WATSONX_APIKEY="your-ibm-cloud-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"

# Optional — default: https://us-south.ml.cloud.ibm.com
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"

# Optional — default: yolov8n.pt (auto-downloaded on first run)
export YOLO_MODEL="yolov8n.pt"

# Optional — path to trained UNet checkpoint
# If unset, random-init weights are used (integration testing only)
export UNET_WEIGHTS="/path/to/unet_checkpoint.pt"

# Optional — server port (default: 8000)
export PORT=8000
```

### Obtaining IBM watsonx.ai credentials

1. Create a free [IBM Cloud account](https://cloud.ibm.com/registration)
2. Provision an [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) instance
3. Generate an API key: **IBM Cloud Console → Manage → Access (IAM) → API Keys**
4. Copy the **Project ID** from your watsonx.ai project settings

> **Local development without credentials**: The `GraniteFallbackAgent` activates automatically when `WATSONX_APIKEY` is absent. It generates fully-structured operational briefings locally using the real quantitative metrics — no network required.

---

## 8. Running the Application

### Start the FastAPI backend

```bash
# From the project root, with the virtual environment activated:
python -m backend.app
```

The service starts on `http://localhost:8000`. Verify it is running:

```bash
curl http://localhost:8000/health
# → {"status": "ok", "device": "cpu"}
```

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Start the Next.js frontend

```bash
# In a separate terminal, from the frontend/ directory:
cd frontend
npm run dev
```

The dashboard opens at **`http://localhost:3000`**.

### Running a full analysis

1. Open `http://localhost:3000`
2. Drag and drop a Landsat ST_B10 GeoTIFF (or PNG/JPEG thermal image) onto the upload dropzone
3. Adjust **Monte Carlo Passes** (5–30) and **Confidence Threshold** as desired
4. Optionally enter **Latitude** and **Longitude** for geospatial context
5. Click **Run Reconstruction & Analysis**
6. The dashboard updates with:
   - Split-view thermal ↔ RGB comparison slider
   - Uncertainty heatmap blend toggle
   - Bounding box overlay with colour-coded confidence
   - Telemetry badges (device, uncertainty statistics, execution time)
   - IBM Granite operational intelligence briefing
   - Detection table with raw and adjusted confidence scores

---

## 9. API Reference

### `POST /api/v1/analyze-thermal`

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary | ✓ | Single-channel thermal image (GeoTIFF, PNG, JPEG) |
| `mc_passes` | integer (2–100) | | Monte Carlo forward passes. Default: 10 |
| `latitude` | float | | Scene centre latitude (WGS-84) |
| `longitude` | float | | Scene centre longitude (WGS-84) |

**Response `200 OK`:**

```json
{
  "status": "success",
  "metrics": {
    "mean_uncertainty": 0.312,
    "max_uncertainty":  0.781,
    "mc_passes_executed": 10,
    "device": "cuda"
  },
  "detections": [
    {
      "class_id": 2,
      "class_name": "car",
      "bbox": { "x1": 10, "y1": 20, "x2": 80, "y2": 90 },
      "raw_confidence":     0.87,
      "mean_uncertainty":   0.41,
      "adjusted_confidence": 0.513
    }
  ],
  "images": {
    "reconstructed_rgb":    "data:image/png;base64,...",
    "uncertainty_heatmap":  "data:image/png;base64,...",
    "bbox_overlay":         "data:image/png;base64,..."
  },
  "agent_briefing": "VISUAL RECONSTRUCTION INTEGRITY\n...",
  "agent_meta": {
    "model_id": "ibm/granite-3-8b-instruct",
    "used_fallback": false,
    "retrieved_context_ids": ["mc_dropout_uncertainty_interpretation", "urban_building_detection"]
  }
}
```

**Error responses:**

| Code | Condition |
|---|---|
| `422` | Corrupt image, unsupported format, image < 16×16 px |
| `503` | CUDA out-of-memory (cache cleared automatically) |
| `500` | Unhandled pipeline error |

---

## 10. IBM Bob — Development Tool

See **[docs/ibm_bob_usage.md](docs/ibm_bob_usage.md)** for the complete account of how IBM Bob served as the primary development environment throughout all seven phases of the Chitra.ai build.

---

## 11. IBM AI Builders Challenge Alignment

| Challenge Criterion | Chitra.ai Implementation |
|---|---|
| **Space Exploration Theme** | Landsat 8/9 satellite thermal band processing; orbital sensor physics; remote sensing workflows |
| **IBM AI Integration** | IBM Granite 3 8B Instruct via watsonx.ai + LangChain; GraniteFallbackAgent for resilient offline operation |
| **Novel AI Application** | MC-Dropout aleatoric uncertainty propagated into object detection scores — not found in standard colourisation or detection pipelines |
| **Responsible AI** | Uncertainty maps and adjusted confidence scores explicitly communicate model reliability; high-uncertainty zones flagged in briefing to prevent overconfident operational decisions |
| **End-to-End System** | Sensor → preprocessing → deep learning → uncertainty → detection → RAG → briefing → interactive dashboard |
| **Production Quality** | FastAPI async microservice, startup model caching, CUDA OOM handling, CORS configuration, typed response contracts |

---

<p align="center">
  Built with <strong>IBM Bob</strong> · Powered by <strong>IBM Granite 3.0</strong> on <strong>watsonx.ai</strong>
</p>
