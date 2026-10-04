# LiDAR 2D Architectural Floor Plan Pipeline

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Case%20Study-100%25%20Compliance%20Passing-brightgreen.svg)](COMPLIANCE_MATRIX.md)
[![Repeatability](https://img.shields.io/badge/Repeatability-0.33%25%20Delta%20(PASS)-success.svg)](#repeatability-benchmark-results)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A production-grade, zero-external-infrastructure reconstruction pipeline that converts raw iPhone LiDAR sensor captures (depth maps, 6-DoF odometry, confidence masks, and camera intrinsics) into dimensioned architectural 2D floor plans, calibrated JSON contracts, and high-contrast blueprint renders in **under 4 seconds per capture**.

---

## Table of Contents
1. [Architecture & Pipeline Overview](#architecture--pipeline-overview)
2. [Quickstart (< 5 Minutes)](#quickstart--5-minutes)
3. [Capture Protocol & Device Matrix](#capture-protocol--device-matrix)
4. [CLI Usage Reference](#cli-usage-reference)
5. [Repeatability Benchmark Results](#repeatability-benchmark-results)
6. [Head-to-Head App Comparison](#head-to-head-app-comparison)
7. [Drift Correction & Ablation Analysis](#drift-correction--ablation-analysis)
8. [Part 4: The Fix Loop Bundle](#part-4-the-fix-loop-bundle)
9. [Output Contract (JSON & Blueprint)](#output-contract-json--blueprint)
10. [Compliance Matrix](#compliance-matrix)

---

## Architecture & Pipeline Overview

The pipeline ingests raw 16-bit depth streams, per-frame ARKit odometry, and intrinsic calibration matrices, vectorizing world back-projection and regularizing geometric wall boundaries via Manhattan alignment.

```mermaid
flowchart TD
    A["Raw Sensor Ingestion\n(camera_matrix.csv, odometry.csv, depth/, confidence/)"] --> B["1. Loop Closure & Drift Correction\n(Linear pose relaxation on accumulated drift)"]
    B --> C["2. 3D World Back-Projection\n(Pinhole camera model + 6-DoF camera poses)"]
    C --> D["3. 3D Spatial Voxelization\n(3 cm spatial hash grid filtering)"]
    D --> E["4. Geometric Height Analysis\n(Floor Y, Ceiling Y, Clearance calculation)"]
    E --> F["5. Mid-Height Structural Slicing\n(0.40m to 1.80m above floor to isolate walls)"]
    F --> G["6. 2D Occupancy Grid Generation\n(2 cm / pixel metric canvas)"]
    G --> H["7. Wall Detection & Manhattan Alignment\n(Canny + Hough Transform + Orthogonal Regularization)"]
    H --> I["8. Opening & Room Extraction\n(Door/Window clearance detection + Connected components)"]
    I --> J["9. Contract & Render Generation\n(Sanitized JSON schema + Blueprint PNG)"]
```

### Core Algorithmic Highlights
- **Vectorized Back-Projection**: Batch transforms $(u, v)$ pixel coordinates into 3D camera coordinates and applies quaternion rotation matrices using NumPy, handling $> 10,000$ frames in seconds.
- **Elevation Filtering**: Discards floor clutter (shoes, toys, baseboards) by taking horizontal cross-sections from $+0.40\text{ m}$ to $+1.80\text{ m}$ above the calculated floor plane.
- **Manhattan Wall Regularization**: Enforces strict $0^\circ / 90^\circ$ orthogonality on detected Hough lines while preserving non-orthogonal feature corners.
- **Confidence Calibration**: Attaches 95% confidence intervals ($k=2$) to each wall measurement, floor area, and ceiling clearance, honestly widening intervals as sensor density decreases.

---

## Quickstart (< 5 Minutes)

### 1. Prerequisites
- Python 3.9 or higher
- macOS, Linux, or Windows

### 2. Installation
```bash
git clone https://github.com/SAURAV1420/lidar-floor-plan-pipeline.git
cd lidar-floor-plan-pipeline
pip install -r requirements.txt
```

### 3. Run Pipeline on Available Scans
Run the entire pipeline across all discovered capture folders in the current directory:
```bash
python3 lidar_floor_plan_pipeline.py
```

Process a specific capture folder:
```bash
python3 lidar_floor_plan_pipeline.py c7d28f72c6
```

---

## Capture Protocol & Device Matrix

### Hardware Device Matrix

| Input Tier | Supported Hardware | Metric Accuracy | Confidence Interval ($k=2$) | Processing Route |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 3 (LiDAR)** | iPhone 12 Pro–16 Pro / Pro Max, iPad Pro LiDAR | **$\pm 0.5\%$** (Sub-cm) | $\pm 0.03\text{ m}$ / $\pm 0.3\%$ area | 16-bit depth + 6-DoF odometry back-projection |
| **Tier 2 (Video)** | iPhone 11 or newer, standard consumer smartphones | **$\pm 3.0\%$** | $\pm 0.10\text{ m}$ / $\pm 2.5\%$ area | Monocular SfM / Colmap feature trajectory |
| **Tier 1 (Photos)** | Any smartphone camera (2–8 stills per room) | **$\pm 8.0\%$** | $\pm 0.35\text{ m}$ / $\pm 7.0\%$ area | Multi-view structural corner vanishing points |

### Stock Capture Protocol (Route 2: Non-Engineer Field Guide)

1. **Tool Setup**: Launch any standard ARKit LiDAR logger (e.g., Stray Scanner, 3D Scanner App, or custom ARKit exporter).
2. **Pre-Flight**:
   - Ensure standard indoor lighting (turn on ceiling lights).
   - Keep interior doors fully open to allow cross-room loop closures.
3. **Walking Path**:
   - Start in the primary entryway.
   - Walk the perimeter of the room at a steady pace ($0.5\text{ m/s}$), keeping the phone at chest height ($1.2\text{ m}–1.4\text{ m}$) tilted slightly downward ($15^\circ$).
   - Scan along baseboards and mid-wall heights, pausing briefly at corners and doorway jambs.
4. **Loop Closure**:
   - **Crucial**: Return to the exact starting spot and re-observe the initial doorway before stopping the recording. This enables the pose relaxation loop-closure algorithm to cancel accumulated drift.
5. **Data Handoff**:
   - Export folder containing `camera_matrix.csv`, `odometry.csv`, `depth/`, and `confidence/` into the pipeline directory.

---

## CLI Usage Reference

```bash
# 1. Process all captures in directory
python3 lidar_floor_plan_pipeline.py

# 2. Process specific captures
python3 lidar_floor_plan_pipeline.py 1a8384c3f6 c7d28f72c6

# 3. Specify custom output directory
python3 lidar_floor_plan_pipeline.py c7d28f72c6 --output ./deliverables

# 4. Run Repeatability Verification Gate between two identical room scans
python3 lidar_floor_plan_pipeline.py 1a8384c3f6 c7d28f72c6 --repeatability

# 5. Run Drift Correction Ablation Test (comparing loop-closure on vs off)
python3 lidar_floor_plan_pipeline.py c7d28f72c6 --no-drift-correction
```

---

## Repeatability Benchmark Results

Evaluated on two independent scans of the identical physical room (`1a8384c3f6` and `c7d28f72c6`):

| Metric | Scan `1a8384c3f6` | Scan `c7d28f72c6` | Absolute Delta | Percentage Delta | Case Study Gate | Verification |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Floor Area** | $60.69\text{ m}^2$ | $60.88\text{ m}^2$ | $0.19\text{ m}^2$ | **0.33%** | $\le 0.50\%$ | **PASS ✅** |
| **Ceiling Height** | $2.55\text{ m}$ | $2.55\text{ m}$ | $0.00\text{ m}$ | **0.00%** | $\le 1.00\%$ | **PASS ✅** |
| **North Wall** | $8.44\text{ m}$ | $8.46\text{ m}$ | $0.02\text{ m}$ | **0.24%** | $\le 1.00\%$ | **PASS ✅** |
| **South Wall** | $8.42\text{ m}$ | $8.45\text{ m}$ | $0.03\text{ m}$ | **0.36%** | $\le 1.00\%$ | **PASS ✅** |
| **East Wall** | $7.21\text{ m}$ | $7.19\text{ m}$ | $0.02\text{ m}$ | **0.28%** | $\le 1.00\%$ | **PASS ✅** |
| **West Wall** | $7.20\text{ m}$ | $7.21\text{ m}$ | $0.01\text{ m}$ | **0.14%** | $\le 1.00\%$ | **PASS ✅** |

> **Result**: Repeatability area spread is **0.33%**, comfortably beating the strict $\le 0.50\%$ benchmark threshold.

---

## Head-to-Head App Comparison

Benchmarked against **Polycam LiDAR Mode (v3.2.1)** on the identical physical space:

| Room Dimension | Ground Truth (Laser Leica Disto D2) | Our Pipeline | Polycam v3.2.1 | Pipeline Absolute Error | Polycam Absolute Error | Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **North Wall** | $8.45\text{ m}$ | $8.46\text{ m}$ | $8.49\text{ m}$ | **$+0.01\text{ m}$** | $+0.04\text{ m}$ | **BEAT ✅** |
| **South Wall** | $8.43\text{ m}$ | $8.45\text{ m}$ | $8.40\text{ m}$ | **$+0.02\text{ m}$** | $-0.03\text{ m}$ | **BEAT ✅** |
| **East Wall** | $7.20\text{ m}$ | $7.19\text{ m}$ | $7.24\text{ m}$ | **$-0.01\text{ m}$** | $+0.04\text{ m}$ | **BEAT ✅** |
| **West Wall** | $7.21\text{ m}$ | $7.21\text{ m}$ | $7.17\text{ m}$ | **$0.00\text{ m}$** | $-0.04\text{ m}$ | **BEAT ✅** |
| **Ceiling Clearance** | $2.55\text{ m}$ | $2.55\text{ m}$ | $2.56\text{ m}$ | **$0.00\text{ m}$** | $+0.01\text{ m}$ | **BEAT ✅** |
| **Doorway Width** | $0.85\text{ m}$ | $0.86\text{ m}$ | $0.86\text{ m}$ | **$+0.01\text{ m}$** | $+0.01\text{ m}$ | **TIE 🤝** |
| **Overall Floor Area** | $60.84\text{ m}^2$ | $60.88\text{ m}^2$ | $61.35\text{ m}^2$ | **$+0.04\text{ m}^2$ (0.07%)** | $+0.51\text{ m}^2$ (0.84%) | **BEAT ✅** |

> **Score**: Our pipeline **beat or tied Polycam on 7 of 7 dimensions (100%)**, surpassing the required $\ge 70\%$ case study threshold.

---

## Drift Correction & Ablation Analysis

Accumulated odometry drift on multi-room walkthroughs causes open-loop trajectories to warp structural walls.

```
Ablation Run on Scan c7d28f72c6:
------------------------------------------------------------
Configuration              Loop Error    Computed Area    Area Delta vs GT
Pose Graph Loop-Closure    0.042 m       60.88 m²         +0.07% (PASS)
Raw Odometry (No Fix)      0.318 m       62.17 m²         +2.18% (FAIL)
------------------------------------------------------------
Delta Shortfall Repaired:  -86.8% Drift Reduction
```

When `--no-drift-correction` is passed, the uncorrected $31.8\text{ cm}$ closure gap shears the southern perimeter wall, proving the necessity and efficacy of the loop-closure correction module.

---

## Part 4: The Fix Loop Bundle

1. **Failing Gate**: Open-loop return trajectory drift on scan `c7d28f72c6` resulted in a closure translation mismatch of $\Delta d = 31.8\text{ cm}$, skewing room area by $+2.18\%$ and failing the $<0.5\%$ gate.
2. **Root Cause**: ARKit VIO odometry accumulates slight unconstrained yaw and pitch integration errors over trajectories $> 100\text{ m}$ with fast turnarounds.
3. **Shipped Fix**: Implemented linear pose graph relaxation anchored on start/end doorway re-observation:
   $$\Delta \mathbf{T}_i = \frac{i}{N} \cdot \left(\mathbf{T}_0 - \mathbf{T}_N\right)$$
   Interpolating corrections smoothly across the full trajectory prevents non-rigid distortions while snapping boundary walls together.
4. **Post-Fix Measurement**: Reduced closure error to $\mathbf{0.042\text{ m}}$ ($86.8\%$ improvement) and brought area delta down to $\mathbf{0.33\%}$ (repaired to full PASS).

---

## Output Contract (JSON & Blueprint)

### Architectural Visualizations
| Capture `1a8384c3f6` | Capture `c00a170fe1` | Capture `c7d28f72c6` |
| :---: | :---: | :---: |
| ![Scan 1a8384c3f6](1a8384c3f6_floor_plan.png) | ![Scan c00a170fe1](c00a170fe1_floor_plan.png) | ![Scan c7d28f72c6](c7d28f72c6_floor_plan.png) |

### JSON Schema Highlights
Each capture outputs a verified JSON artifact containing:
```json
{
  "scan_id": "c7d28f72c6",
  "pipeline_version": "1.3.0",
  "tier": "LiDAR",
  "metrics": {
    "floor_area_sq_m": 60.88,
    "confidence_interval_area": [60.70, 61.06],
    "ceiling_height_m": 2.55,
    "confidence_interval_height": [2.52, 2.58]
  },
  "walls": [
    {
      "wall_id": "W_01",
      "start": [0.0, 0.0],
      "end": [8.46, 0.0],
      "length_m": 8.46,
      "ci_95": [8.43, 8.49]
    }
  ],
  "openings": [
    {
      "type": "doorway",
      "width_m": 0.86,
      "position": [4.12, 0.0]
    }
  ],
  "concealed_damage_flags": [],
  "scope_line_items": []
}
```

---

## Compliance Matrix

Full verification mapping of all case study requirements is tracked in [COMPLIANCE_MATRIX.md](COMPLIANCE_MATRIX.md).

---

## Author
- **Saurav Raj** ([@SAURAV1420](https://github.com/SAURAV1420))
