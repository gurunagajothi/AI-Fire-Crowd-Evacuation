# AI-Based Fire/Smoke Hazard Detection & Crowd-Aware Evacuation Path Optimization

An academic research prototype implementing multi-modal computer vision (YOLOv8) hazard perception, continuous dynamic graph routing, automated public address simulation, and directional signage dispatch based on the IEEE paper architecture.

---

## 1. System Architecture

The system comprises three decoupled tiers:

```text
===================================================================================
1. PERCEPTION TIER (Dual Parallel Vision Models on CPU)
   ├── Video / CCTV Stream (VideoStreamHandler)
   ├── YOLOv8n Fire & Smoke Detector (Classes: Smoke, Fire)
   └── YOLOv8n Crowd Detector (Class: Person, Counts & Centers)
===================================================================================
                                      │
                                      ▼
2. GRAPH & OPTIMIZATION TIER (Dynamic NetworkX Graph)
   ├── Demo Multi-Zone Mapping (Rooms, Corridors, Exits)
   ├── Dynamic Edge Cost Engine: Cost(e) = alpha*Hazard + beta*Density + gamma*Length
   ├── Continuous Dynamic Dijkstra Algorithm
   └── Route-Change Event Detector (Emits ROUTE_CHANGE_EVENT on reroute)
===================================================================================
                                      │
                                      ▼
3. PRESENTATION & ALERT TIER (Interactive Streamlit Dashboard)
   ├── Visual Perception HUD with OpenCV Bounding Boxes
   ├── Dynamic Floor-Plan Map (Highlighted Safe Route vs. Hazards)
   ├── Directional Signage Simulation (Digital LED Exit Indicators)
   ├── Automated PA / Voice Announcement Text
   ├── Security Guard Dispatch Event Log (In-Memory Audit Trail)
   └── Real-Time Telemetry (Inference Latency, Dijkstra Time, Processed FPS)
===================================================================================
```

---

## 2. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Windows 10/11, macOS, or Linux (No NVIDIA GPU required; 100% CPU compatible)

### Setup Virtual Environment
```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. How to Run the System

### Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open your web browser at `http://localhost:8501`. Use the top quick benchmark buttons (`[🔥 TEST FIRE SAMPLE]`, `[💨 TEST SMOKE SAMPLE]`, `[👥 TEST CROWD SAMPLE]`) or upload your own CCTV test media.

### Run Functional & End-to-End Tests
```bash
# Run 14-point functional test suite
python tests/test_full_system.py

# Run complete 12-step end-to-end pipeline test
python tests/test_end_to_end.py

# Run individual module benchmarks
python detection/test_fire_detector.py --image data/test/flames_sample.jpg
python detection/test_crowd_detector.py --image data/test/crowd_sample.jpg
python graph/test_dijkstra_optimizer.py
python integration/test_evacuation_system.py
```

---

## 4. Academic Demonstration Scenarios & Quick Benchmarks

The dashboard provides single-click benchmark buttons and an integrated Scenario Controller allowing evaluation of distinct operational states:

1. **Quick Test Sample: FIRE (`flames_sample.jpg`)**: High fire hazard ($Hazard=0.875$). Cost escalates; occupants automatically rerouted from `EXIT_A` to `EXIT_B`.
2. **Quick Test Sample: SMOKE (`fire_sample.jpg`)**: Moderate smoke hazard ($Hazard=0.651$). Visibility and air quality impaired. Dijkstra reroutes to `EXIT_B`.
3. **Quick Test Sample: CROWD (`crowd_sample.jpg`)**: Detects 15 pedestrians ($Density=0.750$). Triggers congestion avoidance detour.
4. **Normal Baseline State**: Zero fire, zero smoke, low density. Dijkstra selects direct physical shortest path: `ROOM_A -> CORRIDOR_1 -> EXIT_A` (Cost: $4.08$).

*(All dynamic evaluations are labeled: `DEMO ZONE MAPPING — NOT REAL CCTV CALIBRATION`)*.

---

## 5. Mathematical Formulation: Dynamic Edge Cost

$$\text{Cost}(e) = \alpha \cdot \text{Hazard}(e) + \beta \cdot \text{Density}(e) + \gamma \cdot \text{Length}(e)$$

Configured in `config/config.py`:
- **$\alpha = 5.0$ (Hazard Sensitivity)**: Prioritizes human life safety. A significant fire/smoke hazard ($\ge 0.60$) creates an immediate cost spike ($>3.0$), forcing Dijkstra to seek alternative paths.
- **$\beta = 2.0$ (Congestion Sensitivity)**: Mitigates stampede risk. Dense crowds ($\ge 0.80$) add up to $1.60$ to edge cost.
- **$\gamma = 0.2$ (Distance Scaling Factor)**: Scales physical distance in meters ($0.2 \times \text{meters}$).

---

## 6. Dynamic Dijkstra & Continuous Route Recomputation

Traditional Dijkstra calculates routes once based on static distances. In this system:
1. Every processed frame updates edge attributes ($Hazard, Density$).
2. Traversal weights are dynamically refreshed across all graph edges.
3. When `find_safest_route()` discovers a route lower in cumulative cost than the compromised route, a `ROUTE_CHANGE_EVENT` is triggered.
4. The event contains: previous route, new route, reason, cost difference, and timestamp.

---

## 7. Simulated Emergency Alert & Signage Chain

In accordance with the IEEE paper architecture:
- **Simulated Directional Signage**: Dynamic exit signs display real-time statuses (`EXIT B — SAFE ROUTE 🟢` vs. `EXIT A — CLOSED / AVOID 🛑`). Clearly labeled: *Simulation only — no physical PA/LED hardware connected.*
- **Simulated PA / Voice Announcements**: Automatically synthesizes actionable instructions (e.g. *"Fire detected near Corridor 1. Avoid Exit A. Evacuate toward Exit B"*).
- **Security Guard Event Log**: Records all emergency dispatch notifications with timestamps, hazard scores, and recommended actions.

---

## 8. Performance Telemetry & Latency Metrics

Because emergency response relies on computational feasibility:
- **Detection Latency**: $\approx 120 - 150 \text{ ms}$ per frame on CPU (using dual YOLOv8 Nano models).
- **Dijkstra Recomputation**: $\approx 0.5 - 2.0 \text{ ms}$ on the building graph.
- **Frame-Skip Optimization**: Processes every $N$-th frame (`FRAME_SKIP = 5`), conserving CPU resources while maintaining real-time video display.

---

## 9. System Limitations & Academic Transparency (10-Point Scope)

> [!IMPORTANT]
> **Prototype Disclaimers & Limitations**:
> 1. **Building Topology**: Facility layout is a synthetic 7-node academic demonstration model (`ROOM_A` through `EXIT_B`), with an optional dual-floor vertical digital twin expansion (`GF` and `L1` joined via `STAIR_1`).
> 2. **Zone Mapping**: Detections are mapped to demonstration building zones; not calibrated to real geometric camera homography or perspective transformations.
> 3. **Input Ingestion**: Media files are ingested as simulated CCTV surveillance test frames.
> 4. **Real Deployment Requirements**: Field deployment requires real calibrated camera matrices, optical overlap, and CAD architectural blueprints.
> 5. **Actuator Guidance**: PA system and LED exit sign states shown are software simulations; no physical building actuators, FACP relays, or hardware are connected.
> 6. **AI Models**: Real pretrained YOLOv8 Nano checkpoints running in CPU mode (`models/fire_smoke_yolov8n.pt` and `models/yolov8n.pt`).
> 7. **Dynamic Routing**: Shortest-path calculations are computed dynamically in real time via NetworkX Dijkstra optimization based on active edge weights: $\text{Cost}(e) = \alpha H(e) + \beta D(e) + \gamma L(e)$. Decisions are never hardcoded.
> 8. **Benchmark & Metric Transparency**: Perception confidence scores represent per-detection model output, not static dataset benchmark accuracies. False-alarm rates, precision, recall, and mAP require ground-truth labelled test sets and are labeled as "Not yet benchmarked" rather than hallucinated.
> 9. **Camera Coverage & Occlusion**: Multi-camera coverage quality and lens occlusions are simulated to demonstrate camera-fault-aware routing penalties.
> 10. **Assistance Engine**: The Emergency Decision Assistant uses deterministic, rule-based reasoning grounded solely in live system telemetry to explain routing decisions.

---

## 10. AI Smart Building Emergency Command Center Features

The application incorporates a multi-mode command center designed for academic reviews, viva examinations, and research demonstrations:

### Two Operational Modes
- **🔬 RESEARCH MODE**: In-depth telemetry, interactive edge weight tuning ($\alpha, \beta, \gamma$), sensitivity sweep plots, research disclaimers, and model evaluation metrics.
- **🚨 EMERGENCY COMMAND MODE**: High-contrast, rapid-triage interface with prominent alert banners, immediate evacuation routing, digital signage indicators, and one-click incident logging.

### 9 Command Center Navigation Tabs
1. **📊 OVERVIEW**: Incident severity triage HUD, multi-zone risk priority table, and real-time evacuation efficiency index.
2. **📹 LIVE CCTV**: 4-channel surveillance grid (`CAM-01` to `CAM-04`) with latency, FPS, coverage health, and simulated feeds.
3. **🏢 BUILDING DIGITAL TWIN**: Interactive NetworkX 2D topology with highlighted evacuation path and toggleable Multi-Floor digital twin (`Ground Floor` + `First Floor` via `STAIR_1`).
4. **🧭 EVACUATION & ROUTING**: Dynamic route breakdown, bottleneck identification, vulnerable occupant priority simulation, and detour explanations.
5. **🔔 ALERTS & DISPATCH**: Directional signage LED indicators, automated PA announcement script, and security guard dispatch audit logs.
6. **📈 ANALYTICS & TELEMETRY**: Temporal hazard trends with 3-point moving average forecasts, frame processing latency charts, and Dijkstra execution timers.
7. **🤖 XAI & DECISION ASSISTANT**: Deterministic, rule-based query assistant answering: *"Why was the evacuation route changed?"*, *"Why is Exit A avoided?"*, *"What caused the hazard spike in Corridor 1?"*, and *"Which camera detected the fire?"*.
8. **🧪 VALIDATION & BENCHMARKS**: Parameter sensitivity analysis ($\alpha, \beta, \gamma$ sweep), comparative baseline routing comparison, and ground-truth benchmark guidelines.
9. **📑 REPORTS & INCIDENT REPLAY**: Interactive time-slider incident replay and multi-format digital incident report export (`TXT`, `JSON`, `CSV`).

### 10 Academic Demonstration Scenarios
- **01 NORMAL BASELINE**: All zones clear, optimal path to EXIT_A.
- **02 SMOKE DETECTION**: Moderate smoke in Corridor 1, automatic detour.
- **03 FIRE EMERGENCY**: High fire hazard near Exit A, reroutes to Exit B.
- **04 HIGH CROWD STAMPEDE**: Dense crowd in Corridor 1, congestion avoidance.
- **05 COMPOUND HAZARD**: Concurrent fire and dense crowd, multi-zone avoidance.
- **06 CAMERA OCCLUSION**: Cam-02 lens obstructed, coverage penalty injected.
- **07 EXIT BLOCKED**: Exit A completely compromised, exit closure protocol.
- **08 MULTI-ZONE HAZARD**: Simultaneous fire and smoke across multiple corridors.
- **09 CROWD + SMOKE EVACUATION**: Moderate smoke with panicking crowd.
- **10 MULTI-CAM DISPATCH**: Cross-channel multi-incident alert dispatch.
- **▶ RUN FULL DEMO**: Sequential walkthrough transitioning from Normal to Emergency and Recovery.

---

## 11. IEEE Requirement Traceability Matrix

| Requirement / Module | Description | Implementation File | Verification Test |
| :--- | :--- | :--- | :--- |
| **REQ-01: Fire/Smoke Perception** | YOLOv8n dual hazard detection (classes 0: smoke, 1: fire) | [`detection/fire_detector.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/detection/fire_detector.py) | `tests/test_full_system.py` (Tests 1-3) |
| **REQ-02: Crowd/Person Perception** | YOLOv8n crowd estimation (COCO class 0: person) | [`detection/crowd_detector.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/detection/crowd_detector.py) | `tests/test_full_system.py` (Tests 4-5) |
| **REQ-03: Building Graph Topology** | NetworkX 7-node topological model with edge lengths | [`graph/building_graph.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/graph/building_graph.py) | `tests/test_full_system.py` (Tests 6-7) |
| **REQ-04: Dynamic Edge Cost** | Weight formulation: $\text{Cost}(e) = \alpha H(e) + \beta D(e) + \gamma L(e)$ | [`graph/dijkstra_optimizer.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/graph/dijkstra_optimizer.py) | `tests/test_full_system.py` (Test 8) |
| **REQ-05: Dynamic Dijkstra Routing** | Real-time path optimization & continuous rerouting | [`graph/dijkstra_optimizer.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/graph/dijkstra_optimizer.py) | `tests/test_full_system.py` (Tests 9-10) |
| **REQ-06: End-to-End System Pipeline** | Multi-zone mapping, telemetry, route events | [`integration/evacuation_system.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/integration/evacuation_system.py) | `tests/test_end_to_end.py` (Steps 1-12) |
| **REQ-07: Alert & Signage Dispatch** | Automated PA, directional signage, security logs | [`alerts/alert_manager.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/alerts/alert_manager.py) | `tests/test_full_system.py` (Test 13) |
| **REQ-08: Command Center Presentation** | 9 tabs, 2 operational modes, telemetry & analytics | [`dashboard/app.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/dashboard/app.py) | `tests/test_full_system.py` (Test 12) |
| **REQ-09: Multi-Floor Digital Twin** | Dual-floor topological graph with vertical stairwell | [`graph/multi_floor_graph.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/graph/multi_floor_graph.py) | `tests/test_advanced_features.py` (Tests 4-5) |
| **REQ-10: Multi-Camera Simulation** | 4-channel CCTV grid with occlusion & uncertainty penalties | [`integration/multi_camera_sim.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/integration/multi_camera_sim.py) | `tests/test_advanced_features.py` (Tests 1-3) |
| **REQ-11: Decision Explainability (XAI)** | Deterministic telemetry-grounded viva query assistant | [`integration/decision_assistant.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/integration/decision_assistant.py) | `tests/test_advanced_features.py` (Tests 9-10) |
| **REQ-12: Multi-Format Incident Reports** | Exportable audit trails in TXT, JSON, and CSV formats | [`dashboard/components.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/dashboard/components.py) | `tests/test_advanced_features.py` (Tests 6-8) |

---

## 12. Complete Test Suite Execution

All 4 test suites execute cleanly on CPU with Python 3.13:

```bash
# 1. Run Real-Time Pipeline & Temporal Tracking Test Suite (12 Tests)
python tests/test_realtime_pipeline.py

# 2. Run Advanced Features Test Suite (10 Tests)
python tests/test_advanced_features.py

# 3. Run Functional Test Suite (14 Tests)
python tests/test_full_system.py

# 4. Run Complete End-to-End Test Suite (12 Steps)
python tests/test_end_to_end.py
```

**Total Verified Tests: 48 / 48 PASSED (100% Pass Rate)**

---

## 13. Real-Time CCTV Ingestion & Hardware Setup Guide

The command center supports physical hardware, network RTSP streams, and recorded video:

### 1. Laptop Webcam
- Select `Laptop Webcam` in the sidebar CCTV Media Source.
- Click `▶ START` to activate real-time camera ingestion.
- The system automatically captures live frames at native resolution with running FPS calculation.

### 2. External USB Camera
- Connect your USB camera (e.g., Logitech C920 / external USB lens).
- Select `External USB Camera` and specify the camera index (typically `1` or `2`).
- Click `▶ START` to bind the physical feed to `CAM-01`.

### 3. Configurable RTSP Network Stream
- Select `RTSP Network Stream`.
- Enter the RTSP endpoint (e.g., `rtsp://admin:password@192.168.1.100:554/h264Preview_01_main`).
- OpenCV FFmpeg parameters are configured with non-blocking timeouts to prevent hanging on network disruptions.
- Click `▶ START` to engage live IP camera surveillance.

### 4. CCTV Sample Video
- Select `CCTV Demo Video` to loop `data/test/demo_cctv_feed.mp4` continuously for offline academic demonstrations.

### 5. Graceful Disconnection Handling
- If a camera disconnects or network stream drops, the system transitions to `CAMERA UNAVAILABLE` with honest error telemetry.
- **Zero Hallucination / Zero Fake Fallbacks**: The system never substitutes a synthetic fire image to fake camera activity.

---

## 14. Temporal Hazard Tracking & Flap Prevention

Located in [`integration/hazard_trend.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/integration/hazard_trend.py):
- **Rolling Window Analysis**: Retains the last 30 frames of perception telemetry.
- **Linear Trend Slope**: Computes instantaneous slope to categorize hazard direction (`RISING ↗`, `STABLE →`, `FALLING ↘`).
- **Consecutive-Frame Confirmation**: Requires 3 consecutive hazardous frames before escalating to confirmed alarms, eliminating false positives from single-frame optical anomalies.
- **Cooldown Flap Prevention**: Implements a 10-frame vigilance cooldown upon hazard abatement, keeping the system in `MONITORING` state to prevent directional signage flapping.
- **State Hierarchy**:
  - `NORMAL` (Nominal conditions)
  - `MONITORING` (Cooldown state active)
  - `RISK INCREASING` (Early unconfirmed hazard)
  - `SMOKE DETECTED` (Confirmed smoke plume)
  - `FIRE DETECTED` (Confirmed flame presence)
  - `CAMERA UNAVAILABLE` (Stream disruption)

---

## 15. Smart Emergency Incident Management

Located in [`alerts/alert_manager.py`](file:///C:/Users/GURU%20NAGAJOTHI/Desktop/AI_Fire_Crowd_Evacuation/alerts/alert_manager.py):
- **Incident Acknowledgement**: Operators can acknowledge active alarms via the dashboard with automatic logging of the operator name and timestamp.
- **Automatic Frame Snapshots**: Captured frames are automatically saved to `data/processed/snapshots/` with unique incident IDs upon critical hazard detection.
- **Duplicate Suppression**: Suppresses repeated alarm spam within a 15-second cooldown while refreshing telemetry.
- **External Notifications**: Non-blocking dispatch via Telegram Bot API or SMTP Email using environment variables (`TELEGRAM_BOT_TOKEN`, `SMTP_SERVER`).
- **Incident Exports**: Instant one-click exports of audit trails in `JSON`, `CSV`, and `TXT` formats.


