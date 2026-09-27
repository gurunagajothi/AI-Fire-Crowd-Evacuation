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

## 9. System Limitations & Academic Transparency (7-Point Scope)

> [!IMPORTANT]
> **Prototype Disclaimers & Limitations**:
> 1. **Building Topology**: Facility layout is a synthetic 7-node academic demonstration model (`ROOM_A` through `EXIT_B`).
> 2. **Zone Mapping**: Detections are associated with a demonstration building edge; not calibrated to geometric camera homography.
> 3. **Input Ingestion**: Uploaded images/videos are ingested as CCTV surveillance test frames.
> 4. **Real Deployment Requirements**: Field implementation requires camera calibration matrices, optical overlap, and CAD floor-plans.
> 5. **Actuator Guidance**: PA and LED guidance shown in this dashboard are software simulations.
> 6. **AI Models**: Real pretrained YOLOv8 Nano checkpoints running in CPU mode.
> 7. **Dynamic Routing**: Shortest-path calculations are computed dynamically by NetworkX Dijkstra.

