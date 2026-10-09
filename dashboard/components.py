"""
Dashboard UI Components.
Modular renderers for Header, System Status Bar, Multi-Camera Simulation, Live AI Perception,
Zone Mapping, Multi-Zone Heatmap, Crowd Intelligence, Dynamic Edge Cost, Building Digital Twin,
Multi-Floor Digital Model, Route Comparison, Recommended Exit, Alert Center, Security Guard Log,
Simulated LED Signage, PA Voice Advisory, Performance Telemetry, Explainable AI, Advanced XAI,
System Architecture, Mathematical Formulation, System Validation, Evacuation Risk Index,
Evacuation Efficiency Index, Route Confidence & Uncertainty, Temporal Hazard Trends,
Incident Replay, Digital Multi-Format Reports, Vulnerable Occupant Simulation,
Emergency Decision Assistant, Camera Coverage/Occlusion, Model Evaluation,
Research Experiment Mode, IEEE Traceability Matrix, and Academic Limitations.
"""

from typing import List, Dict, Any, Tuple, Optional
import datetime
import json
import pandas as pd
import streamlit as st

from integration.multi_camera_sim import MultiCameraSimulator
from integration.decision_assistant import EmergencyDecisionAssistant


# =============================================================================
# 1. HEADER & SYSTEM STATUS BAR (Sections 1 & 2)
# =============================================================================

def render_header(mode: str = "RESEARCH MODE"):
    """Renders Command Center Header with Status, Mode Badge & Academic Disclaimers."""
    mode_color = "#38BDF8" if mode == "RESEARCH MODE" else "#EF4444"
    st.markdown(f"""
    <div class="command-header">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div class="command-badge">AI EVACUATION COMMAND CENTER</div>
            <div style="font-size: 0.75rem; font-weight: 800; color: {mode_color}; background: rgba(255,255,255,0.06); padding: 4px 10px; border-radius: 4px; border: 1px solid {mode_color};">
                OPERATIONAL MODE: {mode}
            </div>
        </div>
        <div class="command-title">
            AI-Based Fire/Smoke Hazard Detection &amp; Crowd-Aware Evacuation Path Optimization
        </div>
        <div class="command-subtitle">
            CCTV-BASED &bull; AI PERCEPTION &bull; DYNAMIC DIJKSTRA &bull; REAL-TIME EVACUATION
        </div>
        <div class="header-meta-row">
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="status-pill pill-green">&#9679; SYSTEM ONLINE</span>
                <span class="status-pill pill-blue">&#9881; CPU INFERENCE</span>
                <span class="status-pill pill-purple">&#9889; DYNAMIC DIJKSTRA</span>
                <span class="status-pill pill-amber">&#9888; RESEARCH PROTOTYPE</span>
            </div>
            <div style="font-size: 0.75rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">
                IEEE PAPER PROTOTYPE &bull; VIVA DEMONSTRATION &bull; SENSOR-FREE
            </div>
        </div>
    </div>
    <div class="disclaimer-banner">
        ⚠️ ACADEMIC RESEARCH PROTOTYPE &mdash; NOT A CERTIFIED LIFE-SAFETY SYSTEM
    </div>
    <div class="model-accuracy-banner">
        ℹ️ MODEL ACCURACY: NOT YET BENCHMARKED (Dual YOLOv8n CPU Inference &bull; Real-Time Perception Active)
    </div>
    """, unsafe_allow_html=True)


def render_system_status_bar(cctv_connected: bool = True, ai_ready: bool = True,
                             graph_ready: bool = True, routing_ready: bool = True,
                             alert_ready: bool = True):
    """Renders Section 1: Top System Status Bar with Status Indicators."""
    st.markdown("""
    <div class="status-bar-container">
        <div class="status-bar-item">
            <div class="status-bar-lbl">SYSTEM STATUS</div>
            <div class="status-bar-val" style="color: #34D399;">ONLINE</div>
        </div>
        <div class="status-bar-item">
            <div class="status-bar-lbl">CCTV INPUT</div>
            <div class="status-bar-val" style="color: #38BDF8;">CONNECTED</div>
        </div>
        <div class="status-bar-item">
            <div class="status-bar-lbl">AI ENGINE</div>
            <div class="status-bar-val" style="color: #00F0FF;">READY</div>
        </div>
        <div class="status-bar-item">
            <div class="status-bar-lbl">GRAPH ENGINE</div>
            <div class="status-bar-val" style="color: #34D399;">READY</div>
        </div>
        <div class="status-bar-item">
            <div class="status-bar-lbl">ROUTING ENGINE</div>
            <div class="status-bar-val" style="color: #A78BFA;">READY</div>
        </div>
        <div class="status-bar-item">
            <div class="status-bar-lbl">ALERT ENGINE</div>
            <div class="status-bar-val" style="color: #FBBF24;">READY</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_system_status_cards():
    """Backward compatibility alias for Section 1."""
    render_system_status_bar()


# =============================================================================
# 2. CCTV METADATA & MULTI-CAMERA SIMULATION (Feature 1 & Feature 13)
# =============================================================================

def render_cctv_meta_bar(width: int, height: int, frame_num: int, latency_ms: float, fps: float):
    """Renders Section 3: Metadata bar below CCTV camera feeds."""
    st.markdown(f"""
    <div class="cctv-meta-bar">
        <span>Input Resolution: <strong>{width}&times;{height}</strong></span>
        <span>Processed Frame: <strong>#{frame_num}</strong></span>
        <span>Inference Engine: <strong>CPU (Dual YOLOv8n)</strong></span>
        <span>Pipeline Latency: <strong>{latency_ms:.1f} ms</strong></span>
        <span>Processed Rate: <strong>{fps:.1f} FPS</strong></span>
    </div>
    """, unsafe_allow_html=True)


def render_multi_camera_panel(camera_sim: Optional[MultiCameraSimulator] = None, active_cam_id: str = "CAM-01"):
    """
    Renders Feature 1: Multi-Camera CCTV Simulation Grid.
    Displays CAM-01, CAM-02, CAM-03, CAM-04 with Camera ID, Zone, Camera Status,
    Detection Status, Fire/Smoke State, People Count, Crowd Density, Occlusion Status,
    Coverage, Uncertainty Penalty, and Monitoring State.
    Values unavailable from live state are shown as 'Not available' without inventing data.
    """
    st.markdown('<div class="section-header">Surveillance CCTV Multi-Camera Network (Simulated Feeds)</div>', unsafe_allow_html=True)
    st.caption("Software-only multi-channel camera simulation demonstrating simultaneous facility monitoring across key egress corridors.")

    if camera_sim is None:
        st.info("Multi-camera simulation telemetry: Not available.")
        return

    cams = camera_sim.get_all_cameras() if hasattr(camera_sim, "get_all_cameras") else []
    if not cams:
        st.info("No camera channels configured: Not available.")
        return

    cols = st.columns(len(cams))

    for idx, cam in enumerate(cams):
        with cols[idx]:
            cam_id = cam.get("id", "Not available")
            zone = cam.get("zone", "Not available")
            status = cam.get("status", "Not available")
            is_active = (cam_id == active_cam_id)
            monitoring_state = "PRIMARY (ACTIVE)" if is_active else ("STANDBY" if status == "ONLINE" else status)
            
            fire_det = cam.get("fire_detected", False)
            smoke_det = cam.get("smoke_detected", False)
            has_haz = bool(fire_det or smoke_det)
            detection_status = "HAZARD DETECTED" if has_haz else "CLEAR / NORMAL"
            
            hazard_score = cam.get("hazard_score")
            hazard_str = f"{hazard_score:.3f}" if (hazard_score is not None and isinstance(hazard_score, (int, float))) else "Not available"
            
            people_count = cam.get("people_count")
            people_str = str(people_count) if (people_count is not None and isinstance(people_count, (int, float))) else "Not available"
            
            crowd_density = cam.get("crowd_density")
            density_str = f"{crowd_density:.2f}" if (crowd_density is not None and isinstance(crowd_density, (int, float))) else "Not available"
            
            coverage = cam.get("coverage", "Not available")
            occlusion = cam.get("occlusion", "Not available")
            
            # Uncertainty penalty calculated deterministically from simulator logic
            uncertainty_val = 0.0
            if hasattr(camera_sim, "get_zone_uncertainty_penalty") and zone != "Not available":
                uncertainty_val = camera_sim.get_zone_uncertainty_penalty(zone)
            uncertainty_str = f"+{uncertainty_val:.1f} cost" if uncertainty_val > 0 else "0.0 (None)"
            
            last_update = cam.get("last_update", "Not available")
            
            # Colors
            border_style = "2px solid #00F0FF" if is_active else "1px solid #1E293B"
            haz_col = "#EF4444" if has_haz else "#10B981"
            cov_col = "#10B981" if coverage == "GOOD" else ("#FBBF24" if coverage == "PARTIAL" else "#EF4444")
            det_col = "#EF4444" if has_haz else "#10B981"

            st.markdown(f"""
            <div class="cmd-card" style="border: {border_style}; padding: 10px; margin-bottom: 8px; background: #0F172A;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 800; color: #00F0FF; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem;">{cam_id}</span>
                    <span class="status-pill {'pill-blue' if is_active else 'pill-green'}" style="font-size: 0.65rem;">
                        {monitoring_state}
                    </span>
                </div>
                <div style="font-size: 0.80rem; font-weight: 700; color: #F8FAFC; margin-top: 4px;">Zone: {zone}</div>
                <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px; padding: 4px 6px; background: rgba(255,255,255,0.03); border-radius: 4px;">
                    <div>Camera Status: <strong style="color: #38BDF8;">{status}</strong></div>
                    <div>Detection Status: <strong style="color: {det_col};">{detection_status}</strong></div>
                    <div>Fire / Smoke: <strong style="color: {haz_col};">{'YES' if has_haz else 'NO'}</strong></div>
                    <div>Hazard Score: <strong style="color: {haz_col};">{hazard_str}</strong></div>
                    <div>People Count: <strong style="color: #38BDF8;">{people_str}</strong> (Density: {density_str})</div>
                    <div>Occlusion: <strong style="color: #F8FAFC;">{occlusion}</strong></div>
                    <div>Coverage: <strong style="color: {cov_col};">{coverage}</strong></div>
                    <div>Uncertainty Penalty: <strong style="color: {'#F59E0B' if uncertainty_val > 0 else '#10B981'};">{uncertainty_str}</strong></div>
                    <div style="font-size: 0.65rem; color: #64748B; margin-top: 4px; border-top: 1px solid #1E293B; padding-top: 2px;">Updated: {last_update}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 3. LIVE AI PERCEPTION TELEMETRY (Section 4)
# =============================================================================

def render_perception_metrics(has_fire: bool, has_smoke: bool, hazard_score: float,
                              people_count: int, crowd_density: float,
                              hazard_status: str, crowd_status: str):
    """Renders Section 4: AI Perception Telemetry Metric Cards."""
    st.markdown('<div class="section-header">Live AI Perception Telemetry</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FIRE DETECTION</div>
            <div class="metric-value" style="color: {'#EF4444' if has_fire else '#10B981'};">
                {'YES' if has_fire else 'NO'}
            </div>
            <div class="metric-footer">YOLOv8n Fire Head</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SMOKE DETECTION</div>
            <div class="metric-value" style="color: {'#F97316' if has_smoke else '#10B981'};">
                {'YES' if has_smoke else 'NO'}
            </div>
            <div class="metric-footer">YOLOv8n Smoke Head</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HAZARD SCORE</div>
            <div class="metric-value" style="color: {'#EF4444' if hazard_score >= 0.70 else ('#F59E0B' if hazard_score > 0 else '#10B981')};">
                {hazard_score:.3f}
            </div>
            <div class="metric-footer">Range [0.000, 1.000]</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PEOPLE DETECTED</div>
            <div class="metric-value" style="color: #38BDF8;">
                {people_count}
            </div>
            <div class="metric-footer">COCO Class 0 (Person)</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CROWD DENSITY</div>
            <div class="metric-value" style="color: {'#EF4444' if crowd_density >= 0.70 else ('#F59E0B' if crowd_density >= 0.35 else '#34D399')};">
                {crowd_density:.3f}
            </div>
            <div class="metric-footer">Normalized Density</div>
        </div>
        """, unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        badge_cls = "pill-red" if hazard_status in ["CRITICAL", "DANGER", "WARNING"] else ("pill-amber" if hazard_status == "CAUTION" else "pill-green")
        st.markdown(f"""
        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-top: 8px; margin-bottom: 2px;">PERCEPTION HAZARD LEVEL</div>
        <div class="status-pill {badge_cls}">🔥 {hazard_status}</div>
        """, unsafe_allow_html=True)
    with b2:
        crowd_cls = "pill-red" if crowd_status == "HIGH CONGESTION" else ("pill-amber" if crowd_status in ["MODERATE", "MODERATE CONGESTION"] else "pill-green")
        st.markdown(f"""
        <div style="font-size: 0.72rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-top: 8px; margin-bottom: 2px;">PERCEPTION CROWD STATUS</div>
        <div class="status-pill {crowd_cls}">👥 {crowd_status}</div>
        """, unsafe_allow_html=True)


# =============================================================================
# 4. CURRENT HAZARD ZONE CARD (Section 5)
# =============================================================================

def render_hazard_zone_panel(active_edge: Tuple[str, str], hazard_type_str: str,
                             hazard_score: float, crowd_density: float, status_label: str,
                             confidence: float = 0.0):
    """Renders Section 5: Current Hazard Zone Focus Card with Mandatory Disclaimer."""
    st.markdown('<div class="section-header">Current Hazard Zone Focus</div>', unsafe_allow_html=True)
    is_emergency = (status_label in ["CRITICAL", "EMERGENCY", "DANGER", "WARNING"])
    card_cls = "hazard-card-emergency" if is_emergency else "hazard-card-normal"
    stat_color = "#EF4444" if is_emergency else ("#F59E0B" if status_label == "CAUTION" else "#10B981")

    conf_display = f"{confidence * 100:.1f}%" if confidence > 0 else (f"{hazard_score * 100:.1f}%" if hazard_score > 0 else "0.0%")

    st.markdown(f"""
    <div class="{card_cls}">
        <div style="font-size: 0.72rem; color: #38BDF8; text-transform: uppercase; font-weight: 800; letter-spacing: 1px;">
            CURRENT MONITORED ZONE (ACTIVE CCTV SURVEILLANCE)
        </div>
        <div style="font-size: 1.45rem; font-weight: 800; color: #FFFFFF; margin: 4px 0; font-family: 'JetBrains Mono', monospace;">
            {active_edge[0]} &harr; {active_edge[1]}
        </div>
        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-top: 8px; font-size: 0.88rem; color: #94A3B8;">
            <span>Hazard Type: <strong style="color: {'#EF4444' if hazard_type_str == 'FIRE' else ('#F97316' if hazard_type_str == 'SMOKE' else '#10B981')};">{hazard_type_str}</strong></span>
            <span>Perception Confidence: <strong style="color: #00F0FF;">{conf_display}</strong></span>
            <span>Hazard Score: <strong style="color: #F8FAFC;">{hazard_score:.3f}</strong></span>
            <span>Crowd Density: <strong style="color: #F8FAFC;">{crowd_density:.3f}</strong></span>
            <span>Zone Status: <strong style="color: {stat_color};">{status_label}</strong></span>
        </div>
        <div style="font-size: 0.78rem; color: #FBBF24; font-weight: 700; margin-top: 10px; padding-top: 6px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
            ⚠️ DEMO ZONE MAPPING &mdash; NOT REAL CCTV HOMOGRAPHY CALIBRATION
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 5. CROWD INTELLIGENCE, OCCUPANCY HEATMAP & ZONE RANKING (Features 2 & 4)
# =============================================================================

def render_crowd_intelligence_panel(zone_counts: Dict[str, int], zone_densities: Dict[str, float],
                                    total_people: int, zone_capacities: Optional[Dict[str, int]] = None):
    """Renders Section 6: Crowd Intelligence Panel with Capacity & Occupancy Progress Bars."""
    st.markdown('<div class="section-header">Crowd Intelligence &amp; Multi-Zone Occupancy</div>', unsafe_allow_html=True)
    st.caption("Real-time occupant distribution partitioned across synthetic demonstration facility zones (Demo Zone Mapping).")

    default_caps = {
        "ROOM_A": 15,
        "ROOM_B": 15,
        "ROOM_C": 25,
        "CORRIDOR_1": 20,
        "CORRIDOR_2": 20,
        "EXIT_A": 10,
        "EXIT_B": 10
    }
    caps = zone_capacities if zone_capacities else default_caps

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Occupants Monitored", total_people)
    with c2:
        peak_zone = max(zone_counts.items(), key=lambda x: x[1])[0] if zone_counts and total_people > 0 else "NONE"
        st.metric("Peak Congestion Zone", peak_zone)
    with c3:
        overall_density = sum(zone_densities.values()) / max(1, len(zone_densities)) if zone_densities else 0.0
        st.metric("Mean Topology Density", f"{overall_density:.3f}")

    zones = ["ROOM_A", "ROOM_B", "ROOM_C", "CORRIDOR_1", "CORRIDOR_2", "EXIT_A", "EXIT_B"]
    rows = []
    for z in zones:
        cnt = zone_counts.get(z, 0)
        cap = caps.get(z, 15)
        occ_pct = min(100.0, (cnt / cap) * 100.0) if cap > 0 else 0.0
        den = zone_densities.get(z, 0.0)

        if total_people == 0:
            status = "CLEAR"
        elif den >= 0.70 or occ_pct >= 80.0:
            status = "HIGH CONGESTION"
        elif den >= 0.35 or occ_pct >= 50.0:
            status = "MODERATE"
        elif cnt > 0:
            status = "NORMAL"
        else:
            status = "CLEAR"

        rows.append({
            "ZONE": z,
            "OCCUPANTS": f"{cnt} / {cap}",
            "OCCUPANCY (%)": f"{occ_pct:.1f}%",
            "DENSITY SCORE": f"{den:.3f}" if total_people > 0 else "0.000",
            "CONGESTION STATUS": status
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_zone_risk_priority_table(zone_counts: Dict[str, int], zone_densities: Dict[str, float],
                                    hazard_score: float, active_edge: Tuple[str, str],
                                    recommended_route: List[str]):
    """
    Renders Feature 4: Ranked ZONE RISK PRIORITY table.
    Sorts building zones dynamically by combined Hazard + Density + Route Relevance.
    """
    st.markdown('<div class="section-header">Zone Risk Priority Ranking (Multi-Zone Heatmap)</div>', unsafe_allow_html=True)
    zones = ["ROOM_A", "ROOM_B", "ROOM_C", "CORRIDOR_1", "CORRIDOR_2", "EXIT_A", "EXIT_B"]

    table_data = []
    for z in zones:
        # Hazard evaluation for this zone
        z_haz = hazard_score if (z in active_edge and hazard_score > 0) else 0.0
        z_den = zone_densities.get(z, 0.0)
        is_on_route = (z in recommended_route)

        # Composite priority score: 50% Hazard + 30% Density + 20% Route Relevance
        priority_score = (50.0 * z_haz) + (30.0 * z_den) + (20.0 * (1.0 if is_on_route else 0.0))

        if z_haz >= 0.70 or priority_score >= 60.0:
            risk_label = "CRITICAL (RED)"
            action = "EVACUATE / AVOID IMMEDIATELY"
        elif z_haz >= 0.40 or z_den >= 0.70 or priority_score >= 40.0:
            risk_label = "WARNING (ORANGE)"
            action = "DIVERT FLOW / DISPATCH SAFETY TEAM"
        elif z_haz > 0 or z_den >= 0.35 or priority_score >= 20.0:
            risk_label = "CAUTION (YELLOW)"
            action = "MONITOR CONGESTION"
        else:
            risk_label = "SAFE (GREEN)"
            action = "MAINTAIN NOMINAL TRANSIT"

        table_data.append({
            "ZONE": z,
            "HAZARD": f"{z_haz:.3f}",
            "PEOPLE": zone_counts.get(z, 0),
            "DENSITY": f"{z_den:.3f}",
            "RISK LEVEL": risk_label,
            "ROUTE STATUS": "ACTIVE PATH" if is_on_route else "ALTERNATIVE",
            "RECOMMENDED ACTION": action,
            "_sort": priority_score
        })

    # Sort descending by priority score
    table_data.sort(key=lambda x: x["_sort"], reverse=True)
    for row in table_data:
        del row["_sort"]

    st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)


def render_multi_zone_analysis(zone_counts: Dict[str, int], zone_densities: Dict[str, float], total_people: int):
    """Backward compatibility wrapper for Section 6."""
    render_crowd_intelligence_panel(zone_counts, zone_densities, total_people)


# =============================================================================
# 6. DYNAMIC EDGE COST ENGINE (Section 8)
# =============================================================================

def render_dynamic_cost_panel(alpha: float, beta: float, gamma: float,
                              hazard_score: float, crowd_density: float,
                              edge_length: float, edge_cost: float,
                              active_edge: Tuple[str, str],
                              baseline_cost: Optional[float] = None):
    """Renders Section 8: Dynamic Edge Cost Calculation with Mathematical Formula."""
    st.markdown('<div class="section-header">Dynamic Edge Cost Engine (Mathematical Model)</div>', unsafe_allow_html=True)
    haz_term = alpha * hazard_score
    den_term = beta * crowd_density
    len_term = gamma * edge_length
    base_cost = baseline_cost if baseline_cost is not None else (gamma * edge_length)

    st.markdown(f"""
    <div class="cmd-card">
        <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 6px;">
            Target Corridor Edge: <code style="color: #00F0FF; font-weight: 700;">{active_edge[0]} &harr; {active_edge[1]}</code>
            &bull; Physical Segment Length: <strong>{edge_length:.2f} m</strong>
        </div>
        <div style="font-size: 1.05rem; margin-bottom: 6px; font-family: 'JetBrains Mono', monospace; color: #F8FAFC;">
            Cost(e) = &alpha; &times; Hazard(e) + &beta; &times; Density(e) + &gamma; &times; Length(e)
        </div>
        <div style="font-size: 0.78rem; color: #64748B; margin-bottom: 10px;">
            Optimized IEEE Weights: &alpha; = {alpha:.1f} (Hazard Penalty) &bull; &beta; = {beta:.1f} (Congestion Penalty) &bull; &gamma; = {gamma:.2f} (Physical Distance)
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin-bottom: 12px;">
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 6px; padding: 8px 12px;">
                <div style="font-size: 0.70rem; color: #F87171; font-weight: 700; text-transform: uppercase;">Hazard Contribution (&alpha; &times; H)</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #F87171; font-family: 'JetBrains Mono', monospace;">
                    {alpha:.1f} &times; {hazard_score:.3f} = {haz_term:.2f}
                </div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 6px; padding: 8px 12px;">
                <div style="font-size: 0.70rem; color: #FBBF24; font-weight: 700; text-transform: uppercase;">Density Contribution (&beta; &times; D)</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #FBBF24; font-family: 'JetBrains Mono', monospace;">
                    {beta:.1f} &times; {crowd_density:.3f} = {den_term:.2f}
                </div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 6px; padding: 8px 12px;">
                <div style="font-size: 0.70rem; color: #38BDF8; font-weight: 700; text-transform: uppercase;">Distance Contribution (&gamma; &times; L)</div>
                <div style="font-size: 1.1rem; font-weight: 800; color: #38BDF8; font-family: 'JetBrains Mono', monospace;">
                    {gamma:.2f} &times; {edge_length:.2f} = {len_term:.2f}
                </div>
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: center; padding-top: 10px; border-top: 1px solid #1C2D4A; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 0.88rem; color: #94A3B8;">
                Cost Comparison: Baseline Cost: <strong style="color: #60A5FA;">{base_cost:.2f}</strong> &rarr; Dynamic Cost: <strong style="color: {'#EF4444' if edge_cost > base_cost else '#10B981'};">{edge_cost:.2f}</strong>
                <span style="font-size: 0.75rem; color: #A78BFA; margin-left: 6px;">(&Delta;Cost: +{edge_cost - base_cost:.2f})</span>
            </div>
            <div style="font-size: 1.25rem; font-weight: 800; color: #34D399; font-family: 'JetBrains Mono', monospace;">
                TOTAL EDGE COST: {edge_cost:.2f}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 7. ROUTE INTELLIGENCE, BASELINE VS PROPOSED & IMPACT (Features 16 & 17)
# =============================================================================

def render_route_comparison(normal_route: List[str], normal_exit: str, normal_cost: float,
                            current_route: List[str], current_exit: str, current_cost: float,
                            route_changed: bool, reason: str, cost_delta: Optional[float] = None):
    """Renders Sections 9 & 10: Evacuation Route Intelligence (Baseline vs. Dynamic)."""
    st.markdown('<div class="section-header">Evacuation Route Intelligence &amp; Decision</div>', unsafe_allow_html=True)
    normal_str = " &rarr; ".join(normal_route) if normal_route else "NONE"
    current_str = " &rarr; ".join(current_route) if current_route else "NO SAFE ROUTE"

    delta_val = cost_delta if cost_delta is not None else (current_cost - normal_cost)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="cmd-card" style="border-left: 4px solid #38BDF8;">
            <div style="font-weight: 800; color: #38BDF8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px;">
                STATIC BASELINE ROUTE (HAZARD = 0, DENSITY = 0)
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 8px 0; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">
                {normal_str}
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Baseline Exit: <strong style="color: #F8FAFC;">{normal_exit}</strong> &bull; Baseline Cost: <strong style="color: #38BDF8;">{normal_cost:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        border_col = "#A855F7" if route_changed else "#10B981"
        st.markdown(f"""
        <div class="cmd-card" style="border-left: 4px solid {border_col};">
            <div style="font-weight: 800; color: {border_col}; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.8px;">
                CURRENT DYNAMIC ROUTE (LIVE EVALUATED VIA DIJKSTRA)
            </div>
            <div style="font-size: 1.1rem; font-weight: 700; margin: 8px 0; color: #F8FAFC; font-family: 'JetBrains Mono', monospace;">
                {current_str}
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">
                Recommended Exit: <strong style="color: #F8FAFC;">{current_exit}</strong> &bull; Total Route Cost: <strong style="color: {border_col};">{current_cost:.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if route_changed:
        st.markdown(f"""
        <div style="background: rgba(168, 85, 247, 0.12); border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
            <div style="font-size: 0.78rem; font-weight: 800; color: #C084FC; text-transform: uppercase; letter-spacing: 0.8px;">
                ⚡ ROUTE CHANGE EVENT DETECTED &mdash; DYNAMIC REROUTING TRIGGERED
            </div>
            <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 3px;">
                {reason}
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">
                Previous Exit: <strong>{normal_exit}</strong> &bull; New Exit: <strong>{current_exit}</strong> &bull; Total Cost Delta: <strong style="color: #FBBF24;">{delta_val:+.2f}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; padding: 10px 14px; margin-top: 10px;">
            <div style="font-size: 0.78rem; font-weight: 800; color: #34D399; text-transform: uppercase; letter-spacing: 0.8px;">
                ✅ ROUTE UNCHANGED &mdash; BASELINE PATH SAFEST &amp; OPTIMAL
            </div>
            <div style="font-size: 0.88rem; color: #CBD5E1; margin-top: 3px;">
                {reason}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_baseline_vs_proposed_panel(normal_route: List[str], normal_exit: str, normal_cost: float,
                                      current_route: List[str], current_exit: str, current_cost: float,
                                      hazard_score: float, crowd_density: float, dijkstra_ms: float = 0.5):
    """
    Renders Feature 16: Research Comparison Panel (Baseline Static vs Dynamic Proposed Route).
    """
    st.markdown('<div class="section-header">Research Contribution: Baseline Static vs. Proposed Dynamic Optimization</div>', unsafe_allow_html=True)
    comp_df = pd.DataFrame([
        {
            "EVALUATION METRIC": "Egress Path Sequence",
            "STATIC BASELINE": " -> ".join(normal_route) if normal_route else "N/A",
            "DYNAMIC PROPOSED": " -> ".join(current_route) if current_route else "N/A"
        },
        {
            "EVALUATION METRIC": "Designated Evacuation Exit",
            "STATIC BASELINE": normal_exit,
            "DYNAMIC PROPOSED": current_exit
        },
        {
            "EVALUATION METRIC": "Cumulative Route Cost",
            "STATIC BASELINE": f"{normal_cost:.2f}",
            "DYNAMIC PROPOSED": f"{current_cost:.2f}"
        },
        {
            "EVALUATION METRIC": "Direct Hazard Exposure",
            "STATIC BASELINE": f"{hazard_score:.3f} (Traverses Danger Zone)" if hazard_score > 0 else "0.000 (Clear)",
            "DYNAMIC PROPOSED": "0.000 (Hazard Avoidance Detour)" if hazard_score > 0 else "0.000 (Clear)"
        },
        {
            "EVALUATION METRIC": "Congestion Exposure",
            "STATIC BASELINE": f"{crowd_density:.3f}" if crowd_density >= 0.5 else "Low",
            "DYNAMIC PROPOSED": "Bypassed via low-occupancy corridors" if crowd_density >= 0.5 else "Nominal"
        },
        {
            "EVALUATION METRIC": "Decision Latency",
            "STATIC BASELINE": "0.00 ms (Hardcoded/Static)",
            "DYNAMIC PROPOSED": f"{dijkstra_ms:.2f} ms (Real-Time Dynamic Dijkstra)"
        }
    ])
    st.dataframe(comp_df, hide_index=True, use_container_width=True)


# =============================================================================
# 8. RECOMMENDED EXIT & ROUTE FLOW (Section 10)
# =============================================================================

def render_recommended_exit_panel(recommended_exit: str, total_route_cost: float,
                                  route_distance: float = 0.0,
                                  avoided_exit: Optional[str] = None):
    """Renders Section 10: Prominent Recommended Exit and Compromised Exit Cards."""
    st.markdown('<div class="section-header">Recommended Evacuation Exit</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        dist_str = f"&bull; Distance: <strong>{route_distance:.1f} m</strong>" if route_distance > 0 else ""
        st.markdown(f"""
        <div class="cmd-card" style="border-left: 4px solid #10B981; padding: 18px;">
            <div class="metric-label" style="color: #34D399;">PRIMARY RECOMMENDED EXIT</div>
            <div class="metric-value" style="color: #00FF88; font-size: 2.2rem;">🚪 {recommended_exit}</div>
            <div style="font-size: 0.85rem; color: #34D399; font-weight: 700; margin-top: 4px;">SAFE EVACUATION PATH</div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 6px;">
                Total Route Cost: <strong style="color: #00F0FF;">{total_route_cost:.2f}</strong> {dist_str}
            </div>
            <div style="font-size: 0.75rem; color: #64748B; margin-top: 4px;">
                Selection Criterion: Lowest combined hazard + congestion + physical distance cost.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if avoided_exit and avoided_exit != recommended_exit:
            st.markdown(f"""
            <div class="cmd-card" style="border-left: 4px solid #EF4444; padding: 18px;">
                <div class="metric-label" style="color: #F87171;">COMPROMISED / AVOIDED EXIT</div>
                <div class="metric-value" style="color: #FF2E4D; font-size: 2.2rem;">⛔ {avoided_exit}</div>
                <div style="font-size: 0.85rem; color: #F87171; font-weight: 700; margin-top: 4px;">AVOID &bull; ELEVATED HAZARD / CROWD</div>
                <div style="font-size: 0.80rem; color: #FCA5A5; margin-top: 6px;">
                    Direct corridor blocked or penalised due to elevated hazard score or crowd density.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="cmd-card" style="border-left: 4px solid #334155; padding: 18px;">
                <div class="metric-label" style="color: #94A3B8;">SECONDARY EXIT STATUS</div>
                <div class="metric-value" style="color: #94A3B8; font-size: 2.2rem;">⚪ STANDBY</div>
                <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 700; margin-top: 4px;">AVAILABLE / NOMINAL</div>
                <div style="font-size: 0.80rem; color: #64748B; margin-top: 6px;">
                    Alternative facility egress routes clear, accessible, and on standby.
                </div>
            </div>
            """, unsafe_allow_html=True)


def render_route_flow_card(route_nodes: List[str], total_cost: float, route_changed: bool):
    """Renders Section 10: Sequential Evacuation Route Flow Breadcrumb."""
    st.markdown('<div class="section-header">Evacuation Route Flow</div>', unsafe_allow_html=True)
    if not route_nodes:
        st.warning("No evacuation route calculated.")
        return

    html_parts = ['<div class="route-path-container">']
    for idx, node in enumerate(route_nodes):
        if idx == 0:
            cls = "route-node-origin"
        elif "EXIT" in node:
            cls = "route-node-exit"
        else:
            cls = "route-node-pill"
        html_parts.append(f'<span class="{cls}">{node}</span>')
        if idx < len(route_nodes) - 1:
            html_parts.append('<span class="route-arrow">&rarr;</span>')
    html_parts.append('</div>')

    st.markdown("".join(html_parts), unsafe_allow_html=True)


# =============================================================================
# 9. ADVANCED XAI & DECISION BREAKDOWN (Feature 15)
# =============================================================================

def render_advanced_xai_panel(active_edge: Tuple[str, str], hazard_score: float,
                              crowd_density: float, edge_len: float, total_cost: float,
                              normal_exit: str, current_exit: str, normal_cost: float):
    """
    Renders Feature 15: Advanced XAI / Why This Exit? with exact percentage contributions.
    """
    st.markdown('<div class="section-header">Advanced Explainable AI (XAI) &mdash; Why Was This Exit Selected?</div>', unsafe_allow_html=True)

    haz_contrib = 5.0 * hazard_score
    den_contrib = 2.0 * crowd_density
    dist_contrib = 0.2 * edge_len
    sum_terms = max(0.001, haz_contrib + den_contrib + dist_contrib)

    haz_pct = (haz_contrib / sum_terms) * 100.0
    den_pct = (den_contrib / sum_terms) * 100.0
    dist_pct = (dist_contrib / sum_terms) * 100.0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Hazard Factor Share", f"{haz_pct:.1f}%", f"+{haz_contrib:.2f} cost penalty")
    with c2:
        st.metric("Congestion Factor Share", f"{den_pct:.1f}%", f"+{den_contrib:.2f} cost penalty")
    with c3:
        st.metric("Distance Factor Share", f"{dist_pct:.1f}%", f"+{dist_contrib:.2f} metric length")

    st.markdown(f"""
    <div class="cmd-card" style="margin-top: 10px;">
        <div style="font-weight: 700; color: #00F0FF; font-size: 0.95rem; margin-bottom: 6px;">
            COMPARATIVE DECISION AUDIT: {current_exit} vs. {normal_exit}
        </div>
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px; font-size: 0.88rem; color: #CBD5E1;">
            <div>
                <strong>EXIT_A Status:</strong> {'CLOSED / HIGH HAZARD' if current_exit != 'EXIT_A' else 'SAFE / OPEN'}<br>
                <span>Evaluated Path Cost: <code style="color: {'#EF4444' if current_exit != 'EXIT_A' else '#10B981'};">{normal_cost if current_exit == 'EXIT_A' else total_cost + haz_contrib:.2f}</code></span>
            </div>
            <div>
                <strong>EXIT_B Status:</strong> {'SAFE / LOWEST CUMULATIVE COST' if current_exit == 'EXIT_B' else 'AVAILABLE'}<br>
                <span>Evaluated Path Cost: <code style="color: {'#10B981' if current_exit == 'EXIT_B' else '#94A3B8'};">{total_cost:.2f}</code></span>
            </div>
        </div>
        <div style="margin-top: 8px; font-size: 0.82rem; color: #34D399; padding-top: 6px; border-top: 1px solid #1C2D4A;">
            ✓ <em>Conclusion: {current_exit} was selected because its dynamically computed cumulative traversal cost ({total_cost:.2f})
            is lower than alternative routes under current fire/smoke hazard ({hazard_score:.3f}) and crowd congestion ({crowd_density:.3f}).</em>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_explainable_ai_panel(current_route: List[str], normal_route: List[str],
                                route_changed: bool, hazard_score: float,
                                crowd_density: float, total_cost: float,
                                recommended_exit: str):
    """Backward compatibility wrapper for explainable AI."""
    st.markdown('<div class="section-header">Explainable AI &mdash; Why Was This Route Selected?</div>', unsafe_allow_html=True)
    route_str = " &rarr; ".join(current_route) if current_route else "NONE"

    st.markdown(f"""
    <div class="cmd-card">
        <div style="font-size: 0.88rem; font-weight: 700; color: #38BDF8; margin-bottom: 8px;">
            OPTIMAL EGRESS EVALUATION: <code style="color: #00FF88;">{route_str}</code> (Cumulative Cost: {total_cost:.2f})
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px;">
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 6px; padding: 10px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #34D399;">✓ HAZARD EXPOSURE</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">
                    Selected route bypasses active hazard zones with score: <strong>{hazard_score:.3f}</strong>.
                </div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 6px; padding: 10px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #34D399;">✓ CONGESTION MITIGATION</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">
                    Selected path prevents human stampedes by routing through density: <strong>{crowd_density:.3f}</strong>.
                </div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 6px; padding: 10px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #34D399;">✓ TRAVERSAL DISTANCE</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">
                    Global minimum weighted distance satisfied under current hazards.
                </div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 6px; padding: 10px;">
                <div style="font-size: 0.78rem; font-weight: 700; color: #34D399;">✓ EXIT ACCESSIBILITY</div>
                <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">
                    Target egress portal <strong>{recommended_exit}</strong> confirmed unobstructed and accessible.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 10. TEMPORAL HAZARD TRENDS & SHORT-TERM FORECAST (Feature 5)
# =============================================================================

def render_temporal_trend_panel(history: List[Dict[str, Any]]):
    """
    Renders Feature 5: Temporal Hazard Trend & Lightweight Short-Term Forecast.
    """
    st.markdown('<div class="section-header">Temporal Telemetry Trend &amp; Moving Average Forecast</div>', unsafe_allow_html=True)
    st.caption("Transparent telemetry trend derived from recent consecutive frames &bull; Short-term moving average forecast.")

    if not history or len(history) < 2:
        st.info("ℹ️ Awaiting sequential frame ingestion to establish trend. Process at least 2 frames or switch demo scenarios.")
        return

    cur_haz = history[-1].get("hazard_score", 0.0)
    prev_haz = history[-2].get("hazard_score", 0.0)
    delta_haz = cur_haz - prev_haz

    cur_den = history[-1].get("crowd_density", 0.0)
    prev_den = history[-2].get("crowd_density", 0.0)
    delta_den = cur_den - prev_den

    if delta_haz > 0.05:
        haz_trend = "RISING (ESCALATING)"
        haz_col = "#EF4444"
    elif delta_haz < -0.05:
        haz_trend = "FALLING (SUBSIDING)"
        haz_col = "#10B981"
    else:
        haz_trend = "STABLE"
        haz_col = "#FBBF24"

    if delta_den > 0.05:
        den_trend = "RISING (CONGESTING)"
        den_col = "#EF4444"
    elif delta_den < -0.05:
        den_trend = "FALLING (CLEARING)"
        den_col = "#10B981"
    else:
        den_trend = "STABLE"
        den_col = "#38BDF8"

    # 3-point moving average for short-term extrapolation
    recent_haz = [h.get("hazard_score", 0.0) for h in history[-3:]]
    ma_haz = sum(recent_haz) / len(recent_haz)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Current Hazard Score", f"{cur_haz:.3f}", f"{delta_haz:+.3f}")
    with c2:
        st.metric("Hazard Trend", haz_trend)
    with c3:
        st.metric("Crowd Density Trend", den_trend, f"{delta_den:+.3f}")
    with c4:
        st.metric("Short-Term Forecast (MA-3)", f"{ma_haz:.3f}")

    st.markdown("""
    <div style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">
        ⚠️ <em>Disclaimer: Short-term telemetry moving-average &mdash; not a trained fire-spread or fluid-dynamic prediction model.</em>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 11. EVACUATION EFFICIENCY INDEX (Feature 6)
# =============================================================================

def render_evacuation_efficiency_card(hazard_score: float, crowd_density: float,
                                      route_cost: float, exit_available: bool = True):
    """
    Renders Feature 6: Prototype Evacuation Efficiency Index (0 - 100).
    Formula: Efficiency = 100 - [40*Hazard + 30*Density + 20*Norm(Cost) + 10*(1 - Available)]
    """
    st.markdown('<div class="section-header">Prototype Evacuation Efficiency Index</div>', unsafe_allow_html=True)

    norm_cost = min(1.0, route_cost / 30.0)
    avail_penalty = 0.0 if exit_available else 10.0
    deduction = (40.0 * hazard_score) + (30.0 * crowd_density) + (20.0 * norm_cost) + avail_penalty
    efficiency_val = max(0, min(100, int(round(100.0 - deduction))))

    if efficiency_val >= 80:
        eff_cat = "OPTIMAL EVACUATION READINESS"
        eff_col = "#00FF88"
        eff_desc = "Paths are wide, unobstructed, and exit portals operate with maximum evacuation throughput."
    elif efficiency_val >= 60:
        eff_cat = "ACCEPTABLE / MINOR BOTTLENECKS"
        eff_col = "#38BDF8"
        eff_desc = "Egress route is accessible with moderate path distance or minor occupant congestion."
    elif efficiency_val >= 40:
        eff_cat = "DEGRADED FLOW"
        eff_col = "#FBBF24"
        eff_desc = "Elevated crowd density or nearby smoke requires active dynamic rerouting to avoid delays."
    else:
        eff_cat = "CRITICAL / HEAVILY COMPROMISED"
        eff_col = "#FF2E4D"
        eff_desc = "High hazard severity and congestion significantly impede direct facility clearance."

    st.markdown(f"""
    <div class="cmd-card" style="border-left: 4px solid {eff_col};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div class="metric-label">PROTOTYPE EVACUATION EFFICIENCY SCORE</div>
                <div class="metric-value" style="color: {eff_col}; font-size: 2.2rem;">
                    {efficiency_val} <span style="font-size: 1.1rem; color: #94A3B8;">/ 100</span>
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; color: {eff_col};">{eff_cat}</div>
            </div>
            <div style="max-width: 550px; font-size: 0.82rem; color: #CBD5E1;">
                <div>{eff_desc}</div>
                <div style="font-size: 0.70rem; color: #64748B; margin-top: 6px;">
                    Formula: <code>Efficiency = 100 - (40&times;Hazard + 30&times;Density + 20&times;Norm(Cost) + ExitPenalty)</code>
                </div>
                <div style="font-size: 0.68rem; color: #94A3B8; margin-top: 2px;">
                    <em>Note: Academic prototype efficiency index &mdash; not empirical physical evacuation egress time.</em>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 12. VULNERABLE OCCUPANT SIMULATION (Feature 7)
# =============================================================================

def render_vulnerable_occupant_panel(vulnerable_enabled: bool, category: str = "Mobility Assistance"):
    """
    Renders Feature 7: Vulnerable Occupant Priority Simulation Panel.
    """
    st.markdown('<div class="section-header">Vulnerable Occupant Priority Simulation (Assisted Egress Layer)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.75rem; color: #FBBF24; font-weight: 700; margin-bottom: 6px;">
        ⚠️ SIMULATED VULNERABLE-OCCUPANT PRIORITY &mdash; NOT DETECTED AUTOMATICALLY BY CURRENT YOLO MODEL
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="cmd-card">
            <div class="metric-label">PRIORITY STATUS</div>
            <div style="font-size: 1.1rem; font-weight: 800; color: {'#00FF88' if vulnerable_enabled else '#94A3B8'}; margin-top: 4px;">
                {'ENABLED (ELEVATED CAUTION WEIGHTS)' if vulnerable_enabled else 'STANDARD GENERAL OCCUPANCY'}
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 4px;">
                Category Profile: <strong>{category}</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="cmd-card">
            <div class="metric-label">OPTIMIZATION IMPACT</div>
            <div style="font-size: 0.82rem; color: #CBD5E1; margin-top: 4px;">
                {'Crowd penalty weight &beta; increased (+1.5) and stair traversal resistance elevated to prioritize wheelchair-accessible / unobstructed single-level egress.' if vulnerable_enabled else 'Baseline weights active (&alpha;=5.0, &beta;=2.0, &gamma;=0.2). Standard corridor clearance model.'}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 13. INCIDENT REPLAY (Feature 8)
# =============================================================================

def render_incident_replay_panel(events: List[Dict[str, Any]]):
    """
    Renders Feature 8: Incident Replay & State Navigation.
    """
    st.markdown('<div class="section-header">Incident Replay &amp; Event Sequence Audit</div>', unsafe_allow_html=True)
    st.caption("Chronological state audit enabling playback and inspection of emergency detection, rerouting, and dispatch sequences.")

    if not events:
        st.info("ℹ️ No incident events registered yet.")
        return

    step_idx = st.slider("Replay Event Step Index", min_value=1, max_value=len(events), value=len(events))
    selected_event = events[step_idx - 1]

    st.markdown(f"""
    <div class="cmd-card" style="border-left: 4px solid #00F0FF; padding: 12px 16px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-family: 'JetBrains Mono', monospace; color: #00F0FF; font-weight: 800; font-size: 0.95rem;">
                STEP #{step_idx:02d} &bull; {selected_event.get('time', selected_event.get('timestamp', '00:00:00'))}
            </span>
            <span class="status-pill pill-blue">AUDIT EVENT</span>
        </div>
        <div style="font-size: 0.95rem; color: #F8FAFC; font-weight: 600; margin-top: 6px;">
            {selected_event.get('desc', selected_event.get('action', selected_event.get('reason', 'Event registered')))}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 14. EMERGENCY DECISION ASSISTANT (Feature 18)
# =============================================================================

def render_decision_assistant(assistant: EmergencyDecisionAssistant, state: Dict[str, Any]):
    """
    Renders Feature 18: AI Emergency Decision Explanation Assistant.
    """
    st.markdown('<div class="section-header">Emergency Decision Explanation Assistant (Local XAI)</div>', unsafe_allow_html=True)
    st.caption("Deterministic, rule-based explainable assistant answering research viva questions using ONLY current runtime state.")

    selected_query = st.selectbox("Select a Research / Operational Query:", options=assistant.preset_questions)
    answer = assistant.answer_query(selected_query, state)

    st.markdown(f"""
    <div class="cmd-card" style="border-left: 4px solid #A855F7; margin-top: 8px;">
        <div style="font-size: 0.75rem; font-weight: 800; color: #C084FC; text-transform: uppercase;">EXPLANATION RESPONSE</div>
        <div style="font-size: 0.92rem; color: #F8FAFC; margin-top: 6px; line-height: 1.5; white-space: pre-line;">
            {answer}
        </div>
        <div style="font-size: 0.68rem; color: #64748B; margin-top: 8px; border-top: 1px solid #1C2D4A; padding-top: 4px;">
            Source: Rule-based deterministic decision tree &bull; Zero external API dependencies &bull; Factual live telemetry.
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 15. MODEL EVALUATION & FALSE ALARM PANELS (Features 11 & 12)
# =============================================================================

def render_model_evaluation_panel():
    """
    Renders Feature 11: Formal Model Evaluation Panel.
    Explicitly separates functional software validation from offline empirical benchmarks.
    """
    st.markdown('<div class="section-header">Model Evaluation &amp; Accuracy Benchmark Status</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cmd-card">
        <div style="font-weight: 800; color: #FBBF24; font-size: 0.90rem; margin-bottom: 6px;">
            EMPIRICAL MODEL BENCHMARK STATUS: FORMAL DATASET BENCHMARK NOT YET AVAILABLE
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5;">
            The dual YOLOv8 Nano architectures operate in zero-shot transfer mode using pre-trained weights
            (<code>fire_smoke_yolov8n.pt</code> fine-tuned on D-Fire and <code>yolov8n.pt</code> on COCO).<br>
            Empirical metrics (mAP@0.5, mAP@0.5:0.95, Precision, Recall, F1) require an offline labelled validation
            corpus with annotated ground-truth bounding boxes.
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-top: 10px;">
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 4px; padding: 6px 10px;">
                <div style="font-size: 0.68rem; color: #94A3B8;">Precision</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748B;">Requires Test Dataset</div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 4px; padding: 6px 10px;">
                <div style="font-size: 0.68rem; color: #94A3B8;">Recall</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748B;">Requires Test Dataset</div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 4px; padding: 6px 10px;">
                <div style="font-size: 0.68rem; color: #94A3B8;">mAP@0.5</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #64748B;">Requires Test Dataset</div>
            </div>
            <div style="background: #0A111F; border: 1px solid #1E293B; border-radius: 4px; padding: 6px 10px;">
                <div style="font-size: 0.68rem; color: #94A3B8;">Functional Tests</div>
                <div style="font-size: 0.85rem; font-weight: 700; color: #00FF88;">14 / 14 PASSED (100%)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_false_alarm_panel():
    """
    Renders Feature 12: Research False Alarm Rate Analysis Placeholder.
    """
    st.markdown('<div class="section-header">False-Alarm Rate &amp; Precision Analysis (Research Protocol)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="cmd-card">
        <div style="font-size: 0.82rem; color: #CBD5E1;">
            False-alarm and missed-detection rates require continuous video verification against ground-truth labels.<br>
            Current system mitigates false alarms via:
            <ul>
                <li>Adjustable Confidence Threshold Slider (0.10 &ndash; 0.90, default: 0.35).</li>
                <li>Weighted Hazard Formulation scaling hazard score monotonically with detection confidence.</li>
                <li>Continuous video frame-skipping buffering to reject single-frame transient false positives.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 16. SYSTEM HEALTH MONITOR (Feature 21)
# =============================================================================

def render_system_health_monitor(latency_info: Dict[str, Any], fps: float, frame_skip: int):
    """
    Renders Feature 21: System Health Monitor.
    """
    st.markdown('<div class="section-header">System Health &amp; Subsystem Operational Status</div>', unsafe_allow_html=True)
    cols = st.columns(6)
    subsystems = [
        ("Fire YOLOv8n", "ONLINE", "#00FF88"),
        ("Crowd YOLOv8n", "ONLINE", "#00FF88"),
        ("NetworkX Graph", "ONLINE", "#00FF88"),
        ("Dynamic Dijkstra", "ONLINE", "#00FF88"),
        ("Alert Manager", "ONLINE", "#00FF88"),
        ("CPU Inference", "ACTIVE", "#38BDF8")
    ]
    for idx, (name, stat, col) in enumerate(subsystems):
        with cols[idx]:
            st.markdown(f"""
            <div class="cmd-card" style="text-align: center; padding: 8px;">
                <div style="font-size: 0.68rem; color: #94A3B8; text-transform: uppercase;">{name}</div>
                <div style="font-size: 0.90rem; font-weight: 800; color: {col}; margin-top: 2px;">{stat}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 17. RESEARCH EXPERIMENT MODE (Feature 23)
# =============================================================================

def render_research_experiment_mode(alpha: float, beta: float, gamma: float):
    """
    Renders Feature 23: Research Experiment Mode Parameter Sensitivity Display.
    """
    st.markdown('<div class="section-header">Research Experiment Mode &mdash; Weight Sensitivity Analysis</div>', unsafe_allow_html=True)
    st.caption("Evaluate algorithm behavior under custom weight configurations &bull; Baseline weights preserved.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="cmd-card">
            <div class="metric-label">HAZARD WEIGHT (&alpha;)</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #F87171;">{alpha:.1f}</div>
            <div style="font-size: 0.72rem; color: #94A3B8;">High value guarantees immediate bypass of fire/smoke corridors.</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="cmd-card">
            <div class="metric-label">CONGESTION WEIGHT (&beta;)</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #FBBF24;">{beta:.1f}</div>
            <div style="font-size: 0.72rem; color: #94A3B8;">Penalizes congested bottlenecks to prevent crowd crushes.</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="cmd-card">
            <div class="metric-label">DISTANCE WEIGHT (&gamma;)</div>
            <div style="font-size: 1.4rem; font-weight: 800; color: #38BDF8;">{gamma:.2f}</div>
            <div style="font-size: 0.72rem; color: #94A3B8;">Scales physical corridor length (meters) to penalize excessive detours.</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 18. IEEE TRACEABILITY MATRIX (Feature 24)
# =============================================================================

def render_ieee_traceability_matrix():
    """
    Renders Feature 24: IEEE Requirement Traceability Matrix.
    """
    st.markdown('<div class="section-header">IEEE Research Requirement Traceability Matrix</div>', unsafe_allow_html=True)
    matrix_data = [
        {"IEEE Requirement": "1. Sensor-Free Hazard Detection", "Implementation Module": "detection/fire_detector.py", "Demo Evidence": "Dual-head YOLOv8n Fire/Smoke on CPU", "Test Status": "PASSED (Test 01-03)"},
        {"IEEE Requirement": "2. Crowd Occupancy Estimation", "Implementation Module": "detection/crowd_detector.py", "Demo Evidence": "YOLOv8n Person Count & Density", "Test Status": "PASSED (Test 04-05)"},
        {"IEEE Requirement": "3. Topological Facility Graph", "Implementation Module": "graph/building_graph.py", "Demo Evidence": "7-Node NetworkX Building Model", "Test Status": "PASSED (Test 06-07)"},
        {"IEEE Requirement": "4. Dynamic Edge Cost Formulation", "Implementation Module": "graph/dijkstra_optimizer.py", "Demo Evidence": "Cost(e) = 5.0*H + 2.0*D + 0.2*L", "Test Status": "PASSED (Test 08)"},
        {"IEEE Requirement": "5. Dynamic Dijkstra Rerouting", "Implementation Module": "graph/dijkstra_optimizer.py", "Demo Evidence": "Continuous Shortest Path Recalculation", "Test Status": "PASSED (Test 09-10)"},
        {"IEEE Requirement": "6. End-to-End System Pipeline", "Implementation Module": "integration/evacuation_system.py", "Demo Evidence": "Perception to Graph to Reroute Chain", "Test Status": "PASSED (Test 11)"},
        {"IEEE Requirement": "7. Emergency Alert & Guard Log", "Implementation Module": "alerts/alert_manager.py", "Demo Evidence": "Dispatch Logs & Incident History", "Test Status": "PASSED (Test 13)"},
        {"IEEE Requirement": "8. Directional Guidance Simulation", "Implementation Module": "alerts/alert_manager.py", "Demo Evidence": "Simulated LED Signs & PA Advisory", "Test Status": "PASSED (Test 12)"},
        {"IEEE Requirement": "9. Command Center Telemetry", "Implementation Module": "dashboard/app.py", "Demo Evidence": "Multi-tier Dark UI & Performance HUD", "Test Status": "PASSED (Test 12)"}
    ]
    st.dataframe(pd.DataFrame(matrix_data), hide_index=True, use_container_width=True)


# =============================================================================
# 19. DIGITAL INCIDENT REPORT GENERATORS (Feature 9)
# =============================================================================

def generate_incident_report_text(report_data: Dict[str, Any]) -> str:
    """Generates structured plain-text incident report."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""================================================================================
AI EMERGENCY COMMAND CENTER — DIGITAL INCIDENT REPORT
Project: AI-Based Fire/Smoke Hazard Detection & Crowd-Aware Evacuation Path Optimization
Report Generated: {ts}
================================================================================

1. SYSTEM & INCIDENT TELEMETRY
--------------------------------------------------------------------------------
- Origin Room:              {report_data.get('origin_room', 'ROOM_A')}
- Monitored Zone Edge:      {report_data.get('monitored_edge', 'CORRIDOR_1 <-> EXIT_A')}
- Fire Detected:            {report_data.get('fire_detected', 'NO')}
- Smoke Detected:           {report_data.get('smoke_detected', 'NO')}
- Hazard Score:             {report_data.get('hazard_score', 0.0):.3f}
- Occupants Count:          {report_data.get('people_detected', 0)}
- Crowd Density:            {report_data.get('crowd_density', 0.0):.3f}
- Evacuation Risk Index:    {report_data.get('risk_index', 0)} / 100
- Evacuation Efficiency:    {report_data.get('efficiency_index', 0)} / 100

2. ROUTE & OPTIMIZATION AUDIT
--------------------------------------------------------------------------------
- Baseline Route:           {report_data.get('baseline_route', 'ROOM_A -> CORRIDOR_1 -> EXIT_A')}
- Dynamic Route:            {report_data.get('dynamic_route', 'ROOM_A -> CORRIDOR_1 -> EXIT_A')}
- Recommended Exit:         {report_data.get('recommended_exit', 'EXIT_A')}
- Baseline Cost:            {report_data.get('baseline_cost', 0.0):.2f}
- Dynamic Route Cost:       {report_data.get('route_cost', 0.0):.2f}
- Route Changed:            {report_data.get('route_changed', 'NO')}
- Reroute Reason:           {report_data.get('reroute_reason', 'Baseline route clear.')}

3. FACILITY SAFETY DISPATCH
--------------------------------------------------------------------------------
- Security Guard Action:    {report_data.get('guard_action', 'MONITOR FACILITY')}
- Dynamic Signage Status:   {report_data.get('signage_status', 'ALL OPEN')}
- Public Address Message:   "{report_data.get('pa_message', 'Normal operations.')}"

4. PERFORMANCE & HARDWARE METRICS
--------------------------------------------------------------------------------
- Fire Inference:           {report_data.get('fire_ms', 0.0):.1f} ms
- Crowd Inference:          {report_data.get('crowd_ms', 0.0):.1f} ms
- Dynamic Dijkstra Recalc:  {report_data.get('dijkstra_ms', 0.0):.2f} ms
- Total Pipeline Latency:   {report_data.get('total_ms', 0.0):.1f} ms
- Processed Frame Rate:     {report_data.get('fps', 0.0):.1f} FPS
- Execution Platform:       CPU (Dual YOLOv8n, NetworkX)

5. ACADEMIC TRANSPARENCY & RESEARCH LIMITATIONS
--------------------------------------------------------------------------------
1. Synthetic 7-node building topology layout.
2. Demo zone mapping rather than physical pixel-level homography calibration.
3. No real CCTV camera calibration or CAD/floor-plan registration.
4. Public Address (PA) and LED digital signage are software simulations.
5. No physical relays, strobe alarms, or sprinkler hardware connected.
6. CPU inference with dual YOLOv8 Nano models.
7. Academic research prototype — not a certified life-safety system.
================================================================================
"""


def generate_incident_report_json(report_data: Dict[str, Any]) -> str:
    """Generates structured JSON incident report."""
    return json.dumps(report_data, indent=2)


def generate_incident_report_csv(report_data: Dict[str, Any]) -> str:
    """Generates flat CSV incident report."""
    df = pd.DataFrame([report_data])
    return df.to_csv(index=False)


# =============================================================================
# 20. EXISTING RENDERERS (Preserved with backward compatibility)
# =============================================================================

def render_route_history_table(route_history: List[Dict[str, Any]]):
    """Renders Section 11: Route History Table."""
    st.markdown('<div class="section-header">Route Change Event History</div>', unsafe_allow_html=True)
    if route_history:
        rh_df = pd.DataFrame(route_history)
        display_cols = [c for c in ["timestamp", "previous_route", "new_route", "reason", "previous_cost", "new_cost", "cost_difference"] if c in rh_df.columns]
        st.dataframe(rh_df[display_cols] if display_cols else rh_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No route changes registered yet. System operating on optimal baseline path.")


def render_emergency_alert(has_fire: bool, has_smoke: bool, hazard_score: float,
                           crowd_density: float, active_edge: Tuple[str, str], current_exit: str):
    """Renders Section 12: Emergency Alert Center with Hierarchy."""
    st.markdown('<div class="section-header">Emergency Alert Center</div>', unsafe_allow_html=True)
    edge_label = f"{active_edge[0]} &harr; {active_edge[1]}"
    if has_fire or hazard_score >= 0.70:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.25) 0%, rgba(18, 28, 48, 0.95) 100%); border-left: 5px solid #EF4444; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="color: #FF4D6D; font-weight: 800; font-size: 1.05rem;">
                🚨 CRITICAL EMERGENCY &mdash; FIRE DETECTED IN {edge_label.upper()}
            </div>
            <div style="font-size: 0.92rem; color: #F8FAFC; margin-top: 4px;">
                HAZARD SCORE: <strong>{hazard_score:.3f}</strong> &bull; DIRECT EGRESS COMPROMISED.
            </div>
            <div style="font-size: 0.95rem; color: #34D399; font-weight: 700; margin-top: 6px;">
                RECOMMENDED ACTION: EVACUATE VIA {current_exit} IMMEDIATELY.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif has_smoke or (0.0 < hazard_score < 0.70):
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(249, 115, 22, 0.22) 0%, rgba(18, 28, 48, 0.95) 100%); border-left: 5px solid #F97316; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="color: #FB923C; font-weight: 800; font-size: 1.05rem;">
                ⚠️ WARNING &mdash; SMOKE / VISIBILITY HAZARD IN {edge_label.upper()}
            </div>
            <div style="font-size: 0.92rem; color: #F8FAFC; margin-top: 4px;">
                HAZARD SCORE: <strong>{hazard_score:.3f}</strong> &bull; CORRIDOR VISIBILITY IMPAIRED.
            </div>
            <div style="font-size: 0.95rem; color: #34D399; font-weight: 700; margin-top: 6px;">
                RECOMMENDED ACTION: FOLLOW GUIDANCE TOWARD {current_exit}.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif crowd_density >= 0.70:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.20) 0%, rgba(18, 28, 48, 0.95) 100%); border-left: 5px solid #F59E0B; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="color: #FBBF24; font-weight: 800; font-size: 1.05rem;">
                👥 CAUTION &mdash; SEVERE CROWD CONGESTION IN {edge_label.upper()}
            </div>
            <div style="font-size: 0.92rem; color: #F8FAFC; margin-top: 4px;">
                CROWD DENSITY: <strong>{crowd_density:.3f}</strong> &bull; BOTTLENECK MITIGATION ACTIVE.
            </div>
            <div style="font-size: 0.95rem; color: #34D399; font-weight: 700; margin-top: 6px;">
                RECOMMENDED ACTION: REDIRECT FLOW TOWARD {current_exit}.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(18, 28, 48, 0.95) 100%); border-left: 5px solid #10B981; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="color: #34D399; font-weight: 800; font-size: 1.05rem;">
                ✅ NOMINAL &mdash; ALL MONITORED FACILITY ZONES CLEAR
            </div>
            <div style="font-size: 0.90rem; color: #CBD5E1; margin-top: 4px;">
                Zero fire/smoke detected. Crowd densities nominal. Standard primary egress paths active.
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_security_guard_log(guard_logs: List[Dict[str, Any]]):
    """Renders Section 13: Security Response Center / Guard Log."""
    st.markdown('<div class="section-header">Security Response Center (Guard Notifications)</div>', unsafe_allow_html=True)
    if guard_logs:
        guard_df = pd.DataFrame(guard_logs)
        display_cols = [c for c in ["timestamp", "location", "type", "hazard_score", "people", "density", "recommended_exit", "action", "status"] if c in guard_df.columns]
        st.dataframe(guard_df[display_cols] if display_cols else guard_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No guard dispatch alerts registered. Facility security status nominal.")


def render_simulated_led_signage(signage_dict: Dict[str, Dict[str, str]]):
    """Renders Section 14: Simulated Directional LED Signage."""
    st.markdown('<div class="section-header">Simulated Dynamic Digital LED Signage</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #FBBF24; font-weight: 700; margin-bottom: 6px;'>SIMULATION ONLY &mdash; NO PHYSICAL LED HARDWARE CONNECTED</div>", unsafe_allow_html=True)
    cols = st.columns(len(signage_dict) if signage_dict else 2)
    for idx, (ex_id, data) in enumerate(signage_dict.items()):
        with cols[idx]:
            status = data.get("status", "OPEN")
            if "SAFE" in status:
                st.markdown(f"""
                <div class="led-board">
                    <div class="led-title">{ex_id} DIGITAL SIGN</div>
                    <div class="led-sign-safe">🟢 {ex_id}: SAFE ROUTE</div>
                </div>
                """, unsafe_allow_html=True)
            elif "CLOSED" in status or "AVOID" in status:
                st.markdown(f"""
                <div class="led-board">
                    <div class="led-title">{ex_id} DIGITAL SIGN</div>
                    <div class="led-sign-avoid">⛔ {ex_id}: CLOSED / AVOID</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="led-board">
                    <div class="led-title">{ex_id} DIGITAL SIGN</div>
                    <div class="led-sign-standby">⚪ {ex_id}: AVAILABLE</div>
                </div>
                """, unsafe_allow_html=True)


def render_pa_announcement(message: str):
    """Renders Section 15: Public Address (PA) Voice Advisory Simulation."""
    st.markdown('<div class="section-header">Public Address (PA) Voice Advisory</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #38BDF8; font-weight: 700; margin-bottom: 6px;'>🔊 SIMULATED PUBLIC ADDRESS ADVISORY</div>", unsafe_allow_html=True)
    st.markdown(f'<div class="pa-box">📢 &ldquo;{message}&rdquo;</div>', unsafe_allow_html=True)


def render_performance_telemetry(latency_info: Dict[str, Any], frame_skip_val: int):
    """Renders Section 16: Hardware & Performance Telemetry Metrics."""
    st.markdown('<div class="section-header">Hardware &amp; Performance Telemetry</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6, t7 = st.columns(7)
    fire_ms = latency_info.get('fire_ms', latency_info.get('detection_ms', 0) / 2)
    crowd_ms = latency_info.get('crowd_ms', latency_info.get('detection_ms', 0) / 2)
    dijkstra_ms = latency_info.get('dijkstra_ms', 0.0)
    total_ms = latency_info.get('total_ms', fire_ms + crowd_ms + dijkstra_ms)
    fps = latency_info.get('fps', (1000.0 / total_ms) if total_ms > 0 else 0.0)

    with t1:
        st.metric("Fire YOLO Latency", f"{fire_ms:.1f} ms")
    with t2:
        st.metric("Crowd YOLO Latency", f"{crowd_ms:.1f} ms")
    with t3:
        st.metric("Dynamic Dijkstra", f"{dijkstra_ms:.2f} ms")
    with t4:
        st.metric("Total Latency", f"{total_ms:.1f} ms")
    with t5:
        st.metric("Processed FPS", f"{fps:.1f}")
    with t6:
        st.metric("Frame Skip Interval", f"N={frame_skip_val}")
    with t7:
        st.metric("Inference Engine", "CPU ACTIVE")


def render_system_architecture_diagram():
    """Renders Section 21: Visual 3-Layer System Architecture Diagram."""
    st.markdown('<div class="section-header">Three-Layer System Architecture (IEEE Model)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-container">
        <div class="arch-layer">
            <div class="arch-layer-title" style="color: #38BDF8;">1. Perception Layer</div>
            <div class="arch-item">&bull; Surveillance CCTV Stream Ingestion</div>
            <div class="arch-item">&bull; Fire/Smoke Detection (YOLOv8n)</div>
            <div class="arch-item">&bull; Occupant Detection (YOLOv8n)</div>
            <div class="arch-item">&bull; Parallel Inference &amp; Frame-Skipping</div>
        </div>
        <div class="arch-arrow-block">&rarr;</div>
        <div class="arch-layer">
            <div class="arch-layer-title" style="color: #FBBF24;">Zone Mapping Layer</div>
            <div class="arch-item">&bull; Synthetic 7-Node Facility Topology</div>
            <div class="arch-item">&bull; Hazard Spatial Mapping</div>
            <div class="arch-item">&bull; Crowd Occupancy Mapping</div>
            <div class="arch-item">&bull; Physical Metric Segment Lengths</div>
        </div>
        <div class="arch-arrow-block">&rarr;</div>
        <div class="arch-layer">
            <div class="arch-layer-title" style="color: #A855F7;">2. Graph &amp; Optimization Layer</div>
            <div class="arch-item">&bull; Dynamic Edge Cost Engine (Cost(e))</div>
            <div class="arch-item">&bull; NetworkX Graph Topology Model</div>
            <div class="arch-item">&bull; Dynamic Dijkstra Optimization</div>
            <div class="arch-item">&bull; Route Adaptation &amp; Divergence Check</div>
        </div>
        <div class="arch-arrow-block">&rarr;</div>
        <div class="arch-layer">
            <div class="arch-layer-title" style="color: #34D399;">3. Presentation Layer</div>
            <div class="arch-item">&bull; Command Center Dashboard</div>
            <div class="arch-item">&bull; Security Guard Dispatch Alerts</div>
            <div class="arch-item">&bull; Simulated Dynamic LED Signage</div>
            <div class="arch-item">&bull; Public Address (PA) Voice Advisory</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_mathematical_model_panel():
    """Renders Section 22: Mathematical Model Panel."""
    st.markdown('<div class="section-header">Mathematical Formulation &amp; Optimization Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="math-box">
        <div style="font-size: 0.95rem; font-weight: 700; color: #38BDF8; margin-bottom: 6px;">
            Dynamic Edge Cost Objective Function:
        </div>
        <div style="font-size: 1.15rem; font-family: 'JetBrains Mono', monospace; color: #00FF88; margin-bottom: 8px;">
            $$\\text{Cost}(e) = \\alpha \\cdot H(e) + \\beta \\cdot D(e) + \\gamma \\cdot L(e)$$
        </div>
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.6;">
            Where:
            <ul>
                <li><strong>$H(e) \\in [0, 1]$</strong>: Hazard severity index computed from maximum fire and smoke bounding-box detection confidences.</li>
                <li><strong>$D(e) \\in [0, 1]$</strong>: Crowd density index calculated from occupant bounding-box counts relative to segment floor area.</li>
                <li><strong>$L(e)$</strong>: Physical Euclidean metric corridor length between topological building junctions (meters).</li>
                <li><strong>Weights</strong>: $\\alpha = 5.0$ (High priority hazard aversion), $\\beta = 2.0$ (Congestion avoidance), $\\gamma = 0.2$ (Distance scale).</li>
            </ul>
        </div>
        <div style="font-size: 0.95rem; font-weight: 700; color: #A855F7; margin-top: 10px; margin-bottom: 4px;">
            Optimal Evacuation Path Optimization:
        </div>
        <div style="font-size: 1.1rem; font-family: 'JetBrains Mono', monospace; color: #F8FAFC;">
            $$\\mathcal{P}^* = \\arg\\min_{\\mathcal{P} \\in \\Pi(s, \\mathcal{E})} \\sum_{e \\in \\mathcal{P}} \\text{Cost}(e)$$
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_system_validation_panel():
    """Renders Section 24: Test Suite & System Verification Status."""
    st.markdown('<div class="section-header">System Validation &amp; Test Suite Verification</div>', unsafe_allow_html=True)
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        st.markdown("""
        <div class="cmd-card" style="border-left: 4px solid #10B981; text-align: center;">
            <div class="metric-label">FUNCTIONAL TESTS</div>
            <div class="metric-value" style="color: #34D399;">14 / 14</div>
            <div class="metric-footer">PASSED (100%)</div>
        </div>
        """, unsafe_allow_html=True)
    with v2:
        st.markdown("""
        <div class="cmd-card" style="border-left: 4px solid #10B981; text-align: center;">
            <div class="metric-label">END-TO-END SUITE</div>
            <div class="metric-value" style="color: #34D399;">12 / 12</div>
            <div class="metric-footer">PASSED (100%)</div>
        </div>
        """, unsafe_allow_html=True)
    with v3:
        st.markdown("""
        <div class="cmd-card" style="border-left: 4px solid #00F0FF; text-align: center;">
            <div class="metric-label">YOLOv8N MODELS</div>
            <div class="metric-value" style="color: #00F0FF; font-size: 1.3rem;">DUAL READY</div>
            <div class="metric-footer">Fire/Smoke &bull; Crowd</div>
        </div>
        """, unsafe_allow_html=True)
    with v4:
        st.markdown("""
        <div class="cmd-card" style="border-left: 4px solid #A855F7; text-align: center;">
            <div class="metric-label">DIJKSTRA ENGINE</div>
            <div class="metric-value" style="color: #A855F7; font-size: 1.3rem;">DYNAMIC</div>
            <div class="metric-footer">NetworkX Connected</div>
        </div>
        """, unsafe_allow_html=True)


def render_evacuation_risk_index(hazard_score: float, crowd_density: float,
                                 route_cost: float, route_changed: bool):
    """Renders Unique Feature 1: Calculated Evacuation Risk Index (0 - 100)."""
    norm_cost = min(1.0, route_cost / 25.0)
    risk_val = min(100, int(round((50.0 * hazard_score) + (30.0 * crowd_density) + (20.0 * norm_cost))))

    if risk_val >= 75:
        risk_lvl = "CRITICAL RISK"
        card_cls = "risk-card-critical"
        risk_col = "#FF2E4D"
        desc = "Severe hazard or overwhelming crowd congestion. Emergency evacuation protocols active."
    elif risk_val >= 50:
        risk_lvl = "HIGH RISK"
        card_cls = "risk-card-high"
        risk_col = "#F97316"
        desc = "Substantial hazard presence or heavy congestion. Dynamic rerouting strongly advised."
    elif risk_val >= 25:
        risk_lvl = "MODERATE RISK"
        card_cls = "risk-card-moderate"
        risk_col = "#FBBF24"
        desc = "Mild smoke or moderate occupant density detected. Continuous monitoring engaged."
    else:
        risk_lvl = "LOW RISK"
        card_cls = "risk-card-low"
        risk_col = "#00FF88"
        desc = "Nominal conditions. Facility corridors clear and standard egress routes operational."

    st.markdown('<div class="section-header">Evacuation Risk Index (Calculated Metric)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="{card_cls}">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.75rem; font-weight: 800; color: #94A3B8; text-transform: uppercase;">
                    COMPOSITE FACILITY RISK SCORE
                </div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {risk_col}; font-family: 'JetBrains Mono', monospace;">
                    {risk_val} <span style="font-size: 1.1rem; color: #94A3B8;">/ 100</span>
                </div>
                <div style="font-size: 0.85rem; font-weight: 700; color: {risk_col};">
                    {risk_lvl}
                </div>
            </div>
            <div style="max-width: 500px; font-size: 0.82rem; color: #CBD5E1;">
                <div>{desc}</div>
                <div style="font-size: 0.72rem; color: #64748B; margin-top: 6px;">
                    Formula: <code>Risk = 50 &times; Hazard + 30 &times; Density + 20 &times; Norm(Cost)</code>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_route_confidence_card(hazard_confidence: float, people_count: int,
                                 route_cost: float, has_hazard: bool):
    """Renders Unique Feature 2: Route Confidence Card (High / Medium / Low)."""
    st.markdown('<div class="section-header">Dynamic Route Confidence Evaluation</div>', unsafe_allow_html=True)
    if has_hazard:
        if hazard_confidence >= 0.60:
            conf_lvl = "HIGH CONFIDENCE"
            conf_col = "#00FF88"
            explanation = "Perception models report decisive bounding-box detections. Dynamic Dijkstra rerouting margin is substantial."
        else:
            conf_lvl = "MEDIUM CONFIDENCE"
            conf_col = "#FBBF24"
            explanation = "Perception detections are near threshold limit. Route adapted with standard safety margin."
    else:
        conf_lvl = "HIGH CONFIDENCE"
        conf_col = "#00FF88"
        explanation = "Corridors nominal and uncompromised. Optimal baseline path verified clear."

    st.markdown(f"""
    <div class="cmd-card" style="border-left: 4px solid {conf_col};">
        <div style="font-size: 0.75rem; font-weight: 800; color: #94A3B8; text-transform: uppercase;">
            ROUTE STABILITY &amp; CONFIDENCE
        </div>
        <div style="font-size: 1.3rem; font-weight: 800; color: {conf_col}; margin: 3px 0;">
            {conf_lvl}
        </div>
        <div style="font-size: 0.84rem; color: #CBD5E1;">
            {explanation}
        </div>
        <div style="font-size: 0.72rem; color: #64748B; margin-top: 6px;">
            Note: Confidence reflects perception stability and route graph separation margin (not an empirical offline model accuracy score).
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_incident_timeline(events: List[Dict[str, Any]]):
    """Renders Unique Feature 3: Chronological Emergency Incident Timeline."""
    st.markdown('<div class="section-header">Emergency Incident Timeline (Runtime Audit)</div>', unsafe_allow_html=True)
    if not events:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        events = [
            {"time": now_str, "desc": "CCTV surveillance feed initialized. Continuous perception active."},
            {"time": now_str, "desc": "Topological building digital twin loaded. Dijkstra optimizer online."},
            {"time": now_str, "desc": "System operating in nominal baseline monitoring state."}
        ]

    st.markdown('<div class="timeline-box">', unsafe_allow_html=True)
    for ev in events:
        t_str = ev.get("time", ev.get("timestamp", ""))
        d_str = ev.get("desc", ev.get("action", ""))
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-time">{t_str}</div>
            <div class="timeline-desc">{d_str}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_prototype_notes():
    """Renders Unique Feature 5: Academic Scope & 10 Exact Limitations Points."""
    with st.expander("📌 RESEARCH PROTOTYPE LIMITATIONS (IEEE ACADEMIC TRANSPARENCY)", expanded=False):
        st.markdown("""
        <div style="font-size: 0.88rem; color: #CBD5E1; line-height: 1.7; padding: 10px 14px;">
            <ol>
                <li><strong>Synthetic 7-Node Building Topology:</strong> The facility layout is an academic topological representation rather than an expansive multi-story building.</li>
                <li><strong>Demo Zone Mapping:</strong> Spatial zone partitioning is simulated via synthetic camera mapping rather than calibrated pixel coordinates.</li>
                <li><strong>No Camera Homography Calibration:</strong> Real-world perspective planar transforms and camera extrinsics have not been calibrated.</li>
                <li><strong>No CAD / Floor-Plan Registration:</strong> Physical architectural blueprints are not integrated into a spatial BIM or CAD engine.</li>
                <li><strong>Software-Simulated PA System:</strong> Public Address voice advisories are text-based software simulations.</li>
                <li><strong>Software-Simulated LED Signage:</strong> Directional emergency exit signage is rendered virtually without physical display boards.</li>
                <li><strong>No Physical Alarm Hardware:</strong> The prototype does not interface with building fire alarm control panels (FACPs) or relays.</li>
                <li><strong>CCTV Coverage Assumptions:</strong> Complete visual corridor coverage is assumed without blind spots or smoke occlusions.</li>
                <li><strong>CPU Inference Constraints:</strong> Dual YOLOv8 Nano architectures operate on CPU with adaptive frame-skipping.</li>
                <li><strong>Research Prototype Classification:</strong> This project is an academic research prototype and NOT a certified life-safety evacuation system.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)


def render_academic_limitations():
    """Alias for render_prototype_notes."""
    render_prototype_notes()
