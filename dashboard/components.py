"""
Dashboard UI Components.
Modular renderers for Header, System Status, Perception Metrics, Zone Mapping,
Multi-Zone Analysis, Dynamic Cost Panel, Route Comparison, Flow Diagram, Alerts,
Signage, PA, and Academic Limitations.
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import streamlit as st


def render_header():
    """Renders Section 1: Professional Header with Status & Academic Disclaimer."""
    st.markdown("""
    <div class="header-container">
        <div class="main-title">
            AI-BASED FIRE/SMOKE HAZARD DETECTION<br>&amp; CROWD-AWARE EVACUATION PATH OPTIMIZATION
        </div>
        <div class="main-subtitle">
            Real-Time Multi-Modal Perception + Dynamic Dijkstra Evacuation Prototype
        </div>
        <div class="system-status-bar">
            <span class="status-pill pill-green">&#9679; SYSTEM ONLINE</span>
            <span class="status-pill pill-blue">&#9881; CPU MODE</span>
            <span class="status-pill pill-amber">&#9888; RESEARCH PROTOTYPE</span>
        </div>
    </div>
    <div class="disclaimer-banner">
        <strong>ACADEMIC RESEARCH PROTOTYPE &mdash; NOT A CERTIFIED LIFE-SAFETY SYSTEM</strong>
    </div>
    """, unsafe_allow_html=True)


def render_system_status_cards():
    """Renders Section 2: Compact System Status Cards."""
    st.markdown('<div class="section-header">System Operational Readiness</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown("""
        <div class="sys-status-card">
            <div class="sys-status-label">SYSTEM</div>
            <div class="sys-status-value" style="color: #34D399;">ONLINE</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="sys-status-card">
            <div class="sys-status-label">PERCEPTION</div>
            <div class="sys-status-value" style="color: #60A5FA;">READY</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="sys-status-card">
            <div class="sys-status-label">GRAPH</div>
            <div class="sys-status-value" style="color: #34D399;">CONNECTED</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="sys-status-card">
            <div class="sys-status-label">ROUTING</div>
            <div class="sys-status-value" style="color: #A855F7;">DYNAMIC</div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown("""
        <div class="sys-status-card">
            <div class="sys-status-label">VIDEO</div>
            <div class="sys-status-value" style="color: #38BDF8;">READY</div>
        </div>
        """, unsafe_allow_html=True)
    with c6:
        st.markdown("""
        <div class="sys-status-card">
            <div class="sys-status-label">CPU</div>
            <div class="sys-status-value" style="color: #FBBF24;">ACTIVE</div>
        </div>
        """, unsafe_allow_html=True)


def render_cctv_meta_bar(width: int, height: int, frame_num: int, latency_ms: float, fps: float):
    """Renders Section 4: Metadata bar below CCTV camera feeds."""
    st.markdown(f"""
    <div class="cctv-meta-bar">
        <span class="cctv-meta-item">Input Resolution: <strong>{width}&times;{height}</strong></span>
        <span class="cctv-meta-item">Processed Frame: <strong>#{frame_num}</strong></span>
        <span class="cctv-meta-item">Inference Mode: <strong>CPU (Dual YOLOv8n)</strong></span>
        <span class="cctv-meta-item">Processing Time: <strong>{latency_ms:.1f} ms</strong></span>
        <span class="cctv-meta-item">Processed FPS: <strong>{fps:.1f}</strong></span>
    </div>
    """, unsafe_allow_html=True)


def render_perception_metrics(has_fire: bool, has_smoke: bool, hazard_score: float,
                              people_count: int, crowd_density: float,
                              hazard_status: str, crowd_status: str):
    """Renders Section 5: AI Perception Metrics."""
    st.markdown('<div class="section-header">AI Perception Telemetry</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">FIRE</div>
            <div class="metric-value" style="color: {'#EF4444' if has_fire else '#10B981'};">
                {'YES' if has_fire else 'NO'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">SMOKE</div>
            <div class="metric-value" style="color: {'#F97316' if has_smoke else '#10B981'};">
                {'YES' if has_smoke else 'NO'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">HAZARD SCORE</div>
            <div class="metric-value" style="color: #F8FAFC;">
                {hazard_score:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">PEOPLE DETECTED</div>
            <div class="metric-value" style="color: #60A5FA;">
                {people_count}
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">CROWD DENSITY</div>
            <div class="metric-value" style="color: #F8FAFC;">
                {crowd_density:.3f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Status Badges
    b1, b2 = st.columns(2)
    with b1:
        badge_cls = "badge-danger" if hazard_status in ["DANGER", "WARNING"] else ("badge-caution" if hazard_status == "CAUTION" else "badge-normal")
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-top: 10px; margin-bottom: 4px;">HAZARD STATUS</div>
        <div class="status-badge {badge_cls}">🔥 {hazard_status}</div>
        """, unsafe_allow_html=True)
    with b2:
        crowd_cls = "badge-danger" if crowd_status == "HIGH CONGESTION" else ("badge-warning" if crowd_status == "MODERATE CONGESTION" else "badge-normal")
        st.markdown(f"""
        <div style="font-size: 0.75rem; color: #94A3B8; text-transform: uppercase; font-weight: 700; margin-top: 10px; margin-bottom: 4px;">CROWD STATUS</div>
        <div class="status-badge {crowd_cls}">👥 {crowd_status}</div>
        """, unsafe_allow_html=True)


def render_hazard_zone_panel(active_edge: Tuple[str, str], hazard_type_str: str,
                             hazard_score: float, crowd_density: float, status_label: str):
    """Renders Section 6: Current Hazard Zone Banner."""
    st.markdown('<div class="section-header">Current Hazard Zone</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background-color: #0F172A; border-left: 4px solid #38BDF8; border-radius: 4px; padding: 14px 18px; margin-bottom: 14px;">
        <div style="font-size: 0.75rem; color: #38BDF8; text-transform: uppercase; font-weight: 700;">CURRENT MONITORED ZONE:</div>
        <div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; margin: 4px 0;">
            {active_edge[0]} &harr; {active_edge[1]}
        </div>
        <div style="display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px; font-size: 0.88rem; color: #94A3B8;">
            <span>Hazard: <strong style="color: #F87171;">{hazard_type_str}</strong></span>
            <span>Hazard Score: <strong style="color: #F8FAFC;">{hazard_score:.3f}</strong></span>
            <span>Crowd Density: <strong style="color: #F8FAFC;">{crowd_density:.3f}</strong></span>
            <span>Zone Status: <strong style="color: {'#EF4444' if status_label == 'EMERGENCY' else ('#F59E0B' if status_label == 'CAUTION' else '#10B981')};">{status_label}</strong></span>
        </div>
        <div style="font-size: 0.80rem; color: #FBBF24; font-weight: 700; margin-top: 8px;">
            ⚠️ DEMO ZONE MAPPING &mdash; NOT REAL CCTV CALIBRATION
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_multi_zone_analysis(zone_counts: Dict[str, int], zone_densities: Dict[str, float], total_people: int):
    """Renders Section 7: Crowd / Multi-Zone Analysis Table."""
    st.markdown('<div class="section-header">Crowd & Multi-Zone Spatial Analysis</div>', unsafe_allow_html=True)
    st.caption("Mapped spatial occupant distribution partitioned across synthetic demonstration zones (Demo Zone Mapping).")

    zones = ["ROOM_A", "ROOM_B", "ROOM_C", "CORRIDOR_1", "CORRIDOR_2", "EXIT_A", "EXIT_B"]
    rows = []
    for z in zones:
        cnt = zone_counts.get(z, 0)
        den = zone_densities.get(z, 0.0)
        if total_people == 0:
            status = "CLEAR"
        elif den >= 0.70:
            status = "HIGH CONGESTION"
        elif den >= 0.35:
            status = "MODERATE"
        elif cnt > 0:
            status = "NORMAL"
        else:
            status = "CLEAR"

        rows.append({
            "ZONE": z,
            "PEOPLE": cnt,
            "DENSITY": f"{den:.2f}" if total_people > 0 else "0.00",
            "STATUS": status
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_dynamic_cost_panel(alpha: float, beta: float, gamma: float,
                              hazard_score: float, crowd_density: float,
                              edge_length: float, edge_cost: float,
                              active_edge: Tuple[str, str]):
    """Renders Section 8: Dynamic Edge Cost Calculation."""
    st.markdown('<div class="section-header">Dynamic Edge Cost Formulation</div>', unsafe_allow_html=True)
    haz_term = alpha * hazard_score
    den_term = beta * crowd_density
    len_term = gamma * edge_length

    st.markdown(f"""
    <div class="comparison-box">
        <div style="font-size: 0.88rem; color: #94A3B8; margin-bottom: 6px;">
            Active Edge: <code>{active_edge[0]} &harr; {active_edge[1]}</code> (Physical Length: {edge_length:.2f}m)
        </div>
        <div style="font-size: 0.95rem; margin-bottom: 8px;">
            $$\\text{{Cost}}(e) = \\alpha \\times \\text{{Hazard}} + \\beta \\times \\text{{Density}} + \\gamma \\times \\text{{Length}}$$
        </div>
        <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 10px;">
            Weights: &alpha; = {alpha:.1f} &bull; &beta; = {beta:.1f} &bull; &gamma; = {gamma:.2f}
        </div>
        <ul style="margin-bottom: 10px; font-size: 0.90rem; color: #CBD5E1; padding-left: 20px;">
            <li>Hazard contribution: {alpha:.1f} &times; {hazard_score:.3f} = <span style="color: #F87171; font-weight: 700;">{haz_term:.2f}</span></li>
            <li>Density contribution: {beta:.1f} &times; {crowd_density:.3f} = <span style="color: #FBBF24; font-weight: 700;">{den_term:.2f}</span></li>
            <li>Distance contribution: {gamma:.2f} &times; {edge_length:.2f}m = <span style="color: #38BDF8; font-weight: 700;">{len_term:.2f}</span></li>
        </ul>
        <div style="font-size: 1.15rem; font-weight: 800; color: #34D399; padding-top: 8px; border-top: 1px solid #1E293B;">
            FINAL EDGE COST: {edge_cost:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_route_comparison(normal_route: List[str], normal_exit: str, normal_cost: float,
                            current_route: List[str], current_exit: str, current_cost: float,
                            route_changed: bool, reason: str):
    """Renders Section 10: Baseline vs. Dynamic Route comparison."""
    st.markdown('<div class="section-header">Route Comparison (Baseline vs. Dynamic)</div>', unsafe_allow_html=True)
    normal_str = " &rarr; ".join(normal_route) if normal_route else "NONE"
    current_str = " &rarr; ".join(current_route) if current_route else "NO SAFE ROUTE"

    badge_cls = "badge-rerouted" if route_changed else "badge-normal"
    status_label = "ROUTE CHANGED" if route_changed else "ROUTE UNCHANGED"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="comparison-box" style="border-left: 4px solid #60A5FA;">
            <div style="font-weight: 700; color: #60A5FA; font-size: 0.82rem; text-transform: uppercase;">BASELINE ROUTE (Hazard = 0, Density = 0)</div>
            <div style="font-size: 1.05rem; font-weight: 700; margin: 6px 0; color: #F8FAFC;">{normal_str}</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Exit: <strong style="color: #F8FAFC;">{normal_exit}</strong> | Cost: <strong style="color: #60A5FA;">{normal_cost:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        border_col = "#A855F7" if route_changed else "#34D399"
        st.markdown(f"""
        <div class="comparison-box" style="border-left: 4px solid {border_col};">
            <div style="font-weight: 700; color: {border_col}; font-size: 0.82rem; text-transform: uppercase;">CURRENT DYNAMIC ROUTE (LIVE EVALUATED)</div>
            <div style="font-size: 1.05rem; font-weight: 700; margin: 6px 0; color: #F8FAFC;">{current_str}</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Exit: <strong style="color: #F8FAFC;">{current_exit}</strong> | Cost: <strong style="color: {border_col};">{current_cost:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-badge {badge_cls}">
        {status_label}: {reason}
    </div>
    """, unsafe_allow_html=True)


def render_recommended_exit_panel(recommended_exit: str, total_route_cost: float, avoided_exit: Optional[str] = None):
    """Renders Section 11: Prominent Recommended Exit."""
    st.markdown('<div class="section-header">Recommended Evacuation Exit</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 4px solid #10B981; padding: 18px;">
            <div class="metric-label">RECOMMENDED EVACUATION EXIT</div>
            <div class="metric-value" style="color: #34D399; font-size: 2.1rem;">🚪 {recommended_exit}</div>
            <div style="font-size: 0.85rem; color: #34D399; font-weight: 700; margin-top: 4px;">SAFE ROUTE</div>
            <div style="font-size: 0.80rem; color: #94A3B8; margin-top: 6px;">Total Route Cost: <strong style="color: #38BDF8;">{total_route_cost:.2f}</strong></div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if avoided_exit and avoided_exit != recommended_exit:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid #EF4444; padding: 18px;">
                <div class="metric-label">AVOIDED / COMPROMISED EXIT</div>
                <div class="metric-value" style="color: #F87171; font-size: 2.1rem;">⛔ {avoided_exit}</div>
                <div style="font-size: 0.85rem; color: #F87171; font-weight: 700; margin-top: 4px;">AVOID / HAZARD DETECTED</div>
                <div style="font-size: 0.80rem; color: #FCA5A5; margin-top: 6px;">Direct pathway blocked or elevated traversal cost.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="padding: 18px;">
                <div class="metric-label">SECONDARY EXIT STATUS</div>
                <div class="metric-value" style="color: #94A3B8; font-size: 2.1rem;">⚪ STANDBY</div>
                <div style="font-size: 0.85rem; color: #94A3B8; font-weight: 700; margin-top: 4px;">AVAILABLE / NOMINAL</div>
                <div style="font-size: 0.80rem; color: #64748B; margin-top: 6px;">Alternative emergency exits clear and accessible.</div>
            </div>
            """, unsafe_allow_html=True)


def render_route_flow_card(route_nodes: List[str], total_cost: float, route_changed: bool):
    """Renders Section 12: Evacuation Route Flow."""
    st.markdown('<div class="section-header">Evacuation Route Flow</div>', unsafe_allow_html=True)
    if not route_nodes:
        st.warning("No evacuation route calculated.")
        return

    html_parts = ['<div class="route-flow-container">']
    for idx, node in enumerate(route_nodes):
        is_exit = "EXIT" in node
        cls = "route-node-exit" if is_exit else "route-node"
        html_parts.append(f'<div class="route-node {cls}">{node}</div>')
        if idx < len(route_nodes) - 1:
            html_parts.append('<div class="route-arrow">&rarr;</div>')
    html_parts.append('</div>')

    st.markdown("".join(html_parts), unsafe_allow_html=True)

    status_str = "⚠ ROUTE UPDATED" if route_changed else "OPTIMAL BASELINE ACTIVE"
    status_color = "#A855F7" if route_changed else "#34D399"

    st.markdown(f"""
    <div style="display: flex; justify-content: space-around; background-color: #0F172A; border: 1px solid #1E293B; border-radius: 4px; padding: 10px; margin-bottom: 12px; font-size: 0.95rem;">
        <span>TOTAL ROUTE COST: <strong style="color: #38BDF8;">{total_cost:.2f}</strong></span>
        <span>ROUTE STATUS: <strong style="color: {status_color};">{status_str}</strong></span>
    </div>
    """, unsafe_allow_html=True)


def render_emergency_alert(has_fire: bool, has_smoke: bool, hazard_score: float,
                           crowd_density: float, active_edge: Tuple[str, str], current_exit: str):
    """Renders Section 13: Alert Center."""
    st.markdown('<div class="section-header">Alert Center</div>', unsafe_allow_html=True)
    edge_label = f"{active_edge[0]} &harr; {active_edge[1]}"
    if has_fire or hazard_score >= 0.70:
        st.markdown(f"""
        <div class="status-card status-emergency">
            🚨 <strong>CRITICAL ALERT: FIRE DETECTED IN {edge_label.upper()}</strong><br>
            <span style="font-size: 0.90rem; font-weight: 500; margin-top: 4px; display: inline-block;">
                AVOID DIRECT CORRIDORS &bull; FOLLOW DYNAMIC GUIDANCE TOWARD <strong>{current_exit}</strong>.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif has_smoke or (0.0 < hazard_score < 0.70):
        st.markdown(f"""
        <div class="status-card status-caution">
            ⚠️ <strong>WARNING: SMOKE DETECTED IN {edge_label.upper()}</strong><br>
            <span style="font-size: 0.90rem; font-weight: 500; margin-top: 4px; display: inline-block;">
                VISIBILITY IMPAIRED &bull; EVACUATION ROUTE DYNAMICALLY UPDATED TO <strong>{current_exit}</strong>.
            </span>
        </div>
        """, unsafe_allow_html=True)
    elif crowd_density >= 0.70:
        st.markdown(f"""
        <div class="status-card status-caution">
            👥 <strong>CAUTION: HIGH CROWD DENSITY DETECTED</strong><br>
            <span style="font-size: 0.90rem; font-weight: 500; margin-top: 4px; display: inline-block;">
                BOTTLENECK DIVERSION ACTIVE &bull; OCCUPANTS REDIRECTED TOWARD <strong>{current_exit}</strong>.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card status-normal">
            ✅ <strong>NORMAL: NO ACTIVE HAZARD DETECTED</strong><br>
            <span style="font-size: 0.90rem; font-weight: 500; margin-top: 4px; display: inline-block;">
                All monitored corridors clear. Normal evacuation paths active.
            </span>
        </div>
        """, unsafe_allow_html=True)


def render_simulated_led_signage(signage_dict: Dict[str, Dict[str, str]]):
    """Renders Section 14: Directional Signage Simulation."""
    st.markdown('<div class="section-header">Directional Signage Simulation</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.80rem; color: #94A3B8; margin-bottom: 6px;'><em>SIMULATION ONLY &mdash; NO PHYSICAL LED HARDWARE CONNECTED</em></div>", unsafe_allow_html=True)
    cols = st.columns(len(signage_dict))
    for idx, (ex_id, data) in enumerate(signage_dict.items()):
        with cols[idx]:
            status = data.get("status", "OPEN")
            if "SAFE" in status:
                css_cls = "led-green"
                indicator = f"🟢 {ex_id}: SAFE ROUTE"
            elif "CLOSED" in status or "AVOID" in status:
                css_cls = "led-red"
                indicator = f"⛔ {ex_id}: CLOSED / AVOID"
            else:
                css_cls = "led-gray"
                indicator = f"⚪ {ex_id}: AVAILABLE"

            st.markdown(f"""
            <div class="led-panel">
                <div class="led-sign {css_cls}">{indicator}</div>
            </div>
            """, unsafe_allow_html=True)


def render_pa_announcement(message: str):
    """Renders Section 15: PA Announcement Simulation."""
    st.markdown('<div class="section-header">Public Address (PA) Announcement</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.80rem; color: #94A3B8; margin-bottom: 6px;'><em>🔊 SIMULATED PA</em></div>", unsafe_allow_html=True)
    st.markdown(f'<div class="pa-announcement-box">📢 "{message}"</div>', unsafe_allow_html=True)


def render_prototype_notes():
    """Renders Section 20: Academic Transparency / Limitations (Exact 10 Points)."""
    with st.expander("📌 ACADEMIC SCOPE & LIMITATIONS (IEEE RESEARCH TRANSPARENCY)", expanded=False):
        st.markdown("""
        <div class="limitations-footer">
            <ol>
                <li>Synthetic 7-node building topology.</li>
                <li>Demo zone mapping.</li>
                <li>No real CCTV homography calibration.</li>
                <li>No CAD/floor-plan registration.</li>
                <li>PA is simulated.</li>
                <li>LED signage is simulated.</li>
                <li>No physical alarm hardware connected.</li>
                <li>Real deployment requires calibrated CCTV and floor plans.</li>
                <li>YOLOv8 Nano models run on CPU.</li>
                <li>This is an academic research prototype and not a certified life-safety system.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
