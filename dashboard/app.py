"""
AI-Based Fire/Smoke Hazard Detection & Crowd-Aware Evacuation Path Optimization
Final IEEE Research Prototype & Smart Building Emergency Command Center Dashboard

Full Architecture & Navigation:
- OVERVIEW: High-level Command Status, Emergency Directives, Multi-Camera Stream Summary
- LIVE CCTV: Raw vs AI Annotated Feeds, Multi-Camera CCTV Sim, HUD Telemetry
- BUILDING DIGITAL TWIN: Topological 7-Node Graph, Multi-Floor Model (Ground & First Floor)
- EVACUATION: Dynamic Dijkstra Routing, Baseline vs Proposed, Exit Ranking, Flow Breadcrumbs
- ALERTS: Emergency Alert Center, Guard Response Log, Digital LED Signage, PA Advisory
- ANALYTICS: Telemetry Time-Series, Moving Average Trends, Latency Budgets, Health Monitor
- XAI & ASSISTANT: Explainable AI Factor Contributions, Local Rule-Based Decision Assistant
- VALIDATION: Test Suites, Model Evaluation Status, False Alarm Protocol, IEEE Traceability
- REPORTS & REPLAY: Incident Replay Step Slider, Multi-Format Digital Report Generator (TXT, JSON, CSV)
"""

import datetime
import importlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import (
    DATA_DIR,
    OUTPUT_DIR,
    ALPHA,
    BETA,
    GAMMA,
    CONF_THRESHOLD_FIRE,
    CONF_THRESHOLD_CROWD,
    PROCESS_EVERY_N_FRAMES,
    DEMO_BUILDING_ZONES,
    ZONE_MAX_CAPACITIES,
)
from integration.evacuation_system import EvacuationSystem
from integration.multi_camera_sim import MultiCameraSimulator
from integration.decision_assistant import EmergencyDecisionAssistant
from graph.multi_floor_graph import MultiFloorBuildingGraph
from video.video_stream import VideoStreamHandler
from integration.hazard_trend import TemporalHazardTracker
from dashboard.styles import get_custom_css
try:
    from dashboard.components import (
        render_header,
        render_system_status_bar,
        render_cctv_meta_bar,
        render_multi_camera_panel,
        render_perception_metrics,
        render_hazard_zone_panel,
        render_crowd_intelligence_panel,
        render_zone_risk_priority_table,
        render_dynamic_cost_panel,
        render_route_comparison,
        render_baseline_vs_proposed_panel,
        render_recommended_exit_panel,
        render_route_flow_card,
        render_route_history_table,
        render_emergency_alert,
        render_security_guard_log,
        render_simulated_led_signage,
        render_pa_announcement,
        render_performance_telemetry,
        render_explainable_ai_panel,
        render_advanced_xai_panel,
        render_system_architecture_diagram,
        render_mathematical_model_panel,
        render_system_validation_panel,
        render_evacuation_risk_index,
        render_evacuation_efficiency_card,
        render_route_confidence_card,
        render_temporal_trend_panel,
        render_vulnerable_occupant_panel,
        render_incident_replay_panel,
        render_decision_assistant,
        render_model_evaluation_panel,
        render_false_alarm_panel,
        render_system_health_monitor,
        render_research_experiment_mode,
        render_ieee_traceability_matrix,
        generate_incident_report_text,
        generate_incident_report_json,
        generate_incident_report_csv,
        render_prototype_notes,
    )
except (ImportError, ModuleNotFoundError):
    from components import (
        render_header,
        render_system_status_bar,
        render_cctv_meta_bar,
        render_multi_camera_panel,
        render_perception_metrics,
        render_hazard_zone_panel,
        render_crowd_intelligence_panel,
        render_zone_risk_priority_table,
        render_dynamic_cost_panel,
        render_route_comparison,
        render_baseline_vs_proposed_panel,
        render_recommended_exit_panel,
        render_route_flow_card,
        render_route_history_table,
        render_emergency_alert,
        render_security_guard_log,
        render_simulated_led_signage,
        render_pa_announcement,
        render_performance_telemetry,
        render_explainable_ai_panel,
        render_advanced_xai_panel,
        render_system_architecture_diagram,
        render_mathematical_model_panel,
        render_system_validation_panel,
        render_evacuation_risk_index,
        render_evacuation_efficiency_card,
        render_route_confidence_card,
        render_temporal_trend_panel,
        render_vulnerable_occupant_panel,
        render_incident_replay_panel,
        render_decision_assistant,
        render_model_evaluation_panel,
        render_false_alarm_panel,
        render_system_health_monitor,
        render_research_experiment_mode,
        render_ieee_traceability_matrix,
        generate_incident_report_text,
        generate_incident_report_json,
        generate_incident_report_csv,
        render_prototype_notes,
    )


# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & STYLING INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Evacuation Command Center — IEEE Prototype",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED SYSTEM INSTANCES & HOT-RELOAD SYNCHRONIZATION (CPU INFERENCE)
# -----------------------------------------------------------------------------
REQUIRED_EMERGENCY_STATE_KEYS: Tuple[str, ...] = (
    "is_emergency",
    "status_level",
    "hazard_type",
    "crowd_status",
    "zone_status",
    "active_edge",
    "active_zone_str",
    "source_room",
    "fire_detected",
    "smoke_detected",
    "hazard_score",
    "people_count",
    "crowd_density",
    "baseline_route",
    "baseline_route_str",
    "baseline_exit",
    "baseline_cost",
    "recommended_route",
    "recommended_route_str",
    "recommended_exit",
    "total_route_cost",
    "cost_delta_vs_baseline",
    "compromised_exits",
    "avoided_exit",
    "is_diverted_from_baseline",
    "route_changed_this_frame",
    "previous_route",
    "adaptation_reason",
    "dynamic_signage",
    "pa_announcement",
    "active_alerts_count",
    "unacknowledged_alerts_count",
)


def _ensure_fresh_system_modules():
    """
    Ensures that `alerts.alert_manager` and `integration.evacuation_system` in
    `sys.modules` are up-to-date and reference the exact same `AlertManager` class
    when Streamlit Cloud hot-reloads `dashboard/app.py` without restarting the process.
    Does not reload modules on normal reruns when class definitions are already aligned.
    """
    global EvacuationSystem
    import alerts.alert_manager as am_mod
    import integration.evacuation_system as es_mod

    am_reloaded = False
    if not hasattr(am_mod.AlertManager, "sync_facility_emergency_state"):
        am_mod = importlib.reload(am_mod)
        am_reloaded = True

    es_needs_reload = (
        am_reloaded
        or getattr(es_mod, "AlertManager", None) is not am_mod.AlertManager
        or not hasattr(getattr(es_mod, "AlertManager", None), "sync_facility_emergency_state")
        or not hasattr(es_mod.EvacuationSystem, "build_facility_emergency_state")
        or not hasattr(es_mod.EvacuationSystem, "ensure_emergency_state")
        or not hasattr(es_mod.EvacuationSystem, "_ensure_compatible_alert_manager")
    )
    if es_needs_reload:
        es_mod = importlib.reload(es_mod)
        es_mod.AlertManager = am_mod.AlertManager

    EvacuationSystem = es_mod.EvacuationSystem
    return EvacuationSystem


def _ensure_compatible_system_instance(system: Any) -> Any:
    """
    Verifies that a cached `EvacuationSystem` instance and its `alert_manager`
    attribute use current, compatible class definitions. If `system` or
    `system.alert_manager` is an instance of a stale pre-reload class, upgrades
    them coherently while preserving loaded YOLO models and existing alert logs.
    """
    if system is None:
        return None

    evac_cls = _ensure_fresh_system_modules()
    import alerts.alert_manager as am_mod

    # 1. Ensure system.alert_manager has sync_facility_emergency_state
    if hasattr(system, "_ensure_compatible_alert_manager"):
        system._ensure_compatible_alert_manager()
    else:
        current_am = getattr(system, "alert_manager", None)
        if current_am is None or not hasattr(current_am, "sync_facility_emergency_state"):
            upgraded_am = am_mod.AlertManager()
            if current_am is not None:
                for attr in (
                    "active_alerts",
                    "guard_alert_log",
                    "route_history",
                    "dynamic_signage",
                    "current_pa_message",
                    "snapshot_dir",
                ):
                    if hasattr(current_am, attr):
                        setattr(upgraded_am, attr, getattr(current_am, attr))
            system.alert_manager = upgraded_am

    # 2. If `system` itself is an instance of a stale EvacuationSystem class, upgrade it
    #    reusing the already-loaded CPU YOLO detectors and building graph.
    if (
        not hasattr(system, "build_facility_emergency_state")
        or not hasattr(system, "ensure_emergency_state")
        or not hasattr(system, "_ensure_compatible_alert_manager")
    ):
        upgraded_sys = evac_cls(
            building_graph=getattr(system, "building_graph", None),
            optimizer=getattr(system, "optimizer", None),
            fire_detector=getattr(system, "fire_detector", None),
            crowd_detector=getattr(system, "crowd_detector", None),
            alert_manager=getattr(system, "alert_manager", None),
        )
        for state_attr in (
            "frame_counter",
            "processed_frames_counter",
            "skipped_frames_counter",
            "last_routes",
            "last_costs",
            "_last_active_edge",
            "_last_cache_signature",
            "current_emergency_state",
            "cached_result",
        ):
            if hasattr(system, state_attr):
                setattr(upgraded_sys, state_attr, getattr(system, state_attr))
        return upgraded_sys

    return system


def ensure_complete_emergency_state(
    system: Any,
    result: Dict[str, Any],
    source_room: str = "ROOM_A",
    active_edge: Tuple[str, str] = ("CORRIDOR_1", "EXIT_A")
) -> Dict[str, Any]:
    """
    Verifies that `result["emergency_state"]` exists and contains every required field.
    If `system` has `ensure_emergency_state`, delegates to it first. If any required
    key is still missing (e.g. from a stale cached object during Streamlit Cloud hot-reload),
    deterministically constructs a complete, valid facility emergency state from `result`
    and `system`'s graph/optimizer/alert_manager.
    """
    if system is not None and hasattr(system, "_ensure_compatible_alert_manager"):
        system._ensure_compatible_alert_manager()
    if system is not None and hasattr(system, "ensure_emergency_state"):
        system.ensure_emergency_state(result, source_room=source_room, active_edge=active_edge)

    em = result.get("emergency_state")
    if isinstance(em, dict) and all(k in em for k in REQUIRED_EMERGENCY_STATE_KEYS):
        return em

    # Deterministic synthesis of a complete, valid emergency state if missing/incomplete
    u, v = active_edge
    has_fire = bool(result.get("fire_detected", False))
    has_smoke = bool(result.get("smoke_detected", False))
    hazard_score = float(result.get("hazard_score", 0.0))
    people_count = int(result.get("people_detected", 0))
    crowd_density = float(result.get("crowd_density", 0.0))
    rec_route = list(result.get("recommended_route", []))
    rec_exit = str(result.get("recommended_exit", "NONE"))
    total_route_cost = float(result.get("total_route_cost", 0.0))
    route_changed_this_frame = bool(result.get("route_changed", False))
    previous_route = result.get("previous_route")

    baseline_route: List[str] = ["ROOM_A", "CORRIDOR_1", "EXIT_A"]
    baseline_exit = "EXIT_A"
    baseline_cost = 0.0
    compromised_exits: List[str] = []
    edge_len = 10.0
    updated_edge_cost = float(result.get("updated_edge_cost", GAMMA * edge_len))
    baseline_edge_cost = GAMMA * edge_len

    if system is not None and hasattr(system, "building_graph") and hasattr(system, "optimizer"):
        graph = system.building_graph.get_graph()
        all_exits = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "EXIT"] or ["EXIT_A", "EXIT_B"]
        clean_graph = graph.copy()
        for u_n, v_n in clean_graph.edges:
            clean_graph[u_n][v_n]["hazard"] = 0.0
            clean_graph[u_n][v_n]["density"] = 0.0
        system.optimizer.update_all_edge_costs(clean_graph)
        b_info = system.optimizer.find_safest_route(clean_graph, source=source_room, exits=all_exits)
        baseline_route = b_info.get("path", [])
        baseline_exit = b_info.get("selected_exit", "NONE")
        baseline_cost = float(b_info.get("total_cost", 0.0))

        if graph.has_edge(u, v):
            edge_len = float(graph[u][v].get("length", 10.0))
            updated_edge_cost = float(graph[u][v].get("cost", updated_edge_cost))
            baseline_edge_cost = system.optimizer.gamma * edge_len

        for ex in all_exits:
            if ex in graph.nodes:
                for nbr in graph.neighbors(ex):
                    edge_data = graph[nbr][ex]
                    if edge_data.get("hazard", 0.0) > 0.0 or edge_data.get("density", 0.0) >= 0.70:
                        if ex not in compromised_exits:
                            compromised_exits.append(ex)

    if has_fire or hazard_score >= 0.70:
        hazard_status, hazard_type = "CRITICAL", "FIRE"
    elif has_smoke or hazard_score >= 0.40:
        hazard_status, hazard_type = "WARNING", "SMOKE"
    elif hazard_score > 0.0:
        hazard_status, hazard_type = "CAUTION", "LOW HAZARD"
    elif crowd_density >= 0.70:
        hazard_status, hazard_type = "WARNING", "CROWD_CONGESTION"
    else:
        hazard_status, hazard_type = "NORMAL", "NONE"

    if crowd_density >= 0.70:
        crowd_status = "HIGH CONGESTION"
    elif crowd_density >= 0.35:
        crowd_status = "MODERATE"
    else:
        crowd_status = "NORMAL"

    is_emergency = bool(has_fire or has_smoke or hazard_score > 0.0 or crowd_density >= 0.70)
    zone_status = (
        "CRITICAL"
        if (hazard_status in ["CRITICAL", "WARNING"] or crowd_status == "HIGH CONGESTION")
        else ("CAUTION" if (hazard_status == "CAUTION" or crowd_status == "MODERATE") else "NORMAL")
    )

    is_diverted_from_baseline = (rec_route != baseline_route)
    if is_emergency and baseline_exit not in ("", "NONE", "UNKNOWN", rec_exit):
        if baseline_exit not in compromised_exits:
            compromised_exits.append(baseline_exit)

    avoided_exit = (
        compromised_exits[0]
        if compromised_exits
        else (baseline_exit if (is_diverted_from_baseline and baseline_exit != rec_exit) else None)
    )

    if is_diverted_from_baseline:
        if has_fire or hazard_score >= 0.70:
            adaptation_reason = (
                f"Corridor edge {u} <-> {v} compromised by Fire Hazard ({hazard_score:.3f}). "
                f"Edge cost increased from {baseline_edge_cost:.2f} to {updated_edge_cost:.2f}. "
                f"Dynamic Dijkstra rerouted origin {source_room} from {baseline_exit} to {rec_exit}."
            )
        elif has_smoke or hazard_score > 0.0:
            adaptation_reason = (
                f"Corridor edge {u} <-> {v} compromised by Smoke Hazard ({hazard_score:.3f}). "
                f"Air quality and visibility impaired. Rerouted from {baseline_exit} to {rec_exit}."
            )
        elif crowd_density >= 0.70:
            adaptation_reason = (
                f"Severe occupant congestion on {u} <-> {v} (Density: {crowd_density:.3f}). "
                f"Dynamic Dijkstra diverted path from {baseline_exit} to {rec_exit} to prevent crowd crush."
            )
        else:
            adaptation_reason = f"Dynamic path cost adjustments made alternate route via {rec_exit} globally optimal."
    else:
        if is_emergency:
            adaptation_reason = (
                f"Hazard/congestion active on {u} <-> {v} (Score: {hazard_score:.3f}), "
                f"and route via {rec_exit} remains the lowest-cost safe egress path (Total Cost: {total_route_cost:.2f})."
            )
        else:
            adaptation_reason = (
                f"Direct egress route to {baseline_exit} remains unobstructed and optimal "
                f"(Total Cost: {total_route_cost:.2f}). No detour required."
            )

    synthesized = {
        "is_emergency": is_emergency,
        "status_level": hazard_status,
        "hazard_type": hazard_type,
        "crowd_status": crowd_status,
        "zone_status": zone_status,
        "active_edge": active_edge,
        "active_zone_str": f"{u} <-> {v}",
        "source_room": source_room,
        "fire_detected": has_fire,
        "smoke_detected": has_smoke,
        "hazard_score": round(hazard_score, 3),
        "people_count": people_count,
        "crowd_density": round(crowd_density, 3),
        "baseline_route": baseline_route,
        "baseline_route_str": " -> ".join(baseline_route) if baseline_route else "NONE",
        "baseline_exit": baseline_exit,
        "baseline_cost": round(baseline_cost, 2),
        "recommended_route": rec_route,
        "recommended_route_str": " -> ".join(rec_route) if rec_route else "NO SAFE ROUTE",
        "recommended_exit": rec_exit,
        "total_route_cost": round(total_route_cost, 2),
        "cost_delta_vs_baseline": round(total_route_cost - baseline_cost, 2),
        "compromised_exits": compromised_exits,
        "avoided_exit": avoided_exit,
        "is_diverted_from_baseline": is_diverted_from_baseline,
        "route_changed_this_frame": route_changed_this_frame,
        "previous_route": previous_route,
        "adaptation_reason": adaptation_reason,
    }

    alert_mgr = getattr(system, "alert_manager", None) if system is not None else None
    if alert_mgr is not None:
        if hasattr(alert_mgr, "sync_facility_emergency_state"):
            alert_mgr.sync_facility_emergency_state(synthesized)
        else:
            if hasattr(alert_mgr, "update_dynamic_signage"):
                alert_mgr.update_dynamic_signage(recommended_exit=rec_exit, dangerous_exit=compromised_exits)
            if is_emergency and hasattr(alert_mgr, "generate_pa_announcement"):
                alert_mgr.generate_pa_announcement(
                    recommended_exit=rec_exit,
                    dangerous_zone=synthesized["active_zone_str"],
                    reason=adaptation_reason,
                    hazard_type=hazard_type,
                    hazard_score=hazard_score,
                    crowd_density=crowd_density,
                    route_str=synthesized["recommended_route_str"],
                )
            elif not is_emergency:
                alert_mgr.current_pa_message = "Facility conditions nominal. Follow standard exit signs if required."
        synthesized["dynamic_signage"] = getattr(alert_mgr, "dynamic_signage", result.get("dynamic_signage", {}))
        synthesized["pa_announcement"] = getattr(alert_mgr, "current_pa_message", result.get("current_pa_message", ""))
        active_alerts = getattr(alert_mgr, "active_alerts", [])
        synthesized["active_alerts_count"] = len(active_alerts)
        synthesized["unacknowledged_alerts_count"] = len([a for a in active_alerts if not a.get("acknowledged", False)])
    else:
        synthesized["dynamic_signage"] = result.get("dynamic_signage", {})
        synthesized["pa_announcement"] = result.get("current_pa_message", "")
        synthesized["active_alerts_count"] = 0
        synthesized["unacknowledged_alerts_count"] = 0

    result["emergency_state"] = synthesized
    result["dynamic_signage"] = synthesized["dynamic_signage"]
    result["current_pa_message"] = synthesized["pa_announcement"]
    return synthesized


@st.cache_resource(show_spinner="Loading Pretrained YOLO Models & Evacuation Engine (CPU)...")
def get_system(_schema_version: str = "v5_coherent_alert_manager_sync"):
    try:
        evac_cls = _ensure_fresh_system_modules()
        sys_inst = evac_cls()
        sys_inst = _ensure_compatible_system_instance(sys_inst)
        return sys_inst, None
    except Exception as e:
        return None, str(e)


# Alias required by tests/test_full_system.py
load_evacuation_system = get_system


@st.cache_resource
def get_multi_floor_model():
    return MultiFloorBuildingGraph()


@st.cache_resource
def get_decision_assistant():
    return EmergencyDecisionAssistant()


# -----------------------------------------------------------------------------
# MEDIA RETRIEVAL HELPERS
# -----------------------------------------------------------------------------
def get_available_test_images() -> List[str]:
    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        return []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    return [f.name for f in test_dir.iterdir() if f.suffix.lower() in valid_ext]


def get_available_test_videos() -> List[str]:
    test_dir = DATA_DIR / "test"
    if not test_dir.exists():
        return []
    valid_ext = {".mp4", ".avi", ".mov", ".mkv"}
    return [f.name for f in test_dir.iterdir() if f.suffix.lower() in valid_ext]


# -----------------------------------------------------------------------------
# MAIN APPLICATION
# -----------------------------------------------------------------------------
def main():
    # Load Core Engines and verify class & instance compatibility
    _ensure_fresh_system_modules()
    system, init_err = get_system()
    if init_err:
        st.error(f"[!] System Engine Initialization Failed: {init_err}")
        return
    system = _ensure_compatible_system_instance(system)

    multi_floor_model = get_multi_floor_model()
    assistant = get_decision_assistant()

    # -------------------------------------------------------------------------
    # SESSION STATE INITIALIZATION
    # -------------------------------------------------------------------------
    # SESSION STATE INITIALIZATION
    # -------------------------------------------------------------------------
    if "camera_simulator" not in st.session_state:
        st.session_state.camera_simulator = MultiCameraSimulator()
    if "hazard_tracker" not in st.session_state:
        st.session_state.hazard_tracker = TemporalHazardTracker()
    if "video_stream" not in st.session_state:
        st.session_state.video_stream = None
    if "monitoring_active" not in st.session_state:
        st.session_state.monitoring_active = False
    if "active_sample_file" not in st.session_state:
        st.session_state.active_sample_file = "flames_sample.jpg"
    if "input_method" not in st.session_state:
        st.session_state.input_method = "Quick Test Sample"
    if "run_benchmark_active" not in st.session_state:
        st.session_state.run_benchmark_active = False
    if "telemetry_history" not in st.session_state:
        st.session_state.telemetry_history = []
    if "incident_events" not in st.session_state:
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        st.session_state.incident_events = [
            {"time": now_str, "desc": "System online. Dual YOLOv8n initialized on CPU."},
            {"time": now_str, "desc": "Topological building digital twin loaded. Dijkstra optimizer online."},
            {"time": now_str, "desc": "CCTV surveillance feed initialized. Continuous perception active."}
        ]

    # -------------------------------------------------------------------------
    # SIDEBAR CONTROLS & CONFIGURATION
    # -------------------------------------------------------------------------
    st.sidebar.header("⚙️ Command Center Configuration")

    # Operational Mode Toggle (Feature 19)
    op_mode = st.sidebar.radio(
        "Operational Mode",
        options=["RESEARCH MODE", "EMERGENCY COMMAND MODE"],
        index=0,
        help="Research Mode displays comprehensive telemetry, algorithms, and XAI. Emergency Mode streamlines high-visibility egress directives."
    )

    # Ingestion Source
    st.sidebar.markdown("---")
    st.sidebar.subheader("📹 CCTV Feed Acquisition")
    input_options = [
        "Quick Test Sample",
        "CCTV Demo Video",
        "Laptop Webcam",
        "External USB Camera",
        "Upload Video",
        "RTSP Network Stream",
        "Upload Image"
    ]
    curr_idx = 0
    if st.session_state.input_method in input_options:
        curr_idx = input_options.index(st.session_state.input_method)

    selected_method = st.sidebar.radio("CCTV Media Source", options=input_options, index=curr_idx)
    st.session_state.input_method = selected_method

    # Extra parameters depending on source
    usb_idx = 1
    rtsp_url = "rtsp://127.0.0.1:8554/live"
    up_vid_file = None
    if selected_method == "External USB Camera":
        usb_idx = st.sidebar.number_input("USB Camera Index", min_value=1, max_value=5, value=1)
    elif selected_method == "RTSP Network Stream":
        rtsp_url = st.sidebar.text_input("RTSP Stream URL", value="rtsp://127.0.0.1:8554/live", help="Format: rtsp://username:password@ip:port/stream")
    elif selected_method == "Upload Video":
        up_vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])

    # Live Stream Monitoring Controls
    if selected_method in ["CCTV Demo Video", "Laptop Webcam", "External USB Camera", "Upload Video", "RTSP Network Stream"]:
        st.sidebar.markdown("**Live Stream Monitoring Controls**")
        mon_col1, mon_col2 = st.sidebar.columns(2)
        with mon_col1:
            if st.button("▶ START", use_container_width=True):
                target_src = 0
                if selected_method == "CCTV Demo Video":
                    target_src = str(DATA_DIR / "test" / "demo_cctv_feed.mp4")
                elif selected_method == "Laptop Webcam":
                    target_src = 0
                elif selected_method == "External USB Camera":
                    target_src = int(usb_idx)
                elif selected_method == "RTSP Network Stream":
                    target_src = rtsp_url.strip()
                elif selected_method == "Upload Video":
                    if up_vid_file is not None:
                        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                        tfile.write(up_vid_file.read())
                        target_src = tfile.name
                    else:
                        target_src = str(DATA_DIR / "test" / "demo_cctv_feed.mp4")

                # Release prior stream if active
                if st.session_state.video_stream is not None:
                    st.session_state.video_stream.release()

                stream = VideoStreamHandler(source=target_src)
                if stream.start_stream():
                    st.session_state.video_stream = stream
                    st.session_state.monitoring_active = True
                    st.session_state.camera_simulator.bind_stream("CAM-01", stream)
                    st.sidebar.success(f"Connected: {stream.source_type}")
                    st.rerun()
                else:
                    st.session_state.monitoring_active = False
                    st.sidebar.error(f"Connection Failed: {stream.error_message}")

        with mon_col2:
            if st.button("⏹ STOP", use_container_width=True):
                if st.session_state.video_stream is not None:
                    st.session_state.video_stream.release()
                    st.session_state.video_stream = None
                st.session_state.monitoring_active = False
                st.session_state.camera_simulator.unbind_stream("CAM-01")
                st.sidebar.info("Monitoring halted.")
                st.rerun()

        # Monitoring status badge in sidebar
        if st.session_state.monitoring_active and st.session_state.video_stream is not None:
            v_tel = st.session_state.video_stream.get_telemetry()
            st.sidebar.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 6px 10px; margin-top: 6px; font-size: 0.8rem; color: #34D399;">
                🟢 <strong>RUNNING:</strong> {v_tel['source_type']}<br>
                FPS: <strong>{v_tel['fps']}</strong> &bull; Res: <strong>{v_tel['resolution']}</strong><br>
                Frames: <strong>{v_tel['frame_count']}</strong>
            </div>
            """, unsafe_allow_html=True)

    # Evacuation Origin Room
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Occupant Origin Room")
    source_room = st.sidebar.selectbox("Active Origin Room", options=["ROOM_A", "ROOM_B", "ROOM_C"], index=0)

    # Monitored Corridor Zone
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Primary Monitored Edge")
    monitored_edge_label = st.sidebar.selectbox(
        "Monitored Corridor Zone (Demo Mapping)",
        options=[
            "CORRIDOR_1 <-> EXIT_A (North Exit Corridor — Default Demo Zone)",
            "CORRIDOR_2 <-> EXIT_B (South Exit Corridor)",
            "CORRIDOR_1 <-> CORRIDOR_2 (Central Hallway Crossway)",
            "ROOM_A <-> CORRIDOR_1",
            "ROOM_B <-> CORRIDOR_1",
            "ROOM_B <-> CORRIDOR_2",
            "ROOM_C <-> CORRIDOR_2"
        ],
        index=0
    )

    if "CORRIDOR_1 <-> EXIT_A" in monitored_edge_label:
        active_edge = ("CORRIDOR_1", "EXIT_A")
    elif "CORRIDOR_2 <-> EXIT_B" in monitored_edge_label:
        active_edge = ("CORRIDOR_2", "EXIT_B")
    elif "CORRIDOR_1 <-> CORRIDOR_2" in monitored_edge_label:
        active_edge = ("CORRIDOR_1", "CORRIDOR_2")
    elif "ROOM_A <-> CORRIDOR_1" in monitored_edge_label:
        active_edge = ("ROOM_A", "CORRIDOR_1")
    elif "ROOM_B <-> CORRIDOR_1" in monitored_edge_label:
        active_edge = ("ROOM_B", "CORRIDOR_1")
    elif "ROOM_B <-> CORRIDOR_2" in monitored_edge_label:
        active_edge = ("ROOM_B", "CORRIDOR_2")
    else:
        active_edge = ("ROOM_C", "CORRIDOR_2")

    # Detection Confidence Threshold Slider (Feature 17)
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Sensitivity & Threshold Control")
    conf_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )
    system.fire_detector.conf_threshold = conf_threshold
    system.crowd_detector.conf_threshold = conf_threshold

    if conf_threshold < 0.30:
        sens_label, sens_col = "HIGH SENSITIVITY", "#EF4444"
    elif conf_threshold <= 0.55:
        sens_label, sens_col = "MEDIUM SENSITIVITY", "#FBBF24"
    else:
        sens_label, sens_col = "LOW SENSITIVITY", "#10B981"
    st.sidebar.markdown(f"Sensitivity: <strong style='color:{sens_col};'>{sens_label}</strong>", unsafe_allow_html=True)

    # Vulnerable Occupant Priority Simulation (Feature 7)
    st.sidebar.markdown("---")
    st.sidebar.subheader("♿ Vulnerable Occupant Simulation")
    vulnerable_priority = st.sidebar.checkbox("Enable Vulnerable Priority Egress", value=False)
    vuln_category = st.sidebar.selectbox("Assistance Profile", options=["Mobility Assistance (Wheelchair)", "Elderly Occupant", "Child Group"], index=0)

    # Frame-Skip Optimization
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ CPU Frame-Skip Sampling")
    enable_frame_skip = st.sidebar.checkbox("Enable Frame-Skip Sampling", value=True)
    frame_skip_val = st.sidebar.slider("Sample Every N Frames", min_value=1, max_value=15, value=PROCESS_EVERY_N_FRAMES)

    # -------------------------------------------------------------------------
    # MEDIA INGESTION
    # -------------------------------------------------------------------------
    input_frame = None
    media_source_description = ""
    is_video_mode = False

    # 1. Active Live Stream Monitoring Path (Persistent Handle)
    if st.session_state.monitoring_active and st.session_state.video_stream is not None:
        is_video_mode = True
        success, v_frame = st.session_state.video_stream.get_frame(loop_video=True)
        if success and v_frame is not None:
            input_frame = v_frame
            v_tel = st.session_state.video_stream.get_telemetry()
            media_source_description = (
                f"🟢 Live Monitoring Active &bull; {v_tel['source_type']} &bull; "
                f"Running FPS: {v_tel['fps']} &bull; Res: {v_tel['resolution']} &bull; Frame: #{v_tel['frame_count']}"
            )
        else:
            # Stream disconnected or encountered error
            st.session_state.monitoring_active = False
            err_msg = st.session_state.video_stream.error_message or "Camera feed interrupted"
            st.session_state.camera_simulator.update_camera_telemetry(
                cam_id="CAM-01",
                fire_detected=False,
                smoke_detected=False,
                hazard_score=0.0,
                people_count=0,
                crowd_density=0.0,
                coverage="UNAVAILABLE",
                occlusion="HIGH"
            )
            st.session_state.hazard_tracker.update(
                hazard_score=0.0,
                camera_status="UNAVAILABLE"
            )
            st.error(f"⚠️ Live camera disconnected / unavailable: {err_msg}")
            st.warning("VideoStreamHandler released safely. Do NOT invent fake detections. Please check your camera device or RTSP endpoint and click '▶ START' to reconnect.")
            return

    # 2. Standby / Static Sample Path
    elif selected_method.startswith("Quick Test"):
        sample_name = st.session_state.active_sample_file
        sample_path = DATA_DIR / "test" / sample_name
        if sample_path.exists():
            input_frame = cv2.imread(str(sample_path))
            media_source_description = f"Local Benchmark Sample: `{sample_name}`"
        else:
            st.error(f"[!] Benchmark file `{sample_name}` not found in `{DATA_DIR / 'test'}`.")

    elif selected_method == "Upload Image":
        up_img = st.sidebar.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "bmp"])
        if up_img is not None:
            try:
                file_bytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
                input_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if input_frame is not None:
                    media_source_description = f"Uploaded Custom Image: `{up_img.name}` ({input_frame.shape[1]}x{input_frame.shape[0]})"
                else:
                    sample_path = DATA_DIR / "test" / "flames_sample.jpg"
                    input_frame = cv2.imread(str(sample_path))
            except Exception as e:
                input_frame = cv2.imread(str(DATA_DIR / "test" / "flames_sample.jpg"))
        else:
            sample_path = DATA_DIR / "test" / st.session_state.active_sample_file
            if sample_path.exists():
                input_frame = cv2.imread(str(sample_path))
                media_source_description = f"Awaiting Upload &bull; Displaying Sample: `{sample_path.name}`"

    elif selected_method == "CCTV Demo Video":
        sample_path = DATA_DIR / "test" / "demo_cctv_feed.mp4"
        if sample_path.exists():
            cap_prev = cv2.VideoCapture(str(sample_path))
            ret_p, frame_p = cap_prev.read()
            cap_prev.release()
            if ret_p and frame_p is not None:
                input_frame = frame_p
                media_source_description = "CCTV Sample Preview (`demo_cctv_feed.mp4`) &bull; Click '▶ START' in sidebar to run live continuous loop."
        if input_frame is None:
            input_frame = cv2.imread(str(DATA_DIR / "test" / "normal_sample.jpg"))
            media_source_description = "Awaiting Video Stream &bull; Click '▶ START' in sidebar."

    else:
        # Standby for Laptop Webcam, External USB Camera, Upload Video, RTSP Network Stream
        sample_path = DATA_DIR / "test" / st.session_state.active_sample_file
        if sample_path.exists():
            input_frame = cv2.imread(str(sample_path))
            media_source_description = f"Standby Mode ({selected_method}) &bull; Click '▶ START' in sidebar to activate real-time feed."

    if input_frame is None:
        st.error("⚠️ No valid surveillance frame available. Please select a Quick Test sample or upload an image.")
        return

    # -------------------------------------------------------------------------
    # PIPELINE EXECUTION (DUAL YOLOV8N CPU INFERENCE & DYNAMIC DIJKSTRA)
    # -------------------------------------------------------------------------
    # Apply vulnerable occupant priority weighting adjustment if active
    if vulnerable_priority:
        system.optimizer.beta = BETA + 1.5
    else:
        system.optimizer.beta = BETA

    with st.spinner("Processing frame (Dual YOLOv8 on CPU + Dynamic Dijkstra)..."):
        try:
            result = system.process_frame(
                frame=input_frame,
                active_edge=active_edge,
                source_room=source_room,
                enable_frame_skip=(enable_frame_skip and is_video_mode),
                skip_interval=frame_skip_val
            )
        except Exception as e:
            st.error(f"⚠️ Error executing pipeline on current frame: {e}")
            return

    # Verify & unpack unified facility emergency state (Single Source of Truth)
    ensure_complete_emergency_state(
        system=system,
        result=result,
        source_room=source_room,
        active_edge=active_edge
    )
    emergency_state = result["emergency_state"]
    has_fire = emergency_state["fire_detected"]
    has_smoke = emergency_state["smoke_detected"]
    hazard_score = emergency_state["hazard_score"]
    people_count = emergency_state["people_count"]
    crowd_density = emergency_state["crowd_density"]
    current_exit = emergency_state["recommended_exit"]
    current_route = emergency_state["recommended_route"]
    current_route_str = emergency_state["recommended_route_str"]
    current_cost = emergency_state["total_route_cost"]
    normal_route = emergency_state["baseline_route"]
    normal_exit = emergency_state["baseline_exit"]
    normal_cost = emergency_state["baseline_cost"]
    hazard_status = emergency_state["status_level"]
    hazard_type_str = emergency_state["hazard_type"]
    crowd_status = emergency_state["crowd_status"]
    zone_status_label = emergency_state["zone_status"]
    route_changed = emergency_state["is_diverted_from_baseline"]
    avoided_exit = emergency_state["avoided_exit"]
    cost_delta = emergency_state["cost_delta_vs_baseline"]
    adaptation_reason = emergency_state["adaptation_reason"]
    signage_dict = emergency_state["dynamic_signage"]
    pa_announcement = emergency_state["pa_announcement"]

    annotated_frame = result["annotated_frame"]
    latency_info = result.get("latency_metrics", {})
    zone_counts = result.get("multi_zone_counts", {})
    zone_densities = result.get("multi_zone_densities", {})

    # Retrieve graph edge metrics
    graph = system.building_graph.get_graph()
    edge_len = graph[active_edge[0]][active_edge[1]]["length"]
    updated_edge_cost = graph[active_edge[0]][active_edge[1]]["cost"]
    baseline_edge_cost = GAMMA * edge_len

    # Update Multi-Camera Simulator Telemetry for CAM-01 (Primary Active Camera)
    st.session_state.camera_simulator.update_camera_telemetry(
        cam_id="CAM-01",
        fire_detected=has_fire,
        smoke_detected=has_smoke,
        hazard_score=hazard_score,
        people_count=people_count,
        crowd_density=crowd_density,
        latency_ms=latency_info.get("total_ms", 0.0),
        sample_file=st.session_state.active_sample_file
    )

    # Update Temporal Hazard Tracker
    f_conf = max([d["confidence"] for d in result.get("fire_detections", []) if d.get("class_name") == "fire"] or [0.0])
    s_conf = max([d["confidence"] for d in result.get("fire_detections", []) if d.get("class_name") == "smoke"] or [0.0])
    st.session_state.hazard_tracker.update(
        hazard_score=hazard_score,
        fire_conf=f_conf,
        smoke_conf=s_conf,
        people_count=people_count,
        crowd_density=crowd_density,
        camera_id="CAM-01",
        zone=f"{active_edge[0]} <-> {active_edge[1]}",
        camera_status="ONLINE"
    )

    # Calculate Evacuation Risk Index & Efficiency Index
    risk_index = min(100, int(round((50.0 * hazard_score) + (30.0 * crowd_density) + (20.0 * min(1.0, current_cost / 25.0)))))
    norm_c = min(1.0, current_cost / 30.0)
    efficiency_index = max(0, min(100, int(round(100.0 - ((40.0 * hazard_score) + (30.0 * crowd_density) + (20.0 * norm_c))))))

    # Calculate route physical distance
    route_dist = 0.0
    if current_route and len(current_route) > 1:
        for idx in range(len(current_route) - 1):
            if graph.has_edge(current_route[idx], current_route[idx + 1]):
                route_dist += graph[current_route[idx]][current_route[idx + 1]]["length"]

    # Update Telemetry History Buffer
    now_ts = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.telemetry_history.append({
        "time": now_ts,
        "hazard_score": hazard_score,
        "crowd_density": crowd_density,
        "people_count": people_count,
        "route_cost": current_cost,
        "efficiency": efficiency_index
    })
    if len(st.session_state.telemetry_history) > 30:
        st.session_state.telemetry_history.pop(0)

    # -------------------------------------------------------------------------
    # TOP HEADER & SYSTEM STATUS BAR (Section 1)
    # -------------------------------------------------------------------------
    render_header(mode=op_mode)
    render_system_status_bar()

    # -------------------------------------------------------------------------
    # QUICK DEMO SCENARIO CONTROLLER (Feature 22)
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">Academic Demo Scenario Controller</div>', unsafe_allow_html=True)
    sc_c1, sc_c2, sc_c3, sc_c4, sc_c5, sc_c6 = st.columns(6)
    with sc_c1:
        if st.button("🟢 01 NORMAL", use_container_width=True):
            st.session_state.active_sample_file = "normal_sample.jpg"
            st.session_state.input_method = "Quick Test: NORMAL"
            st.session_state.run_benchmark_active = False
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 01 NORMAL activated: Clear baseline."})
            st.rerun()
    with sc_c2:
        if st.button("💨 02 SMOKE", use_container_width=True):
            st.session_state.active_sample_file = "fire_sample.jpg"
            st.session_state.input_method = "Quick Test: SMOKE"
            st.session_state.run_benchmark_active = False
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 02 SMOKE activated: Smoke plume detected."})
            st.rerun()
    with sc_c3:
        if st.button("🔥 03 FIRE", use_container_width=True):
            st.session_state.active_sample_file = "flames_sample.jpg"
            st.session_state.input_method = "Quick Test: FIRE"
            st.session_state.run_benchmark_active = False
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 03 FIRE activated: Active flames detected."})
            st.rerun()
    with sc_c4:
        if st.button("👥 04 HIGH CROWD", use_container_width=True):
            st.session_state.active_sample_file = "crowd_sample.jpg"
            st.session_state.input_method = "Quick Test: CROWD"
            st.session_state.run_benchmark_active = False
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 04 HIGH CROWD activated: Congestion bottleneck."})
            st.rerun()
    with sc_c5:
        if st.button("⚡ 05 COMPOUND", use_container_width=True):
            st.session_state.active_sample_file = "compound_sample.jpg"
            st.session_state.input_method = "Quick Test: COMPOUND"
            st.session_state.run_benchmark_active = False
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 05 COMPOUND activated: Simultaneous fire + crowd."})
            st.rerun()
    with sc_c6:
        if st.button("▶ RUN FULL DEMO", use_container_width=True):
            st.session_state.run_benchmark_active = True

    # Advanced Scenarios Sub-Bar
    adv_c1, adv_c2, adv_c3, adv_c4, adv_c5 = st.columns(5)
    with adv_c1:
        if st.button("📹 06 CAMERA OCCLUSION", use_container_width=True):
            st.session_state.camera_simulator.update_camera_telemetry(cam_id="CAM-01", fire_detected=False, smoke_detected=False, hazard_score=0.0, people_count=0, crowd_density=0.0, coverage="UNAVAILABLE", occlusion="HIGH")
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 06: Camera occlusion on CAM-01 triggered coverage uncertainty."})
            st.rerun()
    with adv_c2:
        if st.button("🚫 07 EXIT BLOCKED", use_container_width=True):
            st.session_state.active_sample_file = "flames_sample.jpg"
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 07: EXIT_A access corridor blocked."})
            st.rerun()
    with adv_c3:
        if st.button("🔥 08 MULTI-ZONE HAZARD", use_container_width=True):
            st.session_state.camera_simulator.update_camera_telemetry(cam_id="CAM-01", fire_detected=True, smoke_detected=False, hazard_score=0.88, people_count=2, crowd_density=0.1, sample_file="flames_sample.jpg")
            st.session_state.camera_simulator.update_camera_telemetry(cam_id="CAM-02", fire_detected=False, smoke_detected=True, hazard_score=0.65, people_count=5, crowd_density=0.3, sample_file="fire_sample.jpg")
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 08: Multi-zone hazard across CAM-01 and CAM-02."})
            st.rerun()
    with adv_c4:
        if st.button("👥 09 CROWD + SMOKE", use_container_width=True):
            st.session_state.active_sample_file = "compound_sample.jpg"
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 09: Heavy occupant density fleeing smoke plume."})
            st.rerun()
    with adv_c5:
        if st.button("📡 10 MULTI-CAM INCIDENT", use_container_width=True):
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 10: Multi-camera synchronized surveillance incident."})
            st.rerun()

    # Sequential Benchmark Sweep Execution
    if st.session_state.run_benchmark_active:
        st.markdown("#### Sequential Benchmark Sweep Results (5 IEEE Scenarios)")
        scenarios = [
            ("01 NORMAL", "normal_sample.jpg"),
            ("02 SMOKE", "fire_sample.jpg"),
            ("03 FIRE", "flames_sample.jpg"),
            ("04 HIGH CROWD", "crowd_sample.jpg"),
            ("05 COMPOUND", "compound_sample.jpg")
        ]
        b_res = []
        p_bar = st.progress(0.0)
        for i, (sc_name, sc_img) in enumerate(scenarios):
            img_p = DATA_DIR / "test" / sc_img
            if img_p.exists():
                b_frame = cv2.imread(str(img_p))
                t0_b = time.time()
                res = system.process_frame(b_frame, active_edge=active_edge, source_room=source_room, enable_frame_skip=False)
                ensure_complete_emergency_state(system=system, result=res, source_room=source_room, active_edge=active_edge)
                dt_b = (time.time() - t0_b) * 1000.0
                b_res.append({
                    "Scenario": sc_name,
                    "Fire": "YES" if res["fire_detected"] else "NO",
                    "Smoke": "YES" if res["smoke_detected"] else "NO",
                    "People": res["people_detected"],
                    "Density": f"{res['crowd_density']:.3f}",
                    "Hazard": f"{res['hazard_score']:.3f}",
                    "Dynamic Exit": res["recommended_exit"],
                    "Route": res["recommended_route_str"],
                    "Total Cost": f"{res['total_route_cost']:.2f}",
                    "Latency": f"{dt_b:.1f} ms"
                })
            p_bar.progress((i + 1) / len(scenarios))
            time.sleep(0.04)
        st.dataframe(pd.DataFrame(b_res), hide_index=True, use_container_width=True)
        st.success("✅ **BENCHMARK COMPLETE**: All 5 sequential scenarios evaluated successfully via real dual YOLO perception and Dynamic Dijkstra!")

    # -------------------------------------------------------------------------
    # OPERATIONAL MODE BRANCHING
    # -------------------------------------------------------------------------
    if op_mode == "EMERGENCY COMMAND MODE":
        # Streamlined, high-visibility emergency operations view
        st.markdown("---")
        st.markdown('<div class="section-header" style="color: #FF2E4D; border-left-color: #FF2E4D;">EMERGENCY EVACUATION DIRECTIVES</div>', unsafe_allow_html=True)

        em_col1, em_col2 = st.columns([1.5, 1.0])
        with em_col1:
            render_emergency_alert(
                has_fire=has_fire,
                has_smoke=has_smoke,
                hazard_score=hazard_score,
                crowd_density=crowd_density,
                active_edge=active_edge,
                current_exit=current_exit,
                alert_manager=system.alert_manager
            )
            render_recommended_exit_panel(
                recommended_exit=current_exit,
                total_route_cost=current_cost,
                route_distance=route_dist,
                avoided_exit=avoided_exit
            )
            render_route_flow_card(
                route_nodes=current_route,
                total_cost=current_cost,
                route_changed=route_changed
            )
        with em_col2:
            render_simulated_led_signage(signage_dict=signage_dict)
            render_pa_announcement(message=pa_announcement)
            guard_logs = result.get("security_guard_log") or system.alert_manager.get_guard_log(limit=10)
            render_security_guard_log(guard_logs=guard_logs, alert_manager=system.alert_manager)

        st.markdown("---")
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Surveillance CCTV AI Annotated Feed (HUD)", use_container_width=True)
        return  # End early in Emergency Command Mode

    # -------------------------------------------------------------------------
    # RESEARCH MODE: TOP NAVIGATION TABS (Feature 20)
    # -------------------------------------------------------------------------
    st.markdown("---")
    tabs = st.tabs([
        "OVERVIEW",
        "LIVE CCTV",
        "BUILDING DIGITAL TWIN",
        "EVACUATION",
        "ALERTS",
        "ANALYTICS",
        "XAI & ASSISTANT",
        "VALIDATION",
        "REPORTS & REPLAY"
    ])

    # TAB 1: OVERVIEW
    with tabs[0]:
        st.markdown('<div class="section-header">Operational Overview &amp; Primary Egress Status</div>', unsafe_allow_html=True)
        render_evacuation_risk_index(
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            route_cost=current_cost,
            route_changed=route_changed
        )
        render_evacuation_efficiency_card(
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            route_cost=current_cost,
            exit_available=(current_exit in ["EXIT_A", "EXIT_B"])
        )
        render_perception_metrics(
            has_fire=has_fire,
            has_smoke=has_smoke,
            hazard_score=hazard_score,
            people_count=people_count,
            crowd_density=crowd_density,
            hazard_status=hazard_status,
            crowd_status=crowd_status
        )
        render_hazard_zone_panel(
            active_edge=active_edge,
            hazard_type_str=hazard_type_str,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            status_label=zone_status_label,
            confidence=hazard_score
        )
        render_multi_camera_panel(camera_sim=st.session_state.camera_simulator)

    # TAB 2: LIVE CCTV
    with tabs[1]:
        st.markdown('<div class="section-header">Surveillance CCTV Video Stream Ingestion (Raw vs AI HUD)</div>', unsafe_allow_html=True)
        if media_source_description:
            st.caption(f"Source: {media_source_description}")

        feed_c1, feed_c2 = st.columns(2)
        with feed_c1:
            st.markdown("**RAW CCTV INPUT**")
            raw_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            st.image(raw_rgb, use_container_width=True)

        with feed_c2:
            st.markdown("**AI ANNOTATED OUTPUT (Bounding Boxes, HUD & Confidence)**")
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_container_width=True)

        h_in, w_in = input_frame.shape[:2] if input_frame is not None else (720, 1280)
        render_cctv_meta_bar(
            width=w_in,
            height=h_in,
            frame_num=latency_info.get("frame_number", 1),
            latency_ms=latency_info.get("total_ms", 0.0),
            fps=latency_info.get("fps", 0.0)
        )
        render_multi_camera_panel(camera_sim=st.session_state.camera_simulator)

    # TAB 3: BUILDING DIGITAL TWIN (Features 2, 3 & 4)
    with tabs[2]:
        st.markdown('<div class="section-header">Building Digital Twin &mdash; Topological Graph Map</div>', unsafe_allow_html=True)

        dt_c1, dt_c2 = st.columns([2, 1])
        with dt_c1:
            fig_map = system.building_graph.render_evacuation_map(
                recommended_route=current_route,
                previous_route=result.get("previous_route"),
                source_room=source_room,
                monitored_edge=active_edge,
                hazardous_edges=[active_edge] if hazard_score > 0 else [],
                congested_edges=[active_edge] if crowd_density >= 0.35 else [],
                system_state=hazard_status,
                dark_mode=True
            )
            st.pyplot(fig_map)

        with dt_c2:
            st.markdown("#### Topology Model Selector")
            topology_choice = st.radio("Active Digital Twin Layout", options=["Baseline 7-Node Model (Default)", "Extended Multi-Floor Model (2 Floors)"], index=0)
            if "Multi-Floor" in topology_choice:
                floor_sel = st.selectbox("Floor View", options=["Ground Floor (Level 0)", "First Floor (Level 1)"], index=0)
                fl_num = 0 if "Ground" in floor_sel else 1
                fig_fl = multi_floor_model.render_floor_map(floor=fl_num, recommended_route=["G_ROOM_A", "G_CORRIDOR_1", "G_EXIT_A"] if fl_num == 0 else ["F1_ROOM_C", "F1_CORRIDOR_2", "F1_EXIT_B"], dark_mode=True)
                st.pyplot(fig_fl)
                st.caption("Connected vertically via Central Stairwell (STAIR_1).")
            else:
                st.info("Operating on standard single-level 7-node demonstration facility.")

        # Multi-Zone Occupancy & Heatmap Ranking (Features 2 & 4)
        render_zone_risk_priority_table(
            zone_counts=zone_counts,
            zone_densities=zone_densities,
            hazard_score=hazard_score,
            active_edge=active_edge,
            recommended_route=current_route
        )
        render_crowd_intelligence_panel(
            zone_counts=zone_counts,
            zone_densities=zone_densities,
            total_people=people_count,
            zone_capacities=ZONE_MAX_CAPACITIES
        )

    # TAB 4: EVACUATION & ROUTING
    with tabs[3]:
        st.markdown('<div class="section-header">Dynamic Dijkstra Evacuation Optimization</div>', unsafe_allow_html=True)
        render_dynamic_cost_panel(
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            edge_length=edge_len,
            edge_cost=updated_edge_cost,
            active_edge=active_edge,
            baseline_cost=baseline_edge_cost
        )
        render_route_comparison(
            normal_route=normal_route,
            normal_exit=normal_exit,
            normal_cost=normal_cost,
            current_route=current_route,
            current_exit=current_exit,
            current_cost=current_cost,
            route_changed=route_changed,
            reason=adaptation_reason,
            cost_delta=cost_delta
        )
        render_recommended_exit_panel(
            recommended_exit=current_exit,
            total_route_cost=current_cost,
            route_distance=route_dist,
            avoided_exit=avoided_exit
        )
        render_route_flow_card(
            route_nodes=current_route,
            total_cost=current_cost,
            route_changed=route_changed
        )
        render_baseline_vs_proposed_panel(
            normal_route=normal_route,
            normal_exit=normal_exit,
            normal_cost=normal_cost,
            current_route=current_route,
            current_exit=current_exit,
            current_cost=current_cost,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            dijkstra_ms=latency_info.get("dijkstra_ms", 0.3)
        )
        render_route_confidence_card(
            hazard_confidence=hazard_score,
            people_count=people_count,
            route_cost=current_cost,
            has_hazard=(hazard_score > 0)
        )
        render_vulnerable_occupant_panel(
            vulnerable_enabled=vulnerable_priority,
            category=vuln_category
        )

    # TAB 5: ALERTS & DISPATCH
    with tabs[4]:
        st.markdown('<div class="section-header">Emergency Notification &amp; Facility Guidance Dispatch</div>', unsafe_allow_html=True)
        render_emergency_alert(
            has_fire=has_fire,
            has_smoke=has_smoke,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            active_edge=active_edge,
            current_exit=current_exit,
            alert_manager=system.alert_manager
        )
        al_c1, al_c2 = st.columns(2)
        with al_c1:
            render_simulated_led_signage(signage_dict=signage_dict)
        with al_c2:
            render_pa_announcement(message=pa_announcement)

        guard_logs = result.get("security_guard_log") or system.alert_manager.get_guard_log(limit=10)
        render_security_guard_log(guard_logs=guard_logs, alert_manager=system.alert_manager)
        route_history_list = result.get("route_history") or system.alert_manager.get_route_history(limit=10)
        render_route_history_table(
            route_history=route_history_list,
            current_route=current_route,
            normal_route=normal_route,
            current_exit=current_exit,
            normal_exit=normal_exit,
            reason=adaptation_reason
        )

    # TAB 6: ANALYTICS & TELEMETRY
    with tabs[5]:
        st.markdown('<div class="section-header">Real-Time Performance Analytics &amp; Hardware Telemetry</div>', unsafe_allow_html=True)
        render_performance_telemetry(latency_info=latency_info, frame_skip_val=frame_skip_val)
        render_system_health_monitor(latency_info=latency_info, fps=latency_info.get("fps", 0.0), frame_skip=frame_skip_val)
        render_temporal_trend_panel(history=st.session_state.telemetry_history, tracker=st.session_state.hazard_tracker)

        if st.session_state.telemetry_history and len(st.session_state.telemetry_history) > 1:
            df_hist = pd.DataFrame(st.session_state.telemetry_history)
            ch_c1, ch_c2 = st.columns(2)
            with ch_c1:
                st.markdown("**HAZARD SCORE & CROWD DENSITY OVER TIME**")
                c_df1 = df_hist[["hazard_score", "crowd_density"]].copy()
                c_df1.columns = ["Hazard Score", "Crowd Density"]
                st.line_chart(c_df1, use_container_width=True)
            with ch_c2:
                st.markdown("**DYNAMIC ROUTE COST & EFFICIENCY OVER TIME**")
                c_df2 = df_hist[["route_cost", "efficiency"]].copy()
                c_df2.columns = ["Route Cost", "Efficiency Index"]
                st.line_chart(c_df2, use_container_width=True)

        # Research Experiment Mode (Feature 23)
        st.markdown("---")
        st.subheader("Research Sensitivity Experimentation")
        exp_c1, exp_c2, exp_c3 = st.columns(3)
        with exp_c1:
            exp_alpha = st.slider("Experimental Hazard Weight (α)", 0.0, 15.0, ALPHA, 0.5)
        with exp_c2:
            exp_beta = st.slider("Experimental Crowd Weight (β)", 0.0, 10.0, BETA, 0.5)
        with exp_c3:
            exp_gamma = st.slider("Experimental Distance Weight (γ)", 0.01, 1.0, GAMMA, 0.05)
        render_research_experiment_mode(alpha=exp_alpha, beta=exp_beta, gamma=exp_gamma)

    # TAB 7: XAI & DECISION ASSISTANT
    with tabs[6]:
        st.markdown('<div class="section-header">Explainable AI (XAI) &amp; Decision Justification</div>', unsafe_allow_html=True)
        render_advanced_xai_panel(
            active_edge=active_edge,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            edge_len=edge_len,
            total_cost=current_cost,
            normal_exit=normal_exit,
            current_exit=current_exit,
            normal_cost=normal_cost
        )
        render_explainable_ai_panel(
            current_route=current_route,
            normal_route=normal_route,
            route_changed=route_changed,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            total_cost=current_cost,
            recommended_exit=current_exit
        )
        assistant_state = {
            "fire_detected": has_fire,
            "smoke_detected": has_smoke,
            "hazard_score": hazard_score,
            "people_detected": people_count,
            "crowd_density": crowd_density,
            "route_changed": route_changed,
            "recommended_exit": current_exit,
            "normal_exit": normal_exit,
            "recommended_route_str": current_route_str,
            "active_edge": active_edge,
            "total_route_cost": current_cost,
            "edge_cost": updated_edge_cost,
            "risk_index": risk_index
        }
        render_decision_assistant(assistant=assistant, state=assistant_state)

    # TAB 8: VALIDATION & IEEE ARCHITECTURE
    with tabs[7]:
        st.markdown('<div class="section-header">System Validation &amp; IEEE Paper Alignment</div>', unsafe_allow_html=True)
        render_system_validation_panel()
        render_model_evaluation_panel()
        render_false_alarm_panel()
        render_system_architecture_diagram()
        render_mathematical_model_panel()
        render_ieee_traceability_matrix()

    # TAB 9: REPORTS & REPLAY
    with tabs[8]:
        st.markdown('<div class="section-header">Incident Replay &amp; Digital Audit Report Export</div>', unsafe_allow_html=True)
        render_incident_replay_panel(events=st.session_state.incident_events)

        report_payload = {
            "origin_room": source_room,
            "monitored_edge": f"{active_edge[0]} <-> {active_edge[1]}",
            "fire_detected": "YES" if has_fire else "NO",
            "smoke_detected": "YES" if has_smoke else "NO",
            "hazard_score": hazard_score,
            "people_detected": people_count,
            "crowd_density": crowd_density,
            "risk_index": risk_index,
            "efficiency_index": efficiency_index,
            "baseline_route": " -> ".join(normal_route) if normal_route else "NONE",
            "dynamic_route": current_route_str,
            "recommended_exit": current_exit,
            "baseline_cost": normal_cost,
            "route_cost": current_cost,
            "route_changed": "YES" if route_changed else "NO",
            "reroute_reason": adaptation_reason,
            "guard_action": f"Evacuate via {current_exit}" if route_changed else "Monitor corridor",
            "signage_status": "Dynamic routes active",
            "pa_message": pa_announcement,
            "fire_ms": latency_info.get("fire_ms", 0.0),
            "crowd_ms": latency_info.get("crowd_ms", 0.0),
            "dijkstra_ms": latency_info.get("dijkstra_ms", 0.0),
            "total_ms": latency_info.get("total_ms", 0.0),
            "fps": latency_info.get("fps", 0.0)
        }

        st.markdown("#### Digital Incident Report Downloads")
        rep_c1, rep_c2, rep_c3 = st.columns(3)
        with rep_c1:
            st.download_button(
                label="📄 DOWNLOAD REPORT (.TXT)",
                data=generate_incident_report_text(report_payload),
                file_name=f"Incident_Report_{now_ts.replace(':', '')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with rep_c2:
            st.download_button(
                label="📦 DOWNLOAD AUDIT (.JSON)",
                data=generate_incident_report_json(report_payload),
                file_name=f"Incident_Audit_{now_ts.replace(':', '')}.json",
                mime="application/json",
                use_container_width=True
            )
        with rep_c3:
            st.download_button(
                label="📊 DOWNLOAD TELEMETRY (.CSV)",
                data=generate_incident_report_csv(report_payload),
                file_name=f"Incident_Telemetry_{now_ts.replace(':', '')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        render_prototype_notes()

    # Developer Diagnostics Expander
    with st.expander("🛠️ Developer Diagnostics & Configuration Metadata", expanded=False):
        st.markdown(f"""
        - **Python Version**: `{sys.version.split()[0]}`
        - **Project Directory**: `{project_root}`
        - **Weights**: `ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}`
        - **Detection Threshold**: `{conf_threshold}`
        - **Topological Nodes**: `{list(system.building_graph.graph.nodes())}`
        - **Monitored Edge**: `{active_edge}`
        - **Vulnerable Priority**: `{vulnerable_priority}`
        """)

    # Live continuous monitoring loop auto-refresh
    if st.session_state.monitoring_active and st.session_state.video_stream is not None:
        time.sleep(0.04)
        st.rerun()


if __name__ == "__main__":
    main()
