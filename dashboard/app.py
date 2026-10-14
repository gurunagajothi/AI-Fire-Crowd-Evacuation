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
import threading
from video.video_stream import (
    VideoStreamHandler,
    discover_available_cameras,
    is_cloud_environment,
)
from integration.hazard_trend import TemporalHazardTracker
from dashboard.styles import get_custom_css

try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        RTCConfiguration,
        VideoProcessorBase,
    )
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    VideoProcessorBase = object  # type: ignore


class BrowserWebcamProcessor(VideoProcessorBase):
    """Thread-safe WebRTC video processor capturing browser webcam frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_count: int = 0
        self.last_timestamp: float = 0.0

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self.latest_frame = img.copy()
            self.frame_count += 1
            self.last_timestamp = time.time()
        return frame

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self.latest_frame is None:
                return None, self.frame_count
            return self.latest_frame.copy(), self.frame_count

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
# EVALUATION & VIDEO ENCODING ENGINE
# -----------------------------------------------------------------------------
def encode_frames_to_h264_mp4(frames: List[np.ndarray], fps: float = 15.0) -> bytes:
    """
    Encodes a sequence of BGR OpenCV frames into an H.264 MP4 byte stream using PyAV.
    Ensures even dimensions for yuv420p standard browser compatibility.
    """
    if not frames:
        return b""
    try:
        import av
        import io

        out_buf = io.BytesIO()
        h, w = frames[0].shape[:2]
        if w % 2 != 0:
            w -= 1
        if h % 2 != 0:
            h -= 1

        container = av.open(out_buf, mode="w", format="mp4")
        stream = container.add_stream("libx264", rate=max(1, int(round(fps))))
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}

        for frame_bgr in frames:
            resized = frame_bgr
            if frame_bgr.shape[1] != w or frame_bgr.shape[0] != h:
                resized = cv2.resize(frame_bgr, (w, h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return out_buf.getvalue()
    except Exception:
        return b""


def process_image_evaluation(
    system: Any,
    image_input: Any,
    active_edge: Tuple[str, str] = ("CORRIDOR_1", "EXIT_A"),
    source_room: str = "ROOM_A",
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[str]]:
    """
    Evaluates an uploaded or selected image through Dual YOLOv8 and Dynamic Dijkstra.
    Returns: (result_dict, raw_frame_bgr, error_message)
    """
    if system is None:
        return None, None, "EvacuationSystem engine is not initialized."

    raw_frame: Optional[np.ndarray] = None
    if isinstance(image_input, np.ndarray):
        raw_frame = image_input
    elif isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            return None, None, f"Image file not found: {p}"
        raw_frame = cv2.imread(str(p))
    elif hasattr(image_input, "read"):
        try:
            content = image_input.read()
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            file_bytes = np.frombuffer(content, dtype=np.uint8)
            raw_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            return None, None, f"Failed to read image stream: {e}"
    elif isinstance(image_input, bytes):
        try:
            file_bytes = np.frombuffer(image_input, dtype=np.uint8)
            raw_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            return None, None, f"Failed to decode image bytes: {e}"

    if raw_frame is None or raw_frame.size == 0 or raw_frame.shape[0] < 5 or raw_frame.shape[1] < 5:
        return None, None, "Invalid or unreadable image file. Please provide a valid JPG, PNG, or BMP image."

    try:
        result = system.process_frame(
            frame=raw_frame,
            active_edge=active_edge,
            source_room=source_room,
            enable_frame_skip=False
        )
        ensure_complete_emergency_state(
            system=system,
            result=result,
            source_room=source_room,
            active_edge=active_edge
        )
        return result, raw_frame, None
    except Exception as e:
        return None, raw_frame, f"Error evaluating image: {e}"


def process_video_evaluation(
    system: Any,
    video_source: Any,
    active_edge: Tuple[str, str] = ("CORRIDOR_1", "EXIT_A"),
    source_room: str = "ROOM_A",
    frame_skip: int = 3,
    max_frames: int = 90,
    progress_callback: Optional[Any] = None,
) -> Tuple[Optional[Dict[str, Any]], bytes, Optional[str]]:
    """
    Evaluates a video file sequentially through Dual YOLOv8 and Dynamic Dijkstra,
    respecting frame skipping and max frames limit.
    Returns: (video_summary_dict, annotated_mp4_bytes, error_message)
    """
    if system is None:
        return None, b"", "EvacuationSystem engine is not initialized."

    temp_file = None
    if isinstance(video_source, (str, Path)):
        video_path = str(video_source)
    elif hasattr(video_source, "read"):
        content = video_source.read()
        if hasattr(video_source, "seek"):
            video_source.seek(0)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        video_path = temp_file.name
    elif isinstance(video_source, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(video_source)
        temp_file.flush()
        temp_file.close()
        video_path = temp_file.name
    else:
        return None, b"", "Unsupported video source format."

    if not Path(video_path).exists():
        return None, b"", f"Video file does not exist: {video_path}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if temp_file:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass
        return None, b"", "Unable to open video file. Verify file codec and integrity."

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_frames = min(max_frames, total_video_frames if total_video_frames > 0 else max_frames)
    skip = max(1, frame_skip)

    processed_frames: List[np.ndarray] = []
    frame_results: List[Dict[str, Any]] = []
    peak_hazard = 0.0
    peak_people = 0
    peak_density = 0.0
    route_changes = 0
    last_exit: Optional[str] = None
    last_result: Optional[Dict[str, Any]] = None

    frame_idx = 0
    sampled_idx = 0

    try:
        while cap.isOpened() and sampled_idx < target_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            if frame_idx % skip == 0:
                sampled_idx += 1
                res = system.process_frame(
                    frame=frame,
                    active_edge=active_edge,
                    source_room=source_room,
                    enable_frame_skip=False
                )
                ensure_complete_emergency_state(
                    system=system,
                    result=res,
                    source_room=source_room,
                    active_edge=active_edge
                )
                last_result = res
                annotated = res.get("annotated_frame", frame)
                processed_frames.append(annotated)

                em = res["emergency_state"]
                h_sc = float(em.get("hazard_score", 0.0))
                p_cnt = int(em.get("people_count", 0))
                c_den = float(em.get("crowd_density", 0.0))
                rec_ex = em.get("recommended_exit", "NONE")

                if h_sc > peak_hazard:
                    peak_hazard = h_sc
                if p_cnt > peak_people:
                    peak_people = p_cnt
                if c_den > peak_density:
                    peak_density = c_den

                if last_exit is not None and rec_ex != last_exit:
                    route_changes += 1
                last_exit = rec_ex

                frame_results.append({
                    "frame_idx": frame_idx,
                    "sampled_idx": sampled_idx,
                    "hazard_score": h_sc,
                    "people_count": p_cnt,
                    "crowd_density": c_den,
                    "recommended_exit": rec_ex,
                    "total_cost": float(em.get("total_route_cost", 0.0))
                })

                if progress_callback is not None:
                    try:
                        progress_callback(sampled_idx, target_frames, res)
                    except Exception:
                        pass

            frame_idx += 1
    finally:
        cap.release()
        if temp_file:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass

    if not processed_frames:
        return None, b"", "No frames could be extracted or processed from the video."

    out_fps = max(1.0, fps / skip)
    mp4_bytes = encode_frames_to_h264_mp4(processed_frames, fps=out_fps)

    summary = {
        "total_source_frames": total_video_frames,
        "processed_frames_count": len(processed_frames),
        "source_fps": fps,
        "effective_fps": out_fps,
        "source_resolution": f"{width}x{height}",
        "frame_skip": skip,
        "peak_hazard_score": round(peak_hazard, 3),
        "peak_people_count": peak_people,
        "peak_crowd_density": round(peak_density, 3),
        "route_changes_count": route_changes,
        "primary_recommended_exit": last_exit or "NONE",
        "frame_results": frame_results,
        "latest_result": last_result,
        "first_frame": processed_frames[0] if processed_frames else None,
        "last_frame": processed_frames[-1] if processed_frames else None,
    }

    return summary, mp4_bytes, None


def get_standby_baseline_result(
    system: Any,
    source_room: str = "ROOM_A",
    active_edge: Tuple[str, str] = ("CORRIDOR_1", "EXIT_A"),
    placeholder_frame: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Generates a nominal baseline result dictionary for standby state prior to evaluation."""
    if placeholder_frame is None:
        placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    base_route = ["ROOM_A", "CORRIDOR_1", "EXIT_A"]
    base_exit = "EXIT_A"
    base_cost = 4.08
    if system is not None and hasattr(system, "optimizer") and hasattr(system, "building_graph"):
        try:
            g = system.building_graph.get_graph()
            clean_g = g.copy()
            for u_n, v_n in clean_g.edges:
                clean_g[u_n][v_n]["hazard"] = 0.0
                clean_g[u_n][v_n]["density"] = 0.0
            system.optimizer.update_all_edge_costs(clean_g)
            all_exits = [n for n, d in clean_g.nodes(data=True) if d.get("node_type") == "EXIT"] or ["EXIT_A", "EXIT_B"]
            b_info = system.optimizer.find_safest_route(clean_g, source=source_room, exits=all_exits)
            base_route = b_info.get("path", base_route)
            base_exit = b_info.get("selected_exit", base_exit)
            base_cost = float(b_info.get("total_cost", base_cost))
        except Exception:
            pass

    em_state: Dict[str, Any] = {
        "is_emergency": False,
        "status_level": "NORMAL",
        "hazard_type": "NONE",
        "crowd_status": "NORMAL",
        "zone_status": "NORMAL",
        "active_edge": active_edge,
        "active_zone_str": f"{active_edge[0]} <-> {active_edge[1]}",
        "source_room": source_room,
        "fire_detected": False,
        "smoke_detected": False,
        "hazard_score": 0.0,
        "people_count": 0,
        "crowd_density": 0.0,
        "baseline_route": base_route,
        "baseline_route_str": " -> ".join(base_route),
        "baseline_exit": base_exit,
        "baseline_cost": round(base_cost, 2),
        "recommended_route": base_route,
        "recommended_route_str": " -> ".join(base_route),
        "recommended_exit": base_exit,
        "total_route_cost": round(base_cost, 2),
        "cost_delta_vs_baseline": 0.0,
        "compromised_exits": [],
        "avoided_exit": None,
        "is_diverted_from_baseline": False,
        "route_changed_this_frame": False,
        "previous_route": None,
        "adaptation_reason": (
            f"Direct egress route to {base_exit} remains unobstructed and optimal "
            f"(Total Cost: {base_cost:.2f}). No detour required."
        ),
        "dynamic_signage": {base_exit: "SAFE ROUTE"},
        "pa_announcement": "Facility conditions nominal. Surveillance monitoring on standby.",
        "active_alerts_count": 0,
        "unacknowledged_alerts_count": 0,
    }

    res: Dict[str, Any] = {
        "annotated_frame": placeholder_frame.copy(),
        "fire_detected": False,
        "smoke_detected": False,
        "fire_detections": [],
        "people_detected": 0,
        "crowd_density": 0.0,
        "hazard_score": 0.0,
        "recommended_route": base_route,
        "recommended_route_str": " -> ".join(base_route),
        "recommended_exit": base_exit,
        "total_route_cost": base_cost,
        "latency_metrics": {"fire_ms": 0.0, "crowd_ms": 0.0, "dijkstra_ms": 0.5, "total_ms": 0.5, "fps": 0.0},
        "multi_zone_counts": {k: 0 for k in DEMO_BUILDING_ZONES},
        "multi_zone_densities": {k: 0.0 for k in DEMO_BUILDING_ZONES},
        "route_changed": False,
        "previous_route": None,
        "security_guard_log": [],
        "route_history": [],
        "dynamic_signage": {base_exit: "SAFE ROUTE"},
        "current_pa_message": "Facility conditions nominal. Surveillance monitoring on standby.",
        "emergency_state": em_state,
    }
    return res


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

    now_ts = datetime.datetime.now().strftime("%H:%M:%S")

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
        st.session_state.input_method = "Upload Image"
    if "image_eval_cache" not in st.session_state:
        st.session_state.image_eval_cache = None
    if "video_eval_cache" not in st.session_state:
        st.session_state.video_eval_cache = None
    if "discovered_cameras" not in st.session_state:
        st.session_state.discovered_cameras = None
    if "last_camera_error" not in st.session_state:
        st.session_state.last_camera_error = ""
    if "run_benchmark_active" not in st.session_state:
        st.session_state.run_benchmark_active = False
    if "telemetry_history" not in st.session_state:
        st.session_state.telemetry_history = []
    if "incident_events" not in st.session_state:
        st.session_state.incident_events = [
            {"time": now_ts, "desc": "System online. Dual YOLOv8n initialized on CPU."},
            {"time": now_ts, "desc": "Topological building digital twin loaded. Dijkstra optimizer online."},
            {"time": now_ts, "desc": "CCTV surveillance feed initialized. Continuous perception active."}
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

    # Ingestion Source with Cloud Environment Safeguards
    st.sidebar.markdown("---")
    st.sidebar.subheader("📹 CCTV Feed Acquisition")
    cloud_env = is_cloud_environment()

    if cloud_env:
        input_options = [
            "Upload Image",
            "Upload Video",
            "Browser Webcam (WebRTC / Cloud)",
            "Benchmark Test Images",
            "CCTV Demo Video",
        ]
    else:
        input_options = [
            "Upload Image",
            "Upload Video",
            "Browser Webcam (WebRTC / Cloud)",
            "Benchmark Test Images",
            "CCTV Demo Video",
            "Laptop Webcam",
            "External USB Camera",
            "RTSP Network Stream",
        ]

    # Handle legacy or migrated input selections cleanly
    if st.session_state.input_method not in input_options:
        if "Quick Test" in st.session_state.input_method:
            st.session_state.input_method = "Benchmark Test Images"
        else:
            st.session_state.input_method = "Upload Image"

    curr_idx = input_options.index(st.session_state.input_method)
    selected_method = st.sidebar.radio("CCTV Media Source", options=input_options, index=curr_idx)

    if st.session_state.input_method != selected_method:
        # Cleanly release any active OpenCV camera/stream when switching input modes
        if st.session_state.video_stream is not None:
            st.session_state.video_stream.release()
            st.session_state.video_stream = None
        st.session_state.monitoring_active = False
        st.session_state.camera_simulator.unbind_stream("CAM-01")
        st.session_state.last_camera_error = ""
        st.session_state.input_method = selected_method

    cam_device_idx = 0 if selected_method == "Laptop Webcam" else 1
    rtsp_url = "rtsp://127.0.0.1:8554/live"
    browser_cam_mode = "Live WebRTC Stream (Continuous)"
    webrtc_auto_refresh = True

    # Mode-specific sidebar controls
    if selected_method == "Browser Webcam (WebRTC / Cloud)":
        st.sidebar.caption(
            "Streams video directly from your browser camera. Works on both **Streamlit Cloud** and **Local** browsers."
        )
        browser_modes = ["Live WebRTC Stream (Continuous)", "HTML5 Camera Capture (Firewall-Safe)"]
        if not WEBRTC_AVAILABLE:
            browser_modes = ["HTML5 Camera Capture (Firewall-Safe)"]
        browser_cam_mode = st.sidebar.radio(
            "Browser Capture Engine",
            options=browser_modes,
            index=0,
            help="Use Live WebRTC for continuous streaming, or HTML5 Camera Capture if your firewall blocks WebRTC UDP."
        )
        if browser_cam_mode == "Live WebRTC Stream (Continuous)":
            webrtc_auto_refresh = st.sidebar.checkbox(
                "Continuous AI Telemetry Refresh",
                value=True,
                help="Automatically re-evaluates incoming WebRTC frames through Dual YOLOv8 and Dynamic Dijkstra."
            )

    elif selected_method in ["Laptop Webcam", "External USB Camera"]:
        if cloud_env:
            st.sidebar.warning(
                "☁️ **Streamlit Cloud Detected**: Server-side OpenCV (`cv2.VideoCapture`) runs on a remote cloud container and cannot access your physical camera. "
                "Please select **Browser Webcam (WebRTC / Cloud)** above."
            )
        else:
            if st.sidebar.button("🔍 Scan Local Cameras (0–5)", use_container_width=True):
                with st.sidebar.spinner("Probing camera indices 0–5..."):
                    st.session_state.discovered_cameras = discover_available_cameras(max_index=5)
                    st.session_state.last_camera_error = ""

            discovered = st.session_state.discovered_cameras
            default_idx = 0 if selected_method == "Laptop Webcam" else 1

            if discovered is not None:
                if len(discovered) > 0:
                    cam_options = {
                        f"Index {c['index']} — {c['resolution']} ({c['backend']})": int(c["index"])
                        for c in discovered
                    }
                    option_labels = list(cam_options.keys())
                    preferred_label_idx = 0
                    if selected_method == "External USB Camera":
                        for idx_pos, c in enumerate(discovered):
                            if int(c["index"]) >= 1:
                                preferred_label_idx = idx_pos
                                break
                    selected_cam_label = st.sidebar.selectbox(
                        "Detected Local Cameras",
                        options=option_labels,
                        index=preferred_label_idx,
                        help="Select a verified local camera discovered by OpenCV."
                    )
                    default_idx = cam_options[selected_cam_label]
                else:
                    st.sidebar.warning("⚠️ No active cameras responded on indices 0–5.")

            cam_device_idx = int(
                st.sidebar.number_input(
                    "Camera Device Index (0–5)",
                    min_value=0,
                    max_value=5,
                    value=int(default_idx),
                    help="Index 0 is built-in laptop webcam; indices 1–5 are USB/virtual cameras."
                )
            )

    elif selected_method == "RTSP Network Stream":
        rtsp_url = st.sidebar.text_input("RTSP Stream URL", value="rtsp://127.0.0.1:8554/live", help="Format: rtsp://username:password@ip:port/stream")

    # Live Stream Monitoring Controls (for continuous streaming modes only)
    if selected_method in ["CCTV Demo Video", "Laptop Webcam", "External USB Camera", "RTSP Network Stream"]:
        st.sidebar.markdown("**Live Stream Monitoring Controls**")
        mon_col1, mon_col2 = st.sidebar.columns(2)
        with mon_col1:
            if st.button("▶ START", use_container_width=True):
                target_src: Any = 0
                if selected_method == "CCTV Demo Video":
                    target_src = str(DATA_DIR / "test" / "demo_cctv_feed.mp4")
                elif selected_method in ["Laptop Webcam", "External USB Camera"]:
                    target_src = int(cam_device_idx)
                elif selected_method == "RTSP Network Stream":
                    target_src = rtsp_url.strip()

                if st.session_state.video_stream is not None:
                    st.session_state.video_stream.release()
                    st.session_state.video_stream = None

                stream = VideoStreamHandler(source=target_src)
                if stream.start_stream():
                    st.session_state.video_stream = stream
                    st.session_state.monitoring_active = True
                    st.session_state.last_camera_error = ""
                    st.session_state.camera_simulator.bind_stream("CAM-01", stream)
                    st.sidebar.success(f"Connected: {stream.source_type}")
                    st.rerun()
                else:
                    st.session_state.monitoring_active = False
                    st.session_state.video_stream = None
                    err_detail = stream.error_message
                    if isinstance(target_src, int) and not cloud_env:
                        avail = discover_available_cameras(max_index=5)
                        st.session_state.discovered_cameras = avail
                        if avail:
                            found_str = ", ".join(f"Index {c['index']} ({c['backend']}, {c['resolution']})" for c in avail)
                            err_detail = f"{err_detail} | Detected indices: {found_str}."
                    st.session_state.last_camera_error = err_detail
                    st.sidebar.error(f"Connection Failed: {err_detail}")

        with mon_col2:
            if st.button("⏹ STOP", use_container_width=True):
                if st.session_state.video_stream is not None:
                    st.session_state.video_stream.release()
                    st.session_state.video_stream = None
                st.session_state.monitoring_active = False
                st.session_state.last_camera_error = ""
                st.session_state.camera_simulator.unbind_stream("CAM-01")
                st.sidebar.info("Monitoring halted.")
                st.rerun()

        if st.session_state.last_camera_error and not st.session_state.monitoring_active and selected_method in ["Laptop Webcam", "External USB Camera", "RTSP Network Stream", "CCTV Demo Video"]:
            st.sidebar.error(f"Diagnostic: {st.session_state.last_camera_error}")

        if st.session_state.monitoring_active and st.session_state.video_stream is not None:
            v_tel = st.session_state.video_stream.get_telemetry()
            st.sidebar.markdown(f"""
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 6px; padding: 6px 10px; margin-top: 6px; font-size: 0.8rem; color: #34D399;">
                🟢 <strong>RUNNING:</strong> {v_tel['source_type']}<br>
                Backend: <strong>{v_tel.get('backend', 'AUTO')}</strong> &bull; FPS: <strong>{v_tel['fps']}</strong><br>
                Res: <strong>{v_tel['resolution']}</strong> &bull; Frames: <strong>{v_tel['frame_count']}</strong>
            </div>
            """, unsafe_allow_html=True)

    # Dynamic Occupant Origin Room from Building Graph
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Occupant Origin Room")
    graph = system.building_graph.get_graph()
    graph_rooms = sorted(list(dict.fromkeys([
        n for n, d in graph.nodes(data=True)
        if d.get("node_type") == "ROOM" or str(n).startswith("ROOM_")
    ])))
    if not graph_rooms:
        graph_rooms = ["ROOM_A", "ROOM_B", "ROOM_C"]

    source_room = st.sidebar.selectbox(
        "Active Origin Room",
        options=graph_rooms,
        index=0,
        help="Occupant starting room in building digital twin."
    )

    # Dynamic Primary Monitored Edge with Descriptive Annotations
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Primary Monitored Edge")
    edge_descriptions = {
        ("CORRIDOR_1", "EXIT_A"): "CORRIDOR_1 <-> EXIT_A (North Exit Corridor — Default Demo Zone)",
        ("CORRIDOR_2", "EXIT_B"): "CORRIDOR_2 <-> EXIT_B (South Exit Corridor)",
        ("CORRIDOR_1", "CORRIDOR_2"): "CORRIDOR_1 <-> CORRIDOR_2 (Central Hallway Crossway)",
        ("ROOM_A", "CORRIDOR_1"): "ROOM_A <-> CORRIDOR_1 (Room A Access Corridor)",
        ("ROOM_B", "CORRIDOR_1"): "ROOM_B <-> CORRIDOR_1 (Room B North Corridor)",
        ("ROOM_B", "CORRIDOR_2"): "ROOM_B <-> CORRIDOR_2 (Room B South Corridor)",
        ("ROOM_C", "CORRIDOR_2"): "ROOM_C <-> CORRIDOR_2 (Room C Access Corridor)",
    }
    mon_options = list(edge_descriptions.values())
    monitored_edge_label = st.sidebar.selectbox(
        "Monitored Corridor Zone (Demo Mapping)",
        options=mon_options,
        index=0
    )
    edge_lookup = {v: k for k, v in edge_descriptions.items()}
    active_edge = edge_lookup.get(monitored_edge_label, ("CORRIDOR_1", "EXIT_A"))

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
    if vulnerable_priority:
        system.optimizer.beta = BETA + 1.5
    else:
        system.optimizer.beta = BETA

    # CPU Frame-Skip Sampling (Only applicable for video / live stream sources)
    is_video_input = selected_method in [
        "Upload Video",
        "CCTV Demo Video",
        "Browser Webcam (WebRTC / Cloud)",
        "Laptop Webcam",
        "External USB Camera",
        "RTSP Network Stream"
    ]
    enable_frame_skip = True
    frame_skip_val = PROCESS_EVERY_N_FRAMES
    if is_video_input:
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ CPU Frame-Skip Sampling")
        enable_frame_skip = st.sidebar.checkbox(
            "Enable Frame-Skip Sampling",
            value=True,
            help="Reduces CPU load by processing every N-th frame during video playback."
        )
        frame_skip_val = st.sidebar.slider(
            "Sample Every N Frames",
            min_value=1,
            max_value=15,
            value=PROCESS_EVERY_N_FRAMES,
            help="Higher values increase video throughput by skipping intermediate frames."
        )

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
            st.session_state.input_method = "Benchmark Test Images"
            st.session_state.run_benchmark_active = False
            img_p = DATA_DIR / "test" / "normal_sample.jpg"
            if img_p.exists():
                res_demo, raw_f, _ = process_image_evaluation(system, img_p, active_edge=active_edge, source_room=source_room)
                if res_demo is not None:
                    st.session_state.image_eval_cache = {"result": res_demo, "raw_frame": raw_f, "source_name": "normal_sample.jpg"}
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 01 NORMAL activated: Clear baseline."})
            st.rerun()
    with sc_c2:
        if st.button("💨 02 SMOKE", use_container_width=True):
            st.session_state.active_sample_file = "fire_sample.jpg"
            st.session_state.input_method = "Benchmark Test Images"
            st.session_state.run_benchmark_active = False
            img_p = DATA_DIR / "test" / "fire_sample.jpg"
            if img_p.exists():
                res_demo, raw_f, _ = process_image_evaluation(system, img_p, active_edge=active_edge, source_room=source_room)
                if res_demo is not None:
                    st.session_state.image_eval_cache = {"result": res_demo, "raw_frame": raw_f, "source_name": "fire_sample.jpg"}
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 02 SMOKE activated: Smoke plume detected."})
            st.rerun()
    with sc_c3:
        if st.button("🔥 03 FIRE", use_container_width=True):
            st.session_state.active_sample_file = "flames_sample.jpg"
            st.session_state.input_method = "Benchmark Test Images"
            st.session_state.run_benchmark_active = False
            img_p = DATA_DIR / "test" / "flames_sample.jpg"
            if img_p.exists():
                res_demo, raw_f, _ = process_image_evaluation(system, img_p, active_edge=active_edge, source_room=source_room)
                if res_demo is not None:
                    st.session_state.image_eval_cache = {"result": res_demo, "raw_frame": raw_f, "source_name": "flames_sample.jpg"}
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 03 FIRE activated: Active flames detected."})
            st.rerun()
    with sc_c4:
        if st.button("👥 04 HIGH CROWD", use_container_width=True):
            st.session_state.active_sample_file = "crowd_sample.jpg"
            st.session_state.input_method = "Benchmark Test Images"
            st.session_state.run_benchmark_active = False
            img_p = DATA_DIR / "test" / "crowd_sample.jpg"
            if img_p.exists():
                res_demo, raw_f, _ = process_image_evaluation(system, img_p, active_edge=active_edge, source_room=source_room)
                if res_demo is not None:
                    st.session_state.image_eval_cache = {"result": res_demo, "raw_frame": raw_f, "source_name": "crowd_sample.jpg"}
            st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 04 HIGH CROWD activated: Congestion bottleneck."})
            st.rerun()
    with sc_c5:
        if st.button("⚡ 05 COMPOUND", use_container_width=True):
            st.session_state.active_sample_file = "compound_sample.jpg"
            st.session_state.input_method = "Benchmark Test Images"
            st.session_state.run_benchmark_active = False
            img_p = DATA_DIR / "test" / "compound_sample.jpg"
            if img_p.exists():
                res_demo, raw_f, _ = process_image_evaluation(system, img_p, active_edge=active_edge, source_room=source_room)
                if res_demo is not None:
                    st.session_state.image_eval_cache = {"result": res_demo, "raw_frame": raw_f, "source_name": "compound_sample.jpg"}
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
    # MEDIA INGESTION & EVALUATION WORKFLOW CARDS
    # -------------------------------------------------------------------------
    raw_preview_frame: Optional[np.ndarray] = None
    target_img_name: str = "custom_image"
    target_vid_name: str = "demo_cctv_feed.mp4"
    webrtc_playing_active = False
    is_video_mode = False
    media_source_description = ""

    # WORKFLOW 1: Upload Image (Explicit Gated Evaluation)
    if selected_method == "Upload Image":
        st.markdown('<div class="section-header">SURVEILLANCE IMAGE EVALUATION WORKFLOW</div>', unsafe_allow_html=True)
        st.caption("Upload a floorplan / CCTV snapshot (JPG, PNG, BMP) or choose a benchmark sample. Click **Evaluate Image** to run Dual YOLOv8 fire/smoke + crowd perception and Dynamic Dijkstra egress optimization.")

        img_col1, img_col2 = st.columns([1.2, 1.0])
        with img_col1:
            up_img_file = st.file_uploader(
                "Upload Surveillance Image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="img_eval_uploader",
                help="Accepts JPG, JPEG, PNG, BMP images."
            )
        with img_col2:
            preset_options = [
                "None (Use Uploaded Image)",
                "flames_sample.jpg (Active Fire Hazard)",
                "fire_sample.jpg (Smoke Plume Hazard)",
                "crowd_sample.jpg (Occupant Bottleneck)",
                "compound_sample.jpg (Compound Fire + Crowd)",
                "normal_sample.jpg (Nominal Facility Baseline)"
            ]
            selected_preset = st.selectbox("Or Select Benchmark Image Sample", options=preset_options, index=1 if up_img_file is None else 0)

        target_img_input = None
        if up_img_file is not None:
            target_img_input = up_img_file
            target_img_name = up_img_file.name
        elif selected_preset != "None (Use Uploaded Image)":
            p_name = selected_preset.split()[0]
            target_img_input = DATA_DIR / "test" / p_name
            target_img_name = p_name
        else:
            target_img_input = DATA_DIR / "test" / "flames_sample.jpg"
            target_img_name = "flames_sample.jpg"

        decode_err = None
        if target_img_input is not None:
            if isinstance(target_img_input, (str, Path)):
                raw_preview_frame = cv2.imread(str(target_img_input))
            elif hasattr(target_img_input, "read"):
                try:
                    c = target_img_input.read()
                    if hasattr(target_img_input, "seek"):
                        target_img_input.seek(0)
                    b_arr = np.frombuffer(c, dtype=np.uint8)
                    raw_preview_frame = cv2.imdecode(b_arr, cv2.IMREAD_COLOR)
                except Exception as ex:
                    decode_err = str(ex)

        if raw_preview_frame is None and decode_err is not None:
            st.error(f"⚠️ Uploaded image could not be decoded: {decode_err}. Please provide a valid JPG or PNG file.")

        preview_c1, preview_c2 = st.columns(2)
        with preview_c1:
            st.markdown(f"**ORIGINAL SURVEILLANCE IMAGE PREVIEW** (`{target_img_name}`)")
            if raw_preview_frame is not None:
                st.image(cv2.cvtColor(raw_preview_frame, cv2.COLOR_BGR2RGB), caption=f"Input Image: {target_img_name} ({raw_preview_frame.shape[1]}x{raw_preview_frame.shape[0]})", use_container_width=True)
            else:
                st.info("Awaiting image upload or sample selection.")

        cached_img_eval = st.session_state.image_eval_cache
        has_eval_for_current = (cached_img_eval is not None and cached_img_eval.get("source_name") == target_img_name)

        with preview_c2:
            st.markdown("**AI ANNOTATED PERCEPTION & EGRESS OVERLAY (Dual YOLOv8 HUD)**")
            if has_eval_for_current and cached_img_eval is not None:
                eval_ann = cached_img_eval["result"].get("annotated_frame")
                if eval_ann is not None:
                    st.image(cv2.cvtColor(eval_ann, cv2.COLOR_BGR2RGB), caption="Dual YOLOv8 Perception HUD & Egress Directives", use_container_width=True)
                    is_enc, enc_buf = cv2.imencode(".png", eval_ann)
                    if is_enc:
                        st.download_button(
                            label="📥 Download Annotated Image (.PNG)",
                            data=enc_buf.tobytes(),
                            file_name=f"annotated_{target_img_name.rsplit('.', 1)[0]}.png",
                            mime="image/png",
                            use_container_width=True
                        )
            else:
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.7); border: 2px dashed #475569; border-radius: 8px; padding: 40px 20px; text-align: center; color: #94A3B8;">
                    <h4>⏳ Awaiting Evaluation</h4>
                    <p style="margin: 0; font-size: 0.9rem;">Click the <strong>'🔍 Evaluate Image'</strong> button below to run Dual YOLOv8 fire/smoke + crowd perception and Dynamic Dijkstra routing.</p>
                </div>
                """, unsafe_allow_html=True)

        btn_c1, btn_c2 = st.columns([2, 1])
        with btn_c1:
            eval_img_clicked = st.button("🔍 Evaluate Image", type="primary", use_container_width=True)
        with btn_c2:
            if st.button("🔄 Clear / Reset Cache", use_container_width=True):
                st.session_state.image_eval_cache = None
                st.rerun()

        if eval_img_clicked and raw_preview_frame is not None:
            with st.spinner("Executing Dual YOLOv8 Detection & Dynamic Dijkstra Optimization..."):
                res_img, raw_f, err_eval = process_image_evaluation(
                    system=system,
                    image_input=raw_preview_frame,
                    active_edge=active_edge,
                    source_room=source_room
                )
                if err_eval is not None:
                    st.error(f"Evaluation Failed: {err_eval}")
                else:
                    st.session_state.image_eval_cache = {
                        "result": res_img,
                        "raw_frame": raw_f,
                        "source_name": target_img_name
                    }
                    st.session_state.incident_events.append({
                        "time": now_ts,
                        "desc": f"Image Evaluation completed for {target_img_name}: Hazard={res_img['hazard_score']:.3f}, People={res_img['people_detected']}, Route={res_img['recommended_route_str']}."
                    })
                    st.success("✅ Image Evaluation Complete!")
                    st.rerun()

        if has_eval_for_current and cached_img_eval is not None:
            cur_r = cached_img_eval["result"]
            em = cur_r["emergency_state"]
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            f_conf = max([d["confidence"] for d in cur_r.get("fire_detections", []) if d.get("class_name") == "fire"] or [0.0])
            s_conf = max([d["confidence"] for d in cur_r.get("fire_detections", []) if d.get("class_name") == "smoke"] or [0.0])
            with mc1:
                st.metric("🔥 Fire Detected", "YES" if em["fire_detected"] else "NO", f"Conf: {f_conf:.2f}" if em["fire_detected"] else "Nominal")
            with mc2:
                st.metric("💨 Smoke Detected", "YES" if em["smoke_detected"] else "NO", f"Conf: {s_conf:.2f}" if em["smoke_detected"] else "Nominal")
            with mc3:
                st.metric("👥 Occupant Count", f"{em['people_count']} People", f"Density: {em['crowd_density']:.2f}")
            with mc4:
                st.metric("⚠️ Hazard Score", f"{em['hazard_score']:.3f}", em["status_level"])
            with mc5:
                st.metric("🚪 Recommended Exit", em["recommended_exit"], f"Cost: {em['total_route_cost']:.2f}")

    # WORKFLOW 2: Upload Video (Explicit Gated Evaluation with Frame Skipping & H.264 Export)
    elif selected_method == "Upload Video":
        is_video_mode = True
        st.markdown('<div class="section-header">SURVEILLANCE VIDEO EVALUATION WORKFLOW</div>', unsafe_allow_html=True)
        st.caption("Upload a CCTV video file (MP4, AVI, MOV, MKV) or analyze the demo CCTV surveillance feed. Configure sampling parameters and click **Evaluate Video** to perform sequential AI perception and egress optimization.")

        vid_col1, vid_col2 = st.columns([1.2, 1.0])
        with vid_col1:
            up_vid_file = st.file_uploader(
                "Upload Video File",
                type=["mp4", "avi", "mov", "mkv"],
                key="vid_eval_uploader",
                help="Accepts MP4, AVI, MOV, MKV formats."
            )
        with vid_col2:
            vid_preset_options = [
                "None (Use Uploaded Video)",
                "demo_cctv_feed.mp4 (Corridor Surveillance Demo)"
            ]
            selected_vid_preset = st.selectbox("Or Select Benchmark Demo Video", options=vid_preset_options, index=1 if up_vid_file is None else 0)

        target_vid_source = None
        if up_vid_file is not None:
            target_vid_source = up_vid_file
            target_vid_name = up_vid_file.name
        elif selected_vid_preset != "None (Use Uploaded Video)":
            p_name = selected_vid_preset.split()[0]
            target_vid_source = DATA_DIR / "test" / p_name
            target_vid_name = p_name
        else:
            target_vid_source = DATA_DIR / "test" / "demo_cctv_feed.mp4"
            target_vid_name = "demo_cctv_feed.mp4"

        v_meta = {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0, "duration_s": 0.0}
        probe_path = None
        temp_probe = None
        if isinstance(target_vid_source, (str, Path)):
            probe_path = str(target_vid_source)
        elif hasattr(target_vid_source, "read"):
            c = target_vid_source.read()
            if hasattr(target_vid_source, "seek"):
                target_vid_source.seek(0)
            temp_probe = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_probe.write(c)
            temp_probe.flush()
            temp_probe.close()
            probe_path = temp_probe.name

        if probe_path and Path(probe_path).exists():
            cap_p = cv2.VideoCapture(probe_path)
            if cap_p.isOpened():
                v_meta["width"] = int(cap_p.get(cv2.CAP_PROP_FRAME_WIDTH))
                v_meta["height"] = int(cap_p.get(cv2.CAP_PROP_FRAME_HEIGHT))
                v_meta["fps"] = float(cap_p.get(cv2.CAP_PROP_FPS) or 25.0)
                v_meta["total_frames"] = int(cap_p.get(cv2.CAP_PROP_FRAME_COUNT))
                if v_meta["fps"] > 0:
                    v_meta["duration_s"] = round(v_meta["total_frames"] / v_meta["fps"], 2)
                cap_p.release()

        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        with meta_c1:
            st.metric("📐 Resolution", f"{v_meta['width']}x{v_meta['height']}" if v_meta["width"] else "N/A")
        with meta_c2:
            st.metric("🎞️ Frame Rate", f"{v_meta['fps']:.1f} FPS" if v_meta["fps"] else "N/A")
        with meta_c3:
            st.metric("📊 Total Frames", f"{v_meta['total_frames']} Frames")
        with meta_c4:
            st.metric("⏱️ Duration", f"{v_meta['duration_s']} s")

        set_c1, set_c2 = st.columns(2)
        with set_c1:
            vid_frame_skip = st.slider("Video Frame Skip Interval", min_value=1, max_value=10, value=3, help="Process every N-th frame to accelerate CPU perception throughput.")
        with set_c2:
            max_vid_frames = st.slider("Max Sampled Frames to Process", min_value=10, max_value=150, value=min(60, max(10, v_meta["total_frames"])), help="Limits processing duration to prevent CPU timeouts.")

        vid_p_c1, vid_p_c2 = st.columns(2)
        with vid_p_c1:
            st.markdown(f"**SOURCE VIDEO PREVIEW** (`{target_vid_name}`)")
            if probe_path and Path(probe_path).exists():
                try:
                    with open(probe_path, "rb") as vf:
                        st.video(vf.read())
                except Exception:
                    st.info("Source video format cannot be previewed directly.")
            else:
                st.info("Awaiting video file.")

        cached_vid_eval = st.session_state.video_eval_cache
        has_vid_eval = (cached_vid_eval is not None and cached_vid_eval.get("source_name") == target_vid_name)

        with vid_p_c2:
            st.markdown("**AI ANNOTATED OUTPUT VIDEO (Dual YOLOv8 HUD & Routing)**")
            if has_vid_eval and cached_vid_eval is not None:
                mp4_bytes = cached_vid_eval.get("mp4_bytes", b"")
                if mp4_bytes:
                    st.video(mp4_bytes)
                    st.download_button(
                        label="📥 Download Annotated Video (.MP4)",
                        data=mp4_bytes,
                        file_name=f"annotated_{target_vid_name.rsplit('.', 1)[0]}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                else:
                    st.info("Video processed. Keyframes available in tabs below.")
            else:
                st.markdown("""
                <div style="background: rgba(30, 41, 59, 0.7); border: 2px dashed #475569; border-radius: 8px; padding: 40px 20px; text-align: center; color: #94A3B8;">
                    <h4>⏳ Awaiting Evaluation</h4>
                    <p style="margin: 0; font-size: 0.9rem;">Click the <strong>'🎬 Evaluate Video'</strong> button below to analyze frames sequentially.</p>
                </div>
                """, unsafe_allow_html=True)

        btn_v1, btn_v2 = st.columns([2, 1])
        with btn_v1:
            eval_vid_clicked = st.button("🎬 Evaluate Video", type="primary", use_container_width=True)
        with btn_v2:
            if st.button("🔄 Reset Video Cache", key="btn_reset_vid", use_container_width=True):
                st.session_state.video_eval_cache = None
                st.rerun()

        if eval_vid_clicked and probe_path:
            p_bar = st.progress(0.0)
            status_txt = st.empty()

            def vid_callback(curr, total, cur_res):
                frac = min(1.0, curr / max(1, total))
                p_bar.progress(frac)
                status_txt.text(f"Evaluating frame {curr}/{total} &bull; Hazard: {cur_res['hazard_score']:.3f} &bull; Exit: {cur_res['recommended_exit']}")

            with st.spinner("Evaluating video frames through Dual YOLOv8 (CPU) & Dynamic Dijkstra..."):
                sum_vid, mp4_out, err_v = process_video_evaluation(
                    system=system,
                    video_source=probe_path,
                    active_edge=active_edge,
                    source_room=source_room,
                    frame_skip=vid_frame_skip,
                    max_frames=max_vid_frames,
                    progress_callback=vid_callback
                )

            if err_v is not None:
                st.error(f"Video Evaluation Failed: {err_v}")
            else:
                st.session_state.video_eval_cache = {
                    "summary": sum_vid,
                    "mp4_bytes": mp4_out,
                    "source_name": target_vid_name
                }
                st.session_state.incident_events.append({
                    "time": now_ts,
                    "desc": f"Video Evaluation completed for {target_vid_name}: {sum_vid['processed_frames_count']} frames analyzed, Peak Hazard={sum_vid['peak_hazard_score']}, Exit={sum_vid['primary_recommended_exit']}."
                })
                st.success("✅ Video Evaluation Complete!")
                st.rerun()

        if temp_probe:
            try:
                Path(temp_probe.name).unlink(missing_ok=True)
            except Exception:
                pass

        if has_vid_eval and cached_vid_eval is not None:
            sum_d = cached_vid_eval["summary"]
            st.markdown("#### Video Evaluation Summary Analytics")
            sum_df = pd.DataFrame([{
                "Source Video": cached_vid_eval["source_name"],
                "Total Frames Analyzed": sum_d["processed_frames_count"],
                "Effective FPS": f"{sum_d['effective_fps']:.1f}",
                "Peak Hazard Score": f"{sum_d['peak_hazard_score']:.3f}",
                "Peak Occupant Count": sum_d["peak_people_count"],
                "Peak Crowd Density": f"{sum_d['peak_crowd_density']:.3f}",
                "Route Divergence Events": sum_d["route_changes_count"],
                "Primary Dynamic Exit": sum_d["primary_recommended_exit"]
            }])
            st.dataframe(sum_df, hide_index=True, use_container_width=True)

    # WORKFLOW 3: Benchmark Test Images
    elif selected_method == "Benchmark Test Images":
        st.markdown('<div class="section-header">BENCHMARK TEST IMAGE EVALUATION</div>', unsafe_allow_html=True)
        bench_opts = [
            "flames_sample.jpg (Active Fire)",
            "fire_sample.jpg (Smoke Plume)",
            "crowd_sample.jpg (High Density Crowd)",
            "compound_sample.jpg (Compound Fire + Crowd)",
            "normal_sample.jpg (Nominal Clear Baseline)"
        ]
        curr_b_idx = 0
        for b_i, b_opt in enumerate(bench_opts):
            if st.session_state.active_sample_file in b_opt:
                curr_b_idx = b_i
                break
        sel_bench = st.selectbox("Select Academic Benchmark File", options=bench_opts, index=curr_b_idx)
        st.session_state.active_sample_file = sel_bench.split()[0]
        sample_path = DATA_DIR / "test" / st.session_state.active_sample_file
        if sample_path.exists():
            raw_preview_frame = cv2.imread(str(sample_path))

        b_c1, b_c2 = st.columns(2)
        with b_c1:
            st.markdown(f"**RAW BENCHMARK PREVIEW** (`{st.session_state.active_sample_file}`)")
            if raw_preview_frame is not None:
                st.image(cv2.cvtColor(raw_preview_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with b_c2:
            st.markdown("**AI ANNOTATED OUTPUT**")
            cached_b = st.session_state.image_eval_cache
            has_b_eval = (cached_b is not None and cached_b.get("source_name") == st.session_state.active_sample_file)
            if has_b_eval and cached_b is not None:
                ann_b = cached_b["result"].get("annotated_frame")
                if ann_b is not None:
                    st.image(cv2.cvtColor(ann_b, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.info("Click 'Evaluate Benchmark Image' below to process.")

        if st.button("🔍 Evaluate Benchmark Image", type="primary", use_container_width=True) and raw_preview_frame is not None:
            res_b, raw_f, err_b = process_image_evaluation(system, raw_preview_frame, active_edge=active_edge, source_room=source_room)
            if err_b is None:
                st.session_state.image_eval_cache = {"result": res_b, "raw_frame": raw_f, "source_name": st.session_state.active_sample_file}
                st.rerun()

    # WORKFLOW 4: Browser Webcam (WebRTC / HTML5)
    elif selected_method == "Browser Webcam (WebRTC / Cloud)":
        is_video_mode = True
        st.markdown('<div class="section-header">BROWSER WEBCAM STREAM (CLOUD & LOCAL COMPATIBLE)</div>', unsafe_allow_html=True)
        if browser_cam_mode == "Live WebRTC Stream (Continuous)" and WEBRTC_AVAILABLE:
            webrtc_col1, webrtc_col2 = st.columns([1.3, 1.0])
            with webrtc_col1:
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]})
                webrtc_ctx = webrtc_streamer(
                    key="ai-fire-crowd-browser-webrtc",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_config,
                    video_processor_factory=BrowserWebcamProcessor,
                    media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                    async_processing=True,
                )
            with webrtc_col2:
                st.info("Click **START** in the WebRTC player. Live browser frames route directly into Dual YOLOv8 & Dynamic Dijkstra.")

            if webrtc_ctx.state.playing and webrtc_ctx.video_processor is not None:
                webrtc_playing_active = True
                b_frame, b_count = webrtc_ctx.video_processor.get_latest_frame()
                if b_frame is not None:
                    raw_preview_frame = b_frame
                    media_source_description = f"🟢 Browser WebRTC Live Stream &bull; Frame #{b_count} &bull; Res: {b_frame.shape[1]}x{b_frame.shape[0]}"
        else:
            cam_photo = st.camera_input("📸 Capture Frame from Browser Camera")
            if cam_photo is not None:
                b_arr = np.frombuffer(cam_photo.getvalue(), dtype=np.uint8)
                raw_preview_frame = cv2.imdecode(b_arr, cv2.IMREAD_COLOR)

    # WORKFLOW 5: CCTV Demo Video or Local Cameras (Live Stream Monitoring)
    else:
        if st.session_state.monitoring_active and st.session_state.video_stream is not None:
            is_video_mode = True
            success, v_frame = st.session_state.video_stream.get_frame(loop_video=True)
            if success and v_frame is not None:
                raw_preview_frame = v_frame
                v_tel = st.session_state.video_stream.get_telemetry()
                media_source_description = f"🟢 Live Monitoring Active &bull; {v_tel['source_type']} &bull; FPS: {v_tel['fps']} &bull; Frame: #{v_tel['frame_count']}"
            else:
                st.session_state.monitoring_active = False
                st.session_state.video_stream.release()
                st.session_state.video_stream = None
        else:
            s_p = DATA_DIR / "test" / st.session_state.active_sample_file
            if s_p.exists():
                raw_preview_frame = cv2.imread(str(s_p))
                media_source_description = f"Standby Mode ({selected_method}) &bull; Click '▶ START' in sidebar to activate."

    # -------------------------------------------------------------------------
    # RESOLVE ACTIVE RESULT FOR TELEMETRY & TABS
    # -------------------------------------------------------------------------
    result: Optional[Dict[str, Any]] = None
    input_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None

    if selected_method in ["Upload Image", "Benchmark Test Images"]:
        cached = st.session_state.image_eval_cache
        expected_src = target_img_name if selected_method == "Upload Image" else st.session_state.active_sample_file
        if cached is not None and cached.get("source_name") == expected_src:
            result = cached["result"]
            input_frame = cached.get("raw_frame")
            annotated_frame = result.get("annotated_frame", input_frame)
            media_source_description = f"Evaluated Image: `{expected_src}`"
        else:
            input_frame = raw_preview_frame if raw_preview_frame is not None else cv2.imread(str(DATA_DIR / "test" / "normal_sample.jpg"))
            result = get_standby_baseline_result(system, source_room=source_room, active_edge=active_edge, placeholder_frame=input_frame)
            annotated_frame = result["annotated_frame"]
            media_source_description = f"Standby &bull; Click 'Evaluate' above to trigger AI hazard perception."

    elif selected_method == "Upload Video":
        v_cache = st.session_state.video_eval_cache
        if v_cache is not None and v_cache.get("source_name") == target_vid_name:
            v_sum = v_cache["summary"]
            result = v_sum.get("latest_result")
            input_frame = v_sum.get("first_frame")
            annotated_frame = result.get("annotated_frame") if result else input_frame
            media_source_description = f"Evaluated Video: `{target_vid_name}`"
        else:
            input_frame = cv2.imread(str(DATA_DIR / "test" / "normal_sample.jpg"))
            result = get_standby_baseline_result(system, source_room=source_room, active_edge=active_edge, placeholder_frame=input_frame)
            annotated_frame = result["annotated_frame"]
            media_source_description = f"Standby &bull; Click 'Evaluate Video' above to analyze CCTV frames."

    elif selected_method == "Browser Webcam (WebRTC / Cloud)":
        if raw_preview_frame is not None:
            input_frame = raw_preview_frame
            result = system.process_frame(frame=input_frame, active_edge=active_edge, source_room=source_room, enable_frame_skip=False)
            annotated_frame = result["annotated_frame"]
        else:
            input_frame = cv2.imread(str(DATA_DIR / "test" / "normal_sample.jpg"))
            result = get_standby_baseline_result(system, source_room=source_room, active_edge=active_edge, placeholder_frame=input_frame)
            annotated_frame = result["annotated_frame"]

    else:
        # Live Stream Monitoring / Local Camera / Standby
        if raw_preview_frame is not None and st.session_state.monitoring_active:
            input_frame = raw_preview_frame
            result = system.process_frame(
                frame=input_frame,
                active_edge=active_edge,
                source_room=source_room,
                enable_frame_skip=(enable_frame_skip and is_video_mode),
                skip_interval=frame_skip_val
            )
            annotated_frame = result["annotated_frame"]
        else:
            input_frame = raw_preview_frame if raw_preview_frame is not None else cv2.imread(str(DATA_DIR / "test" / "normal_sample.jpg"))
            result = get_standby_baseline_result(system, source_room=source_room, active_edge=active_edge, placeholder_frame=input_frame)
            annotated_frame = result["annotated_frame"]

    if input_frame is None or result is None or annotated_frame is None:
        st.error("⚠️ Unable to resolve active surveillance frame.")
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

    latency_info = result.get("latency_metrics", {})
    zone_counts = result.get("multi_zone_counts", {})
    zone_densities = result.get("multi_zone_densities", {})

    # Retrieve graph edge metrics
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
    elif webrtc_playing_active and webrtc_auto_refresh:
        time.sleep(0.15)
        st.rerun()



if __name__ == "__main__":
    main()
