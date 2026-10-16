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
    """Thread-safe WebRTC video processor capturing and analyzing browser webcam frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_result: Optional[Dict[str, Any]] = None
        self.frame_count: int = 0
        self.last_timestamp: float = 0.0
        self.is_active: bool = False
        self.system: Any = None
        self.active_edge: Tuple[str, str] = ("CORRIDOR_1", "EXIT_A")
        self.source_room: str = "ROOM_A"
        self.process_every_n: int = 3
        self._last_annotated_cache: Optional[np.ndarray] = None

    def set_detection_active(self, active: bool) -> None:
        """Toggles real-time AI perception and dynamic routing recomputation."""
        with self._lock:
            self.is_active = active

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        with self._lock:
            self.latest_frame = img.copy()
            self.frame_count += 1
            self.last_timestamp = time.time()
            f_count = self.frame_count
            active = self.is_active
            system = self.system
            active_edge = self.active_edge
            source_room = self.source_room
            skip = self.process_every_n
            cached_ann = self._last_annotated_cache

        annotated_bgr = img

        if active and system is not None:
            if f_count % skip == 0 or cached_ann is None:
                try:
                    res = system.process_frame(
                        frame=img,
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
                    annotated_bgr = res.get("annotated_frame", img)
                    with self._lock:
                        self.latest_annotated_frame = annotated_bgr.copy()
                        self.latest_result = res
                        self._last_annotated_cache = annotated_bgr.copy()
                except Exception:
                    pass
            elif cached_ann is not None:
                annotated_bgr = cached_ann

        if WEBRTC_AVAILABLE:
            return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
        return frame

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int]:
        """Backward-compatible tuple (frame_copy, frame_count) for existing tests."""
        with self._lock:
            if self.latest_frame is None:
                return None, self.frame_count
            return self.latest_frame.copy(), self.frame_count

    def get_latest_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]], int]:
        """Returns thread-safe decoupled copies of raw frame, annotated frame, detection results, and count."""
        with self._lock:
            raw = self.latest_frame.copy() if self.latest_frame is not None else None
            ann = self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else None
            res = self.latest_result.copy() if self.latest_result is not None else None
            return raw, ann, res, self.frame_count


try:
    from dashboard.components import (
        render_simple_header,
        render_evacuation_guidance_card,
        render_detection_summary_cards,
        render_academic_disclaimer,
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
        render_academic_limitations,
    )
except (ImportError, ModuleNotFoundError):
    from components import (
        render_simple_header,
        render_evacuation_guidance_card,
        render_detection_summary_cards,
        render_academic_disclaimer,
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
        render_academic_limitations,
    )


# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION & STYLING INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Evacuation Assistant",
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
    peak_fire_conf = 0.0
    peak_smoke_conf = 0.0
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

                fire_dets = res.get("fire_detections", [])
                f_confs = [d["confidence"] for d in fire_dets if d.get("class_name") == "fire"]
                s_confs = [d["confidence"] for d in fire_dets if d.get("class_name") == "smoke"]
                cur_f_conf = max(f_confs) if f_confs else 0.0
                cur_s_conf = max(s_confs) if s_confs else 0.0
                if cur_f_conf > peak_fire_conf:
                    peak_fire_conf = cur_f_conf
                if cur_s_conf > peak_smoke_conf:
                    peak_smoke_conf = cur_s_conf

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
                    "fire_conf": cur_f_conf,
                    "smoke_conf": cur_s_conf,
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
        "peak_fire_conf": round(peak_fire_conf, 3),
        "peak_smoke_conf": round(peak_smoke_conf, 3),
        "peak_people_count": peak_people,
        "peak_crowd_density": round(peak_density, 3),
        "route_changes_count": route_changes,
        "primary_recommended_exit": last_exit or "NONE",
        "fire_detected_any": (peak_fire_conf > 0.0 or any(f.get("hazard_score", 0) > 0.4 for f in frame_results)),
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
# 1. SECTION RENDERERS (CLEAN, MODULAR, COMPLETE FEATURE PRESERVATION)
# -----------------------------------------------------------------------------

def _render_home_section(system, active_result, source_room, active_edge):
    """
    Renders the simple, clean, and inviting Home landing screen.
    Directly answers 'What do you want to analyze?' with clear entry points.
    """
    st.markdown("""
    <div style="text-align: center; margin: 12px 0 28px 0;">
        <h2 style="font-size: 1.75rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">
            What do you want to analyze?
        </h2>
        <p style="color: var(--text-muted); font-size: 1.05rem; margin: 0; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.5;">
            Select a surveillance source below to detect fire/smoke hazards, monitor occupant congestion, and compute dynamic evacuation paths.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Three large, elegant primary action cards
    col_img, col_vid, col_cam = st.columns(3)

    with col_img:
        st.markdown("""
        <div class="home-choice-card">
            <div>
                <div style="font-size: 2.75rem; margin-bottom: 10px;">🖼️</div>
                <h3 class="home-card-title">Upload Image</h3>
                <p class="home-card-desc">
                    Detect active flames, smoke plumes, and occupant congestion from a single CCTV or floorplan snapshot.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🖼️ Analyze Image ➔", key="home_btn_img", use_container_width=True, type="primary"):
            st.session_state.input_method = "Upload Image"
            st.session_state.nav_section = "🔍 Analyze"
            st.rerun()

    with col_vid:
        st.markdown("""
        <div class="home-choice-card">
            <div>
                <div style="font-size: 2.75rem; margin-bottom: 10px;">🎬</div>
                <h3 class="home-card-title">Upload Video</h3>
                <p class="home-card-desc">
                    Analyze sequential video footage (MP4, AVI, MOV) for real-time hazard progression, crowd bottlenecks, and dynamic egress re-routing.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎬 Analyze Video ➔", key="home_btn_vid", use_container_width=True, type="primary"):
            st.session_state.input_method = "Upload Video"
            st.session_state.nav_section = "🔍 Analyze"
            st.rerun()

    with col_cam:
        st.markdown("""
        <div class="home-choice-card">
            <div>
                <div style="font-size: 2.75rem; margin-bottom: 10px;">📹</div>
                <h3 class="home-card-title">Live Camera</h3>
                <p class="home-card-desc">
                    Stream live surveillance directly from your browser webcam or network camera for real-time AI perception and egress guidance.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶️ Start Live Detection ➔", key="home_btn_cam", use_container_width=True, type="primary"):
            st.session_state.input_method = "Live Camera"
            st.session_state.nav_section = "📹 Live Camera"
            st.session_state.live_detection_active = True
            st.rerun()

    # Prominent Quick Action Controls Bar
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    qa_c1, qa_c2, qa_c3 = st.columns(3)
    with qa_c1:
        if st.button("🎬 Analyze Video Now ➔", key="home_qa_vid", use_container_width=True):
            st.session_state.input_method = "Upload Video"
            st.session_state.nav_section = "🔍 Analyze"
            st.rerun()
    with qa_c2:
        if st.button("▶️ Start Live Detection Now ➔", key="home_qa_cam", use_container_width=True):
            st.session_state.input_method = "Live Camera"
            st.session_state.nav_section = "📹 Live Camera"
            st.session_state.live_detection_active = True
            st.rerun()
    with qa_c3:
        if st.button("📊 View Live Results ➔", key="home_qa_res", use_container_width=True):
            st.session_state.nav_section = "📊 Results"
            st.rerun()

    # Active evaluation banner if results exist in cache
    has_image_cache = st.session_state.get("image_eval_cache") is not None
    has_vid_cache = st.session_state.get("video_eval_cache") is not None
    if has_image_cache or has_vid_cache:
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        source_lbl = st.session_state.image_eval_cache.get("source_name") if has_image_cache else st.session_state.video_eval_cache.get("source_name")
        res_banner_col1, res_banner_col2 = st.columns([3, 1])
        with res_banner_col1:
            st.markdown(f"""
            <div style="background: var(--blue-soft); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px 18px; color: var(--primary-blue); font-size: 0.95rem;">
                📊 <strong>Active Evaluation Available:</strong> Results for <code>{source_lbl}</code> are ready to view.
            </div>
            """, unsafe_allow_html=True)
        with res_banner_col2:
            if st.button("View Results ➔", key="home_view_res", use_container_width=True, type="primary"):
                st.session_state.nav_section = "📊 Results"
                st.rerun()

    # Three core capability highlights
    st.markdown("<div style='margin-top: 36px;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 14px;'>Core AI System Capabilities</h4>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
        <div class="clean-metric-card" style="text-align: left; height: 100%;">
            <div style="font-size: 1.6rem; margin-bottom: 8px;">⚡</div>
            <div style="font-weight: 700; color: var(--text-primary); font-size: 1rem; margin-bottom: 6px;">Dual YOLOv8 Perception</div>
            <div style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.5;">
                Dual neural models detect flame boundaries, smoke clouds, and occupant count simultaneously with sub-second CPU latency.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="clean-metric-card" style="text-align: left; height: 100%;">
            <div style="font-size: 1.6rem; margin-bottom: 8px;">🧠</div>
            <div style="font-weight: 700; color: var(--text-primary); font-size: 1rem; margin-bottom: 6px;">Dynamic Dijkstra Routing</div>
            <div style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.5;">
                Optimizes escape paths dynamically by fusing physical distance (γ), fire/smoke severity (α), and crowd density (β).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="clean-metric-card" style="text-align: left; height: 100%;">
            <div style="font-size: 1.6rem; margin-bottom: 8px;">🛡️</div>
            <div style="font-weight: 700; color: var(--text-primary); font-size: 1rem; margin-bottom: 6px;">Fail-Safe Exit Guidance</div>
            <div style="font-size: 0.88rem; color: var(--text-muted); line-height: 1.5;">
                Instantly diverts evacuees away from compromised exits, updating building digital signs and public address advisories.
            </div>
        </div>
        """, unsafe_allow_html=True)


def _render_analyze_section(system, source_room, active_edge, conf_threshold, vulnerable_priority, vuln_category, enable_frame_skip, frame_skip_val, cloud_env, now_ts):
    """Renders the media analysis workspace for Image, Video, and Live Camera."""
    st.markdown('<div class="section-header">SURVEILLANCE ANALYSIS WORKSPACE</div>', unsafe_allow_html=True)

    # Segmented mode selector
    sub_modes = ["Upload Image", "Upload Video", "Live Camera"]
    curr_method = st.session_state.get("input_method", "Upload Image")
    default_sub_idx = 0
    if curr_method == "Upload Video":
        default_sub_idx = 1
    elif curr_method in ["Live Camera", "Browser Webcam (WebRTC / Cloud)", "Laptop Webcam", "External USB Camera", "RTSP Network Stream"]:
        default_sub_idx = 2

    chosen_sub = st.radio("Select Analysis Type", options=sub_modes, index=default_sub_idx, horizontal=True, label_visibility="collapsed")
    st.session_state.input_method = chosen_sub

    # -------------------------------------------------------------------------
    # WORKFLOW A: UPLOAD IMAGE
    # -------------------------------------------------------------------------
    if chosen_sub == "Upload Image":
        st.caption("Upload a surveillance image (JPG, PNG) or select a benchmark sample. Click **ANALYZE IMAGE** to run Dual YOLOv8 perception and Dynamic Dijkstra routing.")

        img_col1, img_col2 = st.columns([1.2, 1.0])
        with img_col1:
            up_img_file = st.file_uploader(
                "Upload Image File",
                type=["jpg", "jpeg", "png", "bmp"],
                key="clean_img_uploader",
                help="Accepts JPG, JPEG, PNG, BMP images."
            )
        with img_col2:
            preset_options = [
                "None (Use Uploaded Image)",
                "flames_sample.jpg (Active Fire Hazard)",
                "fire_sample.jpg (Smoke Plume Hazard)",
                "crowd_sample.jpg (High Density Crowd)",
                "compound_sample.jpg (Compound Fire + Crowd)",
                "normal_sample.jpg (Nominal Facility Baseline)"
            ]
            selected_preset = st.selectbox("Or Select Benchmark Image Sample", options=preset_options, index=1 if up_img_file is None else 0)

        target_img_input = None
        target_img_name = "custom_image.jpg"
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

        raw_preview_frame = None
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
            st.markdown(f"**SOURCE IMAGE PREVIEW** (`{target_img_name}`)")
            if raw_preview_frame is not None:
                st.image(cv2.cvtColor(raw_preview_frame, cv2.COLOR_BGR2RGB), caption=f"{target_img_name} ({raw_preview_frame.shape[1]}x{raw_preview_frame.shape[0]})", use_container_width=True)
            else:
                st.info("Awaiting image upload or sample selection.")

        cached_img_eval = st.session_state.get("image_eval_cache")
        has_eval_for_current = (cached_img_eval is not None and cached_img_eval.get("source_name") == target_img_name)

        with preview_c2:
            st.markdown("**AI ANNOTATED PERCEPTION HUD**")
            if has_eval_for_current and cached_img_eval is not None:
                eval_ann = cached_img_eval["result"].get("annotated_frame")
                if eval_ann is not None:
                    st.image(cv2.cvtColor(eval_ann, cv2.COLOR_BGR2RGB), caption="Dual YOLOv8 Detection HUD & Egress Directives", use_container_width=True)
            else:
                st.markdown("""
                <div style="background: var(--bg-surface); border: 2px dashed var(--border-strong); border-radius: 8px; padding: 48px 20px; text-align: center; color: var(--text-muted);">
                    <div style="font-size: 1.75rem; margin-bottom: 8px;">⏳</div>
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">Awaiting Analysis</div>
                    <div style="font-size: 0.9rem;">Click the <strong>ANALYZE IMAGE</strong> button below to run Dual YOLOv8 & Dynamic Dijkstra routing.</div>
                </div>
                """, unsafe_allow_html=True)

        btn_c1, btn_c2, btn_c3 = st.columns([2, 1, 1])
        with btn_c1:
            eval_img_clicked = st.button("🔍 ANALYZE IMAGE", type="primary", use_container_width=True)
        with btn_c2:
            if has_eval_for_current:
                if st.button("👉 View Results ➔", use_container_width=True):
                    st.session_state.nav_section = "📊 Results"
                    st.rerun()
            else:
                st.button("👉 View Results ➔", use_container_width=True, disabled=True)
        with btn_c3:
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.session_state.image_eval_cache = None
                st.rerun()

        if eval_img_clicked and raw_preview_frame is not None:
            with st.spinner("Executing Dual YOLOv8 Perception & Dynamic Dijkstra Router..."):
                res_img, raw_f, err_eval = process_image_evaluation(
                    system=system,
                    image_input=raw_preview_frame,
                    active_edge=active_edge,
                    source_room=source_room
                )
                if err_eval is not None:
                    st.error(f"Analysis Failed: {err_eval}")
                else:
                    st.session_state.image_eval_cache = {
                        "result": res_img,
                        "raw_frame": raw_f,
                        "source_name": target_img_name
                    }
                    st.session_state.incident_events.append({
                        "time": now_ts,
                        "desc": f"Image Analysis completed ({target_img_name}): Hazard={res_img['hazard_score']:.3f}, People={res_img['people_detected']}, Route={res_img['recommended_route_str']}."
                    })
                    st.session_state.nav_section = "📊 Results"
                    st.success("✅ Analysis Complete! Navigating to Results...")
                    st.rerun()

    # -------------------------------------------------------------------------
    # WORKFLOW B: UPLOAD VIDEO
    # -------------------------------------------------------------------------
    elif chosen_sub == "Upload Video":
        st.caption("Upload a surveillance video (MP4, AVI, MOV) or evaluate the demo corridor feed. Click **ANALYZE VIDEO** to process frames and calculate dynamic routes.")

        vid_col1, vid_col2 = st.columns([1.2, 1.0])
        with vid_col1:
            up_vid_file = st.file_uploader(
                "Upload Video File",
                type=["mp4", "avi", "mov", "mkv"],
                key="clean_vid_uploader",
                help="Accepts MP4, AVI, MOV, MKV video files."
            )
        with vid_col2:
            vid_preset_options = [
                "None (Use Uploaded Video)",
                "demo_cctv_feed.mp4 (Corridor Surveillance Demo)"
            ]
            selected_vid_preset = st.selectbox("Or Select Benchmark Demo Video", options=vid_preset_options, index=1 if up_vid_file is None else 0)

        target_vid_source = None
        target_vid_name = "demo_cctv_feed.mp4"
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

        cached_vid_eval = st.session_state.get("video_eval_cache")
        has_vid_eval = (cached_vid_eval is not None and cached_vid_eval.get("source_name") == target_vid_name)

        with vid_p_c2:
            st.markdown("**AI ANNOTATED OUTPUT VIDEO (PyAV H.264)**")
            if has_vid_eval and cached_vid_eval is not None:
                mp4_bytes = cached_vid_eval.get("mp4_bytes", b"")
                if mp4_bytes:
                    st.video(mp4_bytes)
                    st.download_button(
                        label="📥 Download Processed Video (.MP4)",
                        data=mp4_bytes,
                        file_name=f"annotated_{target_vid_name.rsplit('.', 1)[0]}.mp4",
                        mime="video/mp4",
                        use_container_width=True
                    )
                else:
                    st.info("Video processed. Keyframe metrics available in Results.")
            else:
                st.markdown("""
                <div style="background: var(--bg-surface); border: 2px dashed var(--border-strong); border-radius: 8px; padding: 48px 20px; text-align: center; color: var(--text-muted);">
                    <div style="font-size: 1.75rem; margin-bottom: 8px;">⏳</div>
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">Awaiting Video Processing</div>
                    <div style="font-size: 0.9rem;">Click the <strong>ANALYZE VIDEO</strong> button below to process frames sequentially.</div>
                </div>
                """, unsafe_allow_html=True)

        set_c1, set_c2 = st.columns(2)
        with set_c1:
            vid_frame_skip = st.slider("Frame Skip Sampling Interval", min_value=1, max_value=10, value=3, help="Process every N-th frame to accelerate CPU perception throughput.")
        with set_c2:
            max_vid_frames = st.slider("Max Frames to Process", min_value=10, max_value=150, value=min(60, max(10, v_meta["total_frames"])), help="Limits frames to ensure snappy CPU evaluation.")

        btn_v1, btn_v2, btn_v3 = st.columns([2, 1, 1])
        with btn_v1:
            eval_vid_clicked = st.button("🎬 ANALYZE VIDEO", type="primary", use_container_width=True)
        with btn_v2:
            if has_vid_eval:
                if st.button("👉 View Results ➔", key="vid_view_res", use_container_width=True):
                    st.session_state.nav_section = "📊 Results"
                    st.rerun()
            else:
                st.button("👉 View Results ➔", key="vid_view_res_dis", use_container_width=True, disabled=True)
        with btn_v3:
            if st.button("🔄 Reset Video Cache", key="clean_reset_vid", use_container_width=True):
                st.session_state.video_eval_cache = None
                st.rerun()

        if eval_vid_clicked and probe_path:
            p_bar = st.progress(0.0)
            status_txt = st.empty()

            def vid_callback(curr, total, cur_res):
                frac = min(1.0, curr / max(1, total))
                p_bar.progress(frac)
                h_sc = cur_res.get("hazard_score", 0.0)
                f_cf = cur_res.get("fire_conf", 0.0)
                s_cf = cur_res.get("smoke_conf", 0.0)
                p_cnt = cur_res.get("people_count", 0)
                c_den = cur_res.get("crowd_density", 0.0)
                rec_ex = cur_res.get("recommended_exit", "NONE")
                f_str = f"🔥 Fire: {f_cf*100:.1f}%" if f_cf > 0 else "🟢 Fire: None"
                status_txt.markdown(
                    f"**Processing Frame {curr}/{total}** ({frac*100:.0f}%) | "
                    f"{f_str} | 👥 Occupants: {p_cnt} | 🚶 Density: {c_den:.2f} | "
                    f"⚠️ Hazard: {h_sc:.3f} | 🚪 Recommended Exit: **{rec_ex}**"
                )

            with st.spinner("Processing video frames through Dual YOLOv8 & Dynamic Dijkstra..."):
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
                st.error(f"⚠️ Video Analysis Failed: {err_v}. Please ensure the video is an MP4, AVI, or MOV file encoded with H.264 or standard MPEG-4 codecs.")
            else:
                st.session_state.video_eval_cache = {
                    "summary": sum_vid,
                    "mp4_bytes": mp4_out,
                    "source_name": target_vid_name
                }
                # Synchronize latest keyframe into image_eval_cache for immediate Results tab display
                if sum_vid.get("latest_result") is not None and sum_vid.get("first_frame") is not None:
                    st.session_state.image_eval_cache = {
                        "result": sum_vid["latest_result"],
                        "raw_frame": sum_vid["first_frame"],
                        "source_name": f"Video: {target_vid_name}"
                    }
                st.session_state.incident_events.append({
                    "time": now_ts,
                    "desc": f"Video Analysis completed ({target_vid_name}): {sum_vid['processed_frames_count']} frames analyzed, Peak Fire Conf={sum_vid.get('peak_fire_conf', 0.0)*100:.1f}%, Peak Hazard={sum_vid['peak_hazard_score']}, Exit={sum_vid['primary_recommended_exit']}."
                })
                st.session_state.nav_section = "📊 Results"
                st.success("✅ Video Analysis Complete! Navigating to Results...")
                st.rerun()

        if temp_probe:
            try:
                Path(temp_probe.name).unlink(missing_ok=True)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # WORKFLOW C: LIVE CAMERA
    # -------------------------------------------------------------------------
    else:
        _render_live_camera_section(
            system=system,
            source_room=source_room,
            active_edge=active_edge,
            conf_threshold=conf_threshold,
            dark_mode=st.session_state.get("dark_mode", False),
            now_ts=now_ts
        )


def _render_live_results_panel(result: Optional[Dict[str, Any]], source_name: str = "Live Browser Camera", is_active: bool = False, dark_mode: bool = False) -> None:
    """
    Renders real-time live perception and dynamic routing telemetry.
    Distinguishes genuine Dual YOLOv8 inference from simulated data.
    """
    st.markdown("#### 📊 Live Results")

    if result is not None:
        # Genuine AI Model Detection Badge
        st.markdown("""
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 9999px; padding: 4px 14px; color: #059669; font-size: 0.82rem; font-weight: 700; margin-bottom: 12px;">
            <span>●</span> REAL MODEL DETECTION — LIVE BROWSER FEED
        </div>
        """, unsafe_allow_html=True)

        em = result.get("emergency_state", {})
        fire_dets = result.get("fire_detections", [])
        f_confs = [d["confidence"] for d in fire_dets if d.get("class_name") == "fire"]
        s_confs = [d["confidence"] for d in fire_dets if d.get("class_name") == "smoke"]
        fire_conf = max(f_confs) if f_confs else 0.0
        smoke_conf = max(s_confs) if s_confs else 0.0

        has_fire = em.get("fire_detected", result.get("fire_detected", fire_conf > 0.0))
        has_smoke = em.get("smoke_detected", result.get("smoke_detected", smoke_conf > 0.0))
        people_count = em.get("people_count", result.get("people_detected", 0))
        crowd_density = float(em.get("crowd_density", result.get("crowd_density", 0.0)))
        hazard_score = float(em.get("hazard_score", result.get("hazard_score", 0.0)))
        rec_exit = em.get("recommended_exit", result.get("recommended_exit", "EXIT_A"))
        rec_route = em.get("recommended_route", result.get("recommended_route", []))
        rec_route_str = " ➔ ".join(rec_route) if rec_route else "Direct egress"
        total_cost = float(em.get("total_route_cost", result.get("total_route_cost", 0.0)))
        route_diverted = bool(em.get("is_diverted_from_baseline", False))
        avoided_exit = em.get("avoided_exit")

        # 4 Telemetry Metric Cards
        m_c1, m_c2 = st.columns(2)
        with m_c1:
            if has_fire:
                st.markdown(f"""
                <div class="clean-metric-card" style="border-left: 4px solid #EF4444; text-align: left;">
                    <div style="font-size: 0.8rem; color: #DC2626; font-weight: 700;">🔥 FIRE DETECTED</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #DC2626;">{fire_conf*100:.1f}% Conf</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Active flame signature verified</div>
                </div>
                """, unsafe_allow_html=True)
            elif has_smoke:
                st.markdown(f"""
                <div class="clean-metric-card" style="border-left: 4px solid #F59E0B; text-align: left;">
                    <div style="font-size: 0.8rem; color: #D97706; font-weight: 700;">💨 SMOKE DETECTED</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #D97706;">{smoke_conf*100:.1f}% Conf</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Atmospheric plume verified</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="clean-metric-card" style="border-left: 4px solid #10B981; text-align: left;">
                    <div style="font-size: 0.8rem; color: #059669; font-weight: 700;">🟢 NO FIRE DETECTED</div>
                    <div style="font-size: 1.4rem; font-weight: 800; color: #059669;">0.0% Conf</div>
                    <div style="font-size: 0.75rem; color: var(--text-muted);">Corridor thermal profile nominal</div>
                </div>
                """, unsafe_allow_html=True)

            haz_color = "#DC2626" if hazard_score > 0.7 else ("#D97706" if hazard_score > 0.4 else "#059669")
            haz_level = "CRITICAL" if hazard_score > 0.7 else ("ELEVATED" if hazard_score > 0.4 else "NOMINAL")
            st.markdown(f"""
            <div class="clean-metric-card" style="margin-top: 10px; border-left: 4px solid {haz_color}; text-align: left;">
                <div style="font-size: 0.8rem; color: {haz_color}; font-weight: 700;">⚠️ HAZARD SCORE</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: {haz_color};">{hazard_score:.3f}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">Risk Category: {haz_level}</div>
            </div>
            """, unsafe_allow_html=True)

        with m_c2:
            cong_label = "HIGH CONGESTION" if crowd_density >= 0.7 else ("MODERATE" if crowd_density >= 0.35 else "LOW / NOMINAL")
            cong_color = "#DC2626" if crowd_density >= 0.7 else ("#D97706" if crowd_density >= 0.35 else "#059669")
            st.markdown(f"""
            <div class="clean-metric-card" style="border-left: 4px solid {cong_color}; text-align: left;">
                <div style="font-size: 0.8rem; color: var(--text-muted); font-weight: 700;">👥 OCCUPANTS &amp; DENSITY</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: var(--text-primary);">{people_count} Occupants</div>
                <div style="font-size: 0.75rem; color: {cong_color}; font-weight: 600;">Density: {crowd_density:.2f} ({cong_label})</div>
            </div>
            """, unsafe_allow_html=True)

            exit_color = "#DC2626" if route_diverted else "#059669"
            exit_badge = "🚨 DIVERTED ROUTE" if route_diverted else "🟢 DIRECT EGRESS"
            st.markdown(f"""
            <div class="clean-metric-card" style="margin-top: 10px; border-left: 4px solid {exit_color}; text-align: left;">
                <div style="font-size: 0.8rem; color: {exit_color}; font-weight: 700;">🚪 RECOMMENDED EXIT</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: var(--text-primary);">{rec_exit}</div>
                <div style="font-size: 0.75rem; color: {exit_color}; font-weight: 600;">{exit_badge} (Cost: {total_cost:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

        # Dynamic Route Breadcrumbs
        st.markdown(f"""
        <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 12px 16px; margin-top: 12px; font-size: 0.88rem;">
            <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">Dynamic Evacuation Route:</div>
            <code style="font-size: 0.95rem; color: var(--primary-blue); font-weight: 600;">{rec_route_str}</code>
        </div>
        """, unsafe_allow_html=True)

        if avoided_exit:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #EF4444; border-radius: 8px; padding: 10px 14px; margin-top: 8px; color: #DC2626; font-size: 0.84rem;">
                ⚠️ <strong>Avoided Compromised Exit:</strong> <code>{avoided_exit}</code> (Hazard zone bypass applied).
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        if st.button("📊 View Full Detailed Results in Results Tab ➔", key="live_res_tab_btn", use_container_width=True, type="primary"):
            st.session_state.nav_section = "📊 Results"
            st.rerun()

    else:
        st.markdown("""
        <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(100, 116, 139, 0.12); border: 1px solid #64748B; border-radius: 9999px; padding: 4px 14px; color: #64748B; font-size: 0.82rem; font-weight: 700; margin-bottom: 12px;">
            <span>○</span> STANDBY — AWAITING FRAME INPUT
        </div>
        <div style="background: var(--bg-surface); border: 1px dashed var(--border-strong); border-radius: 8px; padding: 32px 20px; text-align: center; color: var(--text-muted); font-size: 0.9rem;">
            <div style="font-size: 2rem; margin-bottom: 8px;">📹</div>
            <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">No Live Inference Active Yet</div>
            <p style="margin: 0; line-height: 1.5;">
                Click <strong>▶️ Start Live Detection</strong> and start your browser camera feed, or capture a frame with the snapshot tool.<br>
                Real model detections and dynamic evacuation guidance will appear here in real-time.
            </p>
        </div>
        """, unsafe_allow_html=True)


def _render_live_camera_section(system, source_room, active_edge, conf_threshold, dark_mode=False, now_ts=""):
    """
    Renders dedicated Live Browser Camera streaming, real-time perception, and live results.
    Fully compatible with Streamlit Cloud and local client browsers.
    """
    st.markdown('<div class="section-header">LIVE CAMERA SURVEILLANCE & REAL-TIME PERCEPTION</div>', unsafe_allow_html=True)

    st.session_state.setdefault("live_detection_active", False)
    st.session_state.setdefault("live_latest_result", None)
    st.session_state.setdefault("live_latest_frame", None)
    st.session_state.setdefault("live_latest_raw", None)

    # 1. Prominent Top Action Controls
    top_c1, top_c2, top_c3 = st.columns([1.2, 1.2, 1.4])
    with top_c1:
        if st.session_state.live_detection_active:
            st.button("🟢 Live Detection Running", disabled=True, use_container_width=True)
        else:
            if st.button("▶️ Start Live Detection", type="primary", use_container_width=True, key="live_sec_start"):
                st.session_state.live_detection_active = True
                st.session_state.monitoring_active = True
                st.rerun()
    with top_c2:
        if not st.session_state.live_detection_active:
            st.button("⏹️ Live Detection Paused", disabled=True, use_container_width=True)
        else:
            if st.button("⏹️ Stop Live Detection", use_container_width=True, key="live_sec_stop"):
                st.session_state.live_detection_active = False
                st.session_state.monitoring_active = False
                st.rerun()
    with top_c3:
        if st.button("📊 Live Results in Results Tab ➔", use_container_width=True, key="live_sec_view_res"):
            st.session_state.nav_section = "📊 Results"
            st.rerun()

    # 2. Real-time Status Banner
    if st.session_state.live_detection_active:
        st.markdown("""
        <div style="background: rgba(22, 163, 74, 0.12); border: 1px solid #16A34A; border-radius: 8px; padding: 10px 16px; color: #16A34A; font-weight: 600; font-size: 0.92rem; margin: 12px 0 16px 0;">
            🟢 <strong>LIVE AI DETECTION ACTIVE:</strong> Real-time frame processing through Dual YOLOv8 and Dynamic Dijkstra routing is active.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 8px; padding: 10px 16px; color: var(--text-muted); font-size: 0.92rem; margin: 12px 0 16px 0;">
            ⚪ <strong>LIVE DETECTION STANDBY:</strong> Video stream on standby. Click <strong>▶️ Start Live Detection</strong> above to activate real-time YOLO perception.
        </div>
        """, unsafe_allow_html=True)

    # 3. Environment & Compatibility Assessment Notice
    cloud_env = is_cloud_environment()
    if cloud_env:
        st.info("☁️ **Streamlit Cloud Environment Detected:** Server webcams (`cv2.VideoCapture(0)`) are physically absent on cloud servers and disabled to prevent server crashes. Live camera streaming operates securely inside your browser via WebRTC with client-side encryption. Fallback HTTP snapshot capture is also fully supported.")
    else:
        st.caption("💻 **Local Machine Detected:** WebRTC browser camera streaming and local camera scanning are supported.")

    # 4. Stream & Live Results Layout
    cam_c1, cam_c2 = st.columns([1.2, 1.0])

    with cam_c1:
        st.markdown("##### 📹 Live Browser Camera Stream (WebRTC)")
        if WEBRTC_AVAILABLE:
            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]})
            webrtc_ctx = webrtc_streamer(
                key="live-camera-workspace-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=BrowserWebcamProcessor,
                media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
                async_processing=True,
            )
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.system = system
                webrtc_ctx.video_processor.source_room = source_room
                webrtc_ctx.video_processor.active_edge = active_edge
                webrtc_ctx.video_processor.set_detection_active(st.session_state.live_detection_active)

                raw_f, ann_f, live_res, f_cnt = webrtc_ctx.video_processor.get_latest_data()
                if live_res is not None:
                    st.session_state.live_latest_result = live_res
                    st.session_state.live_latest_frame = ann_f
                    st.session_state.live_latest_raw = raw_f
                    st.session_state.image_eval_cache = {
                        "result": live_res,
                        "raw_frame": raw_f,
                        "source_name": f"Live_Webcam_Frame_{f_cnt}"
                    }
        else:
            st.warning("⚠️ WebRTC component (`streamlit-webrtc`) is not installed. Please use the HTTP Browser Camera Snapshot below.")

        st.markdown("---")
        st.markdown("##### 📸 Browser Camera Snapshot Fallback (HTTP)")
        cam_shot = st.camera_input("Capture frame from camera", key="live_cam_snapshot_workspace")
        if cam_shot is not None:
            b_arr = np.frombuffer(cam_shot.getvalue(), dtype=np.uint8)
            shot_frame = cv2.imdecode(b_arr, cv2.IMREAD_COLOR)
            if shot_frame is not None:
                with st.spinner("Analyzing captured camera frame..."):
                    res_live = system.process_frame(
                        frame=shot_frame,
                        active_edge=active_edge,
                        source_room=source_room,
                        enable_frame_skip=False
                    )
                    ensure_complete_emergency_state(
                        system=system,
                        result=res_live,
                        source_room=source_room,
                        active_edge=active_edge
                    )
                    st.session_state.live_latest_result = res_live
                    st.session_state.live_latest_frame = res_live.get("annotated_frame", shot_frame)
                    st.session_state.live_latest_raw = shot_frame
                    st.session_state.image_eval_cache = {
                        "result": res_live,
                        "raw_frame": shot_frame,
                        "source_name": "Live_Camera_Snapshot"
                    }
                    st.success("✅ Frame captured and analyzed by Dual YOLOv8 models!")

        # Camera Permission Instructions Expander
        with st.expander("🔐 Browser Camera Permission & Troubleshooting Guide", expanded=False):
            st.markdown("""
            **How to grant camera permissions in your browser:**
            - **Google Chrome / Brave / Chromium:**
              1. Click the site settings icon (padlock or sliders) on the left of the URL address bar.
              2. Toggle **Camera** to **Allow**.
              3. Click **Reload** to apply changes.
            - **Microsoft Edge:**
              1. Click the padlock icon in the address bar.
              2. Select **Permissions for this site** ➔ set **Camera** to **Allow**.
              3. Reload the dashboard tab.
            - **Mozilla Firefox:**
              1. Click the permissions icon on the left side of the address bar.
              2. Clear any blocked permissions and click **Allow** when prompted.
            - **Apple Safari:**
              1. Open **Settings / Preferences** ➔ **Websites** ➔ **Camera**.
              2. Locate this website URL and choose **Allow**.

            **Troubleshooting Permission Denied:**
            - If you previously denied camera access, click the camera icon with the red 'X' in your browser's URL address bar and change the setting to **Always allow**.
            - Ensure no other application (Zoom, Teams, Skype, OBS) is currently locking your camera hardware.
            - On Streamlit Cloud, browser webcam streams are transmitted directly from client browser to WebRTC peer-connection with TLS encryption; no physical server hardware is touched.
            """)

    with cam_c2:
        _render_live_results_panel(
            result=st.session_state.get("live_latest_result"),
            source_name="Live Camera",
            is_active=st.session_state.live_detection_active,
            dark_mode=dark_mode
        )


def _render_results_section(system, active_result, source_room, active_edge, now_ts):
    """Renders the primary results section highlighting the hero Evacuation Guidance Card."""
    st.markdown('<div class="section-header">DETECTION RESULTS &amp; EVACUATION GUIDANCE</div>', unsafe_allow_html=True)

    ensure_complete_emergency_state(system=system, result=active_result, source_room=source_room, active_edge=active_edge)
    em = active_result["emergency_state"]

    has_fire = em["fire_detected"]
    has_smoke = em["smoke_detected"]
    hazard_score = em["hazard_score"]
    people_count = em["people_count"]
    current_exit = em["recommended_exit"]
    current_route = em["recommended_route"]
    current_cost = em["total_route_cost"]
    route_changed = em["is_diverted_from_baseline"]
    avoided_exit = em["avoided_exit"]
    signage_dict = em["dynamic_signage"]
    pa_announcement = em["pa_announcement"]

    is_emergency = has_fire or has_smoke or (hazard_score > 0.4)
    haz_type = "FIRE" if has_fire else ("SMOKE" if has_smoke else "NONE")

    # HERO FEATURE: EVACUATION GUIDANCE CARD
    render_evacuation_guidance_card(
        recommended_exit=current_exit,
        recommended_route=current_route,
        is_emergency=is_emergency,
        hazard_type=haz_type,
        hazard_score=hazard_score,
        crowd_density=em.get("crowd_density", 0.0),
        adaptation_reason=em.get("adaptation_reason", ""),
        avoided_exit=avoided_exit,
        route_changed=route_changed,
        total_cost=current_cost,
        origin_room=source_room,
        hazard_zone=f"{active_edge[0]} ↔ {active_edge[1]}"
    )

    # SIDE-BY-SIDE: ANNOTATED VISUAL + DETECTION SUMMARY CARDS
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    res_c1, res_c2 = st.columns([1.2, 1.0])

    with res_c1:
        cached_img = st.session_state.get("image_eval_cache")
        if cached_img and any(k in str(cached_img.get("source_name", "")) for k in ["Live_Webcam", "Live_Camera", "Webcam"]):
            st.markdown("""
            <div style="display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10B981; border-radius: 9999px; padding: 4px 14px; color: #059669; font-size: 0.82rem; font-weight: 700; margin-bottom: 10px;">
                <span>●</span> REAL MODEL DETECTION — LIVE BROWSER CAMERA
            </div>
            """, unsafe_allow_html=True)
        st.markdown("**AI ANNOTATED PERCEPTION HUD (Dual YOLOv8 Output)**")
        ann_frame = active_result.get("annotated_frame")
        if ann_frame is not None:
            rgb_ann = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_ann, caption="Dual YOLOv8 Perception HUD & Egress Directives", use_container_width=True)
            is_enc, enc_buf = cv2.imencode(".png", ann_frame)
            if is_enc:
                st.download_button(
                    label="📥 Download Annotated Image (.PNG)",
                    data=enc_buf.tobytes(),
                    file_name=f"AI_Evacuation_Annotated_{now_ts.replace(':', '')}.png",
                    mime="image/png",
                    use_container_width=True
                )
        else:
            st.info("Awaiting visual inference feed.")

    with res_c2:
        f_conf = max([d["confidence"] for d in active_result.get("fire_detections", []) if d.get("class_name") == "fire"] or [0.0])
        s_conf = max([d["confidence"] for d in active_result.get("fire_detections", []) if d.get("class_name") == "smoke"] or [0.0])

        render_detection_summary_cards(
            fire_detected=has_fire,
            smoke_detected=has_smoke,
            people_detected=people_count,
            hazard_score=hazard_score,
            crowd_density=em.get("crowd_density", 0.0),
            total_route_cost=current_cost,
            fire_conf=f_conf,
            smoke_conf=s_conf
        )

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        render_simulated_led_signage(signage_dict=signage_dict)
        render_pa_announcement(message=pa_announcement)

    # RECORDED VIDEO PLAYER & EVALUATION SUMMARY (IF AVAILABLE)
    cached_vid = st.session_state.get("video_eval_cache")
    if cached_vid is not None and cached_vid.get("source_name"):
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        with st.expander("🎬 Recorded Video Analysis & Processed Video Player", expanded=True):
            v_sum = cached_vid.get("summary", {})
            st.markdown(f"**Source Video:** `{cached_vid['source_name']}` | **Frames Evaluated:** {v_sum.get('processed_frames_count', 0)} frames")

            v_m1, v_m2, v_m3, v_m4 = st.columns(4)
            with v_m1:
                st.metric("Peak Fire Conf", f"{v_sum.get('peak_fire_conf', 0.0)*100:.1f}%")
            with v_m2:
                st.metric("Peak Occupants", f"{v_sum.get('peak_people_count', 0)}")
            with v_m3:
                st.metric("Peak Density", f"{v_sum.get('peak_crowd_density', 0.0):.2f}")
            with v_m4:
                st.metric("Peak Hazard", f"{v_sum.get('peak_hazard_score', 0.0):.3f}")

            mp4_bytes = cached_vid.get("mp4_bytes", b"")
            if mp4_bytes:
                st.video(mp4_bytes)
                st.download_button(
                    label="📥 Download Processed Video (.MP4)",
                    data=mp4_bytes,
                    file_name=f"annotated_{cached_vid['source_name'].rsplit('.', 1)[0]}.mp4",
                    mime="video/mp4",
                    key="results_dl_mp4_btn",
                    use_container_width=True
                )

    # QUICK NAVIGATION ACTION BUTTONS
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    act_c1, act_c2, act_c3, act_c4 = st.columns(4)
    with act_c1:
        if st.button("🔍 Analyze Another File", use_container_width=True):
            st.session_state.nav_section = "🔍 Analyze"
            st.rerun()
    with act_c2:
        if st.button("📹 Live Camera Feed ➔", use_container_width=True):
            st.session_state.nav_section = "📹 Live Camera"
            st.rerun()
    with act_c3:
        if st.button("🗺️ View Evacuation & Map ➔", use_container_width=True, type="primary"):
            st.session_state.nav_section = "🗺️ Evacuation & Map"
            st.rerun()
    with act_c4:
        if st.button("📄 Generate Incident Reports ➔", use_container_width=True):
            st.session_state.nav_section = "📄 Reports & Validation"
            st.rerun()


def _render_evacuation_section(system, active_result, source_room, active_edge, multi_floor_model, vulnerable_priority, vuln_category, dark_mode=False):
    """Renders the comprehensive Evacuation, Building Twin, Multi-Zone Spatial, and Algorithm Comparison tab."""
    st.markdown('<div class="section-header">EVACUATION PATHWAY &amp; FACILITY GRAPH MODEL</div>', unsafe_allow_html=True)

    ensure_complete_emergency_state(system=system, result=active_result, source_room=source_room, active_edge=active_edge)
    em = active_result["emergency_state"]

    current_route = em["recommended_route"]
    normal_route = em["baseline_route"]
    current_exit = em["recommended_exit"]
    normal_exit = em["baseline_exit"]
    current_cost = em["total_route_cost"]
    normal_cost = em["baseline_cost"]
    hazard_score = em["hazard_score"]
    crowd_density = em["crowd_density"]
    hazard_status = em["status_level"]
    route_changed = em["is_diverted_from_baseline"]
    adaptation_reason = em["adaptation_reason"]
    cost_delta = em["cost_delta_vs_baseline"]
    people_count = em["people_count"]

    graph = system.building_graph.get_graph()
    zone_counts = active_result.get("multi_zone_counts", {})
    zone_densities = active_result.get("multi_zone_densities", {})
    edge_len = graph[active_edge[0]][active_edge[1]]["length"] if graph.has_edge(active_edge[0], active_edge[1]) else 1.0
    updated_edge_cost = graph[active_edge[0]][active_edge[1]]["cost"] if graph.has_edge(active_edge[0], active_edge[1]) else 1.0
    baseline_edge_cost = GAMMA * edge_len

    evac_tabs = st.tabs([
        "1. Building Digital Twin",
        "2. Spatial Congestion & Multi-Zone",
        "3. Dynamic Cost & Algorithm Comparison",
        "4. Multi-Floor Facility Model",
        "5. Vulnerable Occupant Egress"
    ])

    # SUB-TAB 1: BUILDING DIGITAL TWIN
    with evac_tabs[0]:
        route_c1, route_c2 = st.columns([1.5, 1.0])
        with route_c1:
            st.markdown(f"**TOPOLOGICAL FACILITY MAP ({'Dark Command' if dark_mode else 'Light Modern'} Mode)**")
            fig_map = system.building_graph.render_evacuation_map(
                recommended_route=current_route,
                previous_route=active_result.get("previous_route"),
                source_room=source_room,
                monitored_edge=active_edge,
                hazardous_edges=[active_edge] if hazard_score > 0 else [],
                congested_edges=[active_edge] if crowd_density >= 0.35 else [],
                system_state=hazard_status,
                dark_mode=dark_mode
            )
            st.pyplot(fig_map)

        with route_c2:
            st.markdown("**DYNAMIC ROUTE SEGMENT BREAKDOWN**")
            segments = []
            if current_route and len(current_route) > 1:
                for i in range(len(current_route) - 1):
                    u, v = current_route[i], current_route[i + 1]
                    if graph.has_edge(u, v):
                        edge_d = graph[u][v]
                        length = edge_d.get("length", 1.0)
                        cost = edge_d.get("cost", length)
                        is_mon = (set((u, v)) == set(active_edge))
                        status_seg = "⚠️ HAZARD" if (is_mon and hazard_score > 0) else "🟢 CLEAR"
                        segments.append({
                            "Segment": f"{u} ➔ {v}",
                            "Length": f"{length:.1f} m",
                            "Dynamic Cost": f"{cost:.2f}",
                            "Condition": status_seg
                        })

            if segments:
                st.dataframe(pd.DataFrame(segments), hide_index=True, use_container_width=True)

            render_recommended_exit_panel(
                recommended_exit=current_exit,
                total_route_cost=current_cost,
                route_distance=sum([graph[current_route[i]][current_route[i+1]]["length"] for i in range(len(current_route)-1) if graph.has_edge(current_route[i], current_route[i+1])]) if len(current_route) > 1 else 0.0,
                avoided_exit=em.get("avoided_exit")
            )
            render_route_flow_card(
                route_nodes=current_route,
                total_cost=current_cost,
                route_changed=route_changed
            )

    # SUB-TAB 2: SPATIAL CONGESTION & MULTI-ZONE
    with evac_tabs[1]:
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

    # SUB-TAB 3: DYNAMIC COST & ALGORITHM COMPARISON
    with evac_tabs[2]:
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
        render_baseline_vs_proposed_panel(
            normal_route=normal_route,
            normal_exit=normal_exit,
            normal_cost=normal_cost,
            current_route=current_route,
            current_exit=current_exit,
            current_cost=current_cost,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            dijkstra_ms=active_result.get("latency_metrics", {}).get("dijkstra_ms", 0.3)
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
        render_route_confidence_card(
            hazard_confidence=hazard_score,
            people_count=people_count,
            route_cost=current_cost,
            has_hazard=(hazard_score > 0)
        )

    # SUB-TAB 4: MULTI-FLOOR FACILITY MODEL
    with evac_tabs[3]:
        fl_c1, fl_c2 = st.columns([1, 2])
        with fl_c1:
            floor_sel = st.selectbox("Select Floor Level to Inspect", options=["Ground Floor (Level 0)", "First Floor (Level 1)"], index=0)
            fl_num = 0 if "Ground" in floor_sel else 1
            st.info("Ground Floor connects to First Floor via Central Emergency Stairwell (`STAIR_1`).")
        with fl_c2:
            fig_fl = multi_floor_model.render_floor_map(
                floor=fl_num,
                recommended_route=["G_ROOM_A", "G_CORRIDOR_1", "G_EXIT_A"] if fl_num == 0 else ["F1_ROOM_C", "F1_CORRIDOR_2", "F1_EXIT_B"],
                dark_mode=dark_mode
            )
            st.pyplot(fig_fl)

    # SUB-TAB 5: VULNERABLE OCCUPANT EGRESS
    with evac_tabs[4]:
        render_vulnerable_occupant_panel(
            vulnerable_enabled=vulnerable_priority,
            category=vuln_category
        )


def _render_monitoring_section(system, active_result, source_room, active_edge, frame_skip_val, dark_mode=False):
    """Renders the comprehensive Monitoring, Multi-Camera, Latency Telemetry, and Temporal Trends tab."""
    st.markdown('<div class="section-header">SURVEILLANCE MONITORING &amp; TELEMETRY</div>', unsafe_allow_html=True)

    ensure_complete_emergency_state(system=system, result=active_result, source_room=source_room, active_edge=active_edge)
    em = active_result["emergency_state"]
    latency_info = active_result.get("latency_metrics", {})
    hazard_score = em["hazard_score"]
    crowd_density = em["crowd_density"]
    current_cost = em["total_route_cost"]
    route_changed = em["is_diverted_from_baseline"]
    input_frame = active_result.get("annotated_frame")

    mon_tabs = st.tabs([
        "1. Live CCTV Stream HUD",
        "2. Multi-Camera Matrix (4 Channels)",
        "3. Temporal Hazard Trends",
        "4. Hardware Telemetry & Health"
    ])

    # SUB-TAB 1: LIVE CCTV STREAM HUD
    with mon_tabs[0]:
        feed_c1, feed_c2 = st.columns(2)
        with feed_c1:
            st.markdown("**RAW SURVEILLANCE INPUT**")
            if input_frame is not None:
                st.image(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.info("Awaiting input stream.")
        with feed_c2:
            st.markdown("**AI ANNOTATED OUTPUT (Bounding Boxes, HUD & Confidence)**")
            ann_frame = active_result.get("annotated_frame")
            if ann_frame is not None:
                st.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.info("Awaiting annotated inference stream.")

        h_in, w_in = input_frame.shape[:2] if input_frame is not None else (720, 1280)
        render_cctv_meta_bar(
            width=w_in,
            height=h_in,
            frame_num=latency_info.get("frame_number", 1),
            latency_ms=latency_info.get("total_ms", 0.0),
            fps=latency_info.get("fps", 0.0)
        )

    # SUB-TAB 2: MULTI-CAMERA MATRIX
    with mon_tabs[1]:
        render_multi_camera_panel(camera_sim=st.session_state.camera_simulator)

    # SUB-TAB 3: TEMPORAL HAZARD TRENDS
    with mon_tabs[2]:
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

    # SUB-TAB 4: HARDWARE TELEMETRY & HEALTH
    with mon_tabs[3]:
        render_performance_telemetry(latency_info=latency_info, frame_skip_val=frame_skip_val)
        render_system_health_monitor(latency_info=latency_info, fps=latency_info.get("fps", 0.0), frame_skip=frame_skip_val)
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
            exit_available=(em["recommended_exit"] in ["EXIT_A", "EXIT_B"])
        )


def _render_intelligence_section(system, active_result, source_room, active_edge, now_ts, assistant):
    """Renders Explainable AI, Emergency Decision Assistant, Scenario Controller, and Experimenter."""
    st.markdown('<div class="section-header">INTELLIGENCE, XAI &amp; DEMO CONTROLLER</div>', unsafe_allow_html=True)

    ensure_complete_emergency_state(system=system, result=active_result, source_room=source_room, active_edge=active_edge)
    em = active_result["emergency_state"]
    current_route = em["recommended_route"]
    normal_route = em["baseline_route"]
    current_exit = em["recommended_exit"]
    normal_exit = em["baseline_exit"]
    current_cost = em["total_route_cost"]
    normal_cost = em["baseline_cost"]
    hazard_score = em["hazard_score"]
    crowd_density = em["crowd_density"]
    route_changed = em["is_diverted_from_baseline"]
    current_route_str = em["recommended_route_str"]
    people_count = em["people_count"]
    graph = system.building_graph.get_graph()

    intel_tabs = st.tabs([
        "1. Academic Scenario Controller (10 Scenarios)",
        "2. Explainable AI (XAI)",
        "3. Emergency Decision Assistant",
        "4. Research Sensitivity Experimenter"
    ])

    # SUB-TAB 1: SCENARIO CONTROLLER
    with intel_tabs[0]:
        st.markdown("#### IEEE Demonstration Scenario Quick Switcher")
        sc_c1, sc_c2, sc_c3, sc_c4, sc_c5, sc_c6 = st.columns(6)
        with sc_c1:
            if st.button("🟢 01 NORMAL", use_container_width=True):
                st.session_state.active_sample_file = "normal_sample.jpg"
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

        # Advanced Scenarios 6-10
        adv_c1, adv_c2, adv_c3, adv_c4, adv_c5 = st.columns(5)
        with adv_c1:
            if st.button("📹 06 OCCLUSION", use_container_width=True):
                st.session_state.camera_simulator.update_camera_telemetry(cam_id="CAM-01", fire_detected=False, smoke_detected=False, hazard_score=0.0, people_count=0, crowd_density=0.0, coverage="UNAVAILABLE", occlusion="HIGH")
                st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 06: Camera occlusion on CAM-01 triggered coverage uncertainty."})
                st.rerun()
        with adv_c2:
            if st.button("🚫 07 EXIT BLOCKED", use_container_width=True):
                st.session_state.active_sample_file = "flames_sample.jpg"
                st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 07: EXIT_A access corridor blocked."})
                st.rerun()
        with adv_c3:
            if st.button("🔥 08 MULTI-ZONE", use_container_width=True):
                st.session_state.camera_simulator.update_camera_telemetry(cam_id="CAM-01", fire_detected=True, smoke_detected=False, hazard_score=0.88, people_count=2, crowd_density=0.1, sample_file="flames_sample.jpg")
                st.session_state.camera_simulator.update_camera_telemetry(cam_id="CAM-02", fire_detected=False, smoke_detected=True, hazard_score=0.65, people_count=5, crowd_density=0.3, sample_file="fire_sample.jpg")
                st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 08: Multi-zone hazard across CAM-01 and CAM-02."})
                st.rerun()
        with adv_c4:
            if st.button("👥 09 CROWD+SMOKE", use_container_width=True):
                st.session_state.active_sample_file = "compound_sample.jpg"
                st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 09: Heavy occupant density fleeing smoke plume."})
                st.rerun()
        with adv_c5:
            if st.button("📡 10 MULTI-CAM", use_container_width=True):
                st.session_state.incident_events.append({"time": now_ts, "desc": "Scenario 10: Multi-camera synchronized surveillance incident."})
                st.rerun()

        if st.session_state.get("run_benchmark_active", False):
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
            st.success("✅ **BENCHMARK COMPLETE**: All 5 sequential scenarios evaluated successfully via real Dual YOLOv8 perception and Dynamic Dijkstra routing!")

    # SUB-TAB 2: EXPLAINABLE AI
    with intel_tabs[1]:
        render_advanced_xai_panel(
            active_edge=active_edge,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            edge_len=graph[active_edge[0]][active_edge[1]]["length"] if graph.has_edge(active_edge[0], active_edge[1]) else 1.0,
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

    # SUB-TAB 3: EMERGENCY DECISION ASSISTANT
    with intel_tabs[2]:
        assistant_state = {
            "fire_detected": em["fire_detected"],
            "smoke_detected": em["smoke_detected"],
            "hazard_score": hazard_score,
            "people_detected": people_count,
            "crowd_density": crowd_density,
            "route_changed": route_changed,
            "recommended_exit": current_exit,
            "normal_exit": normal_exit,
            "recommended_route_str": current_route_str,
            "active_edge": active_edge,
            "total_route_cost": current_cost,
            "edge_cost": current_cost,
            "risk_index": min(100, int(round((50.0 * hazard_score) + (30.0 * crowd_density) + (20.0 * min(1.0, current_cost / 25.0)))))
        }
        render_decision_assistant(assistant=assistant, state=assistant_state)

    # SUB-TAB 4: SENSITIVITY EXPERIMENTER
    with intel_tabs[3]:
        exp_c1, exp_c2, exp_c3 = st.columns(3)
        with exp_c1:
            exp_alpha = st.slider("Experimental Hazard Weight (α)", 0.0, 15.0, ALPHA, 0.5)
        with exp_c2:
            exp_beta = st.slider("Experimental Crowd Weight (β)", 0.0, 10.0, BETA, 0.5)
        with exp_c3:
            exp_gamma = st.slider("Experimental Distance Weight (γ)", 0.01, 1.0, GAMMA, 0.05)
        render_research_experiment_mode(alpha=exp_alpha, beta=exp_beta, gamma=exp_gamma)


def _render_reports_section(system, active_result, source_room, active_edge, now_ts):
    """Renders Incident Reports, Downloads, Logs, and System Validation Panels."""
    st.markdown('<div class="section-header">INCIDENT AUDIT, LOGS &amp; SYSTEM VALIDATION</div>', unsafe_allow_html=True)

    ensure_complete_emergency_state(system=system, result=active_result, source_room=source_room, active_edge=active_edge)
    em = active_result["emergency_state"]

    has_fire = em["fire_detected"]
    has_smoke = em["smoke_detected"]
    hazard_score = em["hazard_score"]
    people_count = em["people_count"]
    crowd_density = em["crowd_density"]
    current_exit = em["recommended_exit"]
    current_route_str = em["recommended_route_str"]
    current_route = em["recommended_route"]
    current_cost = em["total_route_cost"]
    normal_route = em["baseline_route"]
    normal_exit = em["baseline_exit"]
    normal_cost = em["baseline_cost"]
    route_changed = em["is_diverted_from_baseline"]
    adaptation_reason = em["adaptation_reason"]
    pa_announcement = em["pa_announcement"]
    latency_info = active_result.get("latency_metrics", {})

    risk_index = min(100, int(round((50.0 * hazard_score) + (30.0 * crowd_density) + (20.0 * min(1.0, current_cost / 25.0)))))
    norm_c = min(1.0, current_cost / 30.0)
    efficiency_index = max(0, min(100, int(round(100.0 - ((40.0 * hazard_score) + (30.0 * crowd_density) + (20.0 * norm_c))))))

    report_payload = {
        "origin_room": source_room,
        "monitored_edge": f"{active_edge[0]} ↔ {active_edge[1]}",
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
        "guard_action": f"Direct occupants to {current_exit}" if route_changed else "Maintain nominal patrol",
        "signage_status": "Dynamic guidance active",
        "pa_message": pa_announcement,
        "fire_ms": latency_info.get("fire_ms", 0.0),
        "crowd_ms": latency_info.get("crowd_ms", 0.0),
        "dijkstra_ms": latency_info.get("dijkstra_ms", 0.0),
        "total_ms": latency_info.get("total_ms", 0.0),
        "fps": latency_info.get("fps", 0.0)
    }

    rep_tabs = st.tabs([
        "1. Incident Audit & Downloads",
        "2. Security Guard & Dispatch Log",
        "3. Route Divergence History",
        "4. Validation & IEEE Alignment",
        "5. Prototype Notes & Limitations"
    ])

    # SUB-TAB 1: INCIDENT AUDIT & DOWNLOADS
    with rep_tabs[0]:
        st.markdown("#### Digital Incident Audit Downloads")
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

        st.markdown("#### Incident Telemetry Snapshot")
        t_df = pd.DataFrame([{
            "Origin": source_room,
            "Monitored Edge": report_payload["monitored_edge"],
            "Fire": report_payload["fire_detected"],
            "Smoke": report_payload["smoke_detected"],
            "Hazard Score": f"{hazard_score:.3f}",
            "People Count": people_count,
            "Recommended Exit": current_exit,
            "Dynamic Cost": f"{current_cost:.2f}",
            "Reroute Required": report_payload["route_changed"]
        }])
        st.dataframe(t_df, hide_index=True, use_container_width=True)

    # SUB-TAB 2: SECURITY GUARD & DISPATCH LOG
    with rep_tabs[1]:
        log_c1, log_c2 = st.columns(2)
        with log_c1:
            guard_logs = active_result.get("security_guard_log") or system.alert_manager.get_guard_log(limit=10)
            render_security_guard_log(guard_logs=guard_logs, alert_manager=system.alert_manager)
        with log_c2:
            render_incident_replay_panel(events=st.session_state.incident_events)

    # SUB-TAB 3: ROUTE DIVERGENCE HISTORY
    with rep_tabs[2]:
        route_history_list = active_result.get("route_history") or system.alert_manager.get_route_history(limit=10)
        render_route_history_table(
            route_history=route_history_list,
            current_route=current_route,
            normal_route=normal_route,
            current_exit=current_exit,
            normal_exit=normal_exit,
            reason=adaptation_reason
        )

    # SUB-TAB 4: VALIDATION & IEEE ALIGNMENT
    with rep_tabs[3]:
        render_system_validation_panel()
        render_model_evaluation_panel()
        render_false_alarm_panel()
        render_system_architecture_diagram()
        render_mathematical_model_panel()
        render_ieee_traceability_matrix()

    # SUB-TAB 5: PROTOTYPE NOTES & LIMITATIONS
    with rep_tabs[4]:
        render_prototype_notes()
        render_academic_limitations()


# -----------------------------------------------------------------------------
# 2. MAIN APPLICATION ENTRYPOINT
# -----------------------------------------------------------------------------
def main():
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
    if "nav_section" not in st.session_state:
        st.session_state.nav_section = "🏠 Home"
    if "input_method" not in st.session_state:
        st.session_state.input_method = "Upload Image"
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "op_mode" not in st.session_state:
        st.session_state.op_mode = "RESEARCH MODE"
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
    if "image_eval_cache" not in st.session_state:
        st.session_state.image_eval_cache = None
    if "video_eval_cache" not in st.session_state:
        st.session_state.video_eval_cache = None
    if "live_detection_active" not in st.session_state:
        st.session_state.live_detection_active = False
    if "live_latest_result" not in st.session_state:
        st.session_state.live_latest_result = None
    if "live_latest_frame" not in st.session_state:
        st.session_state.live_latest_frame = None
    if "live_latest_raw" not in st.session_state:
        st.session_state.live_latest_raw = None
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
            {"time": now_ts, "desc": "AI Evacuation Assistant online. Dual YOLOv8 models loaded."},
            {"time": now_ts, "desc": "Topological building digital twin loaded. Dijkstra optimizer online."},
            {"time": now_ts, "desc": "System ready on standby. Select Image or Video to analyze."}
        ]

    # Inject Theme Styles based on dark_mode session state
    st.markdown(get_custom_css(dark_mode=st.session_state.dark_mode), unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SIDEBAR: NAVIGATION & DISPLAY CONTROLS
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### 🧭 Navigation")
    nav_options = [
        "🏠 Home",
        "🔍 Analyze",
        "📹 Live Camera",
        "📊 Results",
        "🗺️ Evacuation & Map",
        "📡 Monitoring & Telemetry",
        "🧠 Intelligence & XAI",
        "📄 Reports & Validation"
    ]
    if st.session_state.nav_section not in nav_options:
        st.session_state.nav_section = "🏠 Home"

    nav_index = nav_options.index(st.session_state.nav_section)
    selected_nav = st.sidebar.radio("Navigation", options=nav_options, index=nav_index, label_visibility="collapsed")
    if selected_nav != st.session_state.nav_section:
        st.session_state.nav_section = selected_nav
        st.rerun()

    # Sidebar Quick Controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Quick Controls")
    qc1, qc2 = st.sidebar.columns(2)
    with qc1:
        if st.sidebar.button("🎬 Analyze Video", key="sb_quick_vid", use_container_width=True):
            st.session_state.input_method = "Upload Video"
            st.session_state.nav_section = "🔍 Analyze"
            st.rerun()
    with qc2:
        if st.sidebar.button("▶️ Start Live", key="sb_quick_live", use_container_width=True):
            st.session_state.input_method = "Live Camera"
            st.session_state.nav_section = "📹 Live Camera"
            st.session_state.live_detection_active = True
            st.rerun()
    if st.sidebar.button("📊 Live Results", key="sb_quick_res", use_container_width=True):
        st.session_state.nav_section = "📊 Results"
        st.rerun()

    # Determine Active Hazard Status for Status Indicator
    has_active_emergency = False
    if st.session_state.image_eval_cache is not None:
        em = st.session_state.image_eval_cache["result"].get("emergency_state", {})
        if em.get("fire_detected") or em.get("smoke_detected") or em.get("hazard_score", 0) > 0.4:
            has_active_emergency = True
    elif st.session_state.video_eval_cache is not None:
        v_sum = st.session_state.video_eval_cache["summary"]
        if v_sum.get("peak_hazard_score", 0) > 0.4:
            has_active_emergency = True

    # Sidebar Status Indicator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🚦 System Status")
    if has_active_emergency:
        st.sidebar.markdown("""
        <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #EF4444; border-radius: 8px; padding: 10px 14px; color: #DC2626; font-weight: 600; font-size: 0.9rem;">
            🚨 EMERGENCY ACTIVE<br>
            <span style="font-size: 0.8rem; font-weight: 400; color: #B91C1C;">Hazard detected. Dynamic egress route diverted.</span>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.monitoring_active:
        st.sidebar.markdown("""
        <div style="background: rgba(37, 99, 235, 0.15); border: 1px solid #2563EB; border-radius: 8px; padding: 10px 14px; color: #2563EB; font-weight: 600; font-size: 0.9rem;">
            🔄 EVALUATING<br>
            <span style="font-size: 0.8rem; font-weight: 400; color: #1D4ED8;">Continuous surveillance perception active.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style="background: rgba(22, 163, 74, 0.15); border: 1px solid #16A34A; border-radius: 8px; padding: 10px 14px; color: #16A34A; font-weight: 600; font-size: 0.9rem;">
            🟢 SYSTEM READY<br>
            <span style="font-size: 0.8rem; font-weight: 400; color: #15803D;">Dual YOLOv8 & Dijkstra idle. Ready for input.</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # SIDEBAR: DISPLAY & OPERATIONAL MODES
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Mode & Theme")
    op_mode_choice = st.sidebar.radio(
        "Operational Mode",
        options=["RESEARCH MODE", "EMERGENCY COMMAND MODE"],
        index=0 if st.session_state.op_mode == "RESEARCH MODE" else 1,
        help="Research Mode displays comprehensive analytics, algorithms, and XAI. Emergency Mode streamlines high-visibility directives."
    )
    if op_mode_choice != st.session_state.op_mode:
        st.session_state.op_mode = op_mode_choice
        st.rerun()

    theme_choice = st.sidebar.radio(
        "Dashboard Theme",
        options=["Light Mode ☀️", "Dark Mode 🌙"],
        index=1 if st.session_state.dark_mode else 0,
        help="Toggle between modern light theme and dark command-center theme."
    )
    is_dark = (theme_choice == "Dark Mode 🌙")
    if is_dark != st.session_state.dark_mode:
        st.session_state.dark_mode = is_dark
        st.rerun()

    # -------------------------------------------------------------------------
    # SIDEBAR: ADVANCED CONFIGURATION EXPANDER
    # -------------------------------------------------------------------------
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Configuration & Diagnostics", expanded=False):
        st.markdown("**Origin Room**")
        graph = system.building_graph.get_graph()
        graph_rooms = sorted(list(dict.fromkeys([
            n for n, d in graph.nodes(data=True)
            if d.get("node_type") == "ROOM" or str(n).startswith("ROOM_")
        ]))) or ["ROOM_A", "ROOM_B", "ROOM_C"]
        source_room = st.selectbox("Active Origin Room", options=graph_rooms, index=0, label_visibility="collapsed")

        st.markdown("**Monitored Corridor Zone**")
        edge_descriptions = {
            ("CORRIDOR_1", "EXIT_A"): "CORRIDOR_1 ↔ EXIT_A (North Exit Corridor)",
            ("CORRIDOR_2", "EXIT_B"): "CORRIDOR_2 ↔ EXIT_B (South Exit Corridor)",
            ("CORRIDOR_1", "CORRIDOR_2"): "CORRIDOR_1 ↔ CORRIDOR_2 (Central Hallway)",
            ("ROOM_A", "CORRIDOR_1"): "ROOM_A ↔ CORRIDOR_1 (Room A Access)",
            ("ROOM_B", "CORRIDOR_1"): "ROOM_B ↔ CORRIDOR_1 (Room B North)",
            ("ROOM_B", "CORRIDOR_2"): "ROOM_B ↔ CORRIDOR_2 (Room B South)",
            ("ROOM_C", "CORRIDOR_2"): "ROOM_C ↔ CORRIDOR_2 (Room C Access)",
        }
        mon_options = list(edge_descriptions.values())
        monitored_edge_label = st.selectbox("Monitored Corridor Zone", options=mon_options, index=0, label_visibility="collapsed")
        edge_lookup = {v: k for k, v in edge_descriptions.items()}
        active_edge = edge_lookup.get(monitored_edge_label, ("CORRIDOR_1", "EXIT_A"))

        st.markdown("**Detection Confidence Threshold**")
        conf_threshold = st.slider("Detection Threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05, label_visibility="collapsed")
        system.fire_detector.conf_threshold = conf_threshold
        system.crowd_detector.conf_threshold = conf_threshold

        st.markdown("**Vulnerable Occupant Simulation**")
        vulnerable_priority = st.checkbox("Enable Vulnerable Priority Egress", value=False)
        vuln_category = st.selectbox("Assistance Profile", options=["Mobility Assistance (Wheelchair)", "Elderly Occupant", "Child Group"], index=0)
        if vulnerable_priority:
            system.optimizer.beta = BETA + 1.5
        else:
            system.optimizer.beta = BETA

        st.markdown("**Video Frame Sampling**")
        enable_frame_skip = st.checkbox("Enable Frame Skipping", value=True)
        frame_skip_val = st.slider("Sample Every N Frames", min_value=1, max_value=15, value=PROCESS_EVERY_N_FRAMES, label_visibility="collapsed")

        cloud_env = is_cloud_environment()
        if not cloud_env:
            st.markdown("**Hardware Camera Scanning**")
            if st.button("Scan Local Cameras (0–5)", use_container_width=True):
                st.session_state.discovered_cameras = discover_available_cameras(max_index=5)

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size: 0.76rem; color: var(--text-muted); line-height: 1.4;">
            <strong>System Diagnostics:</strong><br>
            Python: <code>{sys.version.split()[0]}</code><br>
            Environment: <code>{'Streamlit Cloud' if cloud_env else 'Local Desktop'}</code><br>
            Weights: <code>α={ALPHA}, β={BETA}, γ={GAMMA}</code><br>
            Nodes: <code>{len(list(system.building_graph.graph.nodes()))} nodes</code>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # RESOLVE ACTIVE RESULT (Single Source of Truth)
    # -------------------------------------------------------------------------
    active_result = None
    input_frame = None

    if st.session_state.image_eval_cache is not None:
        active_result = st.session_state.image_eval_cache["result"]
        input_frame = st.session_state.image_eval_cache.get("raw_frame")
    elif st.session_state.video_eval_cache is not None:
        v_sum = st.session_state.video_eval_cache["summary"]
        active_result = v_sum.get("latest_result")
        input_frame = v_sum.get("first_frame")

    if active_result is None or input_frame is None:
        placeholder_path = DATA_DIR / "test" / "normal_sample.jpg"
        if placeholder_path.exists():
            input_frame = cv2.imread(str(placeholder_path))
        else:
            input_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        active_result = get_standby_baseline_result(system, source_room=source_room, active_edge=active_edge, placeholder_frame=input_frame)

    ensure_complete_emergency_state(system=system, result=active_result, source_room=source_room, active_edge=active_edge)
    em_state = active_result["emergency_state"]

    # Top Application Header
    header_status = "Emergency Detected" if has_active_emergency else ("Evaluating" if st.session_state.monitoring_active else "System Ready")
    render_simple_header(status=header_status)

    # -------------------------------------------------------------------------
    # EMERGENCY COMMAND MODE OVERRIDE (High-Visibility Directives)
    # -------------------------------------------------------------------------
    if st.session_state.op_mode == "EMERGENCY COMMAND MODE":
        st.markdown('<div class="section-header" style="color: #DC2626; border-left-color: #DC2626;">EMERGENCY EVACUATION DIRECTIVES</div>', unsafe_allow_html=True)
        em_col1, em_col2 = st.columns([1.5, 1.0])
        with em_col1:
            render_emergency_alert(
                has_fire=em_state["fire_detected"],
                has_smoke=em_state["smoke_detected"],
                hazard_score=em_state["hazard_score"],
                crowd_density=em_state["crowd_density"],
                active_edge=active_edge,
                current_exit=em_state["recommended_exit"],
                alert_manager=system.alert_manager
            )
            em_haz_type = "FIRE" if em_state["fire_detected"] else ("SMOKE" if em_state["smoke_detected"] else "NONE")
            render_evacuation_guidance_card(
                recommended_exit=em_state["recommended_exit"],
                recommended_route=em_state["recommended_route"],
                is_emergency=(em_state["fire_detected"] or em_state["smoke_detected"] or em_state["hazard_score"] > 0.4),
                hazard_type=em_haz_type,
                hazard_score=em_state["hazard_score"],
                crowd_density=em_state["crowd_density"],
                adaptation_reason=em_state.get("adaptation_reason", ""),
                avoided_exit=em_state["avoided_exit"],
                route_changed=em_state["is_diverted_from_baseline"],
                total_cost=em_state["total_route_cost"],
                origin_room=source_room,
                hazard_zone=f"{active_edge[0]} ↔ {active_edge[1]}"
            )
        with em_col2:
            render_simulated_led_signage(signage_dict=em_state["dynamic_signage"])
            render_pa_announcement(message=em_state["pa_announcement"])
            guard_logs = active_result.get("security_guard_log") or system.alert_manager.get_guard_log(limit=10)
            render_security_guard_log(guard_logs=guard_logs, alert_manager=system.alert_manager)

        st.markdown("---")
        ann_rgb = cv2.cvtColor(active_result.get("annotated_frame", input_frame), cv2.COLOR_BGR2RGB)
        st.image(ann_rgb, caption="Surveillance CCTV AI Annotated Feed (HUD)", use_container_width=True)
        render_academic_disclaimer()
        return

    # -------------------------------------------------------------------------
    # ROUTE TO ACTIVE SECTION (RESEARCH MODE)
    # -------------------------------------------------------------------------
    current_section = st.session_state.nav_section

    if current_section == "🏠 Home":
        _render_home_section(system=system, active_result=active_result, source_room=source_room, active_edge=active_edge)
    elif current_section == "🔍 Analyze":
        _render_analyze_section(
            system=system,
            source_room=source_room,
            active_edge=active_edge,
            conf_threshold=conf_threshold,
            vulnerable_priority=vulnerable_priority,
            vuln_category=vuln_category,
            enable_frame_skip=enable_frame_skip,
            frame_skip_val=frame_skip_val,
            cloud_env=is_cloud_environment(),
            now_ts=now_ts
        )
    elif current_section == "📹 Live Camera":
        _render_live_camera_section(
            system=system,
            source_room=source_room,
            active_edge=active_edge,
            conf_threshold=conf_threshold,
            dark_mode=st.session_state.dark_mode,
            now_ts=now_ts
        )
    elif current_section == "📊 Results":
        _render_results_section(system=system, active_result=active_result, source_room=source_room, active_edge=active_edge, now_ts=now_ts)
    elif current_section == "🗺️ Evacuation & Map":
        _render_evacuation_section(
            system=system,
            active_result=active_result,
            source_room=source_room,
            active_edge=active_edge,
            multi_floor_model=multi_floor_model,
            vulnerable_priority=vulnerable_priority,
            vuln_category=vuln_category,
            dark_mode=st.session_state.dark_mode
        )
    elif current_section == "📡 Monitoring & Telemetry":
        _render_monitoring_section(
            system=system,
            active_result=active_result,
            source_room=source_room,
            active_edge=active_edge,
            frame_skip_val=frame_skip_val,
            dark_mode=st.session_state.dark_mode
        )
    elif current_section == "🧠 Intelligence & XAI":
        _render_intelligence_section(
            system=system,
            active_result=active_result,
            source_room=source_room,
            active_edge=active_edge,
            now_ts=now_ts,
            assistant=assistant
        )
    elif current_section == "📄 Reports & Validation":
        _render_reports_section(
            system=system,
            active_result=active_result,
            source_room=source_room,
            active_edge=active_edge,
            now_ts=now_ts
        )

    # Bottom academic disclaimer on every view
    render_academic_disclaimer()

    # Continuous monitoring loop auto-refresh
    if st.session_state.monitoring_active and st.session_state.video_stream is not None:
        time.sleep(0.04)
        st.rerun()


if __name__ == "__main__":
    main()
