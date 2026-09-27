"""
AI-Based Fire/Smoke Hazard Detection & Crowd-Aware Evacuation Path Optimization
Final IEEE Paper Demonstration Dashboard

Complete 20-Section Architecture:
1. HEADER
2. SYSTEM STATUS
3. INPUT / CCTV INGESTION
4. RAW VS AI ANNOTATED CCTV
5. AI PERCEPTION
6. CURRENT HAZARD ZONE
7. CROWD / ZONE ANALYSIS
8. DYNAMIC EDGE COST
9. BUILDING GRAPH
10. ROUTE COMPARISON
11. RECOMMENDED EXIT
12. EVACUATION ROUTE FLOW
13. ALERT CENTER
14. DIRECTIONAL SIGNAGE
15. PA ANNOUNCEMENT
16. SECURITY GUARD LOG
17. ROUTE HISTORY
18. PERFORMANCE TELEMETRY
19. DEMO SCENARIOS
20. ACADEMIC TRANSPARENCY / LIMITATIONS
"""

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
    FRAME_SKIP,
    PROCESS_EVERY_N_FRAMES,
    DEMO_BUILDING_ZONES,
    ZONE_MAX_CAPACITIES,
)
from integration.evacuation_system import EvacuationSystem
from video.video_stream import VideoStreamHandler
from dashboard.styles import get_custom_css
from dashboard.components import (
    render_header,
    render_system_status_cards,
    render_cctv_meta_bar,
    render_perception_metrics,
    render_hazard_zone_panel,
    render_multi_zone_analysis,
    render_dynamic_cost_panel,
    render_route_comparison,
    render_recommended_exit_panel,
    render_route_flow_card,
    render_emergency_alert,
    render_simulated_led_signage,
    render_pa_announcement,
    render_prototype_notes,
)


# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING INJECTION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Evacuation System — IEEE Paper Demo",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(get_custom_css(), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# CACHED SYSTEM INSTANCE (CPU INFERENCE)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Pretrained YOLO Models & Evacuation Engine (CPU)...")
def get_system():
    try:
        sys_inst = EvacuationSystem()
        return sys_inst, None
    except Exception as e:
        return None, str(e)


# Alias for test suite compatibility
load_evacuation_system = get_system


# -----------------------------------------------------------------------------
# HELPERS
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
    # -------------------------------------------------------------------------
    # 1. HEADER
    # -------------------------------------------------------------------------
    render_header()

    system, init_err = get_system()
    if init_err:
        st.error(f"[!] System Engine Initialization Failed: {init_err}")
        return

    # -------------------------------------------------------------------------
    # 2. SYSTEM STATUS
    # -------------------------------------------------------------------------
    render_system_status_cards()

    # -------------------------------------------------------------------------
    # SESSION STATE INITIALIZATION
    # -------------------------------------------------------------------------
    if "active_sample_file" not in st.session_state:
        st.session_state.active_sample_file = "flames_sample.jpg"
    if "input_method" not in st.session_state:
        st.session_state.input_method = "Quick Test: FIRE"
    if "run_benchmark_active" not in st.session_state:
        st.session_state.run_benchmark_active = False

    # -------------------------------------------------------------------------
    # 3. INPUT / CCTV INGESTION
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">1. Surveillance CCTV Ingestion & Control</div>', unsafe_allow_html=True)
    st.caption("Select an input source or use Quick Test benchmark buttons to execute genuine dual-YOLO perception on CPU.")

    q_col1, q_col2, q_col3, q_col4, q_col5 = st.columns(5)
    with q_col1:
        if st.button("🟢 Quick Test: NORMAL", use_container_width=True):
            st.session_state.active_sample_file = "normal_sample.jpg"
            st.session_state.input_method = "Quick Test: NORMAL"
            st.session_state.run_benchmark_active = False
    with q_col2:
        if st.button("🔥 Quick Test: FIRE", use_container_width=True):
            st.session_state.active_sample_file = "flames_sample.jpg"
            st.session_state.input_method = "Quick Test: FIRE"
            st.session_state.run_benchmark_active = False
    with q_col3:
        if st.button("💨 Quick Test: SMOKE", use_container_width=True):
            st.session_state.active_sample_file = "fire_sample.jpg"
            st.session_state.input_method = "Quick Test: SMOKE"
            st.session_state.run_benchmark_active = False
    with q_col4:
        if st.button("👥 Quick Test: CROWD", use_container_width=True):
            st.session_state.active_sample_file = "crowd_sample.jpg"
            st.session_state.input_method = "Quick Test: CROWD"
            st.session_state.run_benchmark_active = False
    with q_col5:
        if st.button("⚡ Quick Test: COMPOUND", use_container_width=True):
            st.session_state.active_sample_file = "compound_sample.jpg"
            st.session_state.input_method = "Quick Test: COMPOUND"
            st.session_state.run_benchmark_active = False

    # Sidebar Controls
    st.sidebar.header("⚙️ CCTV Ingestion & Topology Controls")

    input_method_options = [
        "Quick Test Sample",
        "Upload Image",
        "Upload Video",
        "Webcam"
    ]
    current_method_idx = 0
    if st.session_state.input_method in ["Upload Image", "Upload Video", "Webcam"]:
        current_method_idx = input_method_options.index(st.session_state.input_method)

    selected_method = st.sidebar.radio("CCTV Media Source", options=input_method_options, index=current_method_idx)
    if selected_method != "Quick Test Sample":
        st.session_state.input_method = selected_method

    # Evacuation Origin Room
    st.sidebar.markdown("---")
    st.sidebar.subheader("📍 Origin & Monitored Zone")
    source_room = st.sidebar.selectbox("Evacuation Origin", options=["ROOM_A", "ROOM_B", "ROOM_C"], index=0)

    # Monitored Corridor Zone
    monitored_edge_label = st.sidebar.selectbox(
        "Monitored Edge (Demo Zone Mapping)",
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

    # Perception Sensitivity Slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Confidence & False-Alarm Control")
    conf_threshold = st.sidebar.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05
    )
    system.fire_detector.conf_threshold = conf_threshold
    system.crowd_detector.conf_threshold = conf_threshold

    # Continuous Video Frame Skip
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ CPU Frame-Skip Optimization")
    enable_frame_skip = st.sidebar.checkbox("Enable Frame-Skip Sampling", value=True)
    frame_skip_val = st.sidebar.slider("Process Every N Frames", min_value=1, max_value=15, value=PROCESS_EVERY_N_FRAMES)
    st.sidebar.caption(f"Current Configured Frame Skip: **{frame_skip_val if enable_frame_skip else 1}**")

    # -------------------------------------------------------------------------
    # ACQUIRE MEDIA FRAME (SECTION 19, 20, 21, 22)
    # -------------------------------------------------------------------------
    input_frame = None
    media_source_description = ""
    is_video_mode = False
    video_capture_obj = None

    if st.session_state.input_method.startswith("Quick Test"):
        sample_name = st.session_state.active_sample_file
        sample_path = DATA_DIR / "test" / sample_name
        if sample_path.exists():
            input_frame = cv2.imread(str(sample_path))
            media_source_description = f"Local Benchmark Sample: `{sample_name}`"
        else:
            st.error(f"[!] Benchmark file `{sample_name}` not found in `{DATA_DIR / 'test'}`.")

    elif st.session_state.input_method == "Upload Image":
        up_img = st.sidebar.file_uploader("Upload Image File", type=["jpg", "jpeg", "png", "bmp"])
        if up_img is not None:
            try:
                file_bytes = np.asarray(bytearray(up_img.read()), dtype=np.uint8)
                input_frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if input_frame is not None:
                    media_source_description = f"Uploaded Custom Image: `{up_img.name}` ({input_frame.shape[1]}x{input_frame.shape[0]})"
                else:
                    st.warning("⚠️ Could not decode uploaded image. Falling back to default benchmark sample.")
                    sample_path = DATA_DIR / "test" / "flames_sample.jpg"
                    input_frame = cv2.imread(str(sample_path))
            except Exception as e:
                st.warning(f"⚠️ Error processing uploaded image: {e}. Falling back to benchmark sample.")
                input_frame = cv2.imread(str(DATA_DIR / "test" / "flames_sample.jpg"))
        else:
            sample_path = DATA_DIR / "test" / st.session_state.active_sample_file
            if sample_path.exists():
                input_frame = cv2.imread(str(sample_path))
                media_source_description = f"Awaiting Image Upload — Displaying Benchmark Sample: `{sample_path.name}`"

    elif st.session_state.input_method == "Upload Video":
        is_video_mode = True
        up_vid = st.sidebar.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
        video_path = None
        if up_vid is not None:
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(up_vid.read())
                video_path = tfile.name
                media_source_description = f"Uploaded Video: `{up_vid.name}`"
            except Exception as e:
                st.warning(f"⚠️ Error buffering video file: {e}")

        if not video_path:
            available_vids = get_available_test_videos()
            if available_vids:
                video_path = str(DATA_DIR / "test" / available_vids[0])
                media_source_description = f"Awaiting Video Upload — Using Demo Video: `{available_vids[0]}`"

        if video_path:
            video_capture_obj = cv2.VideoCapture(video_path)
            if video_capture_obj.isOpened():
                ret, frame = video_capture_obj.read()
                if ret and frame is not None:
                    input_frame = frame
            else:
                st.warning("⚠️ Unable to open video source. Using benchmark fallback image.")
                input_frame = cv2.imread(str(DATA_DIR / "test" / "flames_sample.jpg"))

    elif st.session_state.input_method == "Webcam":
        webcam_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0)
        run_cam = st.sidebar.checkbox("Start Live Webcam Capture", value=False)
        if run_cam:
            try:
                cap = cv2.VideoCapture(int(webcam_idx))
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        input_frame = frame
                        media_source_description = f"Live CCTV Webcam Feed (Device #{webcam_idx})"
                    else:
                        st.warning(f"⚠️ Unable to read frame from camera #{webcam_idx}. Using benchmark fallback.")
                        input_frame = cv2.imread(str(DATA_DIR / "test" / "flames_sample.jpg"))
                    cap.release()
                else:
                    st.warning(f"⚠️ WEBCAM NOT AVAILABLE: Device index #{webcam_idx} is not accessible. Please connect a camera or select Quick Test Sample.")
                    input_frame = cv2.imread(str(DATA_DIR / "test" / "flames_sample.jpg"))
            except Exception as e:
                st.warning(f"⚠️ WEBCAM NOT AVAILABLE: {e}. Using benchmark fallback.")
                input_frame = cv2.imread(str(DATA_DIR / "test" / "flames_sample.jpg"))
        else:
            st.info("ℹ️ Check 'Start Live Webcam Capture' to begin camera stream, or choose a Quick Test sample.")
            input_frame = cv2.imread(str(DATA_DIR / "test" / "flames_sample.jpg"))
            media_source_description = "Webcam Inactive — Displaying Benchmark Sample (flames_sample.jpg)"

    if input_frame is None:
        st.error("⚠️ No valid media frame available. Please select a Quick Test sample or upload an image.")
        return

    # -------------------------------------------------------------------------
    # BASELINE NORMAL ROUTE CALCULATION (DYNAMIC EVALUATION)
    # -------------------------------------------------------------------------
    clean_graph = system.building_graph.graph.copy()
    for u_node, v_node in clean_graph.edges:
        clean_graph[u_node][v_node]["hazard"] = 0.0
        clean_graph[u_node][v_node]["density"] = 0.0
    system.optimizer.update_all_edge_costs(clean_graph)
    normal_route_info = system.optimizer.find_safest_route(
        clean_graph, source=source_room, exits=["EXIT_A", "EXIT_B"]
    )
    normal_route = normal_route_info.get("path", [])
    normal_exit = normal_route_info.get("selected_exit", "NONE")
    normal_cost = normal_route_info.get("total_cost", 0.0)

    # -------------------------------------------------------------------------
    # EXECUTE FULL PIPELINE (GENUINE CPU INFERENCE & DYNAMIC DIJKSTRA)
    # -------------------------------------------------------------------------
    with st.spinner("Processing frame (Dual YOLOv8 on CPU + Dynamic Dijkstra)..."):
        try:
            result = system.process_frame(
                frame=input_frame,
                active_edge=active_edge,
                source_room=source_room,
                enable_frame_skip=enable_frame_skip,
                skip_interval=frame_skip_val
            )
        except Exception as e:
            st.error(f"⚠️ Error executing pipeline on current frame: {e}")
            return

    # Unpack live pipeline metrics
    has_fire = result["fire_detected"]
    has_smoke = result["smoke_detected"]
    hazard_score = result["hazard_score"]
    people_count = result["people_detected"]
    crowd_density = result["crowd_density"]
    current_exit = result["recommended_exit"]
    current_route = result["recommended_route"]
    current_route_str = result["recommended_route_str"]
    current_cost = result["total_route_cost"]
    annotated_frame = result["annotated_frame"]
    latency_info = result.get("latency_metrics", {})
    signage_dict = result.get("dynamic_signage", {})
    pa_announcement = result.get("current_pa_message", "")
    zone_counts = result.get("multi_zone_counts", {})
    zone_densities = result.get("multi_zone_densities", {})

    # Graph edge metrics
    graph = system.building_graph.get_graph()
    edge_len = graph[active_edge[0]][active_edge[1]]["length"]
    updated_edge_cost = graph[active_edge[0]][active_edge[1]]["cost"]

    # Compute Hazard and Crowd Status strings
    if hazard_score >= 0.70 or has_fire:
        hazard_status = "DANGER"
        hazard_type_str = "FIRE"
    elif hazard_score >= 0.40 or has_smoke:
        hazard_status = "WARNING"
        hazard_type_str = "SMOKE"
    elif hazard_score > 0.0:
        hazard_status = "CAUTION"
        hazard_type_str = "LOW HAZARD"
    else:
        hazard_status = "NORMAL"
        hazard_type_str = "NONE"

    if crowd_density >= 0.70:
        crowd_status = "HIGH CONGESTION"
    elif crowd_density >= 0.35:
        crowd_status = "MODERATE CONGESTION"
    else:
        crowd_status = "NORMAL"

    zone_status_label = "EMERGENCY" if hazard_status in ["DANGER", "WARNING"] or crowd_status == "HIGH CONGESTION" else ("CAUTION" if hazard_status == "CAUTION" or crowd_status == "MODERATE CONGESTION" else "NORMAL")

    # Compute Route Adaptation dynamically
    route_changed = (current_route != normal_route)
    if route_changed:
        if has_fire or hazard_score >= 0.70:
            adaptation_reason = f"Active corridor edge {active_edge[0]} <-> {active_edge[1]} compromised by Fire Hazard ({hazard_score:.3f}). Edge cost increased to {updated_edge_cost:.2f}. Dynamic Dijkstra rerouted origin {source_room} from {normal_exit} to {current_exit}."
        elif has_smoke or hazard_score > 0.0:
            adaptation_reason = f"Active corridor edge {active_edge[0]} <-> {active_edge[1]} compromised by Smoke Hazard ({hazard_score:.3f}). Visibility and air quality impaired. Rerouted from {normal_exit} to {current_exit}."
        elif crowd_density >= 0.70:
            adaptation_reason = f"Severe crowd congestion on {active_edge[0]} <-> {active_edge[1]} (Density: {crowd_density:.3f}). Dynamic Dijkstra diverted path from {normal_exit} to {current_exit} to prevent crowd crush."
        else:
            adaptation_reason = f"Dynamic path cost adjustments made alternate route via {current_exit} globally optimal."
    else:
        adaptation_reason = f"Direct route to {normal_exit} remains safe and optimal (Total Cost: {current_cost:.2f}). No detour necessary."

    # -------------------------------------------------------------------------
    # 4. RAW VS AI ANNOTATED CCTV
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">2. CCTV Surveillance Feed Inspection</div>', unsafe_allow_html=True)
    if media_source_description:
        st.caption(f"Source: {media_source_description}")

    feed_col1, feed_col2 = st.columns(2)
    with feed_col1:
        st.markdown("**RAW CCTV INPUT**")
        raw_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        st.image(raw_rgb, use_container_width=True)

    with feed_col2:
        st.markdown("**AI ANNOTATED OUTPUT (Bounding Boxes, Confidences, Centers, HUD)**")
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_container_width=True)

    # Metadata bar below CCTV images
    h_in, w_in = input_frame.shape[:2]
    render_cctv_meta_bar(
        width=w_in,
        height=h_in,
        frame_num=latency_info.get("frame_number", 1),
        latency_ms=latency_info.get("total_ms", 0.0),
        fps=latency_info.get("fps", 0.0)
    )

    # -------------------------------------------------------------------------
    # 5. AI PERCEPTION
    # -------------------------------------------------------------------------
    render_perception_metrics(
        has_fire=has_fire,
        has_smoke=has_smoke,
        hazard_score=hazard_score,
        people_count=people_count,
        crowd_density=crowd_density,
        hazard_status=hazard_status,
        crowd_status=crowd_status
    )

    # -------------------------------------------------------------------------
    # 6. CURRENT HAZARD ZONE
    # -------------------------------------------------------------------------
    render_hazard_zone_panel(
        active_edge=active_edge,
        hazard_type_str=hazard_type_str,
        hazard_score=hazard_score,
        crowd_density=crowd_density,
        status_label=zone_status_label
    )

    # -------------------------------------------------------------------------
    # 7. CROWD / ZONE ANALYSIS
    # -------------------------------------------------------------------------
    render_multi_zone_analysis(
        zone_counts=zone_counts,
        zone_densities=zone_densities,
        total_people=people_count
    )

    # -------------------------------------------------------------------------
    # 8. DYNAMIC EDGE COST
    # -------------------------------------------------------------------------
    render_dynamic_cost_panel(
        alpha=ALPHA,
        beta=BETA,
        gamma=GAMMA,
        hazard_score=hazard_score,
        crowd_density=crowd_density,
        edge_length=edge_len,
        edge_cost=updated_edge_cost,
        active_edge=active_edge
    )

    # -------------------------------------------------------------------------
    # 9. BUILDING GRAPH
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">Topological Building Evacuation Graph</div>', unsafe_allow_html=True)
    fig_map = system.building_graph.render_evacuation_map(
        recommended_route=current_route,
        previous_route=result.get("previous_route"),
        source_room=source_room,
        monitored_edge=active_edge,
        hazardous_edges=[active_edge] if hazard_score > 0 else [],
        congested_edges=[active_edge] if crowd_density >= 0.35 else [],
        system_state=hazard_status
    )
    st.pyplot(fig_map)

    # -------------------------------------------------------------------------
    # 10. ROUTE COMPARISON
    # -------------------------------------------------------------------------
    render_route_comparison(
        normal_route=normal_route,
        normal_exit=normal_exit,
        normal_cost=normal_cost,
        current_route=current_route,
        current_exit=current_exit,
        current_cost=current_cost,
        route_changed=route_changed,
        reason=adaptation_reason
    )

    # -------------------------------------------------------------------------
    # 11. RECOMMENDED EXIT
    # -------------------------------------------------------------------------
    render_recommended_exit_panel(
        recommended_exit=current_exit,
        total_route_cost=current_cost,
        avoided_exit=normal_exit if route_changed else None
    )

    # -------------------------------------------------------------------------
    # 12. EVACUATION ROUTE FLOW
    # -------------------------------------------------------------------------
    render_route_flow_card(
        route_nodes=current_route,
        total_cost=current_cost,
        route_changed=route_changed
    )

    # -------------------------------------------------------------------------
    # 13. ALERT CENTER
    # -------------------------------------------------------------------------
    render_emergency_alert(
        has_fire=has_fire,
        has_smoke=has_smoke,
        hazard_score=hazard_score,
        crowd_density=crowd_density,
        active_edge=active_edge,
        current_exit=current_exit
    )

    # -------------------------------------------------------------------------
    # 14 & 15. DIRECTIONAL SIGNAGE & PA ANNOUNCEMENT
    # -------------------------------------------------------------------------
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        render_simulated_led_signage(signage_dict=signage_dict)

    with act_col2:
        render_pa_announcement(message=pa_announcement)

    # -------------------------------------------------------------------------
    # 16. SECURITY GUARD ALERT LOG
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">Security Guard Alert Log</div>', unsafe_allow_html=True)
    guard_logs = system.alert_manager.get_guard_log(limit=10)
    if guard_logs:
        guard_df = pd.DataFrame(guard_logs)
        display_cols = [c for c in ["timestamp", "location", "type", "hazard_score", "people", "density", "recommended_exit", "action", "status"] if c in guard_df.columns]
        st.dataframe(guard_df[display_cols] if display_cols else guard_df, hide_index=True, use_container_width=True)
        if st.button("CLEAR LOG", key="clear_guard_log_btn"):
            system.alert_manager.guard_alert_log.clear()
            st.rerun()
    else:
        st.caption("No emergency alerts logged yet. System status normal.")

    # -------------------------------------------------------------------------
    # 17. ROUTE HISTORY
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">Route Change Event History</div>', unsafe_allow_html=True)
    route_hist = system.alert_manager.get_route_history(limit=10)
    if route_hist:
        rh_df = pd.DataFrame(route_hist)
        display_cols = [c for c in ["timestamp", "previous_route", "new_route", "reason", "previous_cost", "new_cost", "cost_difference"] if c in rh_df.columns]
        st.dataframe(rh_df[display_cols] if display_cols else rh_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No route changes registered yet.")

    # -------------------------------------------------------------------------
    # 18. PERFORMANCE TELEMETRY
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">Performance & Hardware Telemetry</div>', unsafe_allow_html=True)
    t1, t2, t3, t4, t5, t6, t7 = st.columns(7)
    with t1:
        st.metric("Fire/Smoke Inference", f"{latency_info.get('fire_ms', latency_info.get('detection_ms', 0) / 2):.1f} ms")
    with t2:
        st.metric("Crowd Inference", f"{latency_info.get('crowd_ms', latency_info.get('detection_ms', 0) / 2):.1f} ms")
    with t3:
        st.metric("Dijkstra Recalc", f"{latency_info.get('dijkstra_ms', 0):.2f} ms")
    with t4:
        st.metric("Total Processing", f"{latency_info.get('total_ms', 0):.1f} ms")
    with t5:
        st.metric("Processed FPS", f"{latency_info.get('fps', 0):.1f}")
    with t6:
        st.metric("Frame Skip", f"{latency_info.get('frame_skip', frame_skip_val)}")
    with t7:
        st.metric("CPU Mode", "ACTIVE")

    # Clean release of any video capture
    if video_capture_obj is not None:
        video_capture_obj.release()

    # -------------------------------------------------------------------------
    # 19. DEMO SCENARIOS & FULL DEMO RUNNER
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-header">Academic Demo Scenario Controller</div>', unsafe_allow_html=True)
    sc_c1, sc_c2, sc_c3, sc_c4, sc_c5, sc_c6 = st.columns(6)
    with sc_c1:
        if st.button("01 NORMAL", use_container_width=True):
            st.session_state.active_sample_file = "normal_sample.jpg"
            st.session_state.input_method = "Quick Test: NORMAL"
            st.session_state.run_benchmark_active = False
            st.rerun()
    with sc_c2:
        if st.button("02 SMOKE", use_container_width=True):
            st.session_state.active_sample_file = "fire_sample.jpg"
            st.session_state.input_method = "Quick Test: SMOKE"
            st.session_state.run_benchmark_active = False
            st.rerun()
    with sc_c3:
        if st.button("03 FIRE", use_container_width=True):
            st.session_state.active_sample_file = "flames_sample.jpg"
            st.session_state.input_method = "Quick Test: FIRE"
            st.session_state.run_benchmark_active = False
            st.rerun()
    with sc_c4:
        if st.button("04 HIGH CROWD", use_container_width=True):
            st.session_state.active_sample_file = "crowd_sample.jpg"
            st.session_state.input_method = "Quick Test: CROWD"
            st.session_state.run_benchmark_active = False
            st.rerun()
    with sc_c5:
        if st.button("05 COMPOUND", use_container_width=True):
            st.session_state.active_sample_file = "compound_sample.jpg"
            st.session_state.input_method = "Quick Test: COMPOUND"
            st.session_state.run_benchmark_active = False
            st.rerun()
    with sc_c6:
        if st.button("▶ RUN FULL DEMO", use_container_width=True):
            st.session_state.run_benchmark_active = True

    # Sequential Benchmark Execution
    if st.session_state.run_benchmark_active:
        st.markdown("#### Sequential Benchmark Sweep Results")
        benchmark_scenarios = [
            ("01 NORMAL", "normal_sample.jpg", "Clear pathway baseline (Zero hazard, low crowd)"),
            ("02 SMOKE", "fire_sample.jpg", "Smoke plume detection triggering heightened caution"),
            ("03 FIRE", "flames_sample.jpg", "Critical flame hazard forcing dynamic path diversion"),
            ("04 HIGH CROWD", "crowd_sample.jpg", "Occupant congestion bottleneck avoidance"),
            ("05 COMPOUND", "compound_sample.jpg", "Simultaneous fire hazard and severe congestion")
        ]

        bench_results = []
        progress_bar = st.progress(0.0)

        for idx, (sc_code, sc_file, sc_desc) in enumerate(benchmark_scenarios):
            img_p = DATA_DIR / "test" / sc_file
            if img_p.exists():
                bench_img = cv2.imread(str(img_p))
                res = system.process_frame(
                    frame=bench_img,
                    active_edge=active_edge,
                    source_room=source_room,
                    enable_frame_skip=False
                )
                bench_results.append({
                    "Scenario": sc_code,
                    "Fire": "YES" if res["fire_detected"] else "NO",
                    "Smoke": "YES" if res["smoke_detected"] else "NO",
                    "People": res["people_detected"],
                    "Density": f"{res['crowd_density']:.3f}",
                    "Hazard": f"{res['hazard_score']:.3f}",
                    "Edge Cost": f"{res['updated_edge_cost']:.2f}",
                    "Exit": res["recommended_exit"],
                    "Route": res["recommended_route_str"],
                    "Cost": f"{res['total_route_cost']:.2f}",
                    "Status": "REROUTED" if res["recommended_exit"] != "EXIT_A" else "NORMAL"
                })
            progress_bar.progress((idx + 1) / len(benchmark_scenarios))
            time.sleep(0.05)

        st.dataframe(pd.DataFrame(bench_results), hide_index=True, use_container_width=True)
        st.success("✅ **BENCHMARK COMPLETE**: All 5 sequential scenarios evaluated successfully via real YOLO perception and Dynamic Dijkstra!")

    # -------------------------------------------------------------------------
    # 20. ACADEMIC TRANSPARENCY / LIMITATIONS
    # -------------------------------------------------------------------------
    render_prototype_notes()

    # Developer / Diagnostics Information Expander
    with st.expander("🛠️ Developer / Diagnostics Information", expanded=False):
        st.markdown(f"""
        - **Python Version**: `{sys.version.split()[0]}`
        - **Project Root**: `{project_root}`
        - **Weights**: `ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}`
        - **Graph Nodes**: `{list(system.building_graph.graph.nodes())}`
        - **Graph Edges**: `{list(system.building_graph.graph.edges())}`
        - **Active Frame Skip Interval**: `{frame_skip_val}`
        - **Active Origin Room**: `{source_room}`
        - **Active Edge**: `{active_edge}`
        """)


if __name__ == "__main__":
    main()
