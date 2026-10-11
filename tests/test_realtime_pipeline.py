"""
Unit & Integration Test Suite: Real-Time Perception, Temporal Hazard Tracking,
Persistent Video Streaming, and Emergency Incident Management.
Validates:
1. VideoStreamHandler initialization and graceful failure on invalid streams.
2. VideoStreamHandler playback from CCTV video with real FPS and metadata.
3. TemporalHazardTracker nominal state tracking.
4. TemporalHazardTracker transient anomaly vs consecutive confirmation.
5. TemporalHazardTracker cooldown and flap prevention.
6. TemporalHazardTracker camera unavailable detection.
7. AlertManager alert ID assignment and operator acknowledgement.
8. AlertManager automatic snapshot creation on disk.
9. AlertManager duplicate suppression within cooldown window.
10. MultiCameraSimulator hybrid stream binding and unbinding.
11. Incident export serialization (JSON, CSV, TXT).
12. Full EvacuationSystem frame processing with integrated snapshot generation.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DATA_DIR
from video.video_stream import VideoStreamHandler
from integration.hazard_trend import TemporalHazardTracker
from alerts.alert_manager import AlertManager
from integration.multi_camera_sim import MultiCameraSimulator
from integration.evacuation_system import EvacuationSystem


def run_tests():
    total_tests = 19
    passed = 0
    failed = 0

    print("=" * 70)
    print("REAL-TIME PIPELINE & TEMPORAL HAZARD TRACKING TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TEST 1: VideoStreamHandler Failure Handling (Invalid Source)
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 01: VideoStreamHandler Graceful Failure (Invalid Source)...")
    try:
        stream = VideoStreamHandler(source="rtsp://invalid.dummy.host:8554/stream")
        success = stream.start_stream()
        assert not success, "Expected start_stream to fail for invalid RTSP source"
        assert stream.status == "ERROR", f"Expected status ERROR, got {stream.status}"
        assert stream.error_message is not None, "Expected descriptive error message"
        ret, frame = stream.get_frame()
        assert not ret and frame is None, "Expected get_frame to return (False, None)"
        stream.release()
        assert stream.status == "DISCONNECTED"
        print("[PASS] Test 01: Graceful failure on invalid stream source verified (No fake fallbacks).")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 01: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 2: VideoStreamHandler File Playback & Properties
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 02: VideoStreamHandler File Playback & FPS Tracking...")
    try:
        sample_vid = DATA_DIR / "test" / "demo_cctv_feed.mp4"
        assert sample_vid.exists(), f"Missing demo video at {sample_vid}"
        stream = VideoStreamHandler(source=str(sample_vid))
        assert stream.start_stream(), "Failed to open demo_cctv_feed.mp4"
        props = stream.get_properties()
        assert props["width"] > 0 and props["height"] > 0, "Invalid video resolution"
        ret, frame = stream.get_frame(loop_video=True)
        assert ret and frame is not None and frame.size > 0, "Failed to read first video frame"
        assert stream.frame_count == 1
        tel = stream.get_telemetry()
        assert tel["status"] == "CONNECTED"
        assert tel["source_type"] == "VIDEO_FILE"
        stream.release()
        assert stream.status == "DISCONNECTED"
        print(f"[PASS] Test 02: CCTV video playback ({props['width']}x{props['height']}) verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 02: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 3: TemporalHazardTracker Nominal Baseline State
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 03: TemporalHazardTracker Nominal Baseline State...")
    try:
        tracker = TemporalHazardTracker(window_size=20, confirmation_threshold=3, cooldown_frames=5)
        for _ in range(5):
            summary = tracker.update(hazard_score=0.0, fire_conf=0.0, smoke_conf=0.0)
        assert summary["current_state"] == "NORMAL", f"Expected NORMAL, got {summary['current_state']}"
        assert summary["consecutive_hazard"] == 0
        assert not summary["is_confirmed"]
        assert summary["slope"] == "STABLE"
        print("[PASS] Test 03: Baseline nominal state and slope calculation verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 03: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 4: TemporalHazardTracker Transient Anomaly (Single Frame)
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 04: TemporalHazardTracker Transient Anomaly Suppression...")
    try:
        tracker = TemporalHazardTracker(window_size=20, confirmation_threshold=3, cooldown_frames=5)
        tracker.update(hazard_score=0.0)
        summary = tracker.update(hazard_score=0.85, fire_conf=0.88)
        assert summary["current_state"] == "RISK INCREASING", f"Expected RISK INCREASING, got {summary['current_state']}"
        assert summary["consecutive_hazard"] == 1
        assert not summary["is_confirmed"], "Single frame should not confirm hazard"
        assert summary["confidence_label"] == "TRANSIENT / ANOMALY"
        print("[PASS] Test 04: Single-frame transient anomaly suppressed correctly without alarm flapping.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 04: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 5: TemporalHazardTracker Consecutive Confirmation
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 05: TemporalHazardTracker Multi-Frame Consecutive Confirmation...")
    try:
        tracker = TemporalHazardTracker(window_size=20, confirmation_threshold=3, cooldown_frames=5)
        tracker.update(hazard_score=0.80, fire_conf=0.85)
        tracker.update(hazard_score=0.85, fire_conf=0.90)
        summary = tracker.update(hazard_score=0.90, fire_conf=0.92)
        assert summary["consecutive_hazard"] == 3
        assert summary["is_confirmed"] is True
        assert summary["current_state"] == "FIRE DETECTED", f"Expected FIRE DETECTED, got {summary['current_state']}"
        assert summary["confidence_label"] == "CONFIRMED HAZARD"
        print("[PASS] Test 05: Confirmed hazard state achieved after 3 consecutive frames.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 05: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 6: TemporalHazardTracker Cooldown & Flap Prevention
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 06: TemporalHazardTracker Cooldown & Flap Prevention...")
    try:
        tracker = TemporalHazardTracker(window_size=20, confirmation_threshold=3, cooldown_frames=3)
        # Establish confirmed fire
        for _ in range(3):
            tracker.update(hazard_score=0.80, fire_conf=0.85)
        # First clear frame -> transitions to MONITORING
        s1 = tracker.update(hazard_score=0.0)
        assert s1["current_state"] == "MONITORING", f"Expected MONITORING, got {s1['current_state']}"
        assert s1["cooldown_remaining"] == 2
        # Second clear frame -> stays MONITORING
        s2 = tracker.update(hazard_score=0.0)
        assert s2["current_state"] == "MONITORING"
        assert s2["cooldown_remaining"] == 1
        # Third clear frame -> stays MONITORING
        s3 = tracker.update(hazard_score=0.0)
        assert s3["cooldown_remaining"] == 0
        # Fourth clear frame -> returns to NORMAL
        s4 = tracker.update(hazard_score=0.0)
        assert s4["current_state"] == "NORMAL", f"Expected NORMAL, got {s4['current_state']}"
        print("[PASS] Test 06: Cooldown flap prevention verified across state decay.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 06: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 7: TemporalHazardTracker Camera Unavailable Detection
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 07: TemporalHazardTracker Camera Disconnection State...")
    try:
        tracker = TemporalHazardTracker()
        summary = tracker.update(hazard_score=0.0, camera_status="UNAVAILABLE")
        assert summary["current_state"] == "CAMERA UNAVAILABLE"
        print("[PASS] Test 07: CAMERA UNAVAILABLE transparent state verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 07: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 8: AlertManager ID Generation & Acknowledgement
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 08: AlertManager Acknowledgement Protocol...")
    try:
        alert_mgr = AlertManager()
        alert = alert_mgr.trigger_hazard_alert(
            location="CORRIDOR_1 <-> EXIT_A",
            hazard_type="FIRE",
            severity="CRITICAL",
            hazard_score=0.85
        )
        assert "alert_id" in alert
        assert not alert["acknowledged"]
        assert alert["status"] == "ACTIVE"

        # Acknowledge
        ack_res = alert_mgr.acknowledge_alert(alert["alert_id"], operator_name="Safety Director")
        assert ack_res is True
        assert alert["acknowledged"] is True
        assert alert["acknowledged_by"] == "Safety Director"
        assert alert["status"] == "ACKNOWLEDGED"
        print(f"[PASS] Test 08: Alert {alert['alert_id']} acknowledged by Safety Director.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 08: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 9: AlertManager Frame Snapshot Creation
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 09: AlertManager Frame Snapshot Capture...")
    try:
        alert_mgr = AlertManager()
        dummy_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        alert = alert_mgr.trigger_hazard_alert(
            location="CORRIDOR_2 <-> EXIT_B",
            hazard_type="FIRE",
            hazard_score=0.92,
            frame=dummy_frame
        )
        assert alert.get("snapshot_path") is not None
        snap_path = Path(alert["snapshot_path"])
        assert snap_path.exists(), f"Snapshot file not created at {snap_path}"
        assert snap_path.stat().st_size > 0
        print(f"[PASS] Test 09: Emergency snapshot verified at {snap_path.name}.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 09: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 10: AlertManager Duplicate Suppression
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 10: AlertManager Duplicate Suppression...")
    try:
        alert_mgr = AlertManager()
        alt1 = alert_mgr.trigger_hazard_alert(
            location="ROOM_A <-> CORRIDOR_1",
            hazard_type="SMOKE",
            hazard_score=0.45,
            cooldown_seconds=10.0
        )
        # Immediate subsequent trigger for same zone within cooldown
        alt2 = alert_mgr.trigger_hazard_alert(
            location="ROOM_A <-> CORRIDOR_1",
            hazard_type="SMOKE",
            hazard_score=0.52,
            cooldown_seconds=10.0
        )
        assert alt1["alert_id"] == alt2["alert_id"], "Expected duplicate suppression to reuse alert_id"
        assert len(alert_mgr.active_alerts) == 1, "Duplicate alert appended despite cooldown"
        assert alt1["hazard_score"] == 0.52, "Telemetry was not updated on duplicate suppression"
        print("[PASS] Test 10: Duplicate alert suppressed; existing record updated cleanly.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 10: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 11: MultiCameraSimulator Hybrid Binding
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 11: MultiCameraSimulator Hybrid Stream Binding...")
    try:
        sim = MultiCameraSimulator()
        sample_vid = DATA_DIR / "test" / "demo_cctv_feed.mp4"
        stream = VideoStreamHandler(source=str(sample_vid))
        stream.start_stream()

        sim.bind_stream("CAM-01", stream)
        cam01 = sim.get_camera("CAM-01")
        assert cam01["is_real_stream"] is True
        assert "LIVE" in cam01["source_type"]

        success, frame = sim.capture_frame("CAM-01")
        assert success and frame is not None
        assert cam01["status"] == "ONLINE"

        sim.unbind_stream("CAM-01")
        assert cam01["is_real_stream"] is False
        stream.release()
        print("[PASS] Test 11: Hybrid real stream binding and unbinding verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 11: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 12: Incident Export Formats (JSON, CSV, TXT)
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 12: Incident Audit Log Exports (JSON, CSV, TXT)...")
    try:
        alert_mgr = AlertManager()
        alert_mgr.trigger_hazard_alert("CORRIDOR_1", "FIRE", hazard_score=0.88)
        alert_mgr.register_route_change(["ROOM_A", "EXIT_A"], ["ROOM_A", "EXIT_B"], "Fire", 4.0, 6.0)

        json_out = alert_mgr.export_incidents_json()
        assert "AI Smart Building Emergency Command Center" in json_out

        csv_out = alert_mgr.export_guard_log_csv()
        assert "CORRIDOR_1" in csv_out and "FIRE" in csv_out

        txt_out = alert_mgr.export_summary_txt()
        assert "INCIDENT AUDIT SUMMARY" in txt_out
        print("[PASS] Test 12: JSON, CSV, and TXT incident export structures verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 12: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 13: Regression — First-Frame FIRE + EXIT_B Recommendation & Unified State
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 13: Regression — First-Frame FIRE + EXIT_B Recommendation & Unified Emergency State...")
    system_reg = None
    res_fire_first = None
    fire_frame = None
    try:
        system_reg = EvacuationSystem()
        flames_path = DATA_DIR / "test" / "flames_sample.jpg"
        fire_frame = cv2.imread(str(flames_path))
        assert fire_frame is not None, "Failed to load flames_sample.jpg"

        res_fire_first = system_reg.process_frame(
            frame=fire_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=False
        )
        em_state = res_fire_first.get("emergency_state", {})
        assert res_fire_first["fire_detected"] is True, "Expected fire_detected=True on flames_sample.jpg"
        assert res_fire_first["hazard_score"] >= 0.70, f"Expected hazard_score >= 0.70, got {res_fire_first['hazard_score']}"
        assert res_fire_first["recommended_exit"] == "EXIT_B", f"Expected EXIT_B, got {res_fire_first['recommended_exit']}"
        assert em_state.get("is_emergency") is True
        assert em_state.get("status_level") == "CRITICAL"
        assert em_state.get("hazard_type") == "FIRE"
        assert em_state.get("is_diverted_from_baseline") is True
        assert em_state.get("baseline_exit") == "EXIT_A"
        assert em_state.get("avoided_exit") == "EXIT_A"
        print(f"[PASS] Test 13: First-frame FIRE (Score: {res_fire_first['hazard_score']:.3f}) -> EXIT_B unified emergency state verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 13: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 14: Regression — Emergency Public Address (PA) Advisory Text
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 14: Regression — Emergency Public Address (PA) Advisory Text...")
    try:
        assert res_fire_first is not None, "Prerequisite Test 13 failed"
        pa_msg = res_fire_first.get("current_pa_message", "")
        assert pa_msg == res_fire_first["emergency_state"].get("pa_announcement", "")
        assert "nominal" not in pa_msg.lower(), f"PA message must NOT say nominal during active fire: {pa_msg}"
        assert "fire" in pa_msg.lower(), f"PA message must mention fire hazard: {pa_msg}"
        assert "EXIT_B" in pa_msg, f"PA message must direct occupants to EXIT_B: {pa_msg}"
        assert "CORRIDOR_1 <-> EXIT_A" in pa_msg, f"PA message must identify hazard zone: {pa_msg}"
        print(f"[PASS] Test 14: Emergency PA advisory verified: \"{pa_msg}\"")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 14: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 15: Regression — Exit Signage Consistency & Unknown-Safety Fallback
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 15: Regression — Exit Signage Consistency (CLOSED/AVOID vs SAFE ROUTE & Fallback)...")
    try:
        assert res_fire_first is not None, "Prerequisite Test 13 failed"
        signage = res_fire_first.get("dynamic_signage", {})
        exit_a_status = signage.get("EXIT_A", {}).get("status", "")
        exit_b_status = signage.get("EXIT_B", {}).get("status", "")
        assert exit_a_status == "CLOSED / AVOID", f"Expected EXIT_A status 'CLOSED / AVOID', got '{exit_a_status}'"
        assert exit_b_status == "SAFE ROUTE", f"Expected EXIT_B status 'SAFE ROUTE', got '{exit_b_status}'"

        # Verify fallback when route safety is unknown
        fallback_am = AlertManager()
        fallback_am.update_dynamic_signage(recommended_exit="NONE", dangerous_exit=["EXIT_A"])
        assert fallback_am.dynamic_signage["EXIT_A"]["status"] == "CLOSED / AVOID"
        assert "UNKNOWN" in fallback_am.dynamic_signage["EXIT_B"]["status"], (
            f"Expected UNKNOWN fallback for EXIT_B when recommended_exit='NONE', got {fallback_am.dynamic_signage['EXIT_B']['status']}"
        )
        print(f"[PASS] Test 15: Signage verified (EXIT_A={exit_a_status}, EXIT_B={exit_b_status}) + unknown-route fallback verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 15: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 16: Regression — Route-History Transitions (Only on Genuine Changes)
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 16: Regression — Route-History Transitions on Genuine Route Changes...")
    try:
        assert system_reg is not None and res_fire_first is not None and fire_frame is not None
        # 1. On initial frame, no prior route existed so route_history must be empty
        assert res_fire_first["route_changed"] is False
        assert len(system_reg.alert_manager.get_route_history()) == 0

        # 2. Repeating the same fire frame keeps the same route -> still 0 transitions
        res_fire_repeat = system_reg.process_frame(
            frame=fire_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=False
        )
        assert res_fire_repeat["route_changed"] is False
        assert len(system_reg.alert_manager.get_route_history()) == 0

        # 3. Transition to clear frame -> route genuinely changes from EXIT_B to EXIT_A
        clear_frame = np.zeros_like(fire_frame)
        res_clear = system_reg.process_frame(
            frame=clear_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=False
        )
        assert res_clear["recommended_exit"] == "EXIT_A"
        assert res_clear["route_changed"] is True
        hist_after_clear = system_reg.alert_manager.get_route_history()
        assert len(hist_after_clear) == 1
        assert hist_after_clear[0]["previous_exit"] == "EXIT_B"
        assert hist_after_clear[0]["new_exit"] == "EXIT_A"
        assert "nominal" in res_clear["current_pa_message"].lower()

        # 4. Transition back to fire frame -> route genuinely changes from EXIT_A to EXIT_B
        res_fire_transition = system_reg.process_frame(
            frame=fire_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=False
        )
        assert res_fire_transition["recommended_exit"] == "EXIT_B"
        assert res_fire_transition["route_changed"] is True
        hist_after_fire = system_reg.alert_manager.get_route_history()
        assert len(hist_after_fire) == 2
        latest_ev = hist_after_fire[0]
        assert latest_ev["previous_route"] == "ROOM_A -> CORRIDOR_1 -> EXIT_A"
        assert latest_ev["new_route"] == "ROOM_A -> CORRIDOR_1 -> CORRIDOR_2 -> EXIT_B"
        assert "timestamp" in latest_ev and "reason" in latest_ev and len(latest_ev["reason"]) > 0
        print(f"[PASS] Test 16: Route history transitions verified ({len(hist_after_fire)} genuine transitions recorded).")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 16: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 17: Regression — Security / Guard Notification Generation & Data
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 17: Regression — Security / Guard Notification Consistency...")
    try:
        assert system_reg is not None and res_fire_first is not None
        guard_logs = res_fire_first.get("security_guard_log", [])
        assert len(guard_logs) >= 1, "Expected guard_logs to be populated immediately on first-frame FIRE"
        fire_dispatches = [g for g in guard_logs if g.get("type") == "FIRE"]
        assert len(fire_dispatches) >= 1, "Expected a FIRE dispatch entry in guard_logs"
        top_fire = fire_dispatches[0]
        assert top_fire["location"] == "CORRIDOR_1 <-> EXIT_A"
        assert top_fire["recommended_exit"] == "EXIT_B"
        assert "EVACUATE VIA EXIT_B" in top_fire["action"]

        # Verify empty-state on fresh AlertManager
        empty_am = AlertManager()
        assert empty_am.get_guard_log() == []
        print(f"[PASS] Test 17: Guard notification log verified ({len(guard_logs)} entries; action: '{top_fire['action']}').")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 17: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 18: Regression — Complete emergency_state Across All Return Paths
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 18: Regression — Complete emergency_state Schema Across All Return Paths (Full + Frame-Skip)...")
    try:
        assert system_reg is not None and fire_frame is not None
        req_keys = EvacuationSystem.REQUIRED_EMERGENCY_STATE_KEYS
        assert len(req_keys) == 32, f"Expected 32 required emergency_state keys, got {len(req_keys)}"

        # 1. Full inference path on fire frame
        res_full = system_reg.process_frame(
            frame=fire_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=True,
            skip_interval=5
        )
        assert "emergency_state" in res_full
        for k in req_keys:
            assert k in res_full["emergency_state"], f"Missing key '{k}' in full-inference emergency_state"

        # 2. Frame-skip cached return path (normal cached_result)
        res_skip = system_reg.process_frame(
            frame=fire_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=True,
            skip_interval=5
        )
        assert "emergency_state" in res_skip
        for k in req_keys:
            assert k in res_skip["emergency_state"], f"Missing key '{k}' in frame-skip emergency_state"

        # 3. Frame-skip return path when cached_result had emergency_state stripped (stale cache recovery)
        if system_reg.cached_result is not None and "emergency_state" in system_reg.cached_result:
            del system_reg.cached_result["emergency_state"]
        res_skip_recovered = system_reg.process_frame(
            frame=fire_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=True,
            skip_interval=5
        )
        assert "emergency_state" in res_skip_recovered
        for k in req_keys:
            assert k in res_skip_recovered["emergency_state"], f"Missing key '{k}' in recovered frame-skip emergency_state"
        assert res_skip_recovered["emergency_state"]["recommended_exit"] == "EXIT_B"
        print(f"[PASS] Test 18: All {len(req_keys)} emergency_state keys verified on full, skipped, and cache-recovery return paths.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 18: {e}")
        failed += 1

    # -------------------------------------------------------------------------
    # TEST 19: Regression — Dashboard Hot-Reload & Legacy Result Synthesis
    # -------------------------------------------------------------------------
    print("\n[RUN] Test 19: Regression — Streamlit Cloud Hot-Reload & Legacy Result Synthesis...")
    try:
        from dashboard.app import ensure_complete_emergency_state, _ensure_fresh_system_modules
        assert system_reg is not None

        reloaded_cls = _ensure_fresh_system_modules()
        assert hasattr(reloaded_cls, "build_facility_emergency_state")
        assert hasattr(reloaded_cls, "ensure_emergency_state")

        # Simulate a legacy result dict returned by a stale pre-hot-reload instance without 'emergency_state'
        legacy_result = {
            "fire_detected": True,
            "smoke_detected": False,
            "hazard_score": 0.875,
            "people_detected": 4,
            "crowd_density": 0.20,
            "active_edge": ("CORRIDOR_1", "EXIT_A"),
            "updated_edge_cost": 9.75,
            "recommended_exit": "EXIT_B",
            "recommended_route": ["ROOM_A", "CORRIDOR_1", "CORRIDOR_2", "EXIT_B"],
            "recommended_route_str": "ROOM_A -> CORRIDOR_1 -> CORRIDOR_2 -> EXIT_B",
            "total_route_cost": 2.40,
            "route_changed": False,
            "previous_route": None,
        }
        # Also test fallback synthesis when system is a minimal legacy object without ensure_emergency_state
        class LegacySystemStub:
            def __init__(self, real_sys):
                self.building_graph = real_sys.building_graph
                self.optimizer = real_sys.optimizer
                self.alert_manager = AlertManager()

        legacy_stub = LegacySystemStub(system_reg)
        synthesized = ensure_complete_emergency_state(
            system=legacy_stub,
            result=legacy_result,
            source_room="ROOM_A",
            active_edge=("CORRIDOR_1", "EXIT_A")
        )
        assert "emergency_state" in legacy_result
        for k in EvacuationSystem.REQUIRED_EMERGENCY_STATE_KEYS:
            assert k in synthesized, f"Missing key '{k}' in synthesized emergency_state"
        assert synthesized["is_emergency"] is True
        assert synthesized["status_level"] == "CRITICAL"
        assert synthesized["hazard_type"] == "FIRE"
        assert synthesized["recommended_exit"] == "EXIT_B"
        assert synthesized["avoided_exit"] == "EXIT_A"
        assert synthesized["dynamic_signage"]["EXIT_A"]["status"] == "CLOSED / AVOID"
        assert synthesized["dynamic_signage"]["EXIT_B"]["status"] == "SAFE ROUTE"
        assert "nominal" not in synthesized["pa_announcement"].lower()
        assert "EXIT_B" in synthesized["pa_announcement"]
        print("[PASS] Test 19: Hot-reload module sync and deterministic legacy emergency_state synthesis verified.")
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 19: {e}")
        failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{total_tests} TESTS PASSED ({failed} FAILED)")
    print("=" * 70)

    if failed == 0:
        print(f"[+] ALL {total_tests} REAL-TIME PIPELINE & REGRESSION TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("[!] SOME TESTS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
