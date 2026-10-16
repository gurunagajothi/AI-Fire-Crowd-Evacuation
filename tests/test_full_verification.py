"""
Comprehensive Automated Verification Test Suite for Tasks 2 through 10.
Validates:
Task 2: Image Upload Regression (JPG, JPEG, PNG, fire/smoke, person count, empty/corrupt files)
Task 3: Invalid File Handling (.txt file rejection, error messaging, resource cleanup)
Task 4: Video Upload (MP4 demo feed, frame processing, H.264 export, corrupt video handling)
Task 5: Webcam / Real-Time Monitoring (discovery, index failures, WebRTC processor, cloud safeguards)
Task 6: Evacuation Routing (normal route, fire diversion, cost formula, graph validity, invalid source)
Task 7: Light Mode / Dark Mode (CSS variables, dark theme toggle, matplotlib light/dark figures)
Task 8: XAI and Telemetry (Decision Assistant, Temporal Tracker, explainability calculations)
Task 9: Reports and Export (TXT, JSON, CSV generation, empty payload handling)
Task 10: End-to-End Regression & HTML Safety (XSS escaping, unindented HTML markdown)
"""

import os
import sys
import json
import csv
import io
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch

import cv2
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DATA_DIR, ALPHA, BETA, GAMMA
from integration.evacuation_system import EvacuationSystem
from integration.decision_assistant import EmergencyDecisionAssistant
from integration.hazard_trend import TemporalHazardTracker
from graph.multi_floor_graph import MultiFloorBuildingGraph
from video.video_stream import (
    VideoStreamHandler,
    discover_available_cameras,
    is_cloud_environment,
)
from dashboard.app import (
    load_evacuation_system,
    process_image_evaluation,
    process_video_evaluation,
    get_standby_baseline_result,
    ensure_complete_emergency_state,
    BrowserWebcamProcessor,
    REQUIRED_EMERGENCY_STATE_KEYS,
)
from dashboard.styles import get_custom_css
from dashboard.components import (
    generate_incident_report_text,
    generate_incident_report_json,
    generate_incident_report_csv,
    render_evacuation_guidance_card,
)


class ComprehensiveVerificationRunner:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []

    def start_test(self, num: int, name: str):
        print(f"\n[RUN]  Test {num:02d}: {name}...", flush=True)

    def record(self, num: int, name: str, passed: bool, message: str = ""):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"[PASS] Test {num:02d}: {name} - {message}", flush=True)
            self.results.append((num, name, "PASS", message))
        else:
            self.failed_tests += 1
            print(f"[FAIL] Test {num:02d}: {name} - {message}", flush=True)
            self.results.append((num, name, "FAIL", message))

    def run_all(self) -> int:
        print("=" * 75, flush=True)
        print("COMPREHENSIVE APPLICATION VERIFICATION TEST SUITE (TASKS 2-10)", flush=True)
        print("=" * 75, flush=True)
        t_start = time.time()

        # Initialize shared EvacuationSystem
        sys_inst, err = load_evacuation_system()
        assert err is None and sys_inst is not None, f"Failed to initialize EvacuationSystem: {err}"

        # ---------------------------------------------------------------------
        # TASK 2: Image Upload Regression
        # ---------------------------------------------------------------------
        # Test 1: Valid JPG fire & smoke benchmarks
        self.start_test(1, "Task 2 - Image Upload: Valid JPG Fire & Smoke Detection")
        try:
            flames_p = DATA_DIR / "test" / "flames_sample.jpg"
            res_f, raw_f, err_f = process_image_evaluation(sys_inst, flames_p, active_edge=("CORRIDOR_1", "EXIT_A"), source_room="ROOM_A")
            assert err_f is None and res_f is not None, f"Flames eval failed: {err_f}"
            assert res_f["fire_detected"] is True, "Fire should be detected in flames_sample.jpg"

            smoke_p = DATA_DIR / "test" / "fire_sample.jpg"
            res_s, raw_s, err_s = process_image_evaluation(sys_inst, smoke_p, active_edge=("CORRIDOR_1", "EXIT_A"), source_room="ROOM_A")
            assert err_s is None and res_s is not None, f"Smoke eval failed: {err_s}"
            assert res_s["smoke_detected"] is True, "Smoke should be detected in fire_sample.jpg"
            self.record(1, "Task 2 - Valid JPG Fire/Smoke Detection", True, f"Fire (Haz={res_f['hazard_score']:.3f}) and Smoke (Haz={res_s['hazard_score']:.3f}) confirmed")
        except Exception as e:
            self.record(1, "Task 2 - Valid JPG Fire/Smoke Detection", False, str(e))

        # Test 2: Valid JPG Crowd benchmark & Person count
        self.start_test(2, "Task 2 - Image Upload: Valid JPG Crowd Detection & Count")
        try:
            crowd_p = DATA_DIR / "test" / "crowd_sample.jpg"
            res_c, raw_c, err_c = process_image_evaluation(sys_inst, crowd_p, active_edge=("CORRIDOR_1", "EXIT_A"), source_room="ROOM_A")
            assert err_c is None and res_c is not None, f"Crowd eval failed: {err_c}"
            assert res_c["people_detected"] == 15, f"Expected 15 people, got {res_c['people_detected']}"
            assert res_c["crowd_density"] == 0.75, f"Expected density 0.75, got {res_c['crowd_density']}"
            self.record(2, "Task 2 - Valid JPG Crowd Count", True, f"Detected exactly {res_c['people_detected']} occupants, density={res_c['crowd_density']:.3f}")
        except Exception as e:
            self.record(2, "Task 2 - Valid JPG Crowd Count", False, str(e))

        # Test 3: Valid PNG format upload
        self.start_test(3, "Task 2 - Image Upload: Valid PNG Format Support")
        try:
            temp_png = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            dummy_img = np.full((300, 400, 3), 128, dtype=np.uint8)
            cv2.rectangle(dummy_img, (50, 50), (250, 250), (255, 255, 255), -1)
            cv2.imwrite(temp_png.name, dummy_img)
            temp_png.close()

            res_png, raw_png, err_png = process_image_evaluation(sys_inst, temp_png.name, active_edge=("CORRIDOR_1", "EXIT_A"), source_room="ROOM_A")
            Path(temp_png.name).unlink(missing_ok=True)

            assert err_png is None and res_png is not None, f"PNG eval failed: {err_png}"
            assert raw_png.shape == (300, 400, 3), "Decoded PNG frame dimensions match"
            assert "emergency_state" in res_png
            self.record(3, "Task 2 - Valid PNG Support", True, f"Processed {raw_png.shape[1]}x{raw_png.shape[0]} PNG cleanly")
        except Exception as e:
            self.record(3, "Task 2 - Valid PNG Support", False, str(e))

        # Test 4: Empty, invalid, and corrupted image files
        self.start_test(4, "Task 2 - Image Upload: Corrupted, Empty, and Truncated Files")
        try:
            # Empty bytes
            res_0, _, err_0 = process_image_evaluation(sys_inst, b"")
            assert res_0 is None and err_0 is not None, "Empty bytes should return error"

            # Corrupt garbage header
            res_corrupt, _, err_corrupt = process_image_evaluation(sys_inst, b"\x89PNG\r\n\x1a\nCORRUPTED_TRUNCATED_BYTES")
            assert res_corrupt is None and err_corrupt is not None, "Garbage bytes should return error"

            # Tiny 2x2 image (below 5x5 minimum threshold)
            tiny_img = np.zeros((2, 2, 3), dtype=np.uint8)
            res_tiny, _, err_tiny = process_image_evaluation(sys_inst, tiny_img)
            assert res_tiny is None and err_tiny is not None, "Tiny image should return error"

            self.record(4, "Task 2 - Image Error Handling", True, f"Empty/corrupt inputs safely rejected without crash (err='{err_corrupt}')")
        except Exception as e:
            self.record(4, "Task 2 - Image Error Handling", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 3: Invalid File Handling (.txt file)
        # ---------------------------------------------------------------------
        self.start_test(5, "Task 3 - Invalid File Handling: Rejection of .txt File & Cleanup")
        try:
            temp_txt = tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w")
            temp_txt.write("This is a plain text file, not a CCTV image or surveillance stream.\n")
            temp_txt.close()

            # Pass text file path to image evaluator
            res_txt, raw_txt, err_txt = process_image_evaluation(sys_inst, temp_txt.name)
            assert res_txt is None, "Text file should not produce analysis result"
            assert raw_txt is None
            assert err_txt is not None
            assert "Invalid or unreadable image file" in err_txt, f"Unexpected error: {err_txt}"

            # Pass text file to video evaluator
            v_res_txt, v_bytes_txt, v_err_txt = process_video_evaluation(sys_inst, temp_txt.name)
            assert v_res_txt is None
            assert v_bytes_txt == b""
            assert v_err_txt is not None

            # Clean up test text file
            Path(temp_txt.name).unlink(missing_ok=True)
            assert not Path(temp_txt.name).exists(), "Temporary text file unlinked"

            self.record(5, "Task 3 - Unsupported .txt Rejection", True, f"Rejected .txt without crashing: '{err_txt}'")
        except Exception as e:
            self.record(5, "Task 3 - Unsupported .txt Rejection", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 4: Video Upload
        # ---------------------------------------------------------------------
        self.start_test(6, "Task 4 - Video Upload: Real MP4 Processing & H.264 Export (demo_cctv_feed.mp4)")
        try:
            vid_path = DATA_DIR / "test" / "demo_cctv_feed.mp4"
            assert vid_path.exists(), f"Demo video feed not found at {vid_path}"

            cb_frames = []
            def test_cb(cur, tot, r):
                cb_frames.append(cur)

            v_sum, mp4_out, err_v = process_video_evaluation(
                system=sys_inst,
                video_source=vid_path,
                active_edge=("CORRIDOR_1", "EXIT_A"),
                source_room="ROOM_A",
                frame_skip=2,
                max_frames=8,
                progress_callback=test_cb
            )
            assert err_v is None, f"Video evaluation failed: {err_v}"
            assert v_sum is not None
            assert v_sum["processed_frames_count"] == 8
            assert len(cb_frames) == 8
            assert len(mp4_out) > 1000, f"Expected non-empty H.264 MP4 bytes, got {len(mp4_out)} bytes"
            assert v_sum["primary_recommended_exit"] in ["EXIT_A", "EXIT_B"]
            self.record(6, "Task 4 - Real MP4 Processing", True, f"Processed {v_sum['processed_frames_count']} frames -> {len(mp4_out)} bytes H.264 MP4")
        except Exception as e:
            self.record(6, "Task 4 - Real MP4 Processing", False, str(e))

        self.start_test(7, "Task 4 - Video Upload: Corrupted & Empty Video Rejection")
        try:
            # Corrupted bytes passed as video stream
            corrupt_v_bytes = b"NOT_A_VIDEO_STREAM_DATA"
            v_sum_c, mp4_c, err_vc = process_video_evaluation(sys_inst, corrupt_v_bytes)
            assert v_sum_c is None and mp4_c == b""
            assert err_vc is not None
            assert "Unable to open video file" in err_vc or "No frames" in err_vc

            self.record(7, "Task 4 - Corrupt Video Rejection", True, f"Handled corrupt video safely: '{err_vc}'")
        except Exception as e:
            self.record(7, "Task 4 - Corrupt Video Rejection", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 5: Webcam / Real-Time Monitoring
        # ---------------------------------------------------------------------
        self.start_test(8, "Task 5 - Webcam: Camera Discovery & Invalid Index Graceful Handling")
        try:
            # 1. Discover cameras with safe bounds (0..2)
            cams = discover_available_cameras(max_index=2)
            assert isinstance(cams, list), "Expected list of camera descriptors"

            # 2. Test VideoStreamHandler on an intentionally invalid/nonexistent index (99)
            handler_invalid = VideoStreamHandler(source=99)
            res_open = handler_invalid.start_stream()
            assert res_open is False, "Camera 99 must not report opened"
            assert handler_invalid.status == "ERROR", "Status should be ERROR on invalid source"
            assert handler_invalid.cap is None, "VideoCapture should be None after failed open"
            assert "Unable to open video source: 99" in handler_invalid.error_message
            handler_invalid.release()

            # 3. Test VideoStreamHandler on invalid RTSP URL
            handler_rtsp_bad = VideoStreamHandler(source="rtsp://invalid.dummy.cctv/stream1")
            res_rtsp = handler_rtsp_bad.start_stream()
            assert res_rtsp is False
            assert handler_rtsp_bad.status == "ERROR"
            handler_rtsp_bad.release()

            self.record(8, "Task 5 - Camera Discovery & Fallbacks", True, f"Invalid index 99 returned ERROR: '{handler_invalid.error_message}'")
        except Exception as e:
            self.record(8, "Task 5 - Camera Discovery & Fallbacks", False, str(e))

        self.start_test(9, "Task 5 - Webcam: WebRTC VideoProcessor Thread-Safety")
        try:
            processor = BrowserWebcamProcessor()
            f_init, count_init = processor.get_latest_frame()
            assert f_init is None and count_init == 0

            # Simulate incoming frame
            fake_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            fake_bgr[100:200, 100:200] = [0, 0, 255] # Red square
            with processor._lock:
                processor.latest_frame = fake_bgr.copy()
                processor.frame_count = 1

            ret_frame, ret_count = processor.get_latest_frame()
            assert ret_count == 1
            assert ret_frame is not None and ret_frame.shape == (480, 640, 3)
            # Verify frame is a decoupled copy
            assert ret_frame is not processor.latest_frame

            self.record(9, "Task 5 - WebRTC Processor Safety", True, "Thread-safe frame access & frame copying verified")
        except Exception as e:
            self.record(9, "Task 5 - WebRTC Processor Safety", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 6: Evacuation Routing
        # ---------------------------------------------------------------------
        self.start_test(10, "Task 6 - Evacuation Routing: Normal vs Emergency Fire Diversion")
        try:
            # Baseline Normal Condition (Hazard=0.0, Density=0.0)
            sys_inst.building_graph.create_demo_building()
            norm_res = sys_inst.optimizer.find_safest_route(
                graph=sys_inst.building_graph.get_graph(),
                source="ROOM_A",
                exits=["EXIT_A", "EXIT_B"]
            )
            normal_exit = norm_res["selected_exit"]
            normal_route = norm_res["path"]
            normal_cost = norm_res["total_cost"]

            assert normal_exit == "EXIT_A", f"Normal exit should be EXIT_A, got {normal_exit}"
            assert normal_route == ["ROOM_A", "CORRIDOR_1", "EXIT_A"]
            # Cost formula: 0.2 * 10.20 + 0.2 * 10.20 = 4.08
            assert abs(normal_cost - 4.08) < 1e-3, f"Expected normal cost 4.08, got {normal_cost}"

            # Emergency Fire on North Corridor (CORRIDOR_1 <-> EXIT_A)
            sys_inst.building_graph.update_edge_hazard("CORRIDOR_1", "EXIT_A", 0.875)
            em_res = sys_inst.optimizer.find_safest_route(
                graph=sys_inst.building_graph.get_graph(),
                source="ROOM_A",
                exits=["EXIT_A", "EXIT_B"]
            )
            em_exit = em_res["selected_exit"]
            em_route = em_res["path"]
            em_cost = em_res["total_cost"]

            assert em_exit == "EXIT_B", f"Emergency exit should divert to EXIT_B, got {em_exit}"
            assert em_route == ["ROOM_A", "CORRIDOR_1", "CORRIDOR_2", "EXIT_B"]
            # Detour cost: 2.04 + 1.20 + 2.04 = 5.28
            assert abs(em_cost - 5.28) < 1e-3, f"Expected diverted cost 5.28, got {em_cost}"

            # Graph path validity: check all edges exist in the actual NetworkX graph
            nx_g = sys_inst.building_graph.get_graph()
            for i in range(len(em_route) - 1):
                u, v = em_route[i], em_route[i + 1]
                assert nx_g.has_edge(u, v), f"Edge {u} <-> {v} does not exist in building graph!"

            self.record(10, "Task 6 - Dynamic Dijkstra Routing", True, f"Normal: {normal_exit} (Cost: {normal_cost:.2f}) -> Diverted: {em_exit} (Cost: {em_cost:.2f})")
        except Exception as e:
            self.record(10, "Task 6 - Dynamic Dijkstra Routing", False, str(e))

        self.start_test(11, "Task 6 - Evacuation Routing: Cost Formula & Edge Cases")
        try:
            # Verify Cost(e) = alpha*Hazard + beta*Density + gamma*Length
            alpha, beta, gamma = 5.0, 2.0, 0.2
            test_haz, test_den, test_len = 0.6, 0.4, 10.0
            expected_c = (alpha * test_haz) + (beta * test_den) + (gamma * test_len) # 3.0 + 0.8 + 2.0 = 5.8
            # DynamicDijkstraOptimizer signature is calculate_edge_cost(hazard, density, length)
            calculated_c = sys_inst.optimizer.calculate_edge_cost(test_haz, test_den, test_len)
            assert abs(calculated_c - expected_c) < 1e-4, f"Formula mismatch: {calculated_c} vs {expected_c}"

            # Invalid source room handling: find_safest_route raises KeyError for missing source node
            key_err_raised = False
            try:
                sys_inst.optimizer.find_safest_route(
                    graph=sys_inst.building_graph.get_graph(),
                    source="NON_EXISTENT_ROOM",
                    exits=["EXIT_A", "EXIT_B"]
                )
            except KeyError:
                key_err_raised = True
            assert key_err_raised is True, "Invalid source node must raise KeyError"

            # Reset graph to clean demo baseline
            sys_inst.building_graph.create_demo_building()
            self.record(11, "Task 6 - Cost Formula & Edge Cases", True, f"Formula {expected_c:.2f} exact & invalid source handled safely via KeyError")
        except Exception as e:
            self.record(11, "Task 6 - Cost Formula & Edge Cases", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 7: Light Mode / Dark Mode
        # ---------------------------------------------------------------------
        self.start_test(12, "Task 7 - Light/Dark Theme: CSS Variables & Visual Contrast")
        try:
            css_light = get_custom_css(dark_mode=False)
            css_dark = get_custom_css(dark_mode=True)

            # Assert CSS output
            assert "--bg-base: #F8FAFC" in css_light, "Light mode base background mismatch"
            assert "--text-primary: #0F172A" in css_light, "Light mode primary text mismatch"
            assert "--bg-base: #0B0F19" in css_dark, "Dark mode base background mismatch"
            assert "--text-primary: #F8FAFC" in css_dark, "Dark mode primary text mismatch"

            # Assert building graph renderer supports both themes
            fig_l = sys_inst.building_graph.render_evacuation_map(
                recommended_route=["ROOM_A", "CORRIDOR_1", "EXIT_A"],
                source_room="ROOM_A",
                dark_mode=False
            )
            assert fig_l is not None, "Light mode map figure is None"

            fig_d = sys_inst.building_graph.render_evacuation_map(
                recommended_route=["ROOM_A", "CORRIDOR_1", "EXIT_A"],
                source_room="ROOM_A",
                dark_mode=True
            )
            assert fig_d is not None, "Dark mode map figure is None"

            # Assert multi-floor graph renderer supports both themes
            mf = MultiFloorBuildingGraph()
            fig_mf_l = mf.render_floor_map(floor=0, recommended_route=["G_ROOM_A", "G_CORRIDOR_1", "G_EXIT_A"], dark_mode=False)
            fig_mf_d = mf.render_floor_map(floor=0, recommended_route=["G_ROOM_A", "G_CORRIDOR_1", "G_EXIT_A"], dark_mode=True)
            assert fig_mf_l is not None and fig_mf_d is not None

            self.record(12, "Task 7 - Dual Theme System", True, "Light & Dark CSS themes and Matplotlib visual twin figures validated")
        except Exception as e:
            self.record(12, "Task 7 - Dual Theme System", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 8: XAI and Telemetry
        # ---------------------------------------------------------------------
        self.start_test(13, "Task 8 - XAI & Telemetry: Emergency Decision Assistant & Hazard Tracker")
        try:
            # 1. EmergencyDecisionAssistant
            asst = EmergencyDecisionAssistant()
            assert len(asst.preset_questions) >= 5, "Expected preset research questions"

            # Nominal query
            state_normal = {
                "fire_detected": False,
                "smoke_detected": False,
                "hazard_score": 0.0,
                "people_detected": 0,
                "crowd_density": 0.0,
                "route_changed": False,
                "recommended_exit": "EXIT_A",
                "normal_exit": "EXIT_A",
                "recommended_route_str": "ROOM_A -> CORRIDOR_1 -> EXIT_A",
                "active_edge": ("CORRIDOR_1", "EXIT_A"),
                "total_route_cost": 4.08,
                "edge_cost": 2.04,
                "risk_index": 5
            }
            ans_norm = asst.answer_query("Why did the route change?", state_normal)
            assert "route has NOT changed" in ans_norm, f"Unexpected answer: {ans_norm}"

            # Emergency query
            state_em = {
                "fire_detected": True,
                "smoke_detected": False,
                "hazard_score": 0.875,
                "people_detected": 0,
                "crowd_density": 0.0,
                "route_changed": True,
                "recommended_exit": "EXIT_B",
                "normal_exit": "EXIT_A",
                "recommended_route_str": "ROOM_A -> CORRIDOR_1 -> CORRIDOR_2 -> EXIT_B",
                "active_edge": ("CORRIDOR_1", "EXIT_A"),
                "total_route_cost": 5.28,
                "edge_cost": 6.42,
                "risk_index": 55
            }
            ans_em = asst.answer_query("Why did the route change?", state_em)
            assert "fire hazard" in ans_em and "EXIT_B" in ans_em, f"Unexpected answer: {ans_em}"

            # Risk query
            ans_risk = asst.answer_query("What is the current risk level?", state_em)
            assert "Risk Index" in ans_risk, f"Unexpected answer: {ans_risk}"

            # 2. TemporalHazardTracker (anomaly suppression & consecutive confirmation)
            tracker = TemporalHazardTracker(confirmation_threshold=3, hazard_threshold=0.35)
            # Frame 1: transient spike -> should be suppressed
            st1 = tracker.update(0.85)
            assert st1["is_confirmed"] is False, "Single spike should not trigger confirmed hazard"
            # Frames 2 and 3: sustained spike -> should trigger confirmed hazard
            tracker.update(0.85)
            st3 = tracker.update(0.85)
            assert st3["is_confirmed"] is True, "3 consecutive frames should trigger confirmed hazard"

            self.record(13, "Task 8 - Decision Assistant & Tracker", True, "Decision Assistant rule-based Q&A and 3-frame temporal confirmation validated")
        except Exception as e:
            self.record(13, "Task 8 - Decision Assistant & Tracker", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 9: Reports and Export
        # ---------------------------------------------------------------------
        self.start_test(14, "Task 9 - Reports & Export: TXT, JSON, and CSV Audit Generators")
        try:
            sample_payload = {
                "origin_room": "ROOM_A",
                "monitored_edge": "CORRIDOR_1 ↔ EXIT_A",
                "fire_detected": "YES",
                "smoke_detected": "NO",
                "hazard_score": 0.875,
                "people_detected": 0,
                "crowd_density": 0.0,
                "risk_index": 44,
                "efficiency_index": 65,
                "baseline_route": "ROOM_A -> CORRIDOR_1 -> EXIT_A",
                "dynamic_route": "ROOM_A -> CORRIDOR_1 -> CORRIDOR_2 -> EXIT_B",
                "recommended_exit": "EXIT_B",
                "baseline_cost": 4.08,
                "route_cost": 5.28,
                "route_changed": "YES",
                "reroute_reason": "Dynamic Dijkstra diverted route: CORRIDOR_1 <-> EXIT_A has fire hazard (Score: 0.875)",
                "guard_action": "Direct occupants to EXIT_B",
                "signage_status": "Dynamic guidance active",
                "pa_message": "Attention occupants, evacuate toward EXIT_B",
                "fire_ms": 135.2,
                "crowd_ms": 28.4,
                "dijkstra_ms": 0.42,
                "total_ms": 164.02,
                "fps": 18.2
            }

            # 1. TXT report
            txt_rep = generate_incident_report_text(sample_payload)
            assert "DIGITAL INCIDENT REPORT" in txt_rep, "Header missing from TXT report"
            assert "EXIT_B" in txt_rep, "Recommended exit missing from TXT report"
            assert "0.875" in txt_rep, "Hazard score missing from TXT report"

            # 2. JSON report
            json_rep = generate_incident_report_json(sample_payload)
            parsed_json = json.loads(json_rep)
            assert parsed_json["recommended_exit"] == "EXIT_B", "JSON missing recommended_exit"
            assert parsed_json["hazard_score"] == 0.875, "JSON missing hazard_score"

            # 3. CSV report (filter any empty lines from CRLF output)
            csv_rep = generate_incident_report_csv(sample_payload)
            reader = csv.reader(io.StringIO(csv_rep))
            rows = [r for r in reader if r]
            assert len(rows) == 2, f"Expected 2 CSV rows (header + data), got {len(rows)}"
            assert "origin_room" in rows[0], "CSV header missing origin_room"
            assert "ROOM_A" in rows[1], "CSV data missing ROOM_A"

            # 4. Handle empty payload safely
            txt_empty = generate_incident_report_text({})
            assert len(txt_empty) > 100, "Empty payload TXT report unexpectedly short"
            json_empty = json.loads(generate_incident_report_json({}))
            assert isinstance(json_empty, dict), "Empty payload JSON not a dict"

            self.record(14, "Task 9 - Incident Audit Formats", True, "TXT, JSON, and CSV exports verified with valid structure and empty fallback")
        except Exception as e:
            self.record(14, "Task 9 - Incident Audit Formats", False, str(e))

        # ---------------------------------------------------------------------
        # TASK 10: End-to-End Regression & HTML Safety
        # ---------------------------------------------------------------------
        self.start_test(15, "Task 10 - Regression: HTML Tag Safety & XSS Escaping")
        try:
            with patch("streamlit.markdown") as mock_md:
                malicious_input = '<script>alert("xss")</script>'
                render_evacuation_guidance_card(
                    recommended_exit=malicious_input,
                    recommended_route=["ROOM_A", malicious_input],
                    is_emergency=True,
                    adaptation_reason='Hazard in <corridor> & "danger"',
                    hazard_zone="ZONE_<1>&EXIT",
                    avoided_exit=malicious_input,
                )
                mock_md.assert_called_once()
                rendered_html = mock_md.call_args[0][0]

                # Dynamic inputs must be escaped
                assert "<script>" not in rendered_html, "Raw <script> tag was not escaped"
                assert "&lt;script&gt;" in rendered_html, "Escaped script tag missing from output"
                assert "&lt;corridor&gt;" in rendered_html, "Escaped corridor tag missing"
                assert "&amp;" in rendered_html, "Ampersand not escaped"

                # Framework structural tags must remain intact HTML
                assert '<div class="evac-guidance-emergency">' in rendered_html, "Card container HTML missing"
                assert "<strong>Compromised Corridor Zone:</strong>" in rendered_html, "Framework <strong> tag broken"
                assert mock_md.call_args[1].get("unsafe_allow_html") is True, "Must use unsafe_allow_html=True"
                assert not any(line.startswith("    ") for line in rendered_html.splitlines()), "Found indentation that would trigger code block"

            self.record(15, "Task 10 - HTML Safety & XSS Escaping", True, "Card renders unindented HTML, escapes dynamic XSS strings, preserves styles")
        except Exception as e:
            self.record(15, "Task 10 - HTML Safety & XSS Escaping", False, str(e))

        # -------------------------------------------------------------
        # Summary
        # -------------------------------------------------------------
        dt = time.time() - t_start
        print("\n" + "=" * 75, flush=True)
        print("COMPREHENSIVE VERIFICATION RESULTS SUMMARY", flush=True)
        print("=" * 75, flush=True)
        print(f"TOTAL TESTS:  {self.total_tests}")
        print(f"PASSED:       {self.passed_tests}")
        print(f"FAILED:       {self.failed_tests}")
        print(f"TOTAL TIME:   {dt:.2f}s")
        print("=" * 75, flush=True)

        return 0 if self.failed_tests == 0 else 1


if __name__ == "__main__":
    runner = ComprehensiveVerificationRunner()
    sys.exit(runner.run_all())
