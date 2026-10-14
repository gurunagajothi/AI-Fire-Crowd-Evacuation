"""
Automated Test Suite: Image & Video Evaluation Workflows and Streamlit Cloud Safeguards.
Verifies:
1. Image evaluation workflow produces valid results with complete 32-key emergency_state schema.
2. Image evaluation handles corrupted, empty, or invalid bytes gracefully without crashes.
3. Video evaluation workflow processes frames, respects frame skip, enforces max frames, and exports H.264 MP4.
4. Video evaluation handles corrupted or unreadable video sources gracefully.
5. Streamlit Cloud environment detection and hardware camera safeguards (no auto-open of camera 0/1 or localhost RTSP).
6. File upload and benchmark evaluation workflows never attempt cv2.VideoCapture on hardware cameras.
7. Standby baseline state is available prior to explicit evaluation action without premature YOLO inference.
8. Incident audit reports (TXT, JSON, CSV) validate from evaluated image and video payloads.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DATA_DIR
from integration.evacuation_system import EvacuationSystem
from video.video_stream import (
    VideoStreamHandler,
    discover_available_cameras,
    is_cloud_environment,
)
from dashboard.app import (
    load_evacuation_system,
    process_image_evaluation,
    process_video_evaluation,
    encode_frames_to_h264_mp4,
    get_standby_baseline_result,
    ensure_complete_emergency_state,
    REQUIRED_EMERGENCY_STATE_KEYS,
)
from dashboard.components import (
    generate_incident_report_text,
    generate_incident_report_json,
    generate_incident_report_csv,
)


class UploadAndSafeguardsTestRunner:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def start_test(self, test_num: int, name: str):
        print(f"\n[RUN]  Test {test_num:02d}: {name}...", flush=True)

    def record(self, test_num: int, name: str, passed: bool, message: str = ""):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            print(f"[PASS] Test {test_num:02d}: {name} - {message}", flush=True)
        else:
            self.failed_tests += 1
            print(f"[FAIL] Test {test_num:02d}: {name} - {message}", flush=True)

    def run_all(self) -> int:
        print("=" * 70, flush=True)
        print("UPLOAD WORKFLOWS & STREAMLIT CLOUD SAFEGUARDS TEST SUITE", flush=True)
        print("=" * 70, flush=True)

        t_suite_start = time.time()

        # Load shared EvacuationSystem
        sys_inst, err = load_evacuation_system()
        assert err is None and sys_inst is not None, f"Failed to initialize EvacuationSystem: {err}"

        # -------------------------------------------------------------
        # Test 1: Image Evaluation on Valid Fire Benchmark
        # -------------------------------------------------------------
        self.start_test(1, "Image Evaluation on Valid Fire Benchmark (flames_sample.jpg)")
        try:
            img_path = DATA_DIR / "test" / "flames_sample.jpg"
            res, raw_f, err_eval = process_image_evaluation(
                system=sys_inst,
                image_input=img_path,
                active_edge=("CORRIDOR_1", "EXIT_A"),
                source_room="ROOM_A"
            )
            assert err_eval is None, f"Unexpected evaluation error: {err_eval}"
            assert res is not None, "Evaluation result is None"
            assert raw_f is not None and raw_f.shape[0] > 0, "Raw frame is empty"
            assert "emergency_state" in res, "Missing emergency_state in result"
            em = res["emergency_state"]
            assert len(em) == 32, f"Expected 32 emergency_state keys, got {len(em)}"
            for k in REQUIRED_EMERGENCY_STATE_KEYS:
                assert k in em, f"Missing required key: {k}"
            assert em["fire_detected"] is True, "Fire should be detected in flames_sample.jpg"
            assert em["recommended_exit"] == "EXIT_B", "Fire on EXIT_A corridor should divert to EXIT_B"
            assert res["annotated_frame"] is not None and res["annotated_frame"].shape == raw_f.shape
            self.record(1, "Image Evaluation (Fire)", True, f"Fire detected (Hazard: {em['hazard_score']:.3f}) -> Diverted to {em['recommended_exit']}")
        except Exception as e:
            self.record(1, "Image Evaluation (Fire)", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 2: Image Evaluation on Valid Crowd Benchmark
        # -------------------------------------------------------------
        self.start_test(2, "Image Evaluation on Valid Crowd Benchmark (crowd_sample.jpg)")
        try:
            crowd_path = DATA_DIR / "test" / "crowd_sample.jpg"
            res_c, raw_c, err_c = process_image_evaluation(
                system=sys_inst,
                image_input=crowd_path,
                active_edge=("CORRIDOR_1", "EXIT_A"),
                source_room="ROOM_A"
            )
            assert err_c is None, f"Unexpected error: {err_c}"
            assert res_c is not None, "Result is None"
            assert res_c["people_detected"] > 0, "Expected people detections in crowd_sample.jpg"
            em_c = res_c["emergency_state"]
            assert em_c["crowd_density"] > 0.0, "Expected non-zero crowd density"
            self.record(2, "Image Evaluation (Crowd)", True, f"Detected {res_c['people_detected']} occupants (Density: {em_c['crowd_density']:.3f})")
        except Exception as e:
            self.record(2, "Image Evaluation (Crowd)", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 3: Image Evaluation Corrupted & Invalid Inputs Handling
        # -------------------------------------------------------------
        self.start_test(3, "Image Evaluation Corrupted / Empty Bytes Handling")
        try:
            # Empty bytes
            res_e, raw_e, err_e = process_image_evaluation(sys_inst, b"")
            assert res_e is None and err_e is not None, "Empty bytes should return error"

            # Corrupted garbage bytes
            res_g, raw_g, err_g = process_image_evaluation(sys_inst, b"RIFF\x00\x00\x00NOT_AN_IMAGE_FILE")
            assert res_g is None and err_g is not None, "Garbage bytes should return error"

            # Non-existent file path
            res_nf, raw_nf, err_nf = process_image_evaluation(sys_inst, DATA_DIR / "non_existent_file.png")
            assert res_nf is None and err_nf is not None, "Non-existent file should return error"

            self.record(3, "Image Corrupted Inputs", True, "Handled empty bytes, garbage bytes, and non-existent file cleanly")
        except Exception as e:
            self.record(3, "Image Corrupted Inputs", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 4: Video Evaluation Workflow & Frame-Skipping
        # -------------------------------------------------------------
        self.start_test(4, "Video Evaluation Workflow & Sampling (demo_cctv_feed.mp4)")
        try:
            vid_path = DATA_DIR / "test" / "demo_cctv_feed.mp4"
            callback_calls = []

            def progress_cb(curr, total, cur_res):
                callback_calls.append((curr, total, cur_res.get("hazard_score", 0.0)))

            v_summary, mp4_bytes, err_v = process_video_evaluation(
                system=sys_inst,
                video_source=vid_path,
                active_edge=("CORRIDOR_1", "EXIT_A"),
                source_room="ROOM_A",
                frame_skip=3,
                max_frames=10,
                progress_callback=progress_cb
            )
            assert err_v is None, f"Video evaluation failed: {err_v}"
            assert v_summary is not None, "Video summary is None"
            assert v_summary["processed_frames_count"] == 10, f"Expected 10 processed frames, got {v_summary['processed_frames_count']}"
            assert v_summary["frame_skip"] == 3
            assert v_summary["peak_hazard_score"] >= 0.0
            assert v_summary["latest_result"] is not None
            assert len(callback_calls) == 10, f"Expected 10 callback calls, got {len(callback_calls)}"
            assert isinstance(mp4_bytes, bytes) and len(mp4_bytes) > 500, "Expected valid H.264 MP4 bytes"

            self.record(4, "Video Evaluation Workflow", True, f"Processed {v_summary['processed_frames_count']} frames (Skip: 3) -> Generated {len(mp4_bytes)} bytes H.264 MP4")
        except Exception as e:
            self.record(4, "Video Evaluation Workflow", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 5: Video Evaluation Corrupted File Handling
        # -------------------------------------------------------------
        self.start_test(5, "Video Evaluation Corrupted / Missing Source Handling")
        try:
            v_sum_bad, mp4_bad, err_bad = process_video_evaluation(
                system=sys_inst,
                video_source=b"INVALID_CORRUPTED_VIDEO_BYTES",
                frame_skip=1,
                max_frames=5
            )
            assert v_sum_bad is None and err_bad is not None
            assert mp4_bad == b""

            v_sum_ne, mp4_ne, err_ne = process_video_evaluation(
                system=sys_inst,
                video_source=DATA_DIR / "missing_video.mp4",
                frame_skip=1,
                max_frames=5
            )
            assert v_sum_ne is None and err_ne is not None
            self.record(5, "Video Corrupted Source", True, "Handled corrupted bytes and missing video file cleanly without crash")
        except Exception as e:
            self.record(5, "Video Corrupted Source", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 6: Streamlit Cloud Detection & Camera Safeguards
        # -------------------------------------------------------------
        self.start_test(6, "Streamlit Cloud Detection & Camera Hardware Safeguards")
        try:
            # Simulate Streamlit Cloud environment variable
            old_mode = os.environ.get("STREAMLIT_SHARING_MODE")
            try:
                os.environ["STREAMLIT_SHARING_MODE"] = "true"
                assert is_cloud_environment() is True, "is_cloud_environment should return True when STREAMLIT_SHARING_MODE is set"

                # Camera discovery should return [] on cloud
                cams = discover_available_cameras()
                assert cams == [], f"Expected [] discovered cameras on cloud, got {cams}"

                # VideoStreamHandler for camera index 0 must fail immediately without calling cv2.VideoCapture
                handler_0 = VideoStreamHandler(source=0)
                opened_0 = handler_0.start_stream()
                assert opened_0 is False, "Camera index 0 must be blocked on Streamlit Cloud"
                assert "Server camera index 0 is unavailable" in handler_0.error_message
                assert "Streamlit Cloud" in handler_0.error_message

                # VideoStreamHandler for camera index 1 must fail immediately
                handler_1 = VideoStreamHandler(source=1)
                opened_1 = handler_1.start_stream()
                assert opened_1 is False, "Camera index 1 must be blocked on Streamlit Cloud"
                assert "Server camera index 1 is unavailable" in handler_1.error_message

                # VideoStreamHandler for localhost RTSP must fail immediately on cloud
                handler_rtsp = VideoStreamHandler(source="rtsp://127.0.0.1:8554/live")
                opened_rtsp = handler_rtsp.start_stream()
                assert opened_rtsp is False, "Localhost RTSP must be blocked on Streamlit Cloud"
                assert "Localhost RTSP stream" in handler_rtsp.error_message
            finally:
                if old_mode is not None:
                    os.environ["STREAMLIT_SHARING_MODE"] = old_mode
                else:
                    os.environ.pop("STREAMLIT_SHARING_MODE", None)

            self.record(6, "Cloud Camera Safeguards", True, "Verified is_cloud_environment(), blocked cam 0/1, and blocked localhost RTSP")
        except Exception as e:
            self.record(6, "Cloud Camera Safeguards", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 7: Upload Modes Never Attempt Physical Camera Opens
        # -------------------------------------------------------------
        self.start_test(7, "File Upload Workflows Never Touch Physical Cameras (0/1/RTSP)")
        try:
            real_vc = cv2.VideoCapture

            def forbidden_vc(src, *args, **kwargs):
                if isinstance(src, int) and src in [0, 1, 2, 3, 4, 5]:
                    raise RuntimeError(f"VIOLATION: cv2.VideoCapture opened physical camera index {src} during file workflow!")
                if isinstance(src, str) and ("127.0.0.1" in src or "localhost" in src):
                    raise RuntimeError(f"VIOLATION: cv2.VideoCapture connected to localhost RTSP {src} during file workflow!")
                return real_vc(src, *args, **kwargs)

            cv2.VideoCapture = forbidden_vc
            try:
                # 1. Image evaluation on flames_sample.jpg
                res_i, _, _ = process_image_evaluation(sys_inst, DATA_DIR / "test" / "flames_sample.jpg")
                assert res_i is not None

                # 2. Video evaluation on demo_cctv_feed.mp4
                res_v, _, _ = process_video_evaluation(sys_inst, DATA_DIR / "test" / "demo_cctv_feed.mp4", max_frames=5)
                assert res_v is not None
            finally:
                cv2.VideoCapture = real_vc

            self.record(7, "Upload Mode Isolation", True, "Zero calls to physical cameras or localhost RTSP during image and video evaluation")
        except Exception as e:
            self.record(7, "Upload Mode Isolation", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 8: Standby Baseline State Available Without Premature YOLO Inference
        # -------------------------------------------------------------
        self.start_test(8, "Standby Baseline State (Gated Inference Protocol)")
        try:
            # Prior to user clicking evaluate, standby baseline state provides safe nominal route
            standby_res = get_standby_baseline_result(sys_inst, source_room="ROOM_A", active_edge=("CORRIDOR_1", "EXIT_A"))
            assert "emergency_state" in standby_res
            em_st = standby_res["emergency_state"]
            assert len(em_st) == 32
            assert em_st["is_emergency"] is False
            assert em_st["status_level"] == "NORMAL"
            assert em_st["recommended_exit"] == "EXIT_A"
            assert em_st["baseline_exit"] == "EXIT_A"
            assert em_st["hazard_score"] == 0.0
            assert em_st["people_count"] == 0
            self.record(8, "Standby Baseline State", True, "Nominal baseline state with 32 keys generated without premature inference")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.record(8, "Standby Baseline State", False, f"Error: {type(e).__name__}: {e}")

        # -------------------------------------------------------------
        # Test 9: Digital Incident Reports Generated From Evaluation Payload
        # -------------------------------------------------------------
        self.start_test(9, "Digital Incident Reports Validation (TXT, JSON, CSV)")
        try:
            payload = {
                "origin_room": "ROOM_A",
                "monitored_edge": "CORRIDOR_1 <-> EXIT_A",
                "fire_detected": "YES",
                "smoke_detected": "NO",
                "hazard_score": 0.875,
                "people_detected": 4,
                "crowd_density": 0.20,
                "risk_index": 52,
                "efficiency_index": 58,
                "baseline_route": "ROOM_A -> CORRIDOR_1 -> EXIT_A",
                "dynamic_route": "ROOM_A -> CORRIDOR_1 -> CORRIDOR_2 -> EXIT_B",
                "recommended_exit": "EXIT_B",
                "baseline_cost": 4.08,
                "route_cost": 5.28,
                "route_changed": "YES",
                "reroute_reason": "Corridor edge compromised by Fire Hazard",
                "guard_action": "Evacuate via EXIT_B",
                "signage_status": "EXIT_A CLOSED / EXIT_B SAFE ROUTE",
                "pa_message": "Fire hazard confirmed. Evacuate via EXIT_B.",
                "fire_ms": 11.2,
                "crowd_ms": 14.5,
                "dijkstra_ms": 0.4,
                "total_ms": 26.1,
                "fps": 38.3,
            }
            txt_report = generate_incident_report_text(payload)
            assert "AI EMERGENCY COMMAND CENTER" in txt_report
            assert "Recommended Exit:         EXIT_B" in txt_report

            json_report = generate_incident_report_json(payload)
            import json
            parsed = json.loads(json_report)
            assert parsed["recommended_exit"] == "EXIT_B"
            assert parsed["fire_detected"] == "YES"

            csv_report = generate_incident_report_csv(payload)
            assert "origin_room" in csv_report
            assert "hazard_score" in csv_report
            assert "0.875" in csv_report

            self.record(9, "Digital Incident Reports", True, "All 3 formats (TXT, JSON, CSV) validated from evaluation payload")
        except Exception as e:
            self.record(9, "Digital Incident Reports", False, f"Error: {e}")

        # -------------------------------------------------------------
        # FINAL SUMMARY REPORT
        # -------------------------------------------------------------
        t_total = time.time() - t_suite_start
        print("\n" + "=" * 70, flush=True)
        print("TEST SUMMARY", flush=True)
        print("=" * 70, flush=True)
        print(f"TOTAL TESTS:  {self.total_tests}", flush=True)
        print(f"PASSED:       {self.passed_tests}", flush=True)
        print(f"FAILED:       {self.failed_tests}", flush=True)
        print(f"TOTAL TIME:   {t_total:.2f}s", flush=True)
        print("=" * 70, flush=True)

        if self.failed_tests == 0:
            print("[+] ALL TESTS IN UPLOAD & SAFEGUARDS SUITE PASSED SUCCESSFULLY!\n", flush=True)
            return 0
        else:
            print(f"[!] {self.failed_tests} TESTS FAILED.\n", flush=True)
            return 1


if __name__ == "__main__":
    runner = UploadAndSafeguardsTestRunner()
    code = runner.run_all()
    sys.exit(code)
