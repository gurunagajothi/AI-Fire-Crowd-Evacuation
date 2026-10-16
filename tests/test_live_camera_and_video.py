"""
Unit and integration tests for Live Camera Detection and Recorded Video Analysis.
Tests verify:
1. Recorded video analysis pipeline (MP4, progress callback, peak fire confidence, occupants, hazard metrics).
2. Video error handling for corrupt and empty files.
3. BrowserWebcamProcessor thread-safe lifecycle and active AI perception toggling.
4. Live results extraction and Dynamic Dijkstra routing under hazard conditions.
5. Cloud environment webcam safeguard preventing physical cv2.VideoCapture(0) on cloud servers.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import DATA_DIR
from integration.evacuation_system import EvacuationSystem
from video.video_stream import is_cloud_environment, VideoStreamHandler
from dashboard.app import (
    BrowserWebcamProcessor,
    process_video_evaluation,
    ensure_complete_emergency_state,
)


class TestRecordedVideoAnalysis(unittest.TestCase):
    """Tests recorded video analysis and progress reporting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.system = EvacuationSystem()
        cls.demo_video = DATA_DIR / "test" / "demo_cctv_feed.mp4"

    def test_demo_cctv_video_processing(self) -> None:
        """Verifies frame-by-frame processing, progress telemetry, and peak metrics."""
        self.assertTrue(self.demo_video.exists(), f"Demo video missing at {self.demo_video}")

        progress_calls: List[Dict[str, Any]] = []

        def on_progress(curr: int, total: int, cur_res: Dict[str, Any]) -> None:
            progress_calls.append({
                "curr": curr,
                "total": total,
                "hazard_score": cur_res.get("hazard_score", 0.0),
                "fire_conf": cur_res.get("fire_conf", 0.0),
                "people_count": cur_res.get("people_count", 0),
                "crowd_density": cur_res.get("crowd_density", 0.0),
                "recommended_exit": cur_res.get("recommended_exit", "NONE"),
            })

        summary, mp4_bytes, err = process_video_evaluation(
            system=self.system,
            video_source=str(self.demo_video),
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            frame_skip=2,
            max_frames=6,
            progress_callback=on_progress,
        )

        self.assertIsNone(err, f"Expected no error, got: {err}")
        self.assertIsNotNone(summary)
        self.assertEqual(summary["processed_frames_count"], 6)
        self.assertEqual(len(progress_calls), 6)
        self.assertIn("peak_fire_conf", summary)
        self.assertIn("peak_people_count", summary)
        self.assertIn("peak_crowd_density", summary)
        self.assertIn("peak_hazard_score", summary)
        self.assertIn(summary["primary_recommended_exit"], ["EXIT_A", "EXIT_B"])
        self.assertGreater(len(mp4_bytes), 500, "Expected non-empty H.264 MP4 export bytes")

        # Verify progress callback payloads contain all required UI metrics
        for call in progress_calls:
            self.assertIn("curr", call)
            self.assertIn("total", call)
            self.assertIn("hazard_score", call)
            self.assertIn("fire_conf", call)
            self.assertIn("people_count", call)
            self.assertIn("crowd_density", call)
            self.assertIn("recommended_exit", call)

    def test_corrupted_and_empty_video_handling(self) -> None:
        """Verifies graceful error reporting when video file is corrupt or unopenable."""
        # Test 1: Empty byte stream
        summary_empty, bytes_empty, err_empty = process_video_evaluation(self.system, b"")
        self.assertIsNone(summary_empty)
        self.assertEqual(bytes_empty, b"")
        self.assertIsNotNone(err_empty)
        self.assertIn("Unable to open video", err_empty)

        # Test 2: Corrupted header bytes
        corrupt_bytes = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00CORRUPTED_STREAM_PAYLOAD"
        summary_corrupt, bytes_corrupt, err_corrupt = process_video_evaluation(self.system, corrupt_bytes)
        self.assertIsNone(summary_corrupt)
        self.assertEqual(bytes_corrupt, b"")
        self.assertIsNotNone(err_corrupt)

        # Test 3: Non-existent file path
        summary_nonexist, bytes_nonexist, err_nonexist = process_video_evaluation(self.system, "non_existent_video_path.mp4")
        self.assertIsNone(summary_nonexist)
        self.assertEqual(bytes_nonexist, b"")
        self.assertIsNotNone(err_nonexist)


class TestBrowserWebcamProcessor(unittest.TestCase):
    """Tests the thread-safe WebRTC VideoProcessor and AI perception pipeline."""

    def setUp(self) -> None:
        self.processor = BrowserWebcamProcessor()
        self.system = EvacuationSystem()
        self.processor.system = self.system

    def test_initial_state(self) -> None:
        """Verifies initial idle state before streaming starts."""
        frame, count = self.processor.get_latest_frame()
        self.assertIsNone(frame)
        self.assertEqual(count, 0)
        self.assertFalse(self.processor.is_active)

        raw, ann, res, f_cnt = self.processor.get_latest_data()
        self.assertIsNone(raw)
        self.assertIsNone(ann)
        self.assertIsNone(res)
        self.assertEqual(f_cnt, 0)

    def test_toggle_detection_active(self) -> None:
        """Verifies active AI perception toggle state."""
        self.processor.set_detection_active(True)
        self.assertTrue(self.processor.is_active)
        self.processor.set_detection_active(False)
        self.assertFalse(self.processor.is_active)

    def test_frame_processing_and_decoupling(self) -> None:
        """Simulates incoming video frame and verifies decoupled memory copies."""
        fake_bgr = np.zeros((240, 320, 3), dtype=np.uint8)
        fake_bgr[50:100, 50:100] = [0, 165, 255] # Mock orange patch

        # Simulate frame arrival in processor
        with self.processor._lock:
            self.processor.latest_frame = fake_bgr.copy()
            self.processor.latest_annotated_frame = fake_bgr.copy()
            self.processor.latest_result = {
                "fire_detected": False,
                "people_detected": 0,
                "hazard_score": 0.0,
                "recommended_exit": "EXIT_A",
            }
            self.processor.frame_count = 5

        # Test backward-compatible method
        frame_copy, count = self.processor.get_latest_frame()
        self.assertEqual(count, 5)
        self.assertIsNotNone(frame_copy)
        self.assertIsNot(frame_copy, self.processor.latest_frame)

        # Test decoupled data retrieval
        raw, ann, res, f_cnt = self.processor.get_latest_data()
        self.assertEqual(f_cnt, 5)
        self.assertIsNotNone(raw)
        self.assertIsNotNone(ann)
        self.assertIsNotNone(res)
        self.assertIsNot(raw, self.processor.latest_frame)
        self.assertIsNot(ann, self.processor.latest_annotated_frame)
        self.assertEqual(res["recommended_exit"], "EXIT_A")


class TestLiveResultsAndRouting(unittest.TestCase):
    """Tests emergency exit diversion under fire conditions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.system = EvacuationSystem()

    def test_nominal_vs_hazard_routing(self) -> None:
        """Verifies that hazard on North Exit Corridor (EXIT_A) safely diverts to EXIT_B."""
        # 1. Baseline Nominal State
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        nom_res = self.system.process_frame(
            frame=blank_frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A",
            enable_frame_skip=False
        )
        ensure_complete_emergency_state(self.system, nom_res, "ROOM_A", ("CORRIDOR_1", "EXIT_A"))
        nom_em = nom_res["emergency_state"]
        self.assertEqual(nom_em["recommended_exit"], "EXIT_A")
        self.assertFalse(nom_em["is_diverted_from_baseline"])

        # 2. Fire Hazard State on North Exit Corridor
        fire_sample = DATA_DIR / "test" / "flames_sample.jpg"
        if fire_sample.exists():
            fire_frame = cv2.imread(str(fire_sample))
            haz_res = self.system.process_frame(
                frame=fire_frame,
                active_edge=("CORRIDOR_1", "EXIT_A"),
                source_room="ROOM_A",
                enable_frame_skip=False
            )
            ensure_complete_emergency_state(self.system, haz_res, "ROOM_A", ("CORRIDOR_1", "EXIT_A"))
            haz_em = haz_res["emergency_state"]
            self.assertTrue(haz_em["fire_detected"] or haz_em["hazard_score"] > 0.4)
            self.assertEqual(haz_em["recommended_exit"], "EXIT_B")
            self.assertTrue(haz_em["is_diverted_from_baseline"])
            self.assertEqual(haz_em["avoided_exit"], "EXIT_A")


class TestCloudWebcamSafeguard(unittest.TestCase):
    """Verifies that physical webcam capture is safely guarded in cloud environments."""

    def test_cloud_environment_detection_and_safeguard(self) -> None:
        """Verifies that is_cloud_environment detects cloud settings and disables physical capture."""
        with patch.dict(os.environ, {"IS_STREAMLIT_CLOUD": "1", "STREAMLIT_SHARING_MODE": "1"}):
            self.assertTrue(is_cloud_environment())

        handler = VideoStreamHandler(source=0)
        with patch("video.video_stream.is_cloud_environment", return_value=True):
            opened = handler.start_stream()
            self.assertFalse(opened, "Physical webcam must not open when cloud environment is detected")
            self.assertEqual(handler.status, "ERROR")
            self.assertIn("Streamlit Cloud", handler.error_message)


if __name__ == "__main__":
    unittest.main()
