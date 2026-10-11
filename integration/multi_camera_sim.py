"""
Integration Layer: Multi-Camera CCTV Simulation & Hybrid Real/Simulated Stream Manager.
Implements Feature 1 (Multi-Camera Management) and Feature 13 (Coverage / Occlusion Analysis)
for the IEEE-Paper AI Evacuation System.
Supports real VideoStreamHandler bindings (Webcam, USB, RTSP) alongside simulated feeds.
"""

from typing import Dict, Any, List, Optional, Tuple
import datetime
from pathlib import Path
import cv2
import numpy as np

from config.config import DATA_DIR
from video.video_stream import VideoStreamHandler


class MultiCameraSimulator:
    """
    Manages hybrid software-simulated and physical/RTSP surveillance cameras
    positioned across key egress building zones:
    - CAM-01: ROOM_A / CORRIDOR_1 (North Hallway)
    - CAM-02: CORRIDOR_2 (South Hallway)
    - CAM-03: EXIT_A (Emergency Exit North)
    - CAM-04: EXIT_B (Emergency Exit South)
    """

    def __init__(self):
        self.cameras: Dict[str, Dict[str, Any]] = {
            "CAM-01": {
                "id": "CAM-01",
                "zone": "ROOM_A / CORRIDOR_1",
                "edge": ("ROOM_A", "CORRIDOR_1"),
                "status": "ONLINE",
                "source_type": "SIMULATED CCTV SOURCE",
                "sample_file": "flames_sample.jpg",
                "coverage": "GOOD",
                "occlusion": "LOW",
                "fire_detected": False,
                "smoke_detected": False,
                "hazard_score": 0.0,
                "people_count": 0,
                "crowd_density": 0.0,
                "latency_ms": 0.0,
                "is_real_stream": False,
                "last_update": datetime.datetime.now().strftime("%H:%M:%S")
            },
            "CAM-02": {
                "id": "CAM-02",
                "zone": "CORRIDOR_2",
                "edge": ("CORRIDOR_1", "CORRIDOR_2"),
                "status": "ONLINE",
                "source_type": "SIMULATED CCTV SOURCE",
                "sample_file": "crowd_sample.jpg",
                "coverage": "GOOD",
                "occlusion": "LOW",
                "fire_detected": False,
                "smoke_detected": False,
                "hazard_score": 0.0,
                "people_count": 0,
                "crowd_density": 0.0,
                "latency_ms": 0.0,
                "is_real_stream": False,
                "last_update": datetime.datetime.now().strftime("%H:%M:%S")
            },
            "CAM-03": {
                "id": "CAM-03",
                "zone": "EXIT_A",
                "edge": ("CORRIDOR_1", "EXIT_A"),
                "status": "ONLINE",
                "source_type": "SIMULATED CCTV SOURCE",
                "sample_file": "normal_sample.jpg",
                "coverage": "PARTIAL",
                "occlusion": "MEDIUM",
                "fire_detected": False,
                "smoke_detected": False,
                "hazard_score": 0.0,
                "people_count": 0,
                "crowd_density": 0.0,
                "latency_ms": 0.0,
                "is_real_stream": False,
                "last_update": datetime.datetime.now().strftime("%H:%M:%S")
            },
            "CAM-04": {
                "id": "CAM-04",
                "zone": "EXIT_B",
                "edge": ("CORRIDOR_2", "EXIT_B"),
                "status": "ONLINE",
                "source_type": "SIMULATED CCTV SOURCE",
                "sample_file": "normal_sample.jpg",
                "coverage": "GOOD",
                "occlusion": "LOW",
                "fire_detected": False,
                "smoke_detected": False,
                "hazard_score": 0.0,
                "people_count": 0,
                "crowd_density": 0.0,
                "latency_ms": 0.0,
                "is_real_stream": False,
                "last_update": datetime.datetime.now().strftime("%H:%M:%S")
            }
        }
        # Real stream handles mapped by camera id
        self.stream_handlers: Dict[str, VideoStreamHandler] = {}

    def bind_stream(self, cam_id: str, stream: VideoStreamHandler):
        """Binds an active physical or RTSP stream handler to a camera channel."""
        if cam_id in self.cameras:
            self.stream_handlers[cam_id] = stream
            self.cameras[cam_id]["is_real_stream"] = True
            self.cameras[cam_id]["source_type"] = f"LIVE {stream.source_type}"
            self.cameras[cam_id]["status"] = "ONLINE" if stream.status == "CONNECTED" else stream.status

    def unbind_stream(self, cam_id: str):
        """Unbinds any physical stream from a camera channel and reverts to simulation."""
        if cam_id in self.stream_handlers:
            del self.stream_handlers[cam_id]
        if cam_id in self.cameras:
            self.cameras[cam_id]["is_real_stream"] = False
            self.cameras[cam_id]["source_type"] = "SIMULATED CCTV SOURCE"
            self.cameras[cam_id]["status"] = "ONLINE"

    def capture_frame(self, cam_id: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves a frame from the specified camera channel.
        If a real stream is bound, pulls from the stream handler.
        If simulated, reads the configured sample file.
        """
        if cam_id not in self.cameras:
            return False, None

        # Real stream branch
        if cam_id in self.stream_handlers:
            stream = self.stream_handlers[cam_id]
            success, frame = stream.get_frame()
            if success and frame is not None:
                self.cameras[cam_id]["status"] = "ONLINE"
                self.cameras[cam_id]["coverage"] = "GOOD"
                return True, frame
            else:
                self.cameras[cam_id]["status"] = "UNAVAILABLE"
                self.cameras[cam_id]["coverage"] = "UNAVAILABLE"
                self.cameras[cam_id]["occlusion"] = "HIGH"
                return False, None

        # Simulated sample file branch
        sample_name = self.cameras[cam_id].get("sample_file", "normal_sample.jpg")
        sample_path = DATA_DIR / "test" / sample_name
        if sample_path.exists():
            frame = cv2.imread(str(sample_path))
            if frame is not None:
                return True, frame

        return False, None

    def get_camera(self, cam_id: str) -> Optional[Dict[str, Any]]:
        return self.cameras.get(cam_id)

    def get_all_cameras(self) -> List[Dict[str, Any]]:
        return list(self.cameras.values())

    def update_camera_telemetry(
        self,
        cam_id: str,
        fire_detected: bool,
        smoke_detected: bool,
        hazard_score: float,
        people_count: int,
        crowd_density: float,
        latency_ms: float = 0.0,
        sample_file: Optional[str] = None,
        coverage: Optional[str] = None,
        occlusion: Optional[str] = None
    ):
        """Updates live telemetry for a specific camera channel."""
        if cam_id in self.cameras:
            cam = self.cameras[cam_id]
            cam["fire_detected"] = bool(fire_detected)
            cam["smoke_detected"] = bool(smoke_detected)
            cam["hazard_score"] = round(float(hazard_score), 3)
            cam["people_count"] = int(people_count)
            cam["crowd_density"] = round(float(crowd_density), 3)
            cam["latency_ms"] = round(float(latency_ms), 1)
            cam["last_update"] = datetime.datetime.now().strftime("%H:%M:%S")
            if sample_file:
                cam["sample_file"] = sample_file
            if coverage:
                cam["coverage"] = coverage
            if occlusion:
                cam["occlusion"] = occlusion

    def get_zone_uncertainty_penalty(self, zone: str) -> float:
        """
        Calculates an edge uncertainty cost penalty if the zone's camera coverage
        is degraded or occlusion is high.
        """
        for cam in self.cameras.values():
            if zone in cam["zone"]:
                penalty = 0.0
                if cam["coverage"] == "UNAVAILABLE":
                    penalty += 3.0
                elif cam["coverage"] == "PARTIAL":
                    penalty += 1.0
                if cam["occlusion"] == "HIGH":
                    penalty += 2.0
                elif cam["occlusion"] == "MEDIUM":
                    penalty += 0.5
                return penalty
        return 0.0
