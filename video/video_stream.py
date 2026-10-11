"""
Video Stream Handler: Ingests CCTV, prerecorded video, RTSP, or webcam streams for the perception layer.
Supports:
- Laptop webcam (index 0)
- External USB camera (index 1, 2, ...)
- Uploaded video files (.mp4, .avi, .mov, .mkv)
- Pre-recorded CCTV sample video (demo_cctv_feed.mp4)
- Configurable RTSP network streams (rtsp://...)
Includes persistent stream lifecycle, reconnection handling, real FPS tracking, and disconnection detection.
"""

import os
import time
from typing import Tuple, Optional, Any, Dict
from pathlib import Path
import cv2
import numpy as np


class VideoStreamHandler:
    """Manages video feed capture and frame iteration for files, live cameras, and RTSP streams."""

    def __init__(self, source: Any = 0):
        """
        Initialize video capture handler.
        source can be an integer device index (e.g., 0 for default webcam),
        a file path string, or an RTSP network URL.
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
        self.status: str = "DISCONNECTED"  # DISCONNECTED, CONNECTING, CONNECTED, ERROR
        self.error_message: Optional[str] = None
        self.frame_count: int = 0
        self.current_fps: float = 0.0
        self._last_frame_time: float = 0.0
        self._start_time: float = 0.0
        self._fps_window: list = []

        # Determine source type
        self.source_type: str = self._determine_source_type(source)
        self.is_live: bool = self.source_type in ["WEBCAM", "USB_CAMERA", "RTSP_STREAM"]

    def _determine_source_type(self, source: Any) -> str:
        """Classifies the media source type."""
        if isinstance(source, int):
            return "WEBCAM" if source == 0 else "USB_CAMERA"
        if isinstance(source, str):
            s_lower = source.strip().lower()
            if s_lower.startswith("rtsp://") or s_lower.startswith("rtsps://"):
                return "RTSP_STREAM"
            if s_lower.startswith("http://") or s_lower.startswith("https://"):
                return "HTTP_STREAM"
            return "VIDEO_FILE"
        return "UNKNOWN"

    def start_stream(self) -> bool:
        """Initializes OpenCV video capture from video file, camera device, or RTSP stream."""
        if self.cap is not None:
            self.release()

        self.status = "CONNECTING"
        self.error_message = None

        try:
            # Set timeout hints for network streams where supported
            if self.source_type == "RTSP_STREAM":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

            # Parse device index if string passed as digit
            actual_source = self.source
            if isinstance(self.source, str) and self.source.isdigit():
                actual_source = int(self.source)

            self.cap = cv2.VideoCapture(actual_source)

            # Additional configurations for camera and network streams
            if self.cap.isOpened():
                # Attempt to set frame buffer size to minimize latency
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

                self.status = "CONNECTED"
                self.frame_count = 0
                self._start_time = time.time()
                self._last_frame_time = self._start_time
                self._fps_window = []
                return True
            else:
                self.status = "ERROR"
                self.error_message = f"Unable to open video source: {self.source}"
                self.cap = None
                return False

        except Exception as e:
            self.status = "ERROR"
            self.error_message = f"Video capture initialization exception: {str(e)}"
            self.cap = None
            return False

    def get_frame(self, loop_video: bool = True) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the next frame from the stream.
        Returns (success: bool, frame: np.ndarray or None).
        Handles EOF for video files (with optional looping) and disconnection for live cameras.
        """
        if self.cap is None or not self.cap.isOpened():
            self.status = "DISCONNECTED"
            return False, None

        ret, frame = self.cap.read()
        now = time.time()

        if ret and frame is not None and frame.size > 0:
            self.frame_count += 1
            self.status = "CONNECTED"

            # Compute running FPS
            if self._last_frame_time > 0:
                dt = now - self._last_frame_time
                if dt > 0:
                    fps_inst = 1.0 / dt
                    self._fps_window.append(fps_inst)
                    if len(self._fps_window) > 15:
                        self._fps_window.pop(0)
                    self.current_fps = sum(self._fps_window) / len(self._fps_window)
            self._last_frame_time = now
            return True, frame
        else:
            # Handle end of video file or camera drop
            if self.source_type == "VIDEO_FILE" and loop_video:
                # Loop video from beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_loop, frame_loop = self.cap.read()
                if ret_loop and frame_loop is not None:
                    self.frame_count += 1
                    return True, frame_loop

            # If not looping or stream failed
            if self.is_live:
                self.status = "ERROR"
                self.error_message = f"Live stream disconnected from source: {self.source}"
            else:
                self.status = "DISCONNECTED"
                self.error_message = "End of video stream reached"

            return False, None

    def reconnect(self, max_attempts: int = 2) -> bool:
        """Attempts to reconnect to the stream."""
        self.release()
        for attempt in range(max_attempts):
            time.sleep(0.2)
            if self.start_stream():
                return True
        return False

    def get_properties(self) -> dict:
        """Returns video metadata properties."""
        if self.cap is None or not self.cap.isOpened():
            return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}

        try:
            return {
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0),
                "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            }
        except Exception:
            return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns live stream status telemetry for dashboard and diagnostics."""
        props = self.get_properties()
        return {
            "status": self.status,
            "source": str(self.source),
            "source_type": self.source_type,
            "is_live": self.is_live,
            "fps": round(self.current_fps, 1),
            "configured_fps": round(props.get("fps", 0.0), 1),
            "resolution": f"{props.get('width', 0)}x{props.get('height', 0)}",
            "frame_count": self.frame_count,
            "error_message": self.error_message
        }

    def release(self):
        """Releases video capture resources cleanly."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.status = "DISCONNECTED"
