"""
Video Stream Handler: Ingests CCTV, prerecorded video, RTSP, or webcam streams for the perception layer.
Supports:
- Local webcams across indices 0..5 with multi-backend probing (CAP_DSHOW, CAP_MSMF, CAP_ANY on Windows)
- Browser-based WebRTC / HTML5 camera frames (via Streamlit Cloud & desktop browsers)
- Uploaded video files (.mp4, .avi, .mov, .mkv)
- Pre-recorded CCTV sample video (demo_cctv_feed.mp4)
- Configurable RTSP network streams (rtsp://...)
Includes persistent stream lifecycle, frame-read verification, proper handle release, real FPS tracking,
camera discovery (indices 0..5), and diagnostic reporting.
"""

import os
import sys
import time
from typing import Tuple, Optional, Any, Dict, List
from pathlib import Path
import cv2
import numpy as np


def is_cloud_environment() -> bool:
    """
    Detects whether the application is running inside Streamlit Community Cloud
    or a headless remote Linux container without local physical camera devices.
    """
    if os.environ.get("STREAMLIT_SHARING_MODE") or os.environ.get("IS_STREAMLIT_CLOUD"):
        return True
    if sys.platform.startswith("linux"):
        # Streamlit Cloud mounts workspace under /mount/src or runs as appuser without /dev/video*
        if Path("/mount/src").exists() or os.environ.get("USER") == "appuser":
            return True
        has_video_dev = any(Path("/dev").glob("video*")) if Path("/dev").exists() else False
        if not has_video_dev and os.environ.get("DISPLAY") is None:
            return True
    return False


def get_preferred_capture_backends() -> List[Tuple[str, Optional[int]]]:
    """
    Returns ordered list of (backend_name, backend_id) suitable for the current OS.
    On Windows, DirectShow (CAP_DSHOW) and Media Foundation (CAP_MSMF) are tested
    before CAP_ANY to avoid MSMF driver hangs on integrated and USB webcams.
    """
    if sys.platform.startswith("win") or os.name == "nt":
        backends: List[Tuple[str, Optional[int]]] = []
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(("CAP_DSHOW", int(cv2.CAP_DSHOW)))
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(("CAP_MSMF", int(cv2.CAP_MSMF)))
        backends.append(("CAP_ANY", int(getattr(cv2, "CAP_ANY", 0))))
        return backends

    backends = []
    if hasattr(cv2, "CAP_V4L2") and sys.platform.startswith("linux"):
        backends.append(("CAP_V4L2", int(cv2.CAP_V4L2)))
    backends.append(("CAP_ANY", int(getattr(cv2, "CAP_ANY", 0))))
    return backends


def _open_capture_with_backend(source: Any, backend_id: Optional[int] = None):
    """
    Opens a cv2.VideoCapture handle using the specified backend when supported,
    falling back cleanly to single-argument cv2.VideoCapture(source) if mocked.
    """
    if backend_id is not None:
        try:
            return cv2.VideoCapture(source, backend_id)
        except TypeError:
            return cv2.VideoCapture(source)
    return cv2.VideoCapture(source)


def discover_available_cameras(
    max_index: int = 5,
    backends: Optional[List[Tuple[str, Optional[int]]]] = None
) -> List[Dict[str, Any]]:
    """
    Probes local camera indices 0 through `max_index` (default 0..5).
    A camera index is reported as available ONLY if:
      1. VideoCapture opens (`isOpened() == True`), AND
      2. A valid non-empty frame is read (`cap.read()` returns `(True, non_empty_ndarray)`).
    Every probed VideoCapture handle is always released immediately.
    """
    if backends is not None:
        pass_list = [backends]
    elif (sys.platform.startswith("win") or os.name == "nt") and hasattr(cv2, "CAP_DSHOW"):
        # Fast pass with DirectShow (~20ms per unoccupied index vs ~1.2s with MSMF),
        # with fallback pass to MSMF/CAP_ANY if DirectShow finds no cameras.
        fallback_win = [b for b in get_preferred_capture_backends() if b[0] != "CAP_DSHOW"]
        pass_list = [[("CAP_DSHOW", int(cv2.CAP_DSHOW))], fallback_win]
    else:
        pass_list = [get_preferred_capture_backends()]

    discovered: List[Dict[str, Any]] = []

    for candidate_backends in pass_list:
        if not candidate_backends:
            continue
        for idx in range(0, max_index + 1):
            if any(d["index"] == idx for d in discovered):
                continue
            for backend_name, backend_id in candidate_backends:
                cap = None
                try:
                    cap = _open_capture_with_backend(idx, backend_id)
                    if cap is not None and cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                            h, w = frame.shape[:2]
                            role_hint = "Default / Integrated Camera" if idx == 0 else f"External / Secondary Camera #{idx}"
                            discovered.append({
                                "index": idx,
                                "backend": backend_name,
                                "backend_name": backend_name,
                                "backend_id": backend_id,
                                "width": int(w),
                                "height": int(h),
                                "resolution": f"{w}x{h}",
                                "label": f"Camera #{idx} — {role_hint} ({backend_name}, {w}x{h})",
                            })
                            break
                except Exception:
                    pass
                finally:
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
        if discovered:
            break

    return discovered



class VideoStreamHandler:
    """Manages video feed capture and frame iteration for files, live cameras, and RTSP streams."""

    def __init__(self, source: Any = 0, preferred_backend: Optional[int] = None):
        """
        Initialize video capture handler.
        source can be an integer device index (e.g., 0..5 for local cameras),
        a file path string, or an RTSP network URL.
        """
        self.source = source
        self.preferred_backend = preferred_backend
        self.active_backend: str = "NONE"
        self.cap: Optional[cv2.VideoCapture] = None
        self.status: str = "DISCONNECTED"  # DISCONNECTED, CONNECTING, CONNECTED, ERROR
        self.error_message: Optional[str] = None
        self.frame_count: int = 0
        self.current_fps: float = 0.0
        self._last_frame_time: float = 0.0
        self._start_time: float = 0.0
        self._fps_window: list = []
        self._prefetched_frame: Optional[np.ndarray] = None

        # Determine source type
        self.source_type: str = self._determine_source_type(source)
        self.is_live: bool = self.source_type in ["WEBCAM", "USB_CAMERA", "RTSP_STREAM"]

    @staticmethod
    def discover_cameras(
        max_index: int = 5,
        backends: Optional[List[Tuple[str, Optional[int]]]] = None
    ) -> List[Dict[str, Any]]:
        """Static convenience wrapper for `discover_available_cameras`."""
        return discover_available_cameras(max_index=max_index, backends=backends)

    def _determine_source_type(self, source: Any) -> str:
        """Classifies the media source type."""
        if isinstance(source, int):
            return "WEBCAM" if source == 0 else "USB_CAMERA"
        if isinstance(source, str):
            s_clean = source.strip()
            if s_clean.isdigit():
                return "WEBCAM" if int(s_clean) == 0 else "USB_CAMERA"
            s_lower = s_clean.lower()
            if s_lower.startswith("rtsp://") or s_lower.startswith("rtsps://"):
                return "RTSP_STREAM"
            if s_lower.startswith("http://") or s_lower.startswith("https://"):
                return "HTTP_STREAM"
            return "VIDEO_FILE"
        return "UNKNOWN"

    def start_stream(self) -> bool:
        """
        Initializes OpenCV video capture from video file, camera device, or RTSP stream.
        Verifies both `cap.isOpened()` and an initial `cap.read()` frame before reporting
        a successful connection, and properly releases failed VideoCapture handles.
        """
        if self.cap is not None:
            self.release()

        self.status = "CONNECTING"
        self.error_message = None
        self._prefetched_frame = None

        try:
            # Parse device index if string passed as digit
            actual_source = self.source
            if isinstance(self.source, str) and self.source.strip().isdigit():
                actual_source = int(self.source.strip())

            # 1. Local Camera Index Path (0..5) — Multi-Backend Probing + Frame Verification
            if isinstance(actual_source, int):
                candidate_backends: List[Tuple[str, Optional[int]]] = []
                if self.preferred_backend is not None:
                    candidate_backends.append(("PREFERRED", self.preferred_backend))
                for b_name, b_id in get_preferred_capture_backends():
                    if all(b_id != existing_id for _, existing_id in candidate_backends):
                        candidate_backends.append((b_name, b_id))

                tried_names: List[str] = []
                opened_but_empty = False

                for backend_name, backend_id in candidate_backends:
                    tried_names.append(backend_name)
                    cap = None
                    try:
                        cap = _open_capture_with_backend(actual_source, backend_id)
                        if cap is None or not cap.isOpened():
                            if cap is not None:
                                cap.release()
                            continue

                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass

                        ret, first_frame = cap.read()
                        if ret and first_frame is not None and isinstance(first_frame, np.ndarray) and first_frame.size > 0:
                            self.cap = cap
                            self.active_backend = backend_name
                            self._prefetched_frame = first_frame
                            self.status = "CONNECTED"
                            self.frame_count = 0
                            self._start_time = time.time()
                            self._last_frame_time = self._start_time
                            self._fps_window = []
                            return True
                        else:
                            opened_but_empty = True
                            cap.release()
                    except Exception:
                        if cap is not None:
                            try:
                                cap.release()
                            except Exception:
                                pass

                self.cap = None
                self.status = "ERROR"
                backends_str = ", ".join(tried_names) if tried_names else "DEFAULT"
                if opened_but_empty:
                    self.error_message = (
                        f"Unable to open video source: {self.source} — Camera index {actual_source} opened via "
                        f"[{backends_str}] but returned no video frames (cap.read() returned an empty frame). "
                        f"The camera may be busy in another application (Zoom/Teams/Browser), blocked by OS camera "
                        f"privacy permissions, or disconnected."
                    )
                elif is_cloud_environment():
                    self.error_message = (
                        f"Unable to open video source: {self.source} — Server camera index {actual_source} is unavailable "
                        f"(tried [{backends_str}]). OpenCV on Streamlit Cloud runs on a remote server and cannot directly "
                        f"access the physical webcam on your laptop. Select 'Browser Webcam (WebRTC / Cloud)' in the sidebar."
                    )
                else:
                    self.error_message = (
                        f"Unable to open video source: {self.source} — Camera index {actual_source} unavailable across "
                        f"backends [{backends_str}]. Check that the camera is plugged in, not locked by another app, "
                        f"and allowed in OS Camera Privacy settings, or use 'Scan Local Cameras (0-5)'."
                    )
                return False

            # 2. Video File / RTSP / HTTP Stream Path
            if self.source_type == "RTSP_STREAM":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;5000000"

            cap = _open_capture_with_backend(actual_source, None)
            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                self.cap = None
                self.status = "ERROR"
                self.error_message = f"Unable to open video source: {self.source}"
                return False

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            ret, first_frame = cap.read()
            if not ret or first_frame is None or not isinstance(first_frame, np.ndarray) or first_frame.size == 0:
                try:
                    cap.release()
                except Exception:
                    pass
                self.cap = None
                self.status = "ERROR"
                self.error_message = f"Unable to open video source: {self.source} (stream opened but returned no frames)"
                return False

            self.cap = cap
            self.active_backend = "FFMPEG"
            self._prefetched_frame = first_frame
            self.status = "CONNECTED"
            self.frame_count = 0
            self._start_time = time.time()
            self._last_frame_time = self._start_time
            self._fps_window = []
            return True

        except Exception as e:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self.status = "ERROR"
            self.error_message = f"Video capture initialization exception: {str(e)}"
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

        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            ret = True
        else:
            ret, frame = self.cap.read()

        now = time.time()

        if ret and frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
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
                if ret_loop and frame_loop is not None and isinstance(frame_loop, np.ndarray) and frame_loop.size > 0:
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
        for _ in range(max_attempts):
            time.sleep(0.2)
            if self.start_stream():
                return True
        return False

    def get_properties(self) -> dict:
        """Returns video metadata properties."""
        if self.cap is None or not self.cap.isOpened():
            return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}

        try:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if (w <= 0 or h <= 0) and self._prefetched_frame is not None:
                h, w = self._prefetched_frame.shape[:2]
            return {
                "width": w,
                "height": h,
                "fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0),
                "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            }
        except Exception:
            if self._prefetched_frame is not None:
                h, w = self._prefetched_frame.shape[:2]
                return {"width": int(w), "height": int(h), "fps": 25.0, "total_frames": 0}
            return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns live stream status telemetry for dashboard and diagnostics."""
        props = self.get_properties()
        return {
            "status": self.status,
            "source": str(self.source),
            "source_type": self.source_type,
            "backend": self.active_backend,
            "is_live": self.is_live,
            "fps": round(self.current_fps, 1),
            "configured_fps": round(props.get("fps", 0.0), 1),
            "resolution": f"{props.get('width', 0)}x{props.get('height', 0)}",
            "frame_count": self.frame_count,
            "error_message": self.error_message
        }

    def release(self):
        """Releases video capture resources cleanly."""
        self._prefetched_frame = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.status = "DISCONNECTED"

