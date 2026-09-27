"""
Video Stream Handler: Ingests CCTV, prerecorded video, or webcam streams for the perception layer.
"""

from typing import Tuple, Optional, Any
import cv2
import numpy as np


class VideoStreamHandler:
    """Manages video feed capture and frame iteration for files and live camera devices."""

    def __init__(self, source: Any = 0):
        """
        Initialize video capture handler.
        source can be an integer device index (e.g., 0 for default webcam)
        or a file path / URL string.
        """
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None

    def start_stream(self) -> bool:
        """Initializes OpenCV video capture from video file or camera device."""
        if self.cap is not None:
            self.release()

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.cap = None
            return False
        return True

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the next frame from the stream.
        Returns (success: bool, frame: np.ndarray or None).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        return ret, frame

    def get_properties(self) -> dict:
        """Returns video metadata properties."""
        if self.cap is None or not self.cap.isOpened():
            return {"width": 0, "height": 0, "fps": 0.0, "total_frames": 0}

        return {
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

    def release(self):
        """Releases video capture resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
