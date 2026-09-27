"""
Perception Layer: Person Detection & Crowd Density Estimator.
Uses a lightweight COCO-pretrained YOLOv8 Nano model (yolov8n.pt) on CPU.
Filters specifically for the 'person' class (class ID: 0).
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config.config import (
    CROWD_MODEL_PATH,
    CONF_THRESHOLD_CROWD,
    MAX_EXPECTED_PEOPLE,
)


class CrowdDetector:
    """
    Detects persons in video frames and calculates normalized crowd density.
    Outputs person counts, bounding boxes, confidences, center coordinates,
    and a normalized density metric for dynamic edge cost calculation.
    """

    def __init__(
        self,
        model_path: str = None,
        conf_threshold: float = None,
        max_expected_people: int = None
    ):
        """
        Initialize the CrowdDetector.

        Args:
            model_path: Path to the pretrained YOLO model weights. Defaults to config.CROWD_MODEL_PATH.
            conf_threshold: Minimum confidence threshold. Defaults to config.CONF_THRESHOLD_CROWD.
            max_expected_people: Maximum occupancy count for density normalization (1.0).
        """
        self.model_path = Path(model_path) if model_path else CROWD_MODEL_PATH
        self.conf_threshold = conf_threshold if conf_threshold is not None else CONF_THRESHOLD_CROWD
        self.max_expected_people = max_expected_people or MAX_EXPECTED_PEOPLE

        # Load YOLO model with robust error handling
        self._load_model()

    def _load_model(self):
        """Loads the YOLO model on CPU, auto-downloading COCO weights if missing."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[*] Loading Person/Crowd model from: {self.model_path} (Device: CPU)")
            # Ultralytics auto-downloads 'yolov8n.pt' if the file is not found locally
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(
                f"[!] Failed to load Crowd YOLO model from '{self.model_path}'. "
                f"Original error: {e}"
            )

        # Validate that class 0 is 'person'
        if not hasattr(self.model, "names") or 0 not in self.model.names:
            raise RuntimeError(f"[!] Model loaded from '{self.model_path}' lacks valid COCO class definitions.")

        person_label = self.model.names[0]
        if person_label.lower() != "person":
            print(f"[!] Warning: Class 0 is '{person_label}', expected 'person'.")

        print(f"[*] Crowd Detector initialized. Tracking class 0 ('{person_label}') on CPU.")

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> Dict[str, Any]:
        """
        Analyze a video frame or image for persons.

        Args:
            frame: OpenCV image / BGR numpy array.
            conf_threshold: Optional override for the confidence threshold.

        Returns:
            dict containing:
            - "count": int (number of detected people)
            - "boxes": list of [x1, y1, x2, y2]
            - "confidences": list of float (0.0 - 1.0)
            - "centers": list of (cx, cy)
            - "density": float (0.0 - 1.0)
            - "detections": list of dicts with box, confidence, center, class_name
        """
        # Validate frame
        if frame is None:
            raise ValueError("[!] Input frame is None. Provide a valid OpenCV image.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"[!] Input frame must be a numpy.ndarray, got {type(frame)}.")
        if frame.size == 0 or len(frame.shape) < 2:
            raise ValueError(f"[!] Input frame is empty or invalid shape: {frame.shape}.")

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        try:
            # Run CPU inference filtering strictly for class 0 ('person'), imgsz=640 optimizes CPU speed
            results = self.model.predict(
                source=frame,
                classes=[0],       # Only detect 'person'
                conf=threshold,
                device="cpu",
                verbose=False,
                imgsz=640
            )
        except Exception as e:
            raise RuntimeError(f"[!] Crowd inference failed on input frame: {e}")

        boxes_list: List[List[int]] = []
        confidences_list: List[float] = []
        centers_list: List[Tuple[int, int]] = []
        detections_list: List[Dict[str, Any]] = []

        if results and len(results) > 0:
            first_res = results[0]
            boxes = first_res.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    bbox = [int(round(coord)) for coord in xyxy]
                    x1, y1, x2, y2 = bbox

                    # Calculate person center point (cx, cy)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    boxes_list.append(bbox)
                    confidences_list.append(round(conf, 4))
                    centers_list.append((cx, cy))
                    detections_list.append({
                        "class_name": "person",
                        "confidence": round(conf, 4),
                        "box": bbox,
                        "center": (cx, cy)
                    })

        count = len(boxes_list)
        density = self.calculate_density(count)

        return {
            "count": count,
            "boxes": boxes_list,
            "confidences": confidences_list,
            "centers": centers_list,
            "density": density,
            "detections": detections_list
        }

    def calculate_density(self, count: int, max_expected_people: int = None) -> float:
        """
        Calculate normalized crowd density between 0.0 and 1.0 based on:
        Density = min(1.0, count / max_expected_people)

        Args:
            count: Number of detected persons.
            max_expected_people: Zone capacity threshold.

        Returns:
            Normalized float between 0.0 and 1.0.
        """
        max_cap = max_expected_people or self.max_expected_people
        if max_cap <= 0:
            return 1.0 if count > 0 else 0.0

        density = min(1.0, count / max_cap)
        return round(float(density), 3)

    def draw_detections(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Draw visual bounding boxes, confidence tags, center points, and crowd metrics on the frame.

        Args:
            frame: Original BGR frame.
            result: Result dictionary returned by detect().

        Returns:
            Annotated BGR frame copy.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("[!] Frame to annotate must be a valid numpy.ndarray.")

        annotated = frame.copy()
        box_color = (255, 180, 0)   # Cyan / Light Amber (BGR)
        center_color = (0, 0, 255)   # Red dot for center

        # Draw each detected person
        for det in result["detections"]:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cx, cy = det["center"]

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            # Center point
            cv2.circle(annotated, (cx, cy), 4, center_color, -1)

            # Confidence label
            label = f"PERSON {conf * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_banner_top = max(0, y1 - th - 6)
            cv2.rectangle(annotated, (x1, y_banner_top), (x1 + tw + 6, y1), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )

        # Draw overall crowd summary banner in top-left corner
        count = result["count"]
        density = result["density"]
        summary_text = f"People detected: {count} | Crowd density: {density:.3f}"

        # Dynamic color coding for congestion level:
        # Green: low (<0.4), Yellow: medium (0.4-0.7), Red: high (>0.7)
        if density < 0.4:
            badge_color = (0, 180, 0)      # Green
        elif density < 0.7:
            badge_color = (0, 165, 255)    # Amber / Orange
        else:
            badge_color = (0, 0, 220)      # Red

        (sw, sh), _ = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (10, 10), (20 + sw, 20 + sh + 10), (0, 0, 0), -1)
        cv2.rectangle(annotated, (10, 10), (20 + sw, 20 + sh + 10), badge_color, 2)
        cv2.putText(
            annotated,
            summary_text,
            (15, 15 + sh),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return annotated
