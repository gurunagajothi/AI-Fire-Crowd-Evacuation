"""
Perception Layer: Fire & Smoke Detector.
Uses a lightweight YOLOv8 Nano model fine-tuned on the D-Fire dataset.
Runs smoothly on CPU without requiring GPU acceleration.
"""

import os
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from config.config import (
    FIRE_MODEL_PATH,
    FIRE_MODEL_URL,
    CONF_THRESHOLD_FIRE,
)


class FireSmokeDetector:
    """
    Detects Fire and Smoke hazards from video frames and static images using YOLOv8n.
    Outputs structured detections (class name, confidence, bounding boxes)
    and an aggregated hazard severity score for graph-based routing.
    """

    def __init__(self, model_path: str = None, conf_threshold: float = None):
        """
        Initialize the FireSmokeDetector.

        Args:
            model_path: Path to the .pt model weights file. Defaults to config.FIRE_MODEL_PATH.
            conf_threshold: Minimum confidence threshold for detection. Defaults to config.CONF_THRESHOLD_FIRE.
        """
        self.model_path = Path(model_path) if model_path else FIRE_MODEL_PATH
        self.conf_threshold = conf_threshold if conf_threshold is not None else CONF_THRESHOLD_FIRE
        
        # Verify and ensure model weights exist locally
        self._ensure_model_exists()

        # Load YOLO model with robust error handling
        try:
            print(f"[*] Loading Fire/Smoke model from: {self.model_path} (Device: CPU)")
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(
                f"[!] Failed to load YOLO model from '{self.model_path}'. "
                f"The weight file may be corrupted or incompatible. Original error: {e}"
            )

        # Validate that the model has the expected classes
        if not hasattr(self.model, "names") or not self.model.names:
            raise RuntimeError(f"[!] The model loaded from '{self.model_path}' has no valid class names.")
            
        self.classes = self.model.names
        print(f"[*] Fire/Smoke Detector initialized. Target classes: {self.classes}")

    def _ensure_model_exists(self):
        """
        Ensures model weights exist on disk.
        If missing or empty, attempts to download the pretrained weights from Hugging Face.
        """
        # If the file exists but has 0 bytes (corrupted download), remove it
        if self.model_path.exists() and self.model_path.stat().st_size == 0:
            print(f"[!] Found empty/corrupted weights file at {self.model_path}. Removing...")
            self.model_path.unlink()

        if not self.model_path.exists():
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[*] Pretrained weights not found at {self.model_path}.")
            print(f"[*] Downloading lightweight model from: {FIRE_MODEL_URL} ...")
            try:
                urllib.request.urlretrieve(FIRE_MODEL_URL, self.model_path)
                size_mb = self.model_path.stat().st_size / (1024 * 1024)
                print(f"[+] Download complete: {self.model_path} ({size_mb:.2f} MB)")
            except Exception as e:
                raise FileNotFoundError(
                    f"[!] Model file is missing at '{self.model_path}' and automatic download failed: {e}. "
                    f"Please verify your internet connection or manually place the model at '{self.model_path}'."
                )

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> List[Dict[str, Any]]:
        """
        Analyze a video frame or image for Fire and Smoke.

        Args:
            frame: OpenCV image / BGR numpy array.
            conf_threshold: Optional override for the confidence threshold.

        Returns:
            List of detected hazard dicts:
            [
                {
                    "class_name": "fire" | "smoke",
                    "confidence": float,
                    "box": [x1, y1, x2, y2],
                    "class_id": int
                },
                ...
            ]
        """
        # Proper error handling for invalid image/frame
        if frame is None:
            raise ValueError("[!] Input frame is None. Provide a valid OpenCV image.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"[!] Input frame must be a numpy.ndarray, got {type(frame)}.")
        if frame.size == 0 or len(frame.shape) < 2:
            raise ValueError(f"[!] Input frame is empty or has invalid shape: {frame.shape}.")

        threshold = conf_threshold if conf_threshold is not None else self.conf_threshold

        try:
            # Run inference on CPU (verbose=False keeps console logs clean, imgsz=640 optimizes CPU speed)
            results = self.model.predict(
                source=frame,
                conf=threshold,
                device="cpu",
                verbose=False,
                imgsz=640
            )
        except Exception as e:
            raise RuntimeError(f"[!] Inference failed on input frame: {e}")

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        first_res = results[0]
        boxes = first_res.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            class_name = self.classes.get(cls_id, str(cls_id)).lower()
            confidence = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            
            # Format bounding box coordinates as integers [x1, y1, x2, y2]
            bbox = [int(round(coord)) for coord in xyxy]

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "box": bbox,
                "class_id": cls_id
            })

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw visual bounding boxes, class labels, and confidence tags on the frame using OpenCV.

        Args:
            frame: Original BGR frame.
            detections: List of detection dictionaries returned by detect().

        Returns:
            Annotated BGR frame copy.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("[!] Frame to annotate must be a valid numpy.ndarray.")

        annotated = frame.copy()

        # Color scheme (BGR):
        # Fire  -> Red / Deep Orange (0, 69, 255)
        # Smoke -> Slate Grey (128, 128, 128)
        color_map = {
            "fire": (0, 69, 255),
            "smoke": (128, 128, 128)
        }

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["class_name"]
            conf = det["confidence"]

            color = color_map.get(label, (0, 0, 255))
            
            # 1. Draw bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 2. Draw filled banner for label and confidence score
            text = f"{label.upper()} {conf * 100:.1f}%"
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y_banner_top = max(0, y1 - th - 8)
            cv2.rectangle(annotated, (x1, y_banner_top), (x1 + tw + 8, y1), color, -1)
            cv2.putText(
                annotated,
                text,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return annotated

    def calculate_hazard_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Compute a normalized hazard score Hazard(e) in [0.0, 1.0] for the dynamic edge cost:
        Cost(e) = alpha * Hazard(e) + beta * Density(e) + gamma * Length(e)

        Fire imparts a high danger factor (0.7 - 1.0),
        Smoke imparts an escalating caution factor (0.4 - 0.7).
        """
        if not detections:
            return 0.0

        score = 0.0
        for det in detections:
            cls_name = det["class_name"]
            conf = det["confidence"]
            if cls_name == "fire":
                score = max(score, min(1.0, 0.7 + (0.3 * conf)))
            elif cls_name == "smoke":
                score = max(score, min(0.8, 0.4 + (0.3 * conf)))

        return round(score, 3)
