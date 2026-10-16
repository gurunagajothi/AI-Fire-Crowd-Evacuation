"""
Testing and Evaluation Module for Fire & Smoke Detector.
Supports testing on:
1. Static local images
2. Local video files
3. Live webcam streams

Saves annotated outputs to data/processed/ and prints detection diagnostics.
"""

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

# Add project root to sys.path so config and detection packages are discoverable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import CONF_THRESHOLD_FIRE, OUTPUT_DIR, DATA_DIR
from detection.fire_detector import FireSmokeDetector


def run_image_test(detector: FireSmokeDetector, image_path: Path, output_path: Path, conf: float):
    """Run detection on a single image and save the annotated result."""

    print(f"\n[+] Testing Image: {image_path}")
    if not image_path.exists():
        print(f"[!] Error: Image file not found at: {image_path}")
        return False

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[!] Error: Failed to read or decode image at: {image_path}")
        print("    Please ensure the file is a valid image (e.g., .jpg, .png, .jpeg).")
        return False

    # Measure CPU inference latency
    start_time = time.time()
    try:
        detections = detector.detect(frame, conf_threshold=conf)
    except Exception as e:
        print(f"[!] Error during inference: {e}")
        return False

    elapsed = (time.time() - start_time) * 1000

    print(f"[+] Inference Time: {elapsed:.1f} ms (CPU)")
    print(f"[+] Total Hazards Detected: {len(detections)}")

    # Print detection details
    for idx, det in enumerate(detections, 1):
        print(
            f"    #{idx} Class: {det['class_name'].upper():<6} | "
            f"Confidence: {det['confidence'] * 100:.1f}% | "
            f"Box: {det['box']}"
        )

    # Compute and display Hazard Score for routing layer integration
    hazard_score = detector.calculate_hazard_score(detections)
    print(f"[+] Computed Dynamic Hazard Score: {hazard_score} (range: 0.0 - 1.0)")

    # Draw annotations
    annotated = detector.draw_detections(frame, detections)

    # Add hazard score badge on image
    status_text = f"Hazard Score: {hazard_score}"
    cv2.putText(
        annotated,
        status_text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if hazard_score > 0 else (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"[+] Annotated image successfully saved to: {output_path}")
    return True


def run_video_test(detector: FireSmokeDetector, video_path: Path, output_path: Path, conf: float):
    """Run detection on a video file and save the annotated output video."""
    print(f"\n[+] Testing Video: {video_path}")
    if not video_path.exists():
        print(f"[!] Error: Video file not found at: {video_path}")
        return False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Error: Could not open video file: {video_path}")
        print("    Please ensure the file is an accessible video format (e.g., .mp4, .avi).")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"[+] Processing {total_frames} frames ({width}x{height} @ {fps:.1f} FPS)...")
    frame_idx = 0
    total_hazards = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        detections = detector.detect(frame, conf_threshold=conf)
        total_hazards += len(detections)

        annotated = detector.draw_detections(frame, detections)
        out.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"    Processed frame {frame_idx}/{total_frames} (Active hazards in frame: {len(detections)})")

    cap.release()
    out.release()
    print(f"[+] Video processing complete. Total hazard events recorded: {total_hazards}")
    print(f"[+] Annotated video saved to: {output_path}")
    return True


def run_webcam_test(detector: FireSmokeDetector, device_idx: int = 0, conf: float = 0.35):
    """Run live detection from a connected webcam."""
    print(f"\n[+] Starting Live Webcam Test on device index: {device_idx}")
    print("[*] Press 'q' in the window to quit live detection.")

    cap = cv2.VideoCapture(device_idx)
    if not cap.isOpened():
        print(f"[!] Error: Could not access webcam at index {device_idx}.")
        print("    Please ensure a camera device is connected and not in use by another app.")
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Warning: Failed to receive frame from webcam.")
            break

        detections = detector.detect(frame, conf_threshold=conf)
        annotated = detector.draw_detections(frame, detections)

        hazard_score = detector.calculate_hazard_score(detections)
        status_text = f"Hazard Score: {hazard_score}"
        cv2.putText(
            annotated,
            status_text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255) if hazard_score > 0 else (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Fire/Smoke Hazard Detector - Live Feed (Press 'q' to quit)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Webcam stream closed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Fire & Smoke Detection Module")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image file (e.g., data/test/fire_sample.jpg)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run live inference on default webcam"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONF_THRESHOLD_FIRE,
        help=f"Confidence threshold (default: {CONF_THRESHOLD_FIRE})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output file"
    )

    args = parser.parse_args()

    # Initialize Detector with error handling
    try:
        detector = FireSmokeDetector(conf_threshold=args.conf)
    except Exception as e:
        print(f"\n[!] Failed to initialize FireSmokeDetector: {e}")
        sys.exit(1)

    # Determine execution mode
    if args.webcam:
        run_webcam_test(detector, device_idx=0, conf=args.conf)
    elif args.video:
        in_path = Path(args.video)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"annotated_{in_path.stem}.mp4"
        run_video_test(detector, in_path, out_path, conf=args.conf)
    elif args.image:
        in_path = Path(args.image)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"annotated_{in_path.stem}.jpg"
        run_image_test(detector, in_path, out_path, conf=args.conf)
    else:
        # Default behavior: test on sample images in data/test/ if available
        sample_path = DATA_DIR / "test" / "fire_sample.jpg"
        if sample_path.exists():
            print("[*] No test source specified. Testing on default sample image...")
            out_path = OUTPUT_DIR / "annotated_fire_sample.jpg"
            run_image_test(detector, sample_path, out_path, conf=args.conf)
        else:
            print("[*] No test input specified and no default sample found.")
            print("[*] Usage examples:")
            print("    python detection/test_fire_detector.py --image data/test/fire_sample.jpg")
            print("    python detection/test_fire_detector.py --video data/test/sample_video.mp4")
            print("    python detection/test_fire_detector.py --webcam")


# =============================================================================
# Pytest Unit Test Suite (Mocked weights, no webcam requirement, fast CPU tests)
# =============================================================================

@pytest.fixture
def mock_fire_detector():
    """Pytest fixture providing a FireSmokeDetector instance with mocked YOLO model."""
    with patch.object(FireSmokeDetector, "_ensure_model_exists"), \
         patch("detection.fire_detector.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.names = {0: "smoke", 1: "fire"}
        mock_yolo_cls.return_value = mock_model
        detector = FireSmokeDetector()
        detector.model = mock_model
        detector.classes = {0: "smoke", 1: "fire"}
        return detector


def test_fire_hazard_score_nominal(mock_fire_detector):
    """Test hazard score calculation across zero, smoke, fire, and multi-hazard detections."""
    # Zero detections -> 0.0
    assert mock_fire_detector.calculate_hazard_score([]) == 0.0

    # Smoke detection (0.4 + 0.3 * conf)
    smoke_det = [{"class_name": "smoke", "confidence": 0.5, "box": [0, 0, 10, 10]}]
    smoke_score = mock_fire_detector.calculate_hazard_score(smoke_det)
    assert 0.4 <= smoke_score <= 0.8
    assert smoke_score == round(0.4 + 0.3 * 0.5, 3)

    # Fire detection (0.7 + 0.3 * conf)
    fire_det = [{"class_name": "fire", "confidence": 0.9, "box": [0, 0, 10, 10]}]
    fire_score = mock_fire_detector.calculate_hazard_score(fire_det)
    assert 0.7 <= fire_score <= 1.0
    assert fire_score == round(min(1.0, 0.7 + 0.3 * 0.9), 3)

    # Multi-hazard prioritizes maximum danger
    multi_det = [
        {"class_name": "smoke", "confidence": 0.6, "box": [0, 0, 10, 10]},
        {"class_name": "fire", "confidence": 0.8, "box": [20, 20, 50, 50]}
    ]
    multi_score = mock_fire_detector.calculate_hazard_score(multi_det)
    assert multi_score == round(0.7 + 0.3 * 0.8, 3)


def test_fire_detector_detect_mocked(mock_fire_detector):
    """Test detect() pipeline with mocked YOLO hazard detection predictions."""
    mock_box = MagicMock()
    mock_box.cls = [MagicMock(item=lambda: 1)]
    mock_box.conf = [MagicMock(item=lambda: 0.92)]
    mock_box.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [15.0, 25.0, 60.0, 90.0])))]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    mock_fire_detector.model.predict.return_value = [mock_res]

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = mock_fire_detector.detect(dummy_frame, conf_threshold=0.4)

    assert len(detections) == 1
    assert detections[0]["class_name"] == "fire"
    assert detections[0]["confidence"] == 0.92
    assert detections[0]["box"] == [15, 25, 60, 90]


def test_fire_draw_detections(mock_fire_detector):
    """Test visual annotation drawing on dummy frame."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sample_dets = [{
        "class_name": "fire",
        "confidence": 0.92,
        "box": [15, 25, 60, 90],
        "class_id": 1
    }]
    annotated = mock_fire_detector.draw_detections(dummy_frame, sample_dets)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == dummy_frame.shape


def test_run_image_test_missing_file(mock_fire_detector, tmp_path):
    """Verify run_image_test safely returns False when target image file is missing."""
    missing_file = tmp_path / "nonexistent_fire.jpg"
    out_file = tmp_path / "out_fire.jpg"
    success = run_image_test(mock_fire_detector, missing_file, out_file, conf=0.4)
    assert success is False


def test_run_image_test_valid_file(mock_fire_detector, tmp_path):
    """Verify run_image_test processes valid image, saves output, and returns True."""
    in_file = tmp_path / "test_fire.jpg"
    out_file = tmp_path / "out_fire.jpg"
    cv2.imwrite(str(in_file), np.zeros((100, 100, 3), dtype=np.uint8))

    mock_fire_detector.detect = MagicMock(return_value=[])
    mock_fire_detector.draw_detections = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))

    success = run_image_test(mock_fire_detector, in_file, out_file, conf=0.4)
    assert success is True
    assert out_file.exists()


if __name__ == "__main__":
    main()
