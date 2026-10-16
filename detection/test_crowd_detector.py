"""
Testing and Evaluation Module for Person Detection and Crowd Density Estimator.
Supports testing on:
1. Static local images
2. Local video files
3. Live webcam streams

Displays:
People detected: N
Crowd density: X.XXX

Saves annotated outputs to data/processed/
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

from config.config import CONF_THRESHOLD_CROWD, MAX_EXPECTED_PEOPLE, OUTPUT_DIR, DATA_DIR
from detection.crowd_detector import CrowdDetector


def run_image_test(detector: CrowdDetector, image_path: Path, output_path: Path, conf: float):
    """Run crowd detection on a single image and save the annotated result."""

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
        result = detector.detect(frame, conf_threshold=conf)
    except Exception as e:
        print(f"[!] Error during inference: {e}")
        return False

    elapsed = (time.time() - start_time) * 1000

    print(f"[+] Inference Time: {elapsed:.1f} ms (CPU)")
    print(f"People detected: {result['count']}")
    print(f"Crowd density: {result['density']:.3f}")

    # Print individual person details
    for idx, det in enumerate(result["detections"], 1):
        print(
            f"    #{idx} Person | Confidence: {det['confidence'] * 100:.1f}% | "
            f"Box: {det['box']} | Center: {det['center']}"
        )

    # Draw visual annotations
    annotated = detector.draw_detections(frame, result)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    print(f"[+] Annotated image successfully saved to: {output_path}")
    return True


def run_video_test(detector: CrowdDetector, video_path: Path, output_path: Path, conf: float):
    """Run crowd detection on a video file and save the annotated output video."""
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
    max_people_seen = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        result = detector.detect(frame, conf_threshold=conf)
        max_people_seen = max(max_people_seen, result["count"])

        annotated = detector.draw_detections(frame, result)
        out.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(
                f"    Frame {frame_idx}/{total_frames} | "
                f"People detected: {result['count']} | "
                f"Crowd density: {result['density']:.3f}"
            )

    cap.release()
    out.release()
    print(f"[+] Video processing complete. Peak people detected: {max_people_seen}")
    print(f"[+] Annotated video saved to: {output_path}")
    return True


def run_webcam_test(detector: CrowdDetector, device_idx: int = 0, conf: float = 0.35):
    """Run live crowd detection from a connected webcam."""
    print(f"\n[+] Starting Live Webcam Crowd Detection on device index: {device_idx}")
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

        result = detector.detect(frame, conf_threshold=conf)
        annotated = detector.draw_detections(frame, result)

        cv2.imshow("Crowd Detection & Density - Live Feed (Press 'q' to quit)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[+] Webcam stream closed.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Crowd & Person Detection Module")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image file (e.g., data/test/crowd_sample.jpg)"
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
        default=CONF_THRESHOLD_CROWD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD_CROWD})"
    )
    parser.add_argument(
        "--max_cap",
        type=int,
        default=MAX_EXPECTED_PEOPLE,
        help=f"Max expected people per zone for density calculation (default: {MAX_EXPECTED_PEOPLE})"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output file"
    )

    args = parser.parse_args()

    # Initialize Crowd Detector with error handling
    try:
        detector = CrowdDetector(
            conf_threshold=args.conf,
            max_expected_people=args.max_cap
        )
    except Exception as e:
        print(f"\n[!] Failed to initialize CrowdDetector: {e}")
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
        # Default behavior: test on sample image in data/test/ if available
        sample_path = DATA_DIR / "test" / "crowd_sample.jpg"
        if sample_path.exists():
            print("[*] No test source specified. Testing on default crowd sample image...")
            out_path = OUTPUT_DIR / "annotated_crowd_sample.jpg"
            run_image_test(detector, sample_path, out_path, conf=args.conf)
        else:
            print("[*] No test input specified and no default crowd sample found.")
            print("[*] Usage examples:")
            print("    python detection/test_crowd_detector.py --image data/test/crowd_sample.jpg")
            print("    python detection/test_crowd_detector.py --video data/test/sample_video.mp4")
            print("    python detection/test_crowd_detector.py --webcam")


# =============================================================================
# Pytest Unit Test Suite (Mocked weights, no webcam requirement, fast CPU tests)
# =============================================================================

@pytest.fixture
def mock_crowd_detector():
    """Pytest fixture providing a CrowdDetector instance with mocked YOLO model."""
    with patch.object(CrowdDetector, "_load_model"):
        detector = CrowdDetector()
        mock_model = MagicMock()
        mock_model.names = {0: "person"}
        detector.model = mock_model
        return detector


def test_crowd_density_nominal(mock_crowd_detector):
    """Test density normalization across zero, nominal, and over-capacity limits."""
    # Zero occupants
    assert mock_crowd_detector.calculate_density(0, max_expected_people=20) == 0.0
    # Half capacity
    assert mock_crowd_detector.calculate_density(10, max_expected_people=20) == 0.5
    # Full capacity
    assert mock_crowd_detector.calculate_density(20, max_expected_people=20) == 1.0
    # Over capacity (clamped to 1.0)
    assert mock_crowd_detector.calculate_density(30, max_expected_people=20) == 1.0


def test_crowd_detector_detect_mocked(mock_crowd_detector):
    """Test detect() pipeline with mocked YOLO predictions and bounding box extraction."""
    mock_box = MagicMock()
    mock_box.conf = [MagicMock(item=lambda: 0.88)]
    mock_box.xyxy = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: MagicMock(tolist=lambda: [10.0, 20.0, 50.0, 80.0])))]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box]
    mock_crowd_detector.model.predict.return_value = [mock_res]

    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = mock_crowd_detector.detect(dummy_frame, conf_threshold=0.4)

    assert result["count"] == 1
    assert len(result["detections"]) == 1
    det = result["detections"][0]
    assert det["class_name"] == "person"
    assert det["confidence"] == 0.88
    assert det["box"] == [10, 20, 50, 80]
    assert det["center"] == (30, 50)
    assert result["density"] > 0.0


def test_crowd_draw_detections(mock_crowd_detector):
    """Test visual annotation drawing on dummy frame."""
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    sample_result = {
        "count": 1,
        "boxes": [[10, 20, 50, 80]],
        "confidences": [0.88],
        "centers": [(30, 50)],
        "density": 0.25,
        "detections": [{
            "class_name": "person",
            "confidence": 0.88,
            "box": [10, 20, 50, 80],
            "center": (30, 50)
        }]
    }
    annotated = mock_crowd_detector.draw_detections(dummy_frame, sample_result)
    assert isinstance(annotated, np.ndarray)
    assert annotated.shape == dummy_frame.shape


def test_run_image_test_missing_file(mock_crowd_detector, tmp_path):
    """Verify run_image_test safely returns False when target image file is missing."""
    missing_file = tmp_path / "nonexistent_crowd.jpg"
    out_file = tmp_path / "out_crowd.jpg"
    success = run_image_test(mock_crowd_detector, missing_file, out_file, conf=0.4)
    assert success is False


def test_run_image_test_valid_file(mock_crowd_detector, tmp_path):
    """Verify run_image_test processes valid image, saves output, and returns True."""
    in_file = tmp_path / "test_crowd.jpg"
    out_file = tmp_path / "out_crowd.jpg"
    cv2.imwrite(str(in_file), np.zeros((100, 100, 3), dtype=np.uint8))

    mock_crowd_detector.detect = MagicMock(return_value={
        "count": 0,
        "boxes": [],
        "confidences": [],
        "centers": [],
        "density": 0.0,
        "detections": []
    })
    mock_crowd_detector.draw_detections = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))

    success = run_image_test(mock_crowd_detector, in_file, out_file, conf=0.4)
    assert success is True
    assert out_file.exists()


if __name__ == "__main__":
    main()
