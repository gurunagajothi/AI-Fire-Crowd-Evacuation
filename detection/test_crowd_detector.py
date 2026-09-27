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
import cv2

# Add project root to sys.path so config and detection packages are discoverable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import CONF_THRESHOLD_CROWD, MAX_EXPECTED_PEOPLE, OUTPUT_DIR, DATA_DIR
from detection.crowd_detector import CrowdDetector


def test_image(detector: CrowdDetector, image_path: Path, output_path: Path, conf: float):
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


def test_video(detector: CrowdDetector, video_path: Path, output_path: Path, conf: float):
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


def test_webcam(detector: CrowdDetector, device_idx: int = 0, conf: float = 0.35):
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
        test_webcam(detector, device_idx=0, conf=args.conf)
    elif args.video:
        in_path = Path(args.video)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"annotated_{in_path.stem}.mp4"
        test_video(detector, in_path, out_path, conf=args.conf)
    elif args.image:
        in_path = Path(args.image)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"annotated_{in_path.stem}.jpg"
        test_image(detector, in_path, out_path, conf=args.conf)
    else:
        # Default behavior: test on sample image in data/test/ if available
        sample_path = DATA_DIR / "test" / "crowd_sample.jpg"
        if sample_path.exists():
            print("[*] No test source specified. Testing on default crowd sample image...")
            out_path = OUTPUT_DIR / "annotated_crowd_sample.jpg"
            test_image(detector, sample_path, out_path, conf=args.conf)
        else:
            print("[*] No test input specified and no default crowd sample found.")
            print("[*] Usage examples:")
            print("    python detection/test_crowd_detector.py --image data/test/crowd_sample.jpg")
            print("    python detection/test_crowd_detector.py --video data/test/sample_video.mp4")
            print("    python detection/test_crowd_detector.py --webcam")


if __name__ == "__main__":
    main()
