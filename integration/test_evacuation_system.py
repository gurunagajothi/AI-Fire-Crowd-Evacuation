"""
Test and Verification Script for Integrated Evacuation System.
Demonstrates end-to-end integration:
Perception (Fire/Smoke & Crowd) -> Graph Update (Demo Zone) -> Dynamic Dijkstra Route Optimization.

Supports:
- Single image testing via --image argument
- Automated dual-condition verification (Smoke sample vs. Flames sample)
"""

import argparse
import sys
from pathlib import Path
import cv2

# Add project root to sys.path so config and modules are discoverable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DATA_DIR, OUTPUT_DIR
from integration.evacuation_system import EvacuationSystem


def run_single_test(
    system: EvacuationSystem,
    image_path: Path,
    output_path: Path,
    active_edge=("CORRIDOR_1", "EXIT_A"),
    source_room="ROOM_A"
) -> bool:
    """Runs end-to-end evacuation processing on a given image file."""
    print(f"\n[+] Processing Test Image: {image_path}")
    print(f"[*] Monitored Zone (Demo Mapping): {active_edge[0]} <-> {active_edge[1]}")
    print(f"[*] Evacuation Origin: {source_room}")

    # Validation: image existence
    if not image_path.exists():
        print(f"[!] Error: Test image file not found at: {image_path}")
        return False

    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"[!] Error: Could not read or decode image at: {image_path}")
        return False

    # Execute complete perception-to-route pipeline
    try:
        result = system.process_frame(
            frame=frame,
            active_edge=active_edge,
            source_room=source_room
        )
    except Exception as e:
        print(f"[!] Error during frame processing: {e}")
        return False

    # Print formatted status report
    report = system.format_status_report(result)
    print(report)

    # Save annotated output image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result["annotated_frame"])
    print(f"[+] Annotated visualization saved to: {output_path}")
    return True


def run_dual_benchmark(system: EvacuationSystem):
    """
    Executes the two required benchmark tests:
    TEST 1: Smoke Sample (data/test/fire_sample.jpg)
    TEST 2: Flames Sample (data/test/flames_sample.jpg)
    """
    print("\n" + "=" * 60)
    print("STARTING DUAL-CONDITION EVACUATION BENCHMARK")
    print("DEMO ZONE MAPPING — NOT REAL CCTV CALIBRATION")
    print("=" * 60)

    smoke_img = DATA_DIR / "test" / "fire_sample.jpg"
    flames_img = DATA_DIR / "test" / "flames_sample.jpg"

    # --- TEST 1: SMOKE CONDITION ---
    print("\n------------------------------------------------------------")
    print("TEST 1: SMOKE SAMPLE")
    print("------------------------------------------------------------")
    out_smoke = OUTPUT_DIR / "integrated_smoke_output.jpg"
    run_single_test(
        system=system,
        image_path=smoke_img,
        output_path=out_smoke,
        active_edge=("CORRIDOR_1", "EXIT_A"),
        source_room="ROOM_A"
    )

    # --- TEST 2: FLAMES CONDITION ---
    print("\n------------------------------------------------------------")
    print("TEST 2: FLAMES SAMPLE")
    print("------------------------------------------------------------")
    out_flames = OUTPUT_DIR / "integrated_flames_output.jpg"
    run_single_test(
        system=system,
        image_path=flames_img,
        output_path=out_flames,
        active_edge=("CORRIDOR_1", "EXIT_A"),
        source_room="ROOM_A"
    )

    print("\n" + "=" * 60)
    print("DUAL-CONDITION BENCHMARK COMPLETED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test Integrated Evacuation System")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to test image file (e.g., data/test/flames_sample.jpg)"
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="CORRIDOR_1,EXIT_A",
        help="Comma-separated edge nodes for demo zone mapping (default: CORRIDOR_1,EXIT_A)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="ROOM_A",
        help="Origin room ID for path calculation (default: ROOM_A)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom path to save annotated output image"
    )

    args = parser.parse_args()

    # Parse active edge
    try:
        u, v = [node.strip() for node in args.zone.split(",")]
        active_edge = (u, v)
    except Exception:
        print(f"[!] Invalid zone format: '{args.zone}'. Expected 'NODE_1,NODE_2'. Using default 'CORRIDOR_1,EXIT_A'.")
        active_edge = ("CORRIDOR_1", "EXIT_A")

    # Initialize System with error handling
    try:
        system = EvacuationSystem()
    except Exception as e:
        print(f"[!] Failed to initialize EvacuationSystem: {e}")
        sys.exit(1)

    if args.image:
        img_path = Path(args.image)
        out_path = Path(args.output) if args.output else OUTPUT_DIR / f"integrated_{img_path.stem}.jpg"
        run_single_test(
            system=system,
            image_path=img_path,
            output_path=out_path,
            active_edge=active_edge,
            source_room=args.source
        )
    else:
        # Default behavior: run both smoke and flame benchmark tests
        run_dual_benchmark(system)


if __name__ == "__main__":
    main()
