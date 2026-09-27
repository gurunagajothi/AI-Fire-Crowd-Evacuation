"""
Feature 16: Complete End-to-End Integration Verification Test.

Validates the full IEEE Paper Evacuation Pipeline:
CCTV Frame -> Fire Detection -> Person Detection -> Zone Assignment
-> Hazard/Density Update -> Edge Cost Recalculation -> Dynamic Dijkstra
-> Route Change Event Detection -> Dashboard Data Generation
-> Zone Alert Event -> Directional Signage Update.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DATA_DIR
from integration.evacuation_system import EvacuationSystem


def run_end_to_end_test():
    print("=" * 70)
    print("AI EVACUATION SYSTEM — COMPLETE END-TO-END PIPELINE TEST")
    print("=" * 70)

    steps_total = 0
    steps_passed = 0
    steps_failed = 0

    def record_step(name: str, passed: bool, msg: str = ""):
        nonlocal steps_total, steps_passed, steps_failed
        steps_total += 1
        if passed:
            steps_passed += 1
            print(f"[PASS] Step {steps_total:02d}: {name} - {msg}")
        else:
            steps_failed += 1
            print(f"[FAIL] Step {steps_total:02d}: {name} - {msg}")

    # 1. System Initialization
    try:
        system = EvacuationSystem()
        record_step("System Initialization", True, "All perception, graph, and alert engines initialized")
    except Exception as e:
        record_step("System Initialization", False, f"Initialization failed: {e}")
        return 1

    # 2. CCTV Frame Ingestion
    flames_img_path = DATA_DIR / "test" / "flames_sample.jpg"
    frame = cv2.imread(str(flames_img_path))
    record_step("CCTV Frame Ingestion", frame is not None, f"Loaded {flames_img_path.name} ({frame.shape[1]}x{frame.shape[0]})")

    # 3. Fire / Smoke Perception Detection
    try:
        fire_dets = system.fire_detector.detect(frame)
        hazard_score = system.fire_detector.calculate_hazard_score(fire_dets)
        has_fire = any(d["class_name"] == "fire" for d in fire_dets)
        record_step("Fire/Smoke Perception", has_fire and hazard_score > 0, f"Detected Fire (Score: {hazard_score:.3f})")
    except Exception as e:
        record_step("Fire/Smoke Perception", False, f"Error: {e}")

    # 4. Crowd / Person Perception Detection
    try:
        crowd_res = system.crowd_detector.detect(frame)
        record_step("Crowd Perception Execution", isinstance(crowd_res, dict) and "count" in crowd_res, f"Crowd pipeline executed (Count: {crowd_res['count']})")
    except Exception as e:
        record_step("Crowd Perception Execution", False, f"Error: {e}")

    # 5. Zone Assignment (Demo Mapping)
    try:
        h, w = frame.shape[:2]
        zone_counts = system.map_points_to_demo_zones(crowd_res["centers"], w, h)
        record_step("Multi-Zone Spatial Assignment", isinstance(zone_counts, dict) and len(zone_counts) >= 7, f"Mapped detections across {len(zone_counts)} building zones")
    except Exception as e:
        record_step("Multi-Zone Spatial Assignment", False, f"Error: {e}")

    # 6. Baseline Normal Route Execution
    try:
        # First process with normal condition
        res_baseline = system.process_frame(
            frame=np.zeros_like(frame),
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A"
        )
        norm_exit = res_baseline["recommended_exit"]
        norm_route = res_baseline["recommended_route"]
        record_step("Baseline Normal Route Computation", norm_exit == "EXIT_A", f"Normal route: {' -> '.join(norm_route)} (Exit: {norm_exit})")
    except Exception as e:
        record_step("Baseline Normal Route Computation", False, f"Error: {e}")

    # 7. Hazard / Density Edge Update & Dynamic Edge Cost Recalculation
    try:
        # Process the flames frame on edge CORRIDOR_1 <-> EXIT_A
        res_hazard = system.process_frame(
            frame=frame,
            active_edge=("CORRIDOR_1", "EXIT_A"),
            source_room="ROOM_A"
        )
        edge_cost = res_hazard["updated_edge_cost"]
        record_step("Hazard/Density Edge Cost Recalculation", edge_cost > 2.04, f"Edge cost updated dynamically to: {edge_cost:.2f}")
    except Exception as e:
        record_step("Hazard/Density Edge Cost Recalculation", False, f"Error: {e}")

    # 8. Dynamic Dijkstra Recomputation
    try:
        new_exit = res_hazard["recommended_exit"]
        new_route = res_hazard["recommended_route"]
        record_step("Dynamic Dijkstra Recomputation", new_exit == "EXIT_B", f"Safest route computed: {' -> '.join(new_route)} (Exit: {new_exit})")
    except Exception as e:
        record_step("Dynamic Dijkstra Recomputation", False, f"Error: {e}")

    # 9. Route Change Event Detection
    try:
        route_changed = res_hazard["route_changed"]
        ev = res_hazard["route_change_event"]
        record_step("Route Change Event Detection", route_changed is True and ev is not None, f"ROUTE_CHANGE_EVENT emitted: from {ev['previous_exit']} to {ev['new_exit']}")
    except Exception as e:
        record_step("Route Change Event Detection", False, f"Error: {e}")

    # 10. Dashboard Data Payload Integrity
    expected_keys = [
        "fire_detected", "smoke_detected", "hazard_score", "people_detected", "crowd_density",
        "active_edge", "updated_edge_cost", "recommended_exit", "recommended_route",
        "recommended_route_str", "total_route_cost", "annotated_frame", "dynamic_signage",
        "current_pa_message", "security_guard_log", "latency_metrics"
    ]
    all_keys_present = all(k in res_hazard for k in expected_keys)
    record_step("Dashboard Data Payload Integrity", all_keys_present, f"All {len(expected_keys)} required dashboard fields verified")

    # 11. Zone Alert & Security Guard Event Log
    try:
        guard_log = system.alert_manager.get_guard_log()
        has_log_entries = len(guard_log) > 0
        record_step("Security Guard Event Logging", has_log_entries, f"Logged {len(guard_log)} emergency dispatch events")
    except Exception as e:
        record_step("Security Guard Event Logging", False, f"Error: {e}")

    # 12. Dynamic Directional Signage Update
    try:
        signage = system.alert_manager.dynamic_signage
        exit_b_safe = "SAFE" in signage["EXIT_B"]["status"]
        exit_a_closed = "CLOSED" in signage["EXIT_A"]["status"] or "AVOID" in signage["EXIT_A"]["status"]
        record_step("Dynamic Directional Signage Update", exit_b_safe and exit_a_closed, f"EXIT A: {signage['EXIT_A']['status']} | EXIT B: {signage['EXIT_B']['status']}")
    except Exception as e:
        record_step("Dynamic Directional Signage Update", False, f"Error: {e}")

    # -------------------------------------------------------------
    # FINAL SUMMARY REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("END-TO-END PIPELINE SUMMARY")
    print("=" * 70)
    print(f"TOTAL PIPELINE STEPS: {steps_total}")
    print(f"PASSED STEPS:         {steps_passed}")
    print(f"FAILED STEPS:         {steps_failed}")
    print("=" * 70)

    if steps_failed == 0:
        print("[+] SUCCESS: ALL 12 END-TO-END PIPELINE STEPS PASSED!\n")
        return 0
    else:
        print(f"[!] {steps_failed} PIPELINE STEPS FAILED.\n")
        return 1


if __name__ == "__main__":
    code = run_end_to_end_test()
    sys.exit(code)
