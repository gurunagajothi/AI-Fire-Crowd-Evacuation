"""
Integration Layer: Complete Multi-Tier AI Fire/Smoke, Crowd Density, and Evacuation System.
Connects:
1. Perception Layer (FireSmokeDetector & CrowdDetector)
2. Graph Model (BuildingGraph with Multi-Zone demo partitioning)
3. Optimization Engine (Continuous Dynamic Dijkstra with Route Change Event dispatch)
4. Emergency Dispatch & Alerting (AlertManager with simulated PA, dynamic signage, guard log)
5. Continuous Video Processing & Latency Diagnostics
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import cv2
import numpy as np

from config.config import (
    CONF_THRESHOLD_FIRE,
    CONF_THRESHOLD_CROWD,
    ALPHA,
    BETA,
    GAMMA,
    FRAME_SKIP,
    PROCESS_EVERY_N_FRAMES,
    DEMO_BUILDING_ZONES,
    ZONE_MAX_CAPACITIES,
)
from detection.fire_detector import FireSmokeDetector
from detection.crowd_detector import CrowdDetector
from graph.building_graph import BuildingGraph
from graph.dijkstra_optimizer import DynamicDijkstraOptimizer
from alerts.alert_manager import AlertManager


class EvacuationSystem:
    """
    Coordinates simultaneous perception (fire/smoke and crowd) with graph-based
    dynamic edge cost calculation, continuous Dynamic Dijkstra, and emergency alerts.
    """

    def __init__(
        self,
        building_graph: Optional[BuildingGraph] = None,
        optimizer: Optional[DynamicDijkstraOptimizer] = None,
        fire_detector: Optional[FireSmokeDetector] = None,
        crowd_detector: Optional[CrowdDetector] = None,
        alert_manager: Optional[AlertManager] = None,
    ):
        """
        Initialize the complete evacuation management system.
        """
        print("[*] Initializing Integrated Evacuation System (IEEE Feature Suite)...")

        # 1. Building Graph
        if building_graph is not None:
            self.building_graph = building_graph
        else:
            self.building_graph = BuildingGraph(name="Demo Evacuation Facility")
            self.building_graph.create_demo_building()

        # 2. Dynamic Dijkstra Optimizer
        self.optimizer = optimizer or DynamicDijkstraOptimizer(
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA
        )

        # 3. Vision Detectors
        try:
            self.fire_detector = fire_detector or FireSmokeDetector()
            self.crowd_detector = crowd_detector or CrowdDetector()
        except Exception as e:
            raise RuntimeError(f"[!] Failed to initialize vision models: {e}")

        # 4. Alert & Signage Manager
        self.alert_manager = alert_manager or AlertManager()

        # 5. Continuous Video Processing & State Tracking
        self.frame_counter: int = 0
        self.processed_frames_counter: int = 0
        self.skipped_frames_counter: int = 0
        
        # Route state memory for continuous route change detection
        self.last_routes: Dict[str, List[str]] = {}
        self.last_costs: Dict[str, float] = {}

        # Last perception cache for skipped frames in continuous video
        self.cached_result: Optional[Dict[str, Any]] = None

        print("[+] EvacuationSystem successfully initialized with all perception, optimization, and alert engines.")

    def map_points_to_demo_zones(
        self,
        points: List[Tuple[int, int]],
        frame_width: int,
        frame_height: int
    ) -> Dict[str, int]:
        """
        DEMO ZONE MAPPING — NOT REAL CCTV CALIBRATION.
        Maps 2D pixel coordinates (e.g. person centers) to a set of demonstration zones
        by partitioning the camera view into representative spatial sections.
        """
        zone_counts = {z: 0 for z in DEMO_BUILDING_ZONES}
        if not points or frame_width <= 0 or frame_height <= 0:
            return zone_counts

        for cx, cy in points:
            # Normalized coordinates (0.0 to 1.0)
            nx = cx / frame_width
            ny = cy / frame_height

            # Synthetic spatial mapping for demonstration:
            if nx < 0.35:
                if ny < 0.5:
                    zone_counts["ROOM_A"] += 1
                else:
                    zone_counts["ROOM_B"] += 1
            elif nx < 0.70:
                if ny < 0.5:
                    zone_counts["CORRIDOR_1"] += 1
                else:
                    zone_counts["CORRIDOR_2"] += 1
            else:
                if ny < 0.5:
                    zone_counts["EXIT_A"] += 1
                else:
                    zone_counts["EXIT_B"] += 1

        return zone_counts

    def process_frame(
        self,
        frame: np.ndarray,
        active_edge: Tuple[str, str] = ("CORRIDOR_1", "EXIT_A"),
        source_room: str = "ROOM_A",
        enable_frame_skip: bool = False,
        skip_interval: int = PROCESS_EVERY_N_FRAMES
    ) -> Dict[str, Any]:
        """
        Process a video frame:
        1. Frame-skip sampling (for CPU-friendly continuous video).
        2. Parallel execution of Fire/Smoke + Crowd detectors.
        3. Multi-zone crowd density and hazard mapping.
        4. Dynamic edge cost update.
        5. Continuous Dynamic Dijkstra execution & Route-Change Event detection.
        6. Alert Manager & Dynamic Directional Signage dispatch.
        7. OpenCV HUD overlay rendering.
        """
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("[!] Input frame is empty or invalid.")

        self.frame_counter += 1
        t_start_total = time.time()

        # Handle frame skipping in continuous video stream mode
        if enable_frame_skip and (self.frame_counter % skip_interval != 0) and self.cached_result is not None:
            self.skipped_frames_counter += 1
            # Return cached result with updated frame overlay
            cached = self.cached_result.copy()
            cached["frames_processed"] = self.processed_frames_counter
            cached["frames_skipped"] = self.skipped_frames_counter
            return cached

        self.processed_frames_counter += 1
        h, w = frame.shape[:2]
        graph = self.building_graph.get_graph()
        u, v = active_edge

        if not graph.has_edge(u, v):
            raise KeyError(f"[!] Active edge ({u}, {v}) does not exist in graph.")
        if source_room not in graph.nodes:
            raise KeyError(f"[!] Source room '{source_room}' does not exist in graph.")

        # --- STEP 1 & 2: PERCEPTION (FIRE/SMOKE + CROWD) ---
        t_start_fire = time.time()
        fire_detections = self.fire_detector.detect(frame)
        hazard_score = self.fire_detector.calculate_hazard_score(fire_detections)
        has_fire = any(d["class_name"] == "fire" for d in fire_detections)
        has_smoke = any(d["class_name"] == "smoke" for d in fire_detections)
        t_fire_ms = (time.time() - t_start_fire) * 1000

        t_start_crowd = time.time()
        crowd_result = self.crowd_detector.detect(frame)
        people_count = crowd_result["count"]
        crowd_density = crowd_result["density"]
        crowd_detections = crowd_result["detections"]
        centers = crowd_result["centers"]
        t_crowd_ms = (time.time() - t_start_crowd) * 1000

        t_det_ms = t_fire_ms + t_crowd_ms

        # --- STEP 3: MULTI-ZONE MAPPING ---
        zone_people_counts = self.map_points_to_demo_zones(centers, w, h)
        zone_densities = {}
        for z, cnt in zone_people_counts.items():
            cap = ZONE_MAX_CAPACITIES.get(z, 20)
            zone_densities[z] = round(min(1.0, cnt / cap), 3)

        # Multi-Zone Hazard Record
        hazard_event_record = None
        if has_fire or has_smoke:
            hazard_type = "FIRE" if has_fire else "SMOKE"
            top_conf = max([d["confidence"] for d in fire_detections]) if fire_detections else 0.0
            hazard_event_record = {
                "zone": f"{u} <-> {v}",
                "hazard_type": hazard_type,
                "confidence": top_conf,
                "hazard_score": hazard_score,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # Update primary monitored edge
        self.building_graph.update_edge_hazard(u, v, hazard_score)
        self.building_graph.update_edge_density(u, v, crowd_density)

        # --- STEP 4: RECALCULATE DYNAMIC EDGE COSTS ---
        self.optimizer.update_all_edge_costs(graph)
        updated_edge_cost = graph[u][v]["cost"]

        # --- STEP 5: CONTINUOUS DYNAMIC DIJKSTRA & ROUTE-CHANGE EVENT ---
        t_start_dijkstra = time.time()
        all_exits = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "EXIT"]
        if not all_exits:
            all_exits = ["EXIT_A", "EXIT_B"]

        route_info = self.optimizer.find_safest_route(
            graph=graph,
            source=source_room,
            exits=all_exits
        )
        t_dijkstra_ms = (time.time() - t_start_dijkstra) * 1000

        rec_exit = route_info.get("selected_exit", "NONE")
        rec_route = route_info.get("path", [])
        rec_route_str = " -> ".join(rec_route) if rec_route else "NO SAFE ROUTE"
        total_route_cost = route_info.get("total_cost", 0.0)

        # Route Change Detection
        previous_route = self.last_routes.get(source_room, None)
        previous_cost = self.last_costs.get(source_room, total_route_cost)
        route_changed = False
        route_change_event = None

        if previous_route is not None and previous_route != rec_route:
            route_changed = True
            # Determine causal trigger
            if hazard_score > 0.0:
                reason = f"Hazard increased on {u} <-> {v} (Score: {hazard_score:.2f})"
            elif crowd_density > 0.40:
                reason = f"Crowd density increased on {u} <-> {v} (Density: {crowd_density:.2f})"
            else:
                reason = "Dynamic path cost optimization"

            route_change_event = self.alert_manager.register_route_change(
                previous_route=previous_route,
                new_route=rec_route,
                reason=reason,
                prev_cost=previous_cost,
                new_cost=total_route_cost,
                active_zone=f"{u} <-> {v}",
                hazard_score=hazard_score,
                crowd_density=crowd_density
            )
        elif previous_route is None:
            # First initialization
            self.alert_manager.update_dynamic_signage(recommended_exit=rec_exit)

        # Update route memory
        self.last_routes[source_room] = rec_route
        self.last_costs[source_room] = total_route_cost

        # Trigger zone alert if active fire/smoke
        if has_fire or has_smoke:
            self.alert_manager.trigger_hazard_alert(
                location=f"{u} <-> {v}",
                hazard_type="FIRE" if has_fire else "SMOKE",
                severity="CRITICAL" if has_fire else "WARNING",
                hazard_score=hazard_score,
                people_count=people_count,
                crowd_density=crowd_density,
                recommended_exit=rec_exit,
                route_str=rec_route_str
            )

        # Timing & Latency Metrics
        t_total_ms = (time.time() - t_start_total) * 1000
        approx_fps = round(1000.0 / max(1.0, t_total_ms), 1)

        # --- STEP 6: ANNOTATED OVERLAY RENDERING ---
        annotated_frame = self._render_annotations_and_hud(
            frame=frame,
            fire_detections=fire_detections,
            crowd_detections=crowd_detections,
            has_fire=has_fire,
            has_smoke=has_smoke,
            hazard_score=hazard_score,
            people_count=people_count,
            crowd_density=crowd_density,
            active_edge=active_edge,
            recommended_exit=rec_exit,
            recommended_route_str=rec_route_str,
            total_route_cost=total_route_cost,
            route_changed=route_changed,
            fps=approx_fps
        )

        result_dict = {
            "fire_detected": has_fire,
            "smoke_detected": has_smoke,
            "hazard_score": hazard_score,
            "people_detected": people_count,
            "crowd_density": crowd_density,
            "active_edge": active_edge,
            "updated_edge_cost": updated_edge_cost,
            "recommended_exit": rec_exit,
            "recommended_route": rec_route,
            "recommended_route_str": rec_route_str,
            "total_route_cost": total_route_cost,
            "fire_detections": fire_detections,
            "crowd_detections": crowd_detections,
            "annotated_frame": annotated_frame,
            # IEEE Extended Features:
            "previous_route": previous_route,
            "route_changed": route_changed,
            "route_change_event": route_change_event,
            "hazard_event_record": hazard_event_record,
            "multi_zone_counts": zone_people_counts,
            "multi_zone_densities": zone_densities,
            "dynamic_signage": self.alert_manager.dynamic_signage,
            "current_pa_message": self.alert_manager.current_pa_message,
            "security_guard_log": self.alert_manager.get_guard_log(),
            "route_history": self.alert_manager.get_route_history(),
            "latency_metrics": {
                "fire_ms": round(t_fire_ms, 1),
                "crowd_ms": round(t_crowd_ms, 1),
                "detection_ms": round(t_det_ms, 1),
                "dijkstra_ms": round(t_dijkstra_ms, 2),
                "total_ms": round(t_total_ms, 1),
                "fps": approx_fps,
                "frame_number": self.frame_counter,
                "frames_processed": self.processed_frames_counter,
                "frames_skipped": self.skipped_frames_counter,
                "frame_skip": skip_interval if enable_frame_skip else 1
            }
        }

        self.cached_result = result_dict
        return result_dict

    def _render_annotations_and_hud(
        self,
        frame: np.ndarray,
        fire_detections: List[Dict[str, Any]],
        crowd_detections: List[Dict[str, Any]],
        has_fire: bool,
        has_smoke: bool,
        hazard_score: float,
        people_count: int,
        crowd_density: float,
        active_edge: Tuple[str, str],
        recommended_exit: str,
        recommended_route_str: str,
        total_route_cost: float,
        route_changed: bool = False,
        fps: float = 0.0
    ) -> np.ndarray:
        """Draws bounding boxes and a translucent evacuation HUD onto the frame."""
        annotated = frame.copy()

        # 1. Fire / Smoke Bounding Boxes
        annotated = self.fire_detector.draw_detections(annotated, fire_detections)

        # 2. Person Bounding Boxes and Center Dots
        for det in crowd_detections:
            x1, y1, x2, y2 = det["box"]
            conf = det["confidence"]
            cx, cy = det["center"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            label = f"PERSON {conf * 100:.1f}%"
            cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # 3. Translucent Evacuation HUD
        h, w = annotated.shape[:2]
        hud_h = 165
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 25), -1)
        cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)

        # Title
        cv2.putText(annotated, "AI EVACUATION SYSTEM — IEEE PAPER INTEGRATED PIPELINE", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        # Left Column: Perception Status
        fire_str = "YES" if has_fire else "NO"
        smoke_str = "YES" if has_smoke else "NO"
        fire_col = (0, 0, 255) if has_fire else (0, 255, 0)
        smoke_col = (0, 165, 255) if has_smoke else (0, 255, 0)

        cv2.putText(annotated, f"Fire: {fire_str}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fire_col, 2)
        cv2.putText(annotated, f"Smoke: {smoke_str}", (140, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, smoke_col, 2)
        cv2.putText(annotated, f"Hazard Score: {hazard_score:.3f}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(annotated, f"People: {people_count} | Density: {crowd_density:.3f}", (15, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(annotated, f"Monitored Edge: {active_edge[0]} <-> {active_edge[1]}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 1)
        cv2.putText(annotated, f"Measured FPS: {fps:.1f}", (15, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 200), 1)

        # Right Column: Routing & Route Change Notification
        x_col2 = max(340, int(w * 0.44))
        exit_col = (0, 255, 0) if hazard_score == 0 else (0, 215, 255)
        cv2.putText(annotated, f"RECOMMENDED EXIT: {recommended_exit}", (x_col2, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, exit_col, 2)
        cv2.putText(annotated, f"Route: {recommended_route_str}", (x_col2, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(annotated, f"Total Route Cost: {total_route_cost:.2f}", (x_col2, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 150), 2)

        if route_changed:
            cv2.putText(annotated, "STATUS: [!] ROUTE CHANGED DYNAMICALLY", (x_col2, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
        else:
            cv2.putText(annotated, "STATUS: OPTIMAL ROUTE ACTIVE", (x_col2, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 255, 180), 1)

        return annotated

    def format_status_report(self, result: Dict[str, Any]) -> str:
        """
        Formats console status report matching the user's specification.
        """
        fire_str = "YES" if result["fire_detected"] else "NO"
        smoke_str = "YES" if result["smoke_detected"] else "NO"
        u, v = result["active_edge"]

        report = f"""
============================================================
EVACUATION SYSTEM STATUS
============================================================

Fire Detected: {fire_str}
Smoke Detected: {smoke_str}

Hazard Score: {result['hazard_score']:.3f}

People Detected: {result['people_detected']}
Crowd Density: {result['crowd_density']:.3f}

Active Zone/Edge:
{u} <-> {v}

Hazard:
{result['hazard_score']:.3f}

Density:
{result['crowd_density']:.3f}

Recommended Exit:
{result['recommended_exit']}

Recommended Route:
{result['recommended_route_str']}

Total Route Cost:
{result['total_route_cost']:.2f}

============================================================
"""
        return report
