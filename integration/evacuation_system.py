"""
Integration Layer: Complete Multi-Tier AI Fire/Smoke, Crowd Density, and Evacuation System.
Connects:
1. Perception Layer (FireSmokeDetector & CrowdDetector)
2. Graph Model (BuildingGraph with Multi-Zone demo partitioning)
3. Optimization Engine (Continuous Dynamic Dijkstra with Route Change Event dispatch)
4. Emergency Dispatch & Alerting (AlertManager with simulated PA, dynamic signage, guard log)
5. Continuous Video Processing & Latency Diagnostics
"""

import importlib
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
import alerts.alert_manager as alert_manager_module
from alerts.alert_manager import AlertManager


def _resolve_alert_manager_class():
    """
    Resolves the current `AlertManager` class from `alerts.alert_manager`.
    Reloads `alerts.alert_manager` only if the currently loaded class lacks
    `sync_facility_emergency_state` (e.g. during a Streamlit Cloud hot-reload),
    and keeps module-level `AlertManager` synchronized.
    """
    global AlertManager, alert_manager_module
    cls = getattr(alert_manager_module, "AlertManager", AlertManager)
    if not hasattr(cls, "sync_facility_emergency_state"):
        alert_manager_module = importlib.reload(alert_manager_module)
        cls = alert_manager_module.AlertManager
    AlertManager = cls
    return cls


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

        # 4. Alert & Signage Manager (guarantee compatible AlertManager definition)
        alert_cls = _resolve_alert_manager_class()
        self.alert_manager = alert_manager if alert_manager is not None else alert_cls()
        self._ensure_compatible_alert_manager()

        # 5. Continuous Video Processing & State Tracking
        self.frame_counter: int = 0
        self.processed_frames_counter: int = 0
        self.skipped_frames_counter: int = 0
        
        # Route state memory for continuous route change detection
        self.last_routes: Dict[str, List[str]] = {}
        self.last_costs: Dict[str, float] = {}
        self._last_active_edge: Optional[Tuple[str, str]] = None
        self._last_cache_signature: Optional[Tuple[Any, ...]] = None

        # Unified facility emergency state for the current frame/session
        self.current_emergency_state: Optional[Dict[str, Any]] = None

        # Last perception cache for skipped frames in continuous video
        self.cached_result: Optional[Dict[str, Any]] = None

        print("[+] EvacuationSystem successfully initialized with all perception, optimization, and alert engines.")

    def _ensure_compatible_alert_manager(self) -> AlertManager:
        """
        Ensures `self.alert_manager` is an instance of a compatible `AlertManager`
        class that implements `sync_facility_emergency_state`. If `self.alert_manager`
        was instantiated from a stale pre-reload `AlertManager` class, upgrades it
        in-place while preserving all active alerts, guard logs, signage, PA text,
        and route history.
        """
        alert_cls = _resolve_alert_manager_class()
        current_am = getattr(self, "alert_manager", None)
        if current_am is None:
            self.alert_manager = alert_cls()
            return self.alert_manager

        if hasattr(current_am, "sync_facility_emergency_state"):
            return current_am

        upgraded = alert_cls()
        for attr in (
            "active_alerts",
            "guard_alert_log",
            "route_history",
            "dynamic_signage",
            "current_pa_message",
            "snapshot_dir",
        ):
            if hasattr(current_am, attr):
                setattr(upgraded, attr, getattr(current_am, attr))
        self.alert_manager = upgraded
        return self.alert_manager

    def _sync_alert_manager_state(self, emergency_state: Dict[str, Any]) -> None:
        """
        Synchronizes `self.alert_manager` with the unified facility emergency state,
        upgrading stale `AlertManager` instances automatically if needed.
        """
        am = self._ensure_compatible_alert_manager()
        if hasattr(am, "sync_facility_emergency_state"):
            am.sync_facility_emergency_state(emergency_state)
            return

        # Coherent fallback if a custom external alert_manager object was injected
        rec_exit = emergency_state.get("recommended_exit", "NONE")
        compromised_exits = emergency_state.get("compromised_exits", [])
        if hasattr(am, "update_dynamic_signage"):
            am.update_dynamic_signage(recommended_exit=rec_exit, dangerous_exit=compromised_exits)
        if emergency_state.get("is_emergency", False) and hasattr(am, "generate_pa_announcement"):
            am.generate_pa_announcement(
                recommended_exit=rec_exit,
                dangerous_zone=emergency_state.get("active_zone_str", ""),
                reason=emergency_state.get("adaptation_reason", ""),
                hazard_type=emergency_state.get("hazard_type", "NONE"),
                hazard_score=emergency_state.get("hazard_score", 0.0),
                crowd_density=emergency_state.get("crowd_density", 0.0),
                route_str=emergency_state.get("recommended_route_str", ""),
            )
        else:
            am.current_pa_message = "Facility conditions nominal. Follow standard exit signs if required."

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

    def build_facility_emergency_state(
        self,
        graph: Any,
        source_room: str,
        active_edge: Tuple[str, str],
        all_exits: List[str],
        has_fire: bool,
        has_smoke: bool,
        hazard_score: float,
        people_count: int,
        crowd_density: float,
        rec_route: List[str],
        rec_exit: str,
        total_route_cost: float,
        route_changed_this_frame: bool,
        previous_route: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Constructs a single, authoritative facility emergency state dictionary
        for the current frame/session so all UI panels, signage, PA advisories,
        route history, and guard notifications derive from one consistent state.
        """
        u, v = active_edge

        # 1. Clean baseline route evaluation (zero hazard, zero crowd density)
        clean_graph = graph.copy()
        for u_n, v_n in clean_graph.edges:
            clean_graph[u_n][v_n]["hazard"] = 0.0
            clean_graph[u_n][v_n]["density"] = 0.0
        self.optimizer.update_all_edge_costs(clean_graph)
        baseline_info = self.optimizer.find_safest_route(
            clean_graph, source=source_room, exits=all_exits
        )
        baseline_route = baseline_info.get("path", [])
        baseline_exit = baseline_info.get("selected_exit", "NONE")
        baseline_cost = baseline_info.get("total_cost", 0.0)

        # 2. Unified hazard & crowd status classification
        if has_fire or hazard_score >= 0.70:
            hazard_status = "CRITICAL"
            hazard_type = "FIRE"
        elif has_smoke or hazard_score >= 0.40:
            hazard_status = "WARNING"
            hazard_type = "SMOKE"
        elif hazard_score > 0.0:
            hazard_status = "CAUTION"
            hazard_type = "LOW HAZARD"
        elif crowd_density >= 0.70:
            hazard_status = "WARNING"
            hazard_type = "CROWD_CONGESTION"
        else:
            hazard_status = "NORMAL"
            hazard_type = "NONE"

        if crowd_density >= 0.70:
            crowd_status = "HIGH CONGESTION"
        elif crowd_density >= 0.35:
            crowd_status = "MODERATE"
        else:
            crowd_status = "NORMAL"

        is_emergency = bool(has_fire or has_smoke or hazard_score > 0.0 or crowd_density >= 0.70)
        zone_status = (
            "CRITICAL"
            if (hazard_status in ["CRITICAL", "WARNING"] or crowd_status == "HIGH CONGESTION")
            else ("CAUTION" if (hazard_status == "CAUTION" or crowd_status == "MODERATE") else "NORMAL")
        )

        # 3. Dynamically identify compromised exits from graph edge state & route diversion
        compromised_exits: List[str] = []
        for ex in all_exits:
            if ex in graph.nodes:
                for nbr in graph.neighbors(ex):
                    edge_data = graph[nbr][ex]
                    if edge_data.get("hazard", 0.0) > 0.0 or edge_data.get("density", 0.0) >= 0.70:
                        if ex not in compromised_exits:
                            compromised_exits.append(ex)

        is_diverted_from_baseline = (rec_route != baseline_route)
        if is_emergency and baseline_exit not in ("", "NONE", "UNKNOWN", rec_exit):
            if baseline_exit not in compromised_exits:
                compromised_exits.append(baseline_exit)

        avoided_exit = (
            compromised_exits[0]
            if compromised_exits
            else (baseline_exit if (is_diverted_from_baseline and baseline_exit != rec_exit) else None)
        )

        # 4. Compute causal adaptation reason
        edge_len = graph[u][v]["length"]
        updated_edge_cost = graph[u][v]["cost"]
        baseline_edge_cost = self.optimizer.gamma * edge_len

        if is_diverted_from_baseline:
            if has_fire or hazard_score >= 0.70:
                adaptation_reason = (
                    f"Corridor edge {u} <-> {v} compromised by Fire Hazard ({hazard_score:.3f}). "
                    f"Edge cost increased from {baseline_edge_cost:.2f} to {updated_edge_cost:.2f}. "
                    f"Dynamic Dijkstra rerouted origin {source_room} from {baseline_exit} to {rec_exit}."
                )
            elif has_smoke or hazard_score > 0.0:
                adaptation_reason = (
                    f"Corridor edge {u} <-> {v} compromised by Smoke Hazard ({hazard_score:.3f}). "
                    f"Air quality and visibility impaired. Rerouted from {baseline_exit} to {rec_exit}."
                )
            elif crowd_density >= 0.70:
                adaptation_reason = (
                    f"Severe occupant congestion on {u} <-> {v} (Density: {crowd_density:.3f}). "
                    f"Dynamic Dijkstra diverted path from {baseline_exit} to {rec_exit} to prevent crowd crush."
                )
            else:
                adaptation_reason = (
                    f"Dynamic path cost adjustments made alternate route via {rec_exit} globally optimal."
                )
        else:
            if is_emergency:
                adaptation_reason = (
                    f"Hazard/congestion active on {u} <-> {v} (Score: {hazard_score:.3f}), "
                    f"and route via {rec_exit} remains the lowest-cost safe egress path (Total Cost: {total_route_cost:.2f})."
                )
            else:
                adaptation_reason = (
                    f"Direct egress route to {baseline_exit} remains unobstructed and optimal "
                    f"(Total Cost: {total_route_cost:.2f}). No detour required."
                )

        return {
            "is_emergency": is_emergency,
            "status_level": hazard_status,
            "hazard_type": hazard_type,
            "crowd_status": crowd_status,
            "zone_status": zone_status,
            "active_edge": active_edge,
            "active_zone_str": f"{u} <-> {v}",
            "source_room": source_room,
            "fire_detected": has_fire,
            "smoke_detected": has_smoke,
            "hazard_score": round(hazard_score, 3),
            "people_count": people_count,
            "crowd_density": round(crowd_density, 3),
            "baseline_route": baseline_route,
            "baseline_route_str": " -> ".join(baseline_route) if baseline_route else "NONE",
            "baseline_exit": baseline_exit,
            "baseline_cost": round(baseline_cost, 2),
            "recommended_route": rec_route,
            "recommended_route_str": " -> ".join(rec_route) if rec_route else "NO SAFE ROUTE",
            "recommended_exit": rec_exit,
            "total_route_cost": round(total_route_cost, 2),
            "cost_delta_vs_baseline": round(total_route_cost - baseline_cost, 2),
            "compromised_exits": compromised_exits,
            "avoided_exit": avoided_exit,
            "is_diverted_from_baseline": is_diverted_from_baseline,
            "route_changed_this_frame": route_changed_this_frame,
            "previous_route": previous_route,
            "adaptation_reason": adaptation_reason,
        }

    REQUIRED_EMERGENCY_STATE_KEYS: Tuple[str, ...] = (
        "is_emergency",
        "status_level",
        "hazard_type",
        "crowd_status",
        "zone_status",
        "active_edge",
        "active_zone_str",
        "source_room",
        "fire_detected",
        "smoke_detected",
        "hazard_score",
        "people_count",
        "crowd_density",
        "baseline_route",
        "baseline_route_str",
        "baseline_exit",
        "baseline_cost",
        "recommended_route",
        "recommended_route_str",
        "recommended_exit",
        "total_route_cost",
        "cost_delta_vs_baseline",
        "compromised_exits",
        "avoided_exit",
        "is_diverted_from_baseline",
        "route_changed_this_frame",
        "previous_route",
        "adaptation_reason",
        "dynamic_signage",
        "pa_announcement",
        "active_alerts_count",
        "unacknowledged_alerts_count",
    )

    def ensure_emergency_state(
        self,
        result: Dict[str, Any],
        source_room: str = "ROOM_A",
        active_edge: Tuple[str, str] = ("CORRIDOR_1", "EXIT_A")
    ) -> Dict[str, Any]:
        """
        Guarantees that `result["emergency_state"]` exists and contains all required
        fields of the unified facility emergency state. If `result` is missing
        `emergency_state` or any required key (e.g. from a cached/skipped frame or
        legacy result dict), reconstructs and synchronizes the full emergency state.
        """
        self._ensure_compatible_alert_manager()
        existing = result.get("emergency_state")
        core_keys = self.REQUIRED_EMERGENCY_STATE_KEYS[:-4]

        if not isinstance(existing, dict) or not all(k in existing for k in core_keys):
            graph = self.building_graph.get_graph()
            all_exits = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "EXIT"]
            if not all_exits:
                all_exits = ["EXIT_A", "EXIT_B"]

            edge_val = result.get("active_edge", active_edge)
            if not isinstance(edge_val, tuple) or len(edge_val) != 2:
                edge_val = active_edge

            rec_route = result.get("recommended_route", [])
            rec_exit = result.get("recommended_exit", "NONE")
            total_cost = float(result.get("total_route_cost", 0.0))

            rebuilt = self.build_facility_emergency_state(
                graph=graph,
                source_room=source_room,
                active_edge=edge_val,
                all_exits=all_exits,
                has_fire=bool(result.get("fire_detected", False)),
                has_smoke=bool(result.get("smoke_detected", False)),
                hazard_score=float(result.get("hazard_score", 0.0)),
                people_count=int(result.get("people_detected", 0)),
                crowd_density=float(result.get("crowd_density", 0.0)),
                rec_route=rec_route,
                rec_exit=rec_exit,
                total_route_cost=total_cost,
                route_changed_this_frame=bool(result.get("route_changed", False)),
                previous_route=result.get("previous_route")
            )
            self._sync_alert_manager_state(rebuilt)
            em_state = rebuilt
        else:
            em_state = existing.copy()

        em_state["dynamic_signage"] = self.alert_manager.dynamic_signage
        em_state["pa_announcement"] = self.alert_manager.current_pa_message
        em_state["active_alerts_count"] = len(self.alert_manager.active_alerts)
        em_state["unacknowledged_alerts_count"] = len(
            [a for a in self.alert_manager.active_alerts if not a.get("acknowledged", False)]
        )
        self.current_emergency_state = em_state
        result["emergency_state"] = em_state
        return em_state

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
        6. Unified Facility Emergency State construction, Alert Manager & Signage dispatch.
        7. OpenCV HUD overlay rendering.
        """
        # Validate frame
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("[!] Input frame is empty or invalid.")

        self._ensure_compatible_alert_manager()
        self.frame_counter += 1
        t_start_total = time.time()

        h, w = frame.shape[:2]
        cache_sig = (
            active_edge,
            source_room,
            frame.shape,
            int(frame[0, 0, 0]),
            int(frame[h // 2, w // 2, 0]),
            int(frame[-1, -1, 0])
        )

        # Handle frame skipping in continuous video stream mode ONLY when configuration & frame match
        if (
            enable_frame_skip
            and (self.frame_counter % max(1, skip_interval) != 0)
            and self.cached_result is not None
            and self._last_cache_signature == cache_sig
        ):
            self.skipped_frames_counter += 1
            cached = self.cached_result.copy()
            cached["frames_processed"] = self.processed_frames_counter
            cached["frames_skipped"] = self.skipped_frames_counter
            self.ensure_emergency_state(cached, source_room=source_room, active_edge=active_edge)
            cached["dynamic_signage"] = self.alert_manager.dynamic_signage
            cached["current_pa_message"] = self.alert_manager.current_pa_message
            cached["security_guard_log"] = self.alert_manager.get_guard_log()
            cached["route_history"] = self.alert_manager.get_route_history()
            if "latency_metrics" in cached and isinstance(cached["latency_metrics"], dict):
                lat_copy = cached["latency_metrics"].copy()
                lat_copy["frame_number"] = self.frame_counter
                lat_copy["frames_processed"] = self.processed_frames_counter
                lat_copy["frames_skipped"] = self.skipped_frames_counter
                cached["latency_metrics"] = lat_copy
            return cached

        self.processed_frames_counter += 1
        self._last_cache_signature = cache_sig
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

        # Reset previously monitored edge if active_edge changed
        if self._last_active_edge is not None and self._last_active_edge != active_edge:
            prev_u, prev_v = self._last_active_edge
            if graph.has_edge(prev_u, prev_v):
                self.building_graph.update_edge_hazard(prev_u, prev_v, 0.0)
                self.building_graph.update_edge_density(prev_u, prev_v, 0.0)
        self._last_active_edge = active_edge

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

        # Route Change Detection (recorded ONLY when chosen route genuinely changes from previous route)
        previous_route = self.last_routes.get(source_room, None)
        previous_cost = self.last_costs.get(source_room, total_route_cost)
        route_changed = False
        route_change_event = None

        if previous_route is not None and previous_route != rec_route:
            route_changed = True
            # Determine causal trigger
            if has_fire or hazard_score >= 0.70:
                reason = f"Fire hazard confirmed on {u} <-> {v} (Score: {hazard_score:.2f}); rerouted to {rec_exit}"
            elif hazard_score > 0.0:
                reason = f"Hazard increased on {u} <-> {v} (Score: {hazard_score:.2f}); rerouted to {rec_exit}"
            elif crowd_density > 0.40:
                reason = f"Crowd density increased on {u} <-> {v} (Density: {crowd_density:.2f}); rerouted to {rec_exit}"
            else:
                reason = f"Hazard cleared on {u} <-> {v}; restored optimal route via {rec_exit}"

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

        # Update route memory
        self.last_routes[source_room] = rec_route
        self.last_costs[source_room] = total_route_cost

        # --- STEP 5B: BUILD & SYNCHRONIZE UNIFIED FACILITY EMERGENCY STATE ---
        emergency_state = self.build_facility_emergency_state(
            graph=graph,
            source_room=source_room,
            active_edge=active_edge,
            all_exits=all_exits,
            has_fire=has_fire,
            has_smoke=has_smoke,
            hazard_score=hazard_score,
            people_count=people_count,
            crowd_density=crowd_density,
            rec_route=rec_route,
            rec_exit=rec_exit,
            total_route_cost=total_route_cost,
            route_changed_this_frame=route_changed,
            previous_route=previous_route
        )
        self._sync_alert_manager_state(emergency_state)

        # Trigger zone alert & security guard dispatch if active fire/smoke or severe crowd congestion
        if has_fire or has_smoke:
            self.alert_manager.trigger_hazard_alert(
                location=f"{u} <-> {v}",
                hazard_type="FIRE" if has_fire else "SMOKE",
                severity="CRITICAL" if (has_fire or hazard_score >= 0.70) else "WARNING",
                hazard_score=hazard_score,
                people_count=people_count,
                crowd_density=crowd_density,
                recommended_exit=rec_exit,
                route_str=rec_route_str,
                frame=frame
            )
        elif crowd_density >= 0.70:
            self.alert_manager.trigger_hazard_alert(
                location=f"{u} <-> {v}",
                hazard_type="CROWD_CONGESTION",
                severity="WARNING",
                hazard_score=hazard_score,
                people_count=people_count,
                crowd_density=crowd_density,
                recommended_exit=rec_exit,
                route_str=rec_route_str,
                frame=frame
            )

        # Enrich unified emergency_state with synchronized alert/signage/PA state
        emergency_state["dynamic_signage"] = self.alert_manager.dynamic_signage
        emergency_state["pa_announcement"] = self.alert_manager.current_pa_message
        emergency_state["active_alerts_count"] = len(self.alert_manager.active_alerts)
        emergency_state["unacknowledged_alerts_count"] = len(
            [a for a in self.alert_manager.active_alerts if not a.get("acknowledged", False)]
        )
        self.current_emergency_state = emergency_state

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
            # Unified Emergency State & IEEE Extended Features:
            "emergency_state": emergency_state,
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

        self.ensure_emergency_state(result_dict, source_room=source_room, active_edge=active_edge)
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
