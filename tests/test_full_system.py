"""
Phase B: Comprehensive Functional Test Suite for AI Fire & Crowd Evacuation System.
Verifies all 14 core system capabilities across perception, graph modeling,
dynamic optimization, integration, alerts, and video stream handlers.
Features granular real-time progress logging ([RUN], [INFO], [PASS], [FAIL]).
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DATA_DIR, MODELS_DIR, FIRE_MODEL_PATH, CROWD_MODEL_PATH


class SystemTestRunner:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []

    def start_test(self, test_num: int, name: str):
        print(f"\n[RUN]  Test {test_num:02d}: {name}...", flush=True)

    def log_info(self, message: str):
        print(f"       [INFO] {message}", flush=True)

    def record(self, test_num: int, name: str, passed: bool, message: str = ""):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "PASS"
        else:
            self.failed_tests += 1
            status = "FAIL"
        self.results.append((test_num, name, status, message))
        print(f"[{status}] Test {test_num:02d}: {name} - {message}", flush=True)

    def run_all(self):
        print("=" * 70, flush=True)
        print("AI EVACUATION OPTIMIZATION SYSTEM — FUNCTIONAL TEST SUITE", flush=True)
        print("=" * 70, flush=True)

        t_suite_start = time.time()

        # -------------------------------------------------------------
        # Test 1: Fire Detector Initialization
        # -------------------------------------------------------------
        self.start_test(1, "Fire Detector Initialization")
        t0 = time.time()
        try:
            from detection.fire_detector import FireSmokeDetector
            self.log_info(f"Loading weights from: {FIRE_MODEL_PATH}")
            fire_det = FireSmokeDetector(model_path=str(FIRE_MODEL_PATH), conf_threshold=0.35)
            assert hasattr(fire_det, "model"), "Model not loaded"
            assert "fire" in [c.lower() for c in fire_det.classes.values()], "Missing fire class"
            self.log_info(f"Model loaded with classes: {fire_det.classes} in {time.time() - t0:.2f}s")
            self.record(1, "Fire Detector Initialization", True, "YOLOv8n Fire/Smoke loaded successfully on CPU")
        except Exception as e:
            self.record(1, "Fire Detector Initialization", False, f"Error: {e}")
            fire_det = None

        # -------------------------------------------------------------
        # Test 2: Smoke Image Detection
        # -------------------------------------------------------------
        self.start_test(2, "Smoke Image Detection")
        t0 = time.time()
        try:
            smoke_path = DATA_DIR / "test" / "fire_sample.jpg"
            assert smoke_path.exists(), f"Sample image missing at {smoke_path}"
            import cv2
            self.log_info(f"Reading test smoke image: {smoke_path.name}")
            smoke_frame = cv2.imread(str(smoke_path))
            assert smoke_frame is not None, "Failed to decode smoke image"
            self.log_info(f"Image decoded: shape={smoke_frame.shape}. Running YOLO inference...")
            t_inf = time.time()
            dets = fire_det.detect(smoke_frame)
            self.log_info(f"Inference completed in {time.time() - t_inf:.2f}s. Raw detections: {len(dets)}")
            has_smoke = any(d["class_name"] == "smoke" for d in dets)
            hazard = fire_det.calculate_hazard_score(dets)
            self.log_info(f"Has smoke: {has_smoke} | Computed Hazard Score: {hazard:.3f}")
            assert has_smoke, "Smoke was not detected in smoke benchmark image"
            assert hazard > 0.0, f"Hazard score is zero ({hazard})"
            self.record(2, "Smoke Image Detection", True, f"Detected smoke (Hazard Score: {hazard:.3f}) in {time.time() - t0:.2f}s")
        except Exception as e:
            self.record(2, "Smoke Image Detection", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 3: Fire Image Detection
        # -------------------------------------------------------------
        self.start_test(3, "Fire Image Detection")
        t0 = time.time()
        try:
            flames_path = DATA_DIR / "test" / "flames_sample.jpg"
            assert flames_path.exists(), f"Sample image missing at {flames_path}"
            import cv2
            self.log_info(f"Reading test fire image: {flames_path.name}")
            flames_frame = cv2.imread(str(flames_path))
            assert flames_frame is not None, "Failed to decode flames image"
            self.log_info(f"Image decoded: shape={flames_frame.shape}. Running YOLO inference...")
            t_inf = time.time()
            dets = fire_det.detect(flames_frame)
            self.log_info(f"Inference completed in {time.time() - t_inf:.2f}s. Raw detections: {len(dets)}")
            has_fire = any(d["class_name"] == "fire" for d in dets)
            hazard = fire_det.calculate_hazard_score(dets)
            self.log_info(f"Has fire: {has_fire} | Computed Hazard Score: {hazard:.3f}")
            assert has_fire, "Fire was not detected in flames benchmark image"
            assert hazard > 0.0, f"Hazard score is zero ({hazard})"
            self.record(3, "Fire Image Detection", True, f"Detected fire (Hazard Score: {hazard:.3f}) in {time.time() - t0:.2f}s")
        except Exception as e:
            self.record(3, "Fire Image Detection", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 4: Crowd Detector Initialization
        # -------------------------------------------------------------
        self.start_test(4, "Crowd Detector Initialization")
        t0 = time.time()
        try:
            from detection.crowd_detector import CrowdDetector
            self.log_info(f"Loading weights from: {CROWD_MODEL_PATH}")
            crowd_det = CrowdDetector(model_path=str(CROWD_MODEL_PATH), conf_threshold=0.35)
            assert hasattr(crowd_det, "model"), "Model not loaded"
            assert crowd_det.model.names[0].lower() == "person", "Class 0 is not person"
            self.log_info(f"Person model loaded in {time.time() - t0:.2f}s")
            self.record(4, "Crowd Detector Initialization", True, "YOLOv8n Person/Crowd detector loaded on CPU")
        except Exception as e:
            self.record(4, "Crowd Detector Initialization", False, f"Error: {e}")
            crowd_det = None

        # -------------------------------------------------------------
        # Test 5: Crowd Image Detection
        # -------------------------------------------------------------
        self.start_test(5, "Crowd Image Detection")
        t0 = time.time()
        try:
            crowd_path = DATA_DIR / "test" / "crowd_sample.jpg"
            assert crowd_path.exists(), f"Sample image missing at {crowd_path}"
            import cv2
            self.log_info(f"Reading test crowd image: {crowd_path.name}")
            crowd_frame = cv2.imread(str(crowd_path))
            assert crowd_frame is not None, "Failed to decode crowd image"
            self.log_info(f"Image decoded: shape={crowd_frame.shape}. Running YOLO inference...")
            t_inf = time.time()
            crowd_res = crowd_det.detect(crowd_frame)
            count = crowd_res["count"]
            density = crowd_res["density"]
            self.log_info(f"Inference completed in {time.time() - t_inf:.2f}s. Detected: {count} people, Density: {density:.3f}")
            assert count > 0, "No persons detected in crowd benchmark image"
            assert 0.0 < density <= 1.0, f"Density out of bounds: {density}"
            assert len(crowd_res["centers"]) == count, "Centers count mismatch"
            self.record(5, "Crowd Image Detection", True, f"Detected {count} people (Density: {density:.3f}) in {time.time() - t0:.2f}s")
        except Exception as e:
            self.record(5, "Crowd Image Detection", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 6: Building Graph Creation
        # -------------------------------------------------------------
        self.start_test(6, "Building Graph Creation")
        t0 = time.time()
        try:
            from graph.building_graph import BuildingGraph
            bg = BuildingGraph(name="Demo Evacuation Testbed")
            bg.create_demo_building()
            g = bg.get_graph()
            expected_nodes = {"ROOM_A", "ROOM_B", "ROOM_C", "CORRIDOR_1", "CORRIDOR_2", "EXIT_A", "EXIT_B"}
            assert set(g.nodes) == expected_nodes, f"Node set mismatch: {set(g.nodes)}"
            assert len(g.edges) == 7, f"Expected 7 edges, got {len(g.edges)}"
            self.log_info(f"Graph initialized: {len(g.nodes)} nodes, {len(g.edges)} edges")
            self.record(6, "Building Graph Creation", True, f"7 nodes and 7 edges populated in {time.time() - t0:.3f}s")
        except Exception as e:
            self.record(6, "Building Graph Creation", False, f"Error: {e}")
            bg, g = None, None

        # -------------------------------------------------------------
        # Test 7: Graph Connectivity
        # -------------------------------------------------------------
        self.start_test(7, "Graph Connectivity")
        t0 = time.time()
        try:
            import networkx as nx
            assert nx.is_connected(g), "Graph is disconnected"
            for room in ["ROOM_A", "ROOM_B", "ROOM_C"]:
                for exit_node in ["EXIT_A", "EXIT_B"]:
                    assert nx.has_path(g, room, exit_node), f"No path between {room} and {exit_node}"
            self.log_info("Verified all 3 rooms have paths to both emergency exits")
            self.record(7, "Graph Connectivity", True, "All rooms have paths to all emergency exits")
        except Exception as e:
            self.record(7, "Graph Connectivity", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 8: Dynamic Edge Cost Calculation
        # -------------------------------------------------------------
        self.start_test(8, "Dynamic Edge Cost Calculation")
        t0 = time.time()
        try:
            from graph.dijkstra_optimizer import DynamicDijkstraOptimizer
            optimizer = DynamicDijkstraOptimizer(alpha=5.0, beta=2.0, gamma=0.2)
            # Normal cost: length=10.0 => 0.2*10 = 2.0
            normal_cost = optimizer.calculate_edge_cost(hazard=0.0, density=0.0, length=10.0)
            assert abs(normal_cost - 2.0) < 0.01, f"Unexpected normal cost: {normal_cost}"
            # Emergency cost: hazard=0.9, density=0.8, length=10.0 => 5*0.9 + 2*0.8 + 0.2*10 = 4.5 + 1.6 + 2.0 = 8.1
            emer_cost = optimizer.calculate_edge_cost(hazard=0.9, density=0.8, length=10.0)
            assert abs(emer_cost - 8.1) < 0.01, f"Unexpected emergency cost: {emer_cost}"
            self.log_info(f"Normal cost = {normal_cost:.2f}, Emergency cost = {emer_cost:.2f}")
            self.record(8, "Dynamic Edge Cost Calculation", True, "Cost(e) = 5.0*Haz + 2.0*Den + 0.2*Len verified")
        except Exception as e:
            self.record(8, "Dynamic Edge Cost Calculation", False, f"Error: {e}")
            optimizer = None

        # -------------------------------------------------------------
        # Test 9: Normal-Condition Dijkstra Route
        # -------------------------------------------------------------
        self.start_test(9, "Normal-Condition Dijkstra Route")
        t0 = time.time()
        try:
            bg.create_demo_building()
            g = bg.get_graph()
            routes_normal = optimizer.find_routes_for_all_rooms(g)
            self.log_info(f"ROOM_A path: {routes_normal['ROOM_A']['path']} (Cost: {routes_normal['ROOM_A']['total_cost']:.2f})")
            assert routes_normal["ROOM_A"]["selected_exit"] == "EXIT_A", "ROOM_A should exit via EXIT_A in normal condition"
            assert routes_normal["ROOM_A"]["path"] == ["ROOM_A", "CORRIDOR_1", "EXIT_A"]
            assert routes_normal["ROOM_C"]["selected_exit"] == "EXIT_B", "ROOM_C should exit via EXIT_B in normal condition"
            self.record(9, "Normal-Condition Dijkstra Route", True, f"ROOM_A -> EXIT_A (Cost: {routes_normal['ROOM_A']['total_cost']:.2f})")
        except Exception as e:
            self.record(9, "Normal-Condition Dijkstra Route", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 10: Hazard-Condition Dijkstra Rerouting
        # -------------------------------------------------------------
        self.start_test(10, "Hazard-Condition Dijkstra Rerouting")
        t0 = time.time()
        try:
            # Inject hazard on CORRIDOR_1 <-> EXIT_A
            bg.update_edge_hazard("CORRIDOR_1", "EXIT_A", 0.90)
            bg.update_edge_density("CORRIDOR_1", "EXIT_A", 0.80)
            optimizer.update_all_edge_costs(g)
            routes_emer = optimizer.find_routes_for_all_rooms(g)
            self.log_info(f"Hazard injected on CORRIDOR_1-EXIT_A. New ROOM_A path: {routes_emer['ROOM_A']['path']}")
            # ROOM_A must now reroute to EXIT_B
            assert routes_emer["ROOM_A"]["selected_exit"] == "EXIT_B", f"ROOM_A failed to reroute: {routes_emer['ROOM_A']['selected_exit']}"
            assert routes_emer["ROOM_A"]["path"] == ["ROOM_A", "CORRIDOR_1", "CORRIDOR_2", "EXIT_B"]
            self.record(10, "Hazard-Condition Dijkstra Rerouting", True, f"ROOM_A dynamically rerouted to EXIT_B (Cost: {routes_emer['ROOM_A']['total_cost']:.2f})")
        except Exception as e:
            self.record(10, "Hazard-Condition Dijkstra Rerouting", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 11: EvacuationSystem Integration
        # -------------------------------------------------------------
        self.start_test(11, "EvacuationSystem Integration")
        t0 = time.time()
        try:
            from integration.evacuation_system import EvacuationSystem
            self.log_info("Instantiating EvacuationSystem...")
            system = EvacuationSystem()
            import cv2
            smoke_frame = cv2.imread(str(DATA_DIR / "test" / "fire_sample.jpg"))
            self.log_info("Executing full perception -> graph mapping -> Dijkstra reroute pipeline...")
            t_pipe = time.time()
            res = system.process_frame(smoke_frame, active_edge=("CORRIDOR_1", "EXIT_A"), source_room="ROOM_A")
            self.log_info(f"Pipeline executed in {time.time() - t_pipe:.2f}s. Route: {res.get('recommended_route_str')}")
            assert "smoke_detected" in res and res["smoke_detected"] is True
            assert "recommended_exit" in res and res["recommended_exit"] == "EXIT_B"
            assert res["annotated_frame"] is not None
            self.record(11, "EvacuationSystem Integration", True, f"Full pipeline executed. Route: {res['recommended_route_str']}")
        except Exception as e:
            self.record(11, "EvacuationSystem Integration", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 12: Dashboard Import & Initialization
        # -------------------------------------------------------------
        self.start_test(12, "Dashboard Import & Initialization")
        t0 = time.time()
        try:
            self.log_info("Loading load_evacuation_system from dashboard.app...")
            from dashboard.app import load_evacuation_system
            dash_sys, err = load_evacuation_system()
            assert err is None, f"Dashboard load error: {err}"
            assert dash_sys is not None, "Dashboard system instance is None"
            self.log_info(f"Dashboard system instance initialized successfully in {time.time() - t0:.2f}s")
            self.record(12, "Dashboard Import & Initialization", True, "Streamlit dashboard engine verified")
        except Exception as e:
            self.record(12, "Dashboard Import & Initialization", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 13: Alert Manager
        # -------------------------------------------------------------
        self.start_test(13, "Alert Manager")
        t0 = time.time()
        try:
            from alerts.alert_manager import AlertManager
            am = AlertManager()
            alert = am.trigger_hazard_alert(location="CORRIDOR_1", hazard_type="FIRE", severity="CRITICAL")
            self.log_info(f"Alert triggered: {alert.get('action', alert.get('type'))} at {alert['timestamp']}")
            assert len(am.active_alerts) == 1
            assert alert["location"] == "CORRIDOR_1"
            am.clear_alerts()
            assert len(am.active_alerts) == 0
            self.log_info("Alert cleared successfully")
            self.record(13, "Alert Manager", True, "Alert triggers and state clearing verified")
        except Exception as e:
            self.record(13, "Alert Manager", False, f"Error: {e}")

        # -------------------------------------------------------------
        # Test 14: Video Stream Initialization
        # -------------------------------------------------------------
        self.start_test(14, "Video Stream Initialization")
        t0 = time.time()
        try:
            from video.video_stream import VideoStreamHandler
            v_handler = VideoStreamHandler(source=0)
            assert hasattr(v_handler, "start_stream")
            assert hasattr(v_handler, "get_frame")
            assert hasattr(v_handler, "release")
            props = v_handler.get_properties()
            assert isinstance(props, dict)
            self.log_info(f"VideoStreamHandler properties: {props}")
            self.record(14, "Video Stream Initialization", True, "VideoStreamHandler methods and metadata interface verified")
        except Exception as e:
            self.record(14, "Video Stream Initialization", False, f"Error: {e}")

        # -------------------------------------------------------------
        # FINAL SUMMARY REPORT
        # -------------------------------------------------------------
        t_total = time.time() - t_suite_start
        print("\n" + "=" * 70, flush=True)
        print("TEST SUMMARY", flush=True)
        print("=" * 70, flush=True)
        print(f"TOTAL TESTS:  {self.total_tests}", flush=True)
        print(f"PASSED:       {self.passed_tests}", flush=True)
        print(f"FAILED:       {self.failed_tests}", flush=True)
        print(f"TOTAL TIME:   {t_total:.2f}s", flush=True)
        print("=" * 70, flush=True)

        if self.failed_tests == 0:
            print("[+] ALL 14 CORE FUNCTIONAL TESTS PASSED SUCCESSFULLY!\n", flush=True)
            return 0
        else:
            print(f"[!] {self.failed_tests} TESTS FAILED.\n", flush=True)
            return 1


if __name__ == "__main__":
    runner = SystemTestRunner()
    code = runner.run_all()
    sys.exit(code)
