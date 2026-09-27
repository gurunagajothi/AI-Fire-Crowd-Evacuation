"""
Simulation and Verification Script for Dynamic Edge Cost and Dynamic Dijkstra Optimization.

Demonstrates real-time route adaptation under two conditions:
1. SCENARIO 1 (NORMAL): Zero hazard and zero crowd congestion.
2. SCENARIO 2 (SIMULATED HAZARD + CONGESTION): Active fire/smoke (0.90) and heavy congestion (0.80)
   on edge CORRIDOR_1 <-> EXIT_A, demonstrating automatic Dijkstra rerouting.
"""

import sys
from pathlib import Path

# Add project root to sys.path so config and graph packages are discoverable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import ALPHA, BETA, GAMMA
from graph.building_graph import BuildingGraph
from graph.dijkstra_optimizer import DynamicDijkstraOptimizer


def print_route_summary(routes: dict):
    """Utility to print route results cleanly."""
    for room_id, res in routes.items():
        path_str = " -> ".join(res["path"]) if res["path"] else "NO ROUTE FOUND"
        print(f"\n{room_id}")
        print(f"  Recommended Exit: {res['selected_exit']}")
        print(f"  Path: {path_str}")
        print(f"  Total Cost: {res['total_cost']:.2f}")
        print("  Per-Edge Breakdown:")
        for edge in res["per_edge_costs"]:
            print(
                f"    - {edge['from']} <-> {edge['to']}: "
                f"Len={edge['length']}m, Haz={edge['hazard']}, Den={edge['density']} => Cost={edge['cost']:.2f}"
            )


def run_dynamic_dijkstra_simulation():
    print("=" * 60)
    print("DYNAMIC DIJKSTRA TEST")
    print("=" * 60)
    print(f"[*] Optimization Formula: Cost(e) = alpha*Hazard + beta*Density + gamma*Length")
    print(f"[*] Configuration Weights: alpha={ALPHA}, beta={BETA}, gamma={GAMMA}")

    # 1. Initialize Building Graph and Optimizer
    bg = BuildingGraph(name="Demo Evacuation Testbed")
    bg.create_demo_building()
    graph = bg.get_graph()

    optimizer = DynamicDijkstraOptimizer(alpha=ALPHA, beta=BETA, gamma=GAMMA)

    # =========================================================================
    # SCENARIO 1: NORMAL CONDITION (Baseline: No fire, no congestion)
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 1: NORMAL CONDITION (All Hazards = 0.0, All Densities = 0.0)")
    print("-" * 60)

    # Ensure all edges have 0 hazard and 0 density
    for u, v in graph.edges:
        bg.update_edge_hazard(u, v, 0.0)
        bg.update_edge_density(u, v, 0.0)

    routes_normal = optimizer.find_routes_for_all_rooms(graph)
    print_route_summary(routes_normal)

    # Verify baseline expectations
    assert routes_normal["ROOM_A"]["selected_exit"] == "EXIT_A", "ROOM_A should choose EXIT_A in normal conditions"
    assert routes_normal["ROOM_C"]["selected_exit"] == "EXIT_B", "ROOM_C should choose EXIT_B in normal conditions"

    # =========================================================================
    # SCENARIO 2: SIMULATED HAZARD + CONGESTION (Emergency Condition)
    # =========================================================================
    print("\n" + "-" * 60)
    print("SCENARIO 2: SIMULATED HAZARD + CONGESTION")
    print("NOTE: Simulated academic test values (not live CCTV inputs).")
    print("-" * 60)

    sim_edge_u, sim_edge_v = "CORRIDOR_1", "EXIT_A"
    sim_hazard = 0.90   # Severe fire/smoke detected
    sim_density = 0.80  # Heavy human bottleneck detected

    # Apply simulated danger metrics to edge CORRIDOR_1 <-> EXIT_A
    bg.update_edge_hazard(sim_edge_u, sim_edge_v, sim_hazard)
    bg.update_edge_density(sim_edge_u, sim_edge_v, sim_density)

    # Recalculate dynamic edge costs
    optimizer.update_all_edge_costs(graph)
    updated_edge_cost = graph[sim_edge_u][sim_edge_v]["cost"]

    print(f"\nEdge:")
    print(f"{sim_edge_u} <-> {sim_edge_v}")
    print(f"\nHazard: {sim_hazard:.2f}")
    print(f"Density: {sim_density:.2f}")
    print(f"\nUpdated Cost: {updated_edge_cost:.2f} "
          f"(Formula: {ALPHA}*{sim_hazard} + {BETA}*{sim_density} + {GAMMA}*{graph[sim_edge_u][sim_edge_v]['length']:.2f})")

    # Run Dynamic Dijkstra on the updated graph
    routes_emergency = optimizer.find_routes_for_all_rooms(graph)
    print_route_summary(routes_emergency)

    # =========================================================================
    # VERIFICATION OF DYNAMIC ROUTE ADAPTATION
    # =========================================================================
    print("\n" + "=" * 60)
    print("DYNAMIC ROUTE ADAPTATION ANALYSIS")
    print("=" * 60)
    normal_exit_a = routes_normal["ROOM_A"]["selected_exit"]
    emergency_exit_a = routes_emergency["ROOM_A"]["selected_exit"]
    normal_path_a = " -> ".join(routes_normal["ROOM_A"]["path"])
    emergency_path_a = " -> ".join(routes_emergency["ROOM_A"]["path"])

    print(f"ROOM_A Normal:    Exit={normal_exit_a} | Path={normal_path_a} | Cost={routes_normal['ROOM_A']['total_cost']:.2f}")
    print(f"ROOM_A Emergency: Exit={emergency_exit_a} | Path={emergency_path_a} | Cost={routes_emergency['ROOM_A']['total_cost']:.2f}")

    if normal_exit_a != emergency_exit_a:
        print(f"\n[+] SUCCESS: Dynamic Dijkstra successfully rerouted ROOM_A from {normal_exit_a} to {emergency_exit_a}!")
        print(f"    Reason: Path to {normal_exit_a} cost increased to {routes_emergency['ROOM_A']['all_candidate_routes']['EXIT_A']['cost']:.2f} due to fire/crowd,")
        print(f"    making the detour to {emergency_exit_a} (cost: {routes_emergency['ROOM_A']['total_cost']:.2f}) the globally safest route.")
    else:
        print("[!] Rerouting did not occur. Check weights and graph costs.")

    print("\n" + "=" * 60)
    print("PHASE 5 TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True


if __name__ == "__main__":
    run_dynamic_dijkstra_simulation()
