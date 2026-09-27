"""
Test and Verification Script for Building Zone Mapping and Graph Model.
Creates the demo building graph, inspects nodes/edges/lengths, validates connectivity,
and exports the visualization to data/processed/building_graph.png.
"""

import sys
from pathlib import Path
import networkx as nx

# Add project root to sys.path so config and graph packages are discoverable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import OUTPUT_DIR
from graph.building_graph import BuildingGraph


def test_building_graph():
    print("\n" + "=" * 70)
    print("  PHASE 4: BUILDING ZONE MAPPING AND GRAPH MODEL TEST")
    print("=" * 70)

    # 1. Instantiate BuildingGraph and construct demo facility layout
    bg = BuildingGraph(name="Demo Evacuation Testbed")
    bg.create_demo_building()
    g = bg.get_graph()

    print("\n--- [1] NODE LIST (Total: {}) ---".format(len(g.nodes)))
    print(f"{'Node ID':<15} | {'Type':<10} | {'(X, Y)':<12} | {'Description'}")
    print("-" * 70)
    for node_id, data in g.nodes(data=True):
        coords = f"({data['x']:.1f}, {data['y']:.1f})"
        print(f"{node_id:<15} | {data['node_type']:<10} | {coords:<12} | {data['name']}")

    print("\n--- [2] EDGE LIST & ATTRIBUTES (Total: {}) ---".format(len(g.edges)))
    print(f"{'Connection (U <-> V)':<28} | {'Length':<8} | {'Hazard':<8} | {'Density':<8} | {'Initial Cost'}")
    print("-" * 70)
    for u, v, data in g.edges(data=True):
        conn = f"{u} <-> {v}"
        print(
            f"{conn:<28} | "
            f"{data['length']:<6.2f}m | "
            f"{data['hazard']:<8.2f} | "
            f"{data['density']:<8.2f} | "
            f"{data['cost']:.2f}"
        )

    # 3. Verify Graph Connectivity
    print("\n--- [3] GRAPH TOPOLOGY & CONNECTIVITY VERIFICATION ---")
    is_connected = nx.is_connected(g)
    print(f"[*] Overall Graph Connected: {'YES (Valid)' if is_connected else 'NO (Isolated components detected)'}")

    # Check exit reachability from every room
    rooms = [n for n, d in g.nodes(data=True) if d["node_type"] == BuildingGraph.NODE_TYPE_ROOM]
    exits = [n for n, d in g.nodes(data=True) if d["node_type"] == BuildingGraph.NODE_TYPE_EXIT]

    print(f"[*] Source Rooms: {rooms}")
    print(f"[*] Emergency Exits: {exits}")

    all_paths_valid = True
    for room in rooms:
        for exit_node in exits:
            has_path = nx.has_path(g, room, exit_node)
            shortest_hops = nx.shortest_path_length(g, room, exit_node) if has_path else None
            if not has_path:
                all_paths_valid = False
            print(f"    Path {room:<8} -> {exit_node:<8}: Accessible={has_path} (Min Hops: {shortest_hops})")

    if is_connected and all_paths_valid:
        print("[+] SUCCESS: All rooms have guaranteed physical paths to all emergency exits!")
    else:
        print("[!] WARNING: Some rooms cannot reach an emergency exit.")

    # 4. Generate Graph Visualization
    print("\n--- [4] GRAPH VISUALIZATION GENERATION ---")
    out_file = OUTPUT_DIR / "building_graph.png"
    bg.visualize_graph(output_path=out_file)
    print(f"[+] Visualization exported: {out_file}")

    print("\n" + "=" * 70)
    print("  PHASE 4 TEST COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    test_building_graph()
