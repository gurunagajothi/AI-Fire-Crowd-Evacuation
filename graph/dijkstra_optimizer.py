"""
Graph and Optimization Layer: Dynamic Dijkstra Evacuation Path Optimizer.

Core Formulation:
    Cost(e) = alpha * Hazard(e) + beta * Density(e) + gamma * Length(e)

Finds the safest and least-congested evacuation route dynamically across the building graph.
When fire hazards or crowd bottlenecks appear, edge costs rise, prompting Dijkstra's algorithm
to automatically detour evacuees through safer, alternative corridors.
"""

from typing import Dict, Any, List, Tuple, Optional
import networkx as nx

from config.config import ALPHA, BETA, GAMMA


class DynamicDijkstraOptimizer:
    """
    Computes optimal, hazard-aware evacuation paths using Dynamic Dijkstra on a NetworkX graph.
    Recalculates edge weights in real-time as fire/smoke or crowd density conditions change.
    """

    def __init__(self, alpha: float = None, beta: float = None, gamma: float = None):
        """
        Initialize the optimizer with sensitivity weights.

        Args:
            alpha: Weight for fire/smoke hazard (life-safety priority). Defaults to config.ALPHA.
            beta: Weight for crowd density (congestion penalty). Defaults to config.BETA.
            gamma: Weight for physical distance (meters). Defaults to config.GAMMA.
        """
        self.alpha = float(alpha if alpha is not None else ALPHA)
        self.beta = float(beta if beta is not None else BETA)
        self.gamma = float(gamma if gamma is not None else GAMMA)

    def calculate_edge_cost(self, hazard: float, density: float, length: float) -> float:
        """
        Calculates the dynamic traversal cost for an edge using the project formula:
            Cost(e) = alpha * Hazard(e) + beta * Density(e) + gamma * Length(e)

        Validation: Ensures cost is strictly non-negative (Dijkstra requires non-negative weights).
        """
        cost = (self.alpha * float(hazard)) + (self.beta * float(density)) + (self.gamma * float(length))
        return round(max(0.001, cost), 3)

    def update_all_edge_costs(self, graph: nx.Graph):
        """
        Recalculates and stores the dynamic 'cost' attribute for every edge in the graph
        based on its current hazard, density, and physical length attributes.
        """
        for u, v, data in graph.edges(data=True):
            hazard = data.get("hazard", 0.0)
            density = data.get("density", 0.0)
            length = data.get("length", 1.0)

            # Compute new dynamic cost
            new_cost = self.calculate_edge_cost(hazard, density, length)
            graph[u][v]["cost"] = new_cost

    def find_safest_route(
        self,
        graph: nx.Graph,
        source: str,
        exits: List[str]
    ) -> Dict[str, Any]:
        """
        Computes the safest and least-congested path from 'source' to the closest viable exit.

        Evaluates the dynamic shortest path to all candidate exits using NetworkX Dijkstra,
        then selects the exit that minimizes total cumulative path cost.

        Args:
            graph: NetworkX building graph with updated edge 'cost' attributes.
            source: Starting room or corridor node ID (e.g., 'ROOM_A').
            exits: List of accessible exit node IDs (e.g., ['EXIT_A', 'EXIT_B']).

        Returns:
            dict containing:
            - "source": str
            - "selected_exit": str (best exit node)
            - "path": list of node IDs (the full evacuation route)
            - "total_cost": float (cumulative path cost)
            - "per_edge_costs": list of dicts with details for each edge traversed
            - "all_candidate_routes": dict of all evaluated exit paths
        """
        # 1. Validation checks
        if source not in graph.nodes:
            raise KeyError(f"[!] Source node '{source}' does not exist in graph.")
        if not exits:
            raise ValueError("[!] No emergency exits provided for evacuation calculation.")

        valid_exits = [e for e in exits if e in graph.nodes]
        if not valid_exits:
            raise ValueError(f"[!] None of the specified exits {exits} exist in graph.")

        # Ensure edge costs are synchronized
        self.update_all_edge_costs(graph)

        best_exit: Optional[str] = None
        best_path: Optional[List[str]] = None
        min_cost: float = float("inf")
        candidate_routes: Dict[str, Any] = {}

        # 2. Evaluate shortest path to each candidate exit
        for exit_node in valid_exits:
            try:
                # Use NetworkX Dijkstra shortest path algorithm weighted by dynamic 'cost'
                path = nx.dijkstra_path(graph, source=source, target=exit_node, weight="cost")
                cost = nx.dijkstra_path_length(graph, source=source, target=exit_node, weight="cost")
                cost = round(float(cost), 2)

                candidate_routes[exit_node] = {
                    "path": path,
                    "cost": cost,
                    "reachable": True
                }

                # Select the exit with minimum overall cost
                if cost < min_cost:
                    min_cost = cost
                    best_exit = exit_node
                    best_path = path

            except nx.NetworkXNoPath:
                # Handle unreachable exit scenario safely
                candidate_routes[exit_node] = {
                    "path": None,
                    "cost": float("inf"),
                    "reachable": False
                }

        # 3. Validation: Verify at least one exit is reachable
        if best_path is None or best_exit is None:
            return {
                "source": source,
                "selected_exit": None,
                "path": [],
                "total_cost": float("inf"),
                "per_edge_costs": [],
                "all_candidate_routes": candidate_routes,
                "status": "UNREACHABLE"
            }

        # 4. Extract detailed per-edge breakdown for the winning route
        per_edge_costs = []
        for i in range(len(best_path) - 1):
            u = best_path[i]
            v = best_path[i + 1]
            edge_data = graph[u][v]
            per_edge_costs.append({
                "from": u,
                "to": v,
                "length": edge_data.get("length", 0.0),
                "hazard": edge_data.get("hazard", 0.0),
                "density": edge_data.get("density", 0.0),
                "cost": edge_data.get("cost", 0.0)
            })

        return {
            "source": source,
            "selected_exit": best_exit,
            "path": best_path,
            "total_cost": min_cost,
            "per_edge_costs": per_edge_costs,
            "all_candidate_routes": candidate_routes,
            "status": "OPTIMAL"
        }

    def find_routes_for_all_rooms(
        self,
        graph: nx.Graph,
        rooms: Optional[List[str]] = None,
        exits: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculates the safest evacuation route from all designated rooms (e.g., ROOM_A, ROOM_B, ROOM_C)
        to the optimal available emergency exit.

        Args:
            graph: NetworkX building graph.
            rooms: Optional list of room IDs. Defaults to all nodes with node_type == 'ROOM'.
            exits: Optional list of exit IDs. Defaults to all nodes with node_type == 'EXIT'.

        Returns:
            Dictionary mapping room_id -> safest route results dict.
        """
        # Auto-discover rooms and exits from graph node attributes if omitted
        if rooms is None:
            rooms = [
                n for n, d in graph.nodes(data=True)
                if d.get("node_type") == "ROOM"
            ]
            if not rooms:
                rooms = ["ROOM_A", "ROOM_B", "ROOM_C"]

        if exits is None:
            exits = [
                n for n, d in graph.nodes(data=True)
                if d.get("node_type") == "EXIT"
            ]
            if not exits:
                exits = ["EXIT_A", "EXIT_B"]

        # Ensure all edge weights reflect current dynamic conditions
        self.update_all_edge_costs(graph)

        routes = {}
        for room_id in rooms:
            routes[room_id] = self.find_safest_route(graph, source=room_id, exits=exits)

        return routes


# Backward-compatibility alias
EvacuationOptimizer = DynamicDijkstraOptimizer
