"""
Graph and Optimization Layer: Multi-Floor Building Graph Extension.
Extends the baseline 7-node topological model to support multi-floor digital twin facilities
with stairwell connections between floors, while preserving full compatibility with
the default single-floor 7-node demonstration topology.
"""

from typing import Dict, Any, List, Optional, Tuple
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from graph.building_graph import BuildingGraph


class MultiFloorBuildingGraph:
    """
    Multi-Floor topological digital twin representing:
    - Ground Floor (Floor 0): ROOM_A, ROOM_B, CORRIDOR_1, STAIR_1, EXIT_A
    - First Floor (Floor 1): ROOM_C, ROOM_D, CORRIDOR_2, STAIR_1, EXIT_B
    Connected vertically via STAIR_1.
    """

    NODE_TYPE_ROOM = "ROOM"
    NODE_TYPE_CORRIDOR = "CORRIDOR"
    NODE_TYPE_STAIR = "STAIR"
    NODE_TYPE_EXIT = "EXIT"

    def __init__(self, name: str = "Multi-Floor Facility Digital Twin"):
        self.name = name
        self.graph = nx.Graph()
        self.create_multi_floor_demo()

    def create_multi_floor_demo(self):
        """Constructs the standard 2-floor demonstration topology."""
        self.graph.clear()

        # GROUND FLOOR (Floor 0)
        self.graph.add_node("G_ROOM_A", node_type=self.NODE_TYPE_ROOM, floor=0, name="Office 101", x=0.0, y=10.0)
        self.graph.add_node("G_ROOM_B", node_type=self.NODE_TYPE_ROOM, floor=0, name="Office 102", x=0.0, y=5.0)
        self.graph.add_node("G_CORRIDOR_1", node_type=self.NODE_TYPE_CORRIDOR, floor=0, name="Ground Central Hall", x=10.0, y=8.0)
        self.graph.add_node("STAIR_1", node_type=self.NODE_TYPE_STAIR, floor=0, name="Central Emergency Stairwell", x=10.0, y=4.0)
        self.graph.add_node("G_EXIT_A", node_type=self.NODE_TYPE_EXIT, floor=0, name="Ground Main Exit (North)", x=20.0, y=10.0)

        # FIRST FLOOR (Floor 1)
        self.graph.add_node("F1_ROOM_C", node_type=self.NODE_TYPE_ROOM, floor=1, name="Lab 201", x=0.0, y=8.0)
        self.graph.add_node("F1_ROOM_D", node_type=self.NODE_TYPE_ROOM, floor=1, name="Lab 202", x=0.0, y=2.0)
        self.graph.add_node("F1_CORRIDOR_2", node_type=self.NODE_TYPE_CORRIDOR, floor=1, name="First Floor Hallway", x=10.0, y=2.0)
        self.graph.add_node("F1_EXIT_B", node_type=self.NODE_TYPE_EXIT, floor=1, name="First Floor Fire Escape (South)", x=20.0, y=0.0)

        # Connect Ground Floor Edges
        self._connect("G_ROOM_A", "G_CORRIDOR_1", 10.20)
        self._connect("G_ROOM_B", "G_CORRIDOR_1", 10.44)
        self._connect("G_CORRIDOR_1", "G_EXIT_A", 10.20)
        self._connect("G_CORRIDOR_1", "STAIR_1", 4.00)

        # Connect First Floor Edges
        self._connect("F1_ROOM_C", "F1_CORRIDOR_2", 10.44)
        self._connect("F1_ROOM_D", "F1_CORRIDOR_2", 10.20)
        self._connect("F1_CORRIDOR_2", "F1_EXIT_B", 10.20)
        self._connect("F1_CORRIDOR_2", "STAIR_1", 4.00)

        # Vertical connection through Stairwell (stair traversal length ~ 6.0m)
        self._connect("G_CORRIDOR_1", "F1_CORRIDOR_2", 8.00)

    def _connect(self, u: str, v: str, length: float):
        self.graph.add_edge(
            u, v,
            length=round(float(length), 2),
            hazard=0.0,
            density=0.0,
            cost=round(float(length), 2)
        )

    def get_floor_subgraph(self, floor: int) -> nx.Graph:
        """Returns nodes and edges belonging to a specific floor (including shared stairs)."""
        nodes = [n for n, d in self.graph.nodes(data=True) if d.get("floor") == floor or d.get("node_type") == self.NODE_TYPE_STAIR]
        return self.graph.subgraph(nodes).copy()

    def render_floor_map(self, floor: int = 0, recommended_route: Optional[List[str]] = None,
                         active_hazards: Optional[List[str]] = None, dark_mode: bool = True) -> plt.Figure:
        """Renders 2D topological layout for a specific floor."""
        sub = self.get_floor_subgraph(floor)
        fig, ax = plt.subplots(figsize=(9, 5))

        if dark_mode:
            fig.patch.set_facecolor('#070B14')
            ax.set_facecolor('#0D1424')

        pos = {n: (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in sub.nodes}

        # Colors
        color_map = {
            self.NODE_TYPE_ROOM: "#38BDF8",
            self.NODE_TYPE_CORRIDOR: "#F59E0B",
            self.NODE_TYPE_STAIR: "#A855F7",
            self.NODE_TYPE_EXIT: "#10B981"
        }
        node_colors = [color_map.get(sub.nodes[n].get("node_type"), "#94A3B8") for n in sub.nodes]

        # Draw base edges
        nx.draw_networkx_edges(sub, pos, ax=ax, width=2.5, edge_color="#334155" if dark_mode else "#78909C")

        # Highlight Route
        if recommended_route and len(recommended_route) > 1:
            route_edges = [
                (recommended_route[i], recommended_route[i + 1])
                for i in range(len(recommended_route) - 1)
                if sub.has_edge(recommended_route[i], recommended_route[i + 1])
            ]
            if route_edges:
                nx.draw_networkx_edges(sub, pos, edgelist=route_edges, ax=ax, width=5.0, edge_color="#00FF88")

        # Draw nodes
        nx.draw_networkx_nodes(sub, pos, ax=ax, node_size=1600, node_color=node_colors, edgecolors="#070B14" if dark_mode else "#263238", linewidths=2.0)
        nx.draw_networkx_labels(sub, pos, ax=ax, font_size=8, font_weight="bold", font_color="#FFFFFF")

        # Edge distance labels
        edge_labels = {(u, v): f"{d['length']}m" for u, v, d in sub.edges(data=True)}
        nx.draw_networkx_edge_labels(
            sub, pos, edge_labels=edge_labels, font_size=7.5,
            font_color="#94A3B8" if dark_mode else "#263238",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#121C30" if dark_mode else "#F5F5F5", edgecolor="#283D63" if dark_mode else "#CFD8DC")
        )

        floor_name = "GROUND FLOOR (Level 0)" if floor == 0 else "FIRST FLOOR (Level 1)"
        ax.set_title(f"Multi-Floor Digital Twin — {floor_name}", fontsize=11, fontweight="bold", color="#00F0FF" if dark_mode else "#263238", pad=10)
        ax.tick_params(colors='#64748B' if dark_mode else '#263238')
        ax.grid(True, linestyle="--", alpha=0.2, color='#1E293B' if dark_mode else '#CFD8DC')
        plt.tight_layout()
        return fig
