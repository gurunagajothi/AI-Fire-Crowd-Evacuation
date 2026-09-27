"""
Graph and Optimization Layer: Building Zone Mapping and Graph Representation.
Represents facility architecture as a topological network using NetworkX.

NOTE: This is an academic prototype containing a clearly defined, configurable
DEMO building layout. The spatial coordinates represent a synthetic 2D floor grid
(in meters) that can later be replaced with calibrated real-world floor plans or CAD maps.
"""

import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config.config import OUTPUT_DIR


class BuildingGraph:
    """
    Topological building representation as a NetworkX graph.
    Nodes represent rooms, corridor intersections, and emergency exits.
    Edges represent hallways, doors, and passages with length, hazard, density, and cost attributes.
    """

    NODE_TYPE_ROOM = "ROOM"
    NODE_TYPE_CORRIDOR = "CORRIDOR"
    NODE_TYPE_EXIT = "EXIT"

    def __init__(self, name: str = "Demo Facility Layout"):
        self.name = name
        self.graph = nx.Graph()

    def add_room(self, node_id: str, name: str, x: float, y: float) -> str:
        """Add a room node to the building graph."""
        self.graph.add_node(
            node_id,
            node_type=self.NODE_TYPE_ROOM,
            name=name,
            x=float(x),
            y=float(y)
        )
        return node_id

    def add_corridor(self, node_id: str, name: str, x: float, y: float) -> str:
        """Add a corridor/hallway junction node to the building graph."""
        self.graph.add_node(
            node_id,
            node_type=self.NODE_TYPE_CORRIDOR,
            name=name,
            x=float(x),
            y=float(y)
        )
        return node_id

    def add_exit(self, node_id: str, name: str, x: float, y: float) -> str:
        """Add an emergency exit node to the building graph."""
        self.graph.add_node(
            node_id,
            node_type=self.NODE_TYPE_EXIT,
            name=name,
            x=float(x),
            y=float(y)
        )
        return node_id

    def connect_nodes(self, u: str, v: str, length: Optional[float] = None) -> Tuple[str, str]:
        """
        Connect two nodes with an edge.
        If length is omitted, it is automatically computed as the Euclidean distance
        between the nodes' (x, y) coordinates.

        Initial edge attributes:
        - length: physical distance (meters)
        - hazard: 0.0 (baseline, updated via perception layer)
        - density: 0.0 (baseline, updated via perception layer)
        - cost: baseline traversal cost (initially equals length)
        """
        if u not in self.graph.nodes:
            raise KeyError(f"Node '{u}' does not exist in graph.")
        if v not in self.graph.nodes:
            raise KeyError(f"Node '{v}' does not exist in graph.")

        if length is None:
            # Calculate Euclidean distance from (x, y) coordinates
            x1, y1 = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
            x2, y2 = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
            length = round(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), 2)
        else:
            length = round(float(length), 2)

        # Store attributes on the edge
        self.graph.add_edge(
            u,
            v,
            length=length,
            hazard=0.0,
            density=0.0,
            cost=length  # Initial cost equals physical distance
        )
        return u, v

    def update_edge_hazard(self, u: str, v: str, hazard_score: float):
        """Update the detected fire/smoke hazard score (0.0 to 1.0) on edge (u, v)."""
        if not self.graph.has_edge(u, v):
            raise KeyError(f"Edge ({u}, {v}) does not exist in graph.")
        self.graph[u][v]["hazard"] = round(float(hazard_score), 3)

    def update_edge_density(self, u: str, v: str, density_score: float):
        """Update the estimated crowd density score (0.0 to 1.0) on edge (u, v)."""
        if not self.graph.has_edge(u, v):
            raise KeyError(f"Edge ({u}, {v}) does not exist in graph.")
        self.graph[u][v]["density"] = round(float(density_score), 3)

    def get_graph(self) -> nx.Graph:
        """Returns the underlying NetworkX graph instance."""
        return self.graph

    def create_demo_building(self):
        r"""
        Populates a clean, configurable DEMO building layout with 7 nodes:
        - 3 Rooms: ROOM_A, ROOM_B, ROOM_C
        - 2 Corridors: CORRIDOR_1 (North), CORRIDOR_2 (South)
        - 2 Emergency Exits: EXIT_A (North), EXIT_B (South)

        Topology Layout (Grid in meters):
        [ROOM_A] (0, 10)  ---------\
                                    [CORRIDOR_1] (10, 8) -------- [EXIT_A] (20, 10)
        [ROOM_B] (0, 5)   ---------/        |
                          ---------\        |
                                    [CORRIDOR_2] (10, 2) -------- [EXIT_B] (20, 0)
        [ROOM_C] (0, 0)   ---------/
        """
        self.graph.clear()

        # 1. Define Rooms
        self.add_room("ROOM_A", "Room A (Office 101)", x=0.0, y=10.0)
        self.add_room("ROOM_B", "Room B (Office 102)", x=0.0, y=5.0)
        self.add_room("ROOM_C", "Room C (Conference)", x=0.0, y=0.0)

        # 2. Define Corridors
        self.add_corridor("CORRIDOR_1", "North Corridor Junction", x=10.0, y=8.0)
        self.add_corridor("CORRIDOR_2", "South Corridor Junction", x=10.0, y=2.0)

        # 3. Define Emergency Exits
        self.add_exit("EXIT_A", "Emergency Exit A (North)", x=20.0, y=10.0)
        self.add_exit("EXIT_B", "Emergency Exit B (South)", x=20.0, y=0.0)

        # 4. Connect Physical Pathways (Edges)
        # Rooms connecting to corridors:
        self.connect_nodes("ROOM_A", "CORRIDOR_1")  # distance ~ 10.20m
        self.connect_nodes("ROOM_B", "CORRIDOR_1")  # distance ~ 10.44m
        self.connect_nodes("ROOM_B", "CORRIDOR_2")  # distance ~ 10.44m
        self.connect_nodes("ROOM_C", "CORRIDOR_2")  # distance ~ 10.20m

        # Corridors connecting to each other:
        self.connect_nodes("CORRIDOR_1", "CORRIDOR_2")  # distance = 6.00m

        # Corridors connecting to emergency exits:
        self.connect_nodes("CORRIDOR_1", "EXIT_A")  # distance ~ 10.20m
        self.connect_nodes("CORRIDOR_2", "EXIT_B")  # distance ~ 10.20m

        print(f"[*] Demo building layout created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")

    def visualize_graph(self, output_path: Optional[Path] = None) -> Path:
        """
        Renders a 2D floor-plan visualization of the building graph using Matplotlib.
        Nodes are color-coded:
        - ROOM: Sky Blue
        - CORRIDOR: Amber / Orange
        - EXIT: Emerald Green

        Saves the visualization to disk.
        """
        target_path = Path(output_path) if output_path else OUTPUT_DIR / "building_graph.png"
        target_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(11, 7))

        # Retrieve (x, y) coordinates for layout positioning
        pos = {n: (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in self.graph.nodes}

        # Color mapping by node type
        color_map = {
            self.NODE_TYPE_ROOM: "#42A5F5",       # Bright Blue
            self.NODE_TYPE_CORRIDOR: "#FFA726",   # Amber / Orange
            self.NODE_TYPE_EXIT: "#66BB6A"        # Emerald Green
        }
        node_colors = [color_map.get(self.graph.nodes[n].get("node_type"), "#BDBDBD") for n in self.graph.nodes]

        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            ax=ax,
            width=2.5,
            edge_color="#546E7A",
            style="solid"
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            ax=ax,
            node_size=1800,
            node_color=node_colors,
            edgecolors="#263238",
            linewidths=2.0
        )

        # Draw node labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            ax=ax,
            font_size=9,
            font_weight="bold",
            font_color="#FFFFFF"
        )

        # Draw edge labels (display length in meters)
        edge_labels = {
            (u, v): f"{d['length']}m"
            for u, v, d in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="#37474F",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ECEFF1", edgecolor="#B0BEC5", alpha=0.9)
        )

        # Construct legend
        patches = [
            mpatches.Patch(color=color_map[self.NODE_TYPE_ROOM], label="Room (Occupancy Source)"),
            mpatches.Patch(color=color_map[self.NODE_TYPE_CORRIDOR], label="Corridor Junction (Transit)"),
            mpatches.Patch(color=color_map[self.NODE_TYPE_EXIT], label="Emergency Exit (Evacuation Sink)"),
        ]
        ax.legend(handles=patches, loc="upper right", framealpha=0.95, fontsize=10)

        # Styling
        ax.set_title("Demo Facility Layout - Topological Evacuation Graph (Meters)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xlabel("X Coordinate (meters)", fontsize=10)
        ax.set_ylabel("Y Coordinate (meters)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

        # Add margin around points
        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        ax.set_xlim(min(x_vals) - 3, max(x_vals) + 4)
        ax.set_ylim(min(y_vals) - 3, max(y_vals) + 3)

        plt.tight_layout()
        plt.savefig(str(target_path), dpi=300)
        plt.close(fig)

        print(f"[+] Graph visualization saved to: {target_path}")
        return target_path

    def render_evacuation_map(
        self,
        recommended_route: Optional[List[str]] = None,
        previous_route: Optional[List[str]] = None,
        source_room: Optional[str] = None,
        monitored_edge: Optional[Tuple[str, str]] = None,
        hazardous_edges: Optional[List[Tuple[str, str]]] = None,
        congested_edges: Optional[List[Tuple[str, str]]] = None,
        system_state: str = "NORMAL",
        output_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Feature 14 — Graph Visualization:
        Renders an interactive 2D floor-plan visualization displaying:
        - Rooms, Corridors, Emergency Exits
        - Hazardous edges (Red dashed)
        - Congested edges (Orange)
        - Current evacuation route (Vibrant Green)
        - Previous route (Grey dashed)
        - System states: NORMAL, CAUTION, EMERGENCY, ROUTE CHANGED
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        pos = {n: (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"]) for n in self.graph.nodes}

        color_map = {
            self.NODE_TYPE_ROOM: "#42A5F5",       # Bright Blue
            self.NODE_TYPE_CORRIDOR: "#FFA726",   # Amber / Orange
            self.NODE_TYPE_EXIT: "#66BB6A"        # Emerald Green
        }
        node_colors = []
        for n in self.graph.nodes:
            if source_room and n == source_room:
                node_colors.append("#FFD54F")     # Bright Gold for active origin
            else:
                node_colors.append(color_map.get(self.graph.nodes[n].get("node_type"), "#BDBDBD"))

        # 1. Draw baseline edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            ax=ax,
            width=2.5,
            edge_color="#78909C",
            style="solid"
        )

        # 2. Draw Previous Route (if available and different)
        if previous_route and len(previous_route) > 1 and previous_route != recommended_route:
            prev_edges = [
                (previous_route[i], previous_route[i + 1])
                for i in range(len(previous_route) - 1)
                if self.graph.has_edge(previous_route[i], previous_route[i + 1])
            ]
            if prev_edges:
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=prev_edges,
                    ax=ax,
                    width=4.0,
                    edge_color="#9E9E9E",   # Grey dashed
                    style="dashed"
                )

        # 3. Draw Congested Edges
        if congested_edges:
            c_edges = [(u, v) for u, v in congested_edges if self.graph.has_edge(u, v)]
            if c_edges:
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=c_edges,
                    ax=ax,
                    width=4.5,
                    edge_color="#FF9800",   # Amber / Orange
                    style="solid"
                )

        # 4. Draw Hazardous Edges
        active_haz_edges = list(hazardous_edges) if hazardous_edges else []
        if monitored_edge and monitored_edge not in active_haz_edges:
            active_haz_edges.append(monitored_edge)

        if active_haz_edges:
            h_edges = [(u, v) for u, v in active_haz_edges if self.graph.has_edge(u, v)]
            if h_edges:
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=h_edges,
                    ax=ax,
                    width=4.5,
                    edge_color="#E53935",   # Red Hazard
                    style="dashed"
                )

        # 5. Draw Current Recommended Evacuation Route
        if recommended_route and len(recommended_route) > 1:
            route_edges = [
                (recommended_route[i], recommended_route[i + 1])
                for i in range(len(recommended_route) - 1)
                if self.graph.has_edge(recommended_route[i], recommended_route[i + 1])
            ]
            if route_edges:
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=route_edges,
                    ax=ax,
                    width=5.5,
                    edge_color="#00C853",   # Bright Evacuation Green
                    style="solid"
                )

        # 6. Draw Nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            ax=ax,
            node_size=1900,
            node_color=node_colors,
            edgecolors="#263238",
            linewidths=2.5
        )

        # 7. Draw Node Labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            ax=ax,
            font_size=9,
            font_weight="bold",
            font_color="#FFFFFF"
        )

        # 8. Draw Edge Distance Labels
        edge_labels = {
            (u, v): f"{d['length']}m"
            for u, v, d in self.graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="#263238",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#F5F5F5", edgecolor="#CFD8DC", alpha=0.9)
        )

        # 9. Dynamic State Styling & Legend
        state_colors = {
            "NORMAL": "#2E7D32",
            "CAUTION": "#F57C00",
            "EMERGENCY": "#D32F2F",
            "ROUTE CHANGED": "#E65100"
        }
        banner_color = state_colors.get(system_state.upper(), "#37474F")

        patches = [
            mpatches.Patch(color="#FFD54F", label=f"Origin ({source_room})" if source_room else "Origin Room"),
            mpatches.Patch(color="#00C853", label="Current Optimal Route"),
            mpatches.Patch(color="#9E9E9E", label="Previous Route" if previous_route else "Baseline Edge"),
            mpatches.Patch(color="#E53935", label="Hazardous / Monitored Edge"),
            mpatches.Patch(color="#66BB6A", label="Emergency Exit")
        ]
        ax.legend(handles=patches, loc="upper right", framealpha=0.95, fontsize=8.5)

        ax.set_title(
            f"Topological Evacuation Graph — STATE: {system_state.upper()}",
            fontsize=12,
            fontweight="bold",
            color=banner_color,
            pad=12
        )
        ax.set_xlabel("X (meters)", fontsize=9)
        ax.set_ylabel("Y (meters)", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

        x_vals = [p[0] for p in pos.values()]
        y_vals = [p[1] for p in pos.values()]
        ax.set_xlim(min(x_vals) - 3, max(x_vals) + 4)
        ax.set_ylim(min(y_vals) - 3, max(y_vals) + 3)

        plt.tight_layout()

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(out_p), dpi=200)

        return fig


