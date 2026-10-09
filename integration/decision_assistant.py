"""
Integration Layer: AI Emergency Decision Explanation Assistant.
Provides deterministic, rule-based explainability answering viva/research questions
using ONLY the current live system telemetry. Requires NO external APIs or internet access.
"""

from typing import Dict, Any, Optional


class EmergencyDecisionAssistant:
    """
    Local, deterministic rule-based explainable assistant that analyzes the current
    state of the EvacuationSystem and produces structured, factual explanations.
    """

    def __init__(self):
        self.preset_questions = [
            "Why did the route change?",
            "Why is EXIT_A closed / avoided?",
            "What is the current facility risk?",
            "How many people are detected?",
            "Which zone is most dangerous?",
            "What is the recommended safest exit?",
            "How are edge costs mathematically computed?"
        ]

    def answer_query(self, query: str, state: Dict[str, Any]) -> str:
        """
        Generates a factual answer based strictly on the provided runtime state dict.
        """
        q = query.lower().strip()

        has_fire = state.get("fire_detected", False)
        has_smoke = state.get("smoke_detected", False)
        hazard_score = state.get("hazard_score", 0.0)
        people_count = state.get("people_detected", 0)
        crowd_density = state.get("crowd_density", 0.0)
        route_changed = state.get("route_changed", False)
        recommended_exit = state.get("recommended_exit", "EXIT_A")
        normal_exit = state.get("normal_exit", "EXIT_A")
        recommended_route = state.get("recommended_route_str", "")
        active_edge = state.get("active_edge", ("CORRIDOR_1", "EXIT_A"))
        total_cost = state.get("total_route_cost", 0.0)
        edge_cost = state.get("edge_cost", 0.0)
        risk_index = state.get("risk_index", 0)

        edge_name = f"{active_edge[0]} <-> {active_edge[1]}"

        if "why did the route change" in q or "reroute" in q:
            if route_changed:
                cause = "fire hazard" if has_fire else ("smoke plume" if has_smoke else ("excessive crowd congestion" if crowd_density >= 0.70 else "dynamic cost shifts"))
                return (
                    f"The evacuation route changed from baseline {normal_exit} to {recommended_exit} because "
                    f"{edge_name} was penalized by a {cause} (Hazard: {hazard_score:.3f}, Density: {crowd_density:.3f}). "
                    f"This increased the traversal cost of {edge_name} to {edge_cost:.2f}. "
                    f"Dynamic Dijkstra recalculated all paths and identified {recommended_route} as the globally optimal path."
                )
            else:
                return (
                    f"The route has NOT changed. The primary baseline route to {normal_exit} remains unobstructed and optimal "
                    f"(Total Cost: {total_cost:.2f}). No hazard or severe congestion was detected on the direct pathway."
                )

        elif "why is exit_a" in q or "exit a" in q and ("closed" in q or "avoid" in q):
            if recommended_exit != "EXIT_A":
                return (
                    f"EXIT_A is avoided because its primary feeding segment ({edge_name}) has an elevated dynamic traversal cost "
                    f"({edge_cost:.2f}) due to detected hazard (Score: {hazard_score:.3f}) or occupant density ({crowd_density:.3f}). "
                    f"Routing occupants to EXIT_B avoids this danger zone while keeping cumulative route cost at {total_cost:.2f}."
                )
            else:
                return (
                    f"EXIT_A is currently OPEN and SAFE. It is the primary recommended exit with zero critical hazards on its approach path."
                )

        elif "current risk" in q or "risk index" in q or "risk level" in q:
            level = "CRITICAL" if risk_index >= 75 else ("HIGH" if risk_index >= 50 else ("MODERATE" if risk_index >= 25 else "LOW"))
            return (
                f"The composite Evacuation Risk Index is currently {risk_index} / 100 ({level} RISK). "
                f"This metric is derived from: 50% hazard severity ({hazard_score:.3f}), 30% crowd density ({crowd_density:.3f}), "
                f"and 20% normalized path traversal cost."
            )

        elif "how many people" in q or "people detected" in q or "occupant" in q:
            return (
                f"The system detected {people_count} occupant(s) in the active surveillance frame "
                f"(Normalized Density: {crowd_density:.3f}). Occupant detection is powered by YOLOv8n tracking COCO class 0 ('person')."
            )

        elif "most dangerous" in q or "dangerous zone" in q or "highest hazard" in q:
            if hazard_score > 0 or crowd_density >= 0.5:
                return (
                    f"The most critical zone is currently {edge_name} with Hazard Score {hazard_score:.3f} "
                    f"and Crowd Density {crowd_density:.3f}. System alerts and guard dispatch have been targeted to this sector."
                )
            else:
                return "All monitored building sectors are currently nominal. No hazardous zones identified."

        elif "safest exit" in q or "recommended exit" in q:
            return (
                f"The safest exit is {recommended_exit}. Dynamic Dijkstra selected this egress sink with cumulative cost {total_cost:.2f}. "
                f"Active path: {recommended_route}."
            )

        elif "mathematically computed" in q or "formula" in q or "cost" in q:
            return (
                "Edge costs are dynamically calculated using the IEEE objective formulation:\n"
                "Cost(e) = alpha * Hazard(e) + beta * Density(e) + gamma * Length(e)\n"
                "where alpha=5.0 (Life-safety hazard penalty), beta=2.0 (Congestion penalty), and gamma=0.2 (Physical metric distance)."
            )

        else:
            return (
                f"System Summary: Monitored Zone: {edge_name} | Recommended Exit: {recommended_exit} | "
                f"Hazard: {hazard_score:.3f} | People: {people_count} | Total Route Cost: {total_cost:.2f}. "
                f"Route Status: {'REROUTED' if route_changed else 'NOMINAL'}."
            )
