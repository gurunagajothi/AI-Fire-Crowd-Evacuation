"""
Alert System & Emergency Dispatch Layer (Academic Prototype).
Implements the IEEE Paper Alert & Notification Chain:
1. Zone-specific alarm events
2. Security guard notification event log
3. Directional signage simulation (Simulated dynamic exit signs)
4. Public Address (PA) / voice announcement simulation
5. Route change event tracking & history
"""

import time
from datetime import datetime
from typing import List, Dict, Any, Optional


class AlertManager:
    """
    Manages emergency notifications, directional signage instructions,
    public address announcements, and security audit logs.
    """

    def __init__(self):
        self.active_alerts: List[Dict[str, Any]] = []
        self.guard_alert_log: List[Dict[str, Any]] = []
        self.route_history: List[Dict[str, Any]] = []
        
        # Initial Simulated Signage
        self.dynamic_signage: Dict[str, Dict[str, str]] = {
            "EXIT_A": {"status": "OPEN / AVAILABLE", "indicator": "🟢 EXIT A ACTIVE", "color": "green"},
            "EXIT_B": {"status": "OPEN / AVAILABLE", "indicator": "🟢 EXIT B ACTIVE", "color": "green"}
        }

        # Initial PA Announcement
        self.current_pa_message: str = "Facility conditions nominal. Follow standard exit signs if required."

    def trigger_hazard_alert(
        self,
        location: str,
        hazard_type: str,
        severity: str = "HIGH",
        hazard_score: float = 0.0,
        people_count: int = 0,
        crowd_density: float = 0.0,
        recommended_exit: str = "EXIT_A",
        route_str: str = ""
    ) -> Dict[str, Any]:
        """
        Triggers a zone-specific hazard alert and logs it to the security guard registry.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert = {
            "timestamp": now,
            "type": hazard_type.upper(),
            "location": location,
            "severity": severity,
            "hazard_score": round(hazard_score, 3),
            "people": people_count,
            "density": round(crowd_density, 3),
            "recommended_exit": recommended_exit,
            "route": route_str,
            "action": f"DISPATCH SAFETY TEAM TO {location} | EVACUATE VIA {recommended_exit}",
            "status": "ACTIVE"
        }
        self.active_alerts.append(alert)
        self.guard_alert_log.append(alert)
        return alert

    def register_route_change(
        self,
        previous_route: List[str],
        new_route: List[str],
        reason: str,
        prev_cost: float,
        new_cost: float,
        active_zone: str = "CORRIDOR_1 <-> EXIT_A",
        hazard_score: float = 0.0,
        crowd_density: float = 0.0
    ) -> Dict[str, Any]:
        """
        Creates a ROUTE_CHANGE_EVENT, updates route history, modifies simulated
        directional signage, and generates an automated PA announcement.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prev_str = " -> ".join(previous_route) if previous_route else "INITIAL"
        new_str = " -> ".join(new_route) if new_route else "NONE"
        cost_diff = round(new_cost - prev_cost, 2)
        new_exit = new_route[-1] if new_route else "UNKNOWN"
        old_exit = previous_route[-1] if previous_route else "UNKNOWN"

        event = {
            "timestamp": now,
            "event_type": "ROUTE_CHANGE_EVENT",
            "previous_route": prev_str,
            "new_route": new_str,
            "previous_exit": old_exit,
            "new_exit": new_exit,
            "reason": reason,
            "previous_cost": round(prev_cost, 2),
            "new_cost": round(new_cost, 2),
            "cost_difference": cost_diff,
            "active_zone": active_zone,
            "hazard_score": round(hazard_score, 3),
            "crowd_density": round(crowd_density, 3)
        }
        self.route_history.insert(0, event)  # Most recent first

        # Update Simulated Directional Signage
        self.update_dynamic_signage(recommended_exit=new_exit, dangerous_exit=old_exit if old_exit != new_exit else None)

        # Generate Automated PA Announcement
        self.generate_pa_announcement(recommended_exit=new_exit, dangerous_zone=active_zone, reason=reason)

        # Log into Security Guard Register
        self.guard_alert_log.insert(0, {
            "timestamp": now,
            "type": "REROUTE",
            "location": active_zone,
            "severity": "CRITICAL" if hazard_score > 0.6 else "WARNING",
            "hazard_score": round(hazard_score, 3),
            "people": 0,
            "density": round(crowd_density, 3),
            "recommended_exit": new_exit,
            "route": new_str,
            "action": f"DYNAMIC REROUTE TO {new_exit}: {reason}",
            "status": "REROUTED"
        })

        return event

    def update_dynamic_signage(self, recommended_exit: str, dangerous_exit: Optional[str] = None):
        """
        Updates simulated directional exit indicators (LED digital signage).
        """
        for exit_id in self.dynamic_signage.keys():
            if exit_id == recommended_exit:
                self.dynamic_signage[exit_id] = {
                    "status": "SAFE ROUTE",
                    "indicator": f"🟢 {exit_id} — ACTIVE EVACUATION ROUTE",
                    "color": "#00E676"
                }
            elif dangerous_exit and exit_id == dangerous_exit:
                self.dynamic_signage[exit_id] = {
                    "status": "CLOSED / AVOID",
                    "indicator": f"🛑 {exit_id} — CLOSED / HAZARD DETECTED",
                    "color": "#FF1744"
                }
            else:
                self.dynamic_signage[exit_id] = {
                    "status": "STANDBY / ALTERNATIVE",
                    "indicator": f"⚪ {exit_id} — STANDBY EXIT",
                    "color": "#9E9E9E"
                }

    def generate_pa_announcement(self, recommended_exit: str, dangerous_zone: str, reason: str):
        """
        Synthesizes a public address announcement text based on active routing decisions.
        """
        if "hazard" in reason.lower() or "fire" in reason.lower():
            self.current_pa_message = (
                f"🚨 ATTENTION OCCUPANTS: Fire/smoke hazard detected near {dangerous_zone}. "
                f"Do NOT use direct corridor routes toward compromised exits. "
                f"Follow dynamic green signs and evacuate immediately toward {recommended_exit}."
            )
        elif "density" in reason.lower() or "crowd" in reason.lower():
            self.current_pa_message = (
                f"⚠️ ATTENTION OCCUPANTS: Severe crowd bottleneck identified near {dangerous_zone}. "
                f"To prevent corridor stampedes, occupants are redirected toward {recommended_exit}."
            )
        else:
            self.current_pa_message = f"Evacuation route active toward {recommended_exit}. Maintain calm and proceed."

    def get_guard_log(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns the most recent security guard event logs."""
        return self.guard_alert_log[:limit]

    def get_route_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent route change events."""
        return self.route_history[:limit]

    def clear_alerts(self):
        """Clears active alerts."""
        self.active_alerts.clear()
