"""
Alert System & Emergency Dispatch Layer (Academic Prototype).
Implements the IEEE Paper Alert & Notification Chain:
1. Zone-specific alarm events with acknowledgement and cooldown suppression
2. Security guard notification event log
3. Directional signage simulation (Simulated dynamic exit signs)
4. Public Address (PA) / voice announcement simulation
5. Route change event tracking & history
6. Frame snapshot capture upon critical hazard detection
7. External emergency notification dispatches (Telegram / Email / Webhook via env vars)
8. Multi-format incident log exports (TXT, JSON, CSV)
"""

import os
import io
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from config.config import DATA_DIR, OUTPUT_DIR


class AlertManager:
    """
    Manages emergency notifications, directional signage instructions,
    public address announcements, frame snapshots, and security audit logs.
    """

    def __init__(self):
        self.active_alerts: List[Dict[str, Any]] = []
        self.guard_alert_log: List[Dict[str, Any]] = []
        self.route_history: List[Dict[str, Any]] = []
        self.snapshot_dir = DATA_DIR / "processed" / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

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
        route_str: str = "",
        frame: Optional[np.ndarray] = None,
        camera_id: str = "CAM-01",
        cooldown_seconds: float = 15.0
    ) -> Dict[str, Any]:
        """
        Triggers a zone-specific hazard alert, saves snapshot if frame provided,
        suppresses duplicate alarms within cooldown, and logs to the guard registry.
        """
        now_dt = datetime.now()
        now = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        now_ts = now_dt.timestamp()

        # Check for duplicate suppression within cooldown
        for existing in self.active_alerts:
            if (
                existing.get("location") == location and
                existing.get("type") == hazard_type.upper() and
                not existing.get("acknowledged", False)
            ):
                created_ts = existing.get("_created_ts", 0.0)
                if now_ts - created_ts < cooldown_seconds:
                    # Update existing telemetry without duplicating alert log entry
                    existing["hazard_score"] = round(hazard_score, 3)
                    existing["people"] = people_count
                    existing["density"] = round(crowd_density, 3)
                    existing["timestamp"] = now
                    existing["severity"] = severity
                    existing["recommended_exit"] = recommended_exit
                    existing["route"] = route_str
                    existing["action"] = f"DISPATCH SAFETY TEAM TO {location} | EVACUATE VIA {recommended_exit}"
                    if existing not in self.guard_alert_log:
                        self.guard_alert_log.insert(0, existing)
                    return existing

        # Generate unique alert ID
        alert_id = f"ALT-{now_dt.strftime('%Y%m%d%H%M%S')}-{len(self.active_alerts) + 1:02d}"

        # Save snapshot if frame is provided
        snapshot_path = None
        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
            try:
                clean_loc = location.replace(" ", "_").replace("<->", "_to_").replace("/", "_")
                filename = f"snapshot_{alert_id}_{clean_loc}.jpg"
                filepath = self.snapshot_dir / filename
                cv2.imwrite(str(filepath), frame)
                snapshot_path = str(filepath)
            except Exception as e:
                print(f"[!] Warning: Failed to save alert snapshot: {e}")

        alert = {
            "alert_id": alert_id,
            "timestamp": now,
            "_created_ts": now_ts,
            "type": hazard_type.upper(),
            "location": location,
            "camera_id": camera_id,
            "severity": severity,
            "hazard_score": round(hazard_score, 3),
            "people": people_count,
            "density": round(crowd_density, 3),
            "recommended_exit": recommended_exit,
            "route": route_str,
            "action": f"DISPATCH SAFETY TEAM TO {location} | EVACUATE VIA {recommended_exit}",
            "status": "ACTIVE",
            "acknowledged": False,
            "acknowledged_by": None,
            "acknowledged_at": None,
            "snapshot_path": snapshot_path
        }

        self.active_alerts.append(alert)
        self.guard_alert_log.insert(0, alert)

        # Attempt safe external notification dispatch if configured
        self._notify_external_dispatch(alert)

        return alert

    def acknowledge_alert(self, alert_id: str, operator_name: str = "Command Operator") -> bool:
        """Marks an active emergency alert as acknowledged by a security operator."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        found = False

        for alert in self.active_alerts:
            if alert.get("alert_id") == alert_id:
                alert["acknowledged"] = True
                alert["acknowledged_by"] = operator_name
                alert["acknowledged_at"] = now
                alert["status"] = "ACKNOWLEDGED"
                found = True

        for log in self.guard_alert_log:
            if log.get("alert_id") == alert_id:
                log["acknowledged"] = True
                log["acknowledged_by"] = operator_name
                log["acknowledged_at"] = now
                log["status"] = "ACKNOWLEDGED"
                found = True

        return found

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
        self.update_dynamic_signage(
            recommended_exit=new_exit,
            dangerous_exit=old_exit if (old_exit != new_exit and (hazard_score > 0.0 or crowd_density >= 0.70 or "fire" in reason.lower() or "hazard" in reason.lower() or "crowd" in reason.lower())) else None
        )

        # Generate Automated PA Announcement
        self.generate_pa_announcement(
            recommended_exit=new_exit,
            dangerous_zone=active_zone,
            reason=reason,
            hazard_score=hazard_score,
            crowd_density=crowd_density,
            route_str=new_str
        )

        # Log into Security Guard Register
        self.guard_alert_log.insert(0, {
            "alert_id": f"REROUTE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
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
            "status": "REROUTED",
            "acknowledged": True,
            "acknowledged_by": "SYSTEM AUTO-DISPATCH",
            "acknowledged_at": now
        })

        return event

    def update_dynamic_signage(
        self,
        recommended_exit: Optional[str],
        dangerous_exit: Optional[Any] = None
    ):
        """
        Updates simulated directional exit indicators (LED digital signage)
        based on actual hazard and route-optimizer results.
        Marks compromised exits as CLOSED / AVOID, recommended exit as SAFE ROUTE,
        and provides an explicit fallback when route safety is unknown.
        """
        if isinstance(dangerous_exit, str):
            compromised_set = {dangerous_exit} if dangerous_exit not in ("", "NONE", "UNKNOWN") else set()
        elif isinstance(dangerous_exit, (list, set, tuple)):
            compromised_set = { str(x) for x in dangerous_exit if x and str(x) not in ("", "NONE", "UNKNOWN") }
        else:
            compromised_set = set()

        valid_rec = (
            isinstance(recommended_exit, str)
            and recommended_exit in self.dynamic_signage
            and recommended_exit not in ("", "NONE", "UNKNOWN")
            and recommended_exit not in compromised_set
        )

        for exit_id in list(self.dynamic_signage.keys()):
            if exit_id in compromised_set:
                self.dynamic_signage[exit_id] = {
                    "status": "CLOSED / AVOID",
                    "indicator": f"🛑 {exit_id} — CLOSED / HAZARD DETECTED",
                    "color": "#FF1744"
                }
            elif valid_rec and exit_id == recommended_exit:
                self.dynamic_signage[exit_id] = {
                    "status": "SAFE ROUTE",
                    "indicator": f"🟢 {exit_id} — ACTIVE EVACUATION ROUTE",
                    "color": "#00E676"
                }
            elif not valid_rec:
                self.dynamic_signage[exit_id] = {
                    "status": "UNKNOWN / VERIFY SAFETY",
                    "indicator": f"⚠️ {exit_id} — ROUTE SAFETY UNKNOWN",
                    "color": "#F59E0B"
                }
            else:
                self.dynamic_signage[exit_id] = {
                    "status": "STANDBY / ALTERNATIVE",
                    "indicator": f"⚪ {exit_id} — STANDBY EXIT",
                    "color": "#9E9E9E"
                }

    def generate_pa_announcement(
        self,
        recommended_exit: str,
        dangerous_zone: str,
        reason: str = "",
        hazard_type: str = "NONE",
        hazard_score: float = 0.0,
        crowd_density: float = 0.0,
        route_str: str = ""
    ) -> str:
        """
        Synthesizes a public address announcement text based on active emergency
        and routing decisions. Never announces nominal conditions during an active emergency.
        """
        reason_lower = (reason or "").lower()
        htype = (hazard_type or "NONE").upper()

        if recommended_exit in (None, "", "NONE", "UNKNOWN"):
            self.current_pa_message = (
                f"🚨 EMERGENCY ADVISORY: Hazard detected near {dangerous_zone or 'monitored corridor'}. "
                f"Safe evacuation route is currently undetermined. Shelter in place or follow security instructions."
            )
            return self.current_pa_message

        route_clause = (
            f" via route {route_str}"
            if route_str and route_str not in ("NONE", "NO SAFE ROUTE", "INITIAL")
            else ""
        )

        is_fire = (htype == "FIRE" or "fire" in reason_lower or hazard_score >= 0.70)
        is_smoke = (htype in ("SMOKE", "LOW HAZARD") or "smoke" in reason_lower or "hazard" in reason_lower or hazard_score > 0.0)
        is_crowd = (htype in ("CROWD", "CROWD_CONGESTION") or "density" in reason_lower or "crowd" in reason_lower or crowd_density >= 0.70)

        if is_fire:
            score_txt = f" (Hazard Score: {hazard_score:.3f})" if hazard_score > 0 else ""
            self.current_pa_message = (
                f"🚨 ATTENTION OCCUPANTS: Fire hazard confirmed in {dangerous_zone}{score_txt}. "
                f"Do NOT use compromised corridor routes. "
                f"Follow dynamic green signs and evacuate immediately toward {recommended_exit}{route_clause}."
            )
        elif is_smoke:
            score_txt = f" (Hazard Score: {hazard_score:.3f})" if hazard_score > 0 else ""
            self.current_pa_message = (
                f"⚠️ ATTENTION OCCUPANTS: Smoke/visibility hazard detected near {dangerous_zone}{score_txt}. "
                f"Do NOT enter compromised corridors. "
                f"Follow dynamic green signs and evacuate immediately toward {recommended_exit}{route_clause}."
            )
        elif is_crowd:
            self.current_pa_message = (
                f"⚠️ ATTENTION OCCUPANTS: Severe crowd bottleneck identified near {dangerous_zone}. "
                f"To prevent corridor stampedes, occupants are redirected toward {recommended_exit}{route_clause}."
            )
        elif reason_lower:
            self.current_pa_message = f"Evacuation route active toward {recommended_exit}. Maintain calm and proceed."
        else:
            self.current_pa_message = "Facility conditions nominal. Follow standard exit signs if required."

        return self.current_pa_message

    def sync_facility_emergency_state(self, emergency_state: Dict[str, Any]) -> None:
        """
        Synchronizes directional signage and PA announcement directly with the
        unified facility emergency state for the current frame/session.
        """
        rec_exit = emergency_state.get("recommended_exit", "NONE")
        compromised_exits = emergency_state.get("compromised_exits", [])
        self.update_dynamic_signage(recommended_exit=rec_exit, dangerous_exit=compromised_exits)

        if emergency_state.get("is_emergency", False):
            self.generate_pa_announcement(
                recommended_exit=rec_exit,
                dangerous_zone=emergency_state.get("active_zone_str", ""),
                reason=emergency_state.get("adaptation_reason", ""),
                hazard_type=emergency_state.get("hazard_type", "NONE"),
                hazard_score=emergency_state.get("hazard_score", 0.0),
                crowd_density=emergency_state.get("crowd_density", 0.0),
                route_str=emergency_state.get("recommended_route_str", "")
            )
        else:
            self.current_pa_message = "Facility conditions nominal. Follow standard exit signs if required."

    def get_guard_log(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns the most recent security guard event logs."""
        return self.guard_alert_log[:limit]

    def get_route_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent route change events."""
        return self.route_history[:limit]

    def clear_alerts(self):
        """Clears active alerts."""
        self.active_alerts.clear()

    # -------------------------------------------------------------------------
    # EXTERNAL DISPATCH HELPERS (SAFE FALLBACK IF NO KEYS CONFIGURED)
    # -------------------------------------------------------------------------
    def _notify_external_dispatch(self, alert: Dict[str, Any]):
        """Safe notification dispatcher checking environment variables."""
        msg = f"EMERGENCY ALERT: {alert['type']} at {alert['location']}. Egress via {alert['recommended_exit']}."
        self.send_telegram_alert(msg)

    def send_telegram_alert(self, message: str) -> bool:
        """Sends alert to Telegram if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set."""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return False

        try:
            import urllib.request
            import urllib.parse
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            data = urllib.parse.urlencode({"chat_id": chat_id, "text": message}).encode("utf-8")
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def send_email_alert(self, subject: str, message: str) -> bool:
        """Sends alert via SMTP if SMTP_SERVER and ALERT_EMAIL_RECIPIENT are set."""
        smtp_server = os.getenv("SMTP_SERVER")
        recipient = os.getenv("ALERT_EMAIL_RECIPIENT")
        if not smtp_server or not recipient:
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            msg = MIMEText(message)
            msg["Subject"] = subject
            msg["From"] = os.getenv("SMTP_SENDER", "emergency@smartbuilding.local")
            msg["To"] = recipient

            port = int(os.getenv("SMTP_PORT", 25))
            with smtplib.SMTP(smtp_server, port, timeout=3) as server:
                server.send_message(msg)
            return True
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # INCIDENT EXPORT HELPERS
    # -------------------------------------------------------------------------
    def export_incidents_json(self) -> str:
        """Exports the active and historical incidents as formatted JSON."""
        payload = {
            "system": "AI Smart Building Emergency Command Center",
            "export_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active_alerts": self.active_alerts,
            "guard_alert_log": self.guard_alert_log,
            "route_history": self.route_history
        }
        return json.dumps(payload, indent=2)

    def export_guard_log_csv(self) -> str:
        """Exports the security guard alert log as CSV formatted string."""
        if not self.guard_alert_log:
            return "timestamp,type,location,severity,hazard_score,recommended_exit,status,acknowledged\n"

        output = io.StringIO()
        fieldnames = ["timestamp", "type", "location", "severity", "hazard_score", "recommended_exit", "status", "acknowledged"]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in self.guard_alert_log:
            writer.writerow(row)
        return output.getvalue()

    def export_summary_txt(self) -> str:
        """Exports an incident executive summary text file."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 70,
            "AI EMERGENCY COMMAND CENTER — INCIDENT AUDIT SUMMARY",
            f"Generated: {now}",
            "=" * 70,
            f"Total Active Alerts:     {len(self.active_alerts)}",
            f"Total Security Log Items:{len(self.guard_alert_log)}",
            f"Route Change Events:     {len(self.route_history)}",
            f"Current PA Advisory:     {self.current_pa_message}",
            "-" * 70,
            "ACTIVE ALERTS:"
        ]
        if not self.active_alerts:
            lines.append("  [None — Nominal Facility State]")
        else:
            for alt in self.active_alerts:
                ack = f"[ACK by {alt['acknowledged_by']}]" if alt.get("acknowledged") else "[UNACKNOWLEDGED]"
                lines.append(f"  * {alt['timestamp']} | {alt['type']} at {alt['location']} | Score: {alt['hazard_score']} | Exit: {alt['recommended_exit']} {ack}")

        lines.extend([
            "-" * 70,
            "RECENT ROUTE CHANGE EVENTS:"
        ])
        if not self.route_history:
            lines.append("  [None — Standard Direct Egress Paths Maintained]")
        else:
            for ev in self.route_history[:5]:
                lines.append(f"  * {ev['timestamp']} | {ev['previous_route']} -> {ev['new_route']} | Reason: {ev['reason']}")

        lines.append("=" * 70)
        return "\n".join(lines)
