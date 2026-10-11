"""
Temporal Hazard Trend & Early Warning Analytics.
Tracks rolling-window perception telemetry to calculate:
1. Rate-of-change (slope: RISING, STABLE, FALLING)
2. Consecutive-frame confirmation (prevents false positives from single-frame flickers)
3. Flap prevention via cooldown timer
4. Transparent state transitions:
   - NORMAL
   - MONITORING
   - RISK INCREASING
   - SMOKE DETECTED
   - FIRE DETECTED
   - CAMERA UNAVAILABLE
"""

import time
import datetime
from collections import deque
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class TemporalHazardTracker:
    """
    Maintains a rolling temporal window of perception telemetry and computes
    hazard trends, anomaly confirmation, and cooldown safety states.
    """

    def __init__(
        self,
        window_size: int = 30,
        confirmation_threshold: int = 3,
        hazard_threshold: float = 0.35,
        cooldown_frames: int = 10
    ):
        """
        Args:
            window_size: Maximum number of historical frames to retain.
            confirmation_threshold: Consecutive frames above hazard_threshold required for confirmed alarm.
            hazard_threshold: Score above which a frame is considered hazardous.
            cooldown_frames: Number of consecutive clear frames required to return to NORMAL from MONITORING.
        """
        self.window_size = window_size
        self.confirmation_threshold = confirmation_threshold
        self.hazard_threshold = hazard_threshold
        self.cooldown_frames = cooldown_frames

        self.history: deque = deque(maxlen=window_size)
        self.consecutive_hazard_count: int = 0
        self.consecutive_normal_count: int = 0
        self.cooldown_remaining: int = 0
        self.is_confirmed: bool = False
        self.current_state: str = "NORMAL"
        self.last_slope: str = "STABLE"
        self.last_slope_val: float = 0.0

    def update(
        self,
        hazard_score: float,
        fire_conf: float = 0.0,
        smoke_conf: float = 0.0,
        people_count: int = 0,
        crowd_density: float = 0.0,
        camera_id: str = "CAM-01",
        zone: str = "CORRIDOR_1",
        camera_status: str = "ONLINE"
    ) -> Dict[str, Any]:
        """
        Ingests the latest frame telemetry and computes trend, slope, and safety state.
        """
        now = time.time()
        time_str = datetime.datetime.fromtimestamp(now).strftime("%H:%M:%S")

        record = {
            "timestamp": now,
            "time_str": time_str,
            "hazard_score": float(hazard_score),
            "fire_conf": float(fire_conf),
            "smoke_conf": float(smoke_conf),
            "people_count": int(people_count),
            "crowd_density": float(crowd_density),
            "camera_id": str(camera_id),
            "zone": str(zone),
            "camera_status": str(camera_status)
        }
        self.history.append(record)

        # 1. Check Camera Availability
        if camera_status.upper() in ["UNAVAILABLE", "ERROR", "DISCONNECTED"]:
            self.current_state = "CAMERA UNAVAILABLE"
            self.is_confirmed = False
            self.last_slope = "STABLE"
            self.last_slope_val = 0.0
            return self.get_summary()

        # 2. Slope / Rate of Change Calculation
        slope_val, slope_str = self._compute_slope()
        self.last_slope = slope_str
        self.last_slope_val = slope_val

        # 3. Consecutive-Frame Confirmation & Cooldown Logic
        is_frame_hazard = (hazard_score >= self.hazard_threshold) or (fire_conf >= 0.30) or (smoke_conf >= 0.30)

        if is_frame_hazard:
            self.consecutive_hazard_count += 1
            self.consecutive_normal_count = 0
            self.cooldown_remaining = self.cooldown_frames

            if self.consecutive_hazard_count >= self.confirmation_threshold:
                self.is_confirmed = True
                if fire_conf > smoke_conf or fire_conf >= 0.40:
                    self.current_state = "FIRE DETECTED"
                else:
                    self.current_state = "SMOKE DETECTED"
            else:
                self.is_confirmed = False
                self.current_state = "RISK INCREASING"
        else:
            self.consecutive_hazard_count = 0
            self.consecutive_normal_count += 1

            if self.cooldown_remaining > 0:
                self.cooldown_remaining -= 1
                self.current_state = "MONITORING"
            else:
                self.is_confirmed = False
                self.current_state = "NORMAL"

        return self.get_summary()

    def _compute_slope(self) -> tuple[float, str]:
        """Calculates linear slope over the recent hazard score history."""
        if len(self.history) < 3:
            return 0.0, "STABLE"

        recent = list(self.history)[-min(len(self.history), 10):]
        y = np.array([r["hazard_score"] for r in recent], dtype=np.float32)
        x = np.arange(len(y), dtype=np.float32)

        # Linear regression slope: cov(x, y) / var(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        var_x = np.var(x)
        if var_x == 0:
            return 0.0, "STABLE"

        slope = float(np.sum((x - x_mean) * (y - y_mean)) / (len(x) * var_x))

        if slope > 0.015:
            return round(slope, 4), "RISING"
        elif slope < -0.015:
            return round(slope, 4), "FALLING"
        else:
            return round(slope, 4), "STABLE"

    def get_summary(self) -> Dict[str, Any]:
        """Returns the current state summary and diagnostic telemetry."""
        latest = self.history[-1] if self.history else {}
        return {
            "current_state": self.current_state,
            "slope": self.last_slope,
            "slope_val": self.last_slope_val,
            "consecutive_hazard": self.consecutive_hazard_count,
            "confirmation_threshold": self.confirmation_threshold,
            "is_confirmed": self.is_confirmed,
            "cooldown_remaining": self.cooldown_remaining,
            "current_hazard_score": latest.get("hazard_score", 0.0),
            "current_crowd_density": latest.get("crowd_density", 0.0),
            "confidence_label": "CONFIRMED HAZARD" if self.is_confirmed else (
                "TRANSIENT / ANOMALY" if self.consecutive_hazard_count > 0 else "NOMINAL"
            ),
            "window_length": len(self.history)
        }

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the history deque as a Pandas DataFrame for charting."""
        if not self.history:
            return pd.DataFrame(columns=[
                "timestamp", "time_str", "hazard_score", "fire_conf",
                "smoke_conf", "people_count", "crowd_density", "camera_id", "zone"
            ])
        return pd.DataFrame(list(self.history))

    def reset(self):
        """Resets the history buffer and state counters."""
        self.history.clear()
        self.consecutive_hazard_count = 0
        self.consecutive_normal_count = 0
        self.cooldown_remaining = 0
        self.is_confirmed = False
        self.current_state = "NORMAL"
        self.last_slope = "STABLE"
        self.last_slope_val = 0.0
