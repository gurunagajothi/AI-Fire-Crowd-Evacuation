"""
System configuration settings.
Defines hyperparameters for hazard weighting, detection thresholds, continuous video processing,
and multi-zone demonstration layouts.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Dynamic Edge Cost Weights:
# Cost(e) = alpha * Hazard(e) + beta * Density(e) + gamma * Length(e)
ALPHA = 5.0   # Weight for fire/smoke hazard (life-safety priority)
BETA = 2.0    # Weight for crowd density (congestion penalty)
GAMMA = 0.2   # Weight for physical corridor/edge length (distance factor)

# Fire & Smoke Model Configuration (Phase 2)
FIRE_MODEL_NAME = "fire_smoke_yolov8n.pt"
FIRE_MODEL_PATH = MODELS_DIR / FIRE_MODEL_NAME
FIRE_MODEL_URL = "https://huggingface.co/rabahdev/fire-smoke-yolov8n/resolve/main/best.pt"
CONF_THRESHOLD_FIRE = 0.35

# Crowd / Person Detection Configuration (Phase 3)
CROWD_MODEL_NAME = "yolov8n.pt"
CROWD_MODEL_PATH = MODELS_DIR / CROWD_MODEL_NAME
CONF_THRESHOLD_CROWD = 0.35

# Continuous Video Processing Configuration (CPU-Friendly Frame Sampling)
FRAME_SKIP = 5
PROCESS_EVERY_N_FRAMES = 5

# Maximum expected number of people in a monitored zone/corridor.
MAX_EXPECTED_PEOPLE = 20

# Multi-Zone Demonstration Layout Configurations
DEMO_BUILDING_ZONES = [
    "ROOM_A",
    "ROOM_B",
    "ROOM_C",
    "CORRIDOR_1",
    "CORRIDOR_2",
    "EXIT_A",
    "EXIT_B"
]

ZONE_MAX_CAPACITIES = {
    "ROOM_A": 15,
    "ROOM_B": 15,
    "ROOM_C": 25,
    "CORRIDOR_1": 20,
    "CORRIDOR_2": 20,
    "EXIT_A": 10,
    "EXIT_B": 10,
}
