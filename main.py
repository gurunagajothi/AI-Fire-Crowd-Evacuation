"""
Main Entry Point for AI Fire & Crowd Evacuation Optimization System.
"""

import sys
from pathlib import Path


def display_welcome_banner():
    """Displays project information and system layer architecture."""
    banner = """
================================================================================
  AI-Based Fire/Smoke Hazard Detection & Crowd-Aware Evacuation Optimization
================================================================================
  Layer 1: Perception Layer       -> Fire/Smoke & Crowd Density Detection
  Layer 2: Optimization Layer    -> Dynamic Graph & Dijkstra Safe Path Calculation
  Layer 3: Presentation Layer     -> Live Streamlit Monitoring Dashboard
================================================================================
    """
    print(banner)


def check_environment():
    """Performs a quick check of Python version and system paths."""
    print(f"[*] Python Version: {sys.version.split()[0]}")
    root_dir = Path(__file__).resolve().parent
    print(f"[*] Project Root:   {root_dir}")
    print("[*] Project structure initialized successfully.")
    print("[*] Next steps: Configure virtual environment and install dependencies.")


def main():
    display_welcome_banner()
    check_environment()


if __name__ == "__main__":
    main()
