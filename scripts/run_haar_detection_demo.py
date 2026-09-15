#!/usr/bin/env python3
"""Run the Haar-cascade face/eye detection demo (no identity recognition).

Usage:
    python scripts/run_haar_detection_demo.py

Press 'q' in the video window to quit.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_recognition_system.detection.haar_detector import run_webcam_demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_webcam_demo()
