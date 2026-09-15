#!/usr/bin/env python3
"""Run live webcam face recognition.

Usage:
    python scripts/run_face_recognition.py

Before running, add at least one photo per person under
known_faces/<person_name>/ - see known_faces/README.md.

Press 'q' in the video window to quit.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_recognition_system.camera.live_recognition import LiveRecognizer


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    LiveRecognizer().run()


if __name__ == "__main__":
    main()
