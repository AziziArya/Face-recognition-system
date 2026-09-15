"""Central configuration for the face recognition system.

These values were previously hardcoded inline across the original
notebooks (camera index, resize scale, match tolerance). They are
collected here so they can be tuned in one place instead of edited
inside the recognition loop.
"""

from dataclasses import dataclass
from pathlib import Path


# Repository root (this file lives at src/face_recognition_system/config/)
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RecognitionConfig:
    """Settings for the face_recognition-based recognition pipeline."""

    # Folder containing one sub-folder per known person, each with
    # one or more face images. See known_faces/README.md.
    known_faces_dir: Path = REPO_ROOT / "known_faces"

    # Index passed to cv2.VideoCapture(). 0 is usually the default webcam.
    camera_index: int = 0

    # Detection model used by the `face_recognition` library.
    # "hog" is CPU-friendly (used in the original notebooks).
    # "cnn" is more accurate but requires a GPU/dlib built with CUDA
    # and was not used in the original code - kept as documentation only.
    detector_model: str = "hog"

    # Distance tolerance for face_recognition.compare_faces().
    # Lower = stricter match. This matches the value used in the
    # original notebook (0.45).
    match_tolerance: float = 0.45

    # Downscale factor applied to each frame before detection, for
    # performance. The original notebook used 0.2 (i.e. resized to 20%).
    frame_resize_scale: float = 0.2

    # Process every Nth frame instead of every frame, to reduce load.
    process_every_n_frames: int = 4


@dataclass
class HaarDetectionConfig:
    """Settings for the supplementary Haar-cascade detection demo."""

    face_cascade_path: Path = REPO_ROOT / "models" / "haarcascades" / "haarcascade_frontalface_default.xml"
    eye_cascade_path: Path = REPO_ROOT / "models" / "haarcascades" / "haarcascade_eye.xml"

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    target_fps: int = 30

    face_scale_factor: float = 1.05
    face_min_neighbors: int = 8
    face_min_size: tuple = (100, 100)

    eye_scale_factor: float = 1.1
    eye_min_neighbors: int = 5
    eye_min_size: tuple = (30, 30)

    max_eyes_per_face: int = 2
