"""Face detection + embedding + matching for a single video frame.

Extracted from the `process_faces()` function in
face_recognition_orginall.ipynb. The algorithm is unchanged:

1. Detect face locations in the frame with `face_recognition.face_locations`.
2. Compute a 128-d embedding per detected face with
   `face_recognition.face_encodings`.
3. Compare each embedding against the known embeddings with
   `face_recognition.compare_faces` (tolerance-based) and
   `face_recognition.face_distance` (to pick the closest match among
   any faces that pass the tolerance check).
"""

from __future__ import annotations

from typing import List, Tuple

import face_recognition as fr
import numpy as np

from face_recognition_system.enrollment.known_faces_loader import KnownFaces

UNKNOWN_LABEL = "unknown"

FaceLocation = Tuple[int, int, int, int]  # (top, right, bottom, left)


def recognize_faces_in_frame(
    rgb_frame: np.ndarray,
    known_faces: KnownFaces,
    detector_model: str = "hog",
    tolerance: float = 0.45,
) -> Tuple[List[FaceLocation], List[str]]:
    """Detect and identify faces in an RGB frame.

    Returns (face_locations, face_names), matching the shape of the
    original notebook's `face_locations` / `face_names` variables.
    """
    face_locations = fr.face_locations(rgb_frame, model=detector_model)
    face_encodings = fr.face_encodings(rgb_frame, face_locations, model=detector_model)

    face_names: List[str] = []
    for face_encoding in face_encodings:
        name = UNKNOWN_LABEL

        if len(known_faces) > 0:
            matches = fr.compare_faces(
                known_faces.encodings, face_encoding, tolerance=tolerance
            )
            face_distances = fr.face_distance(known_faces.encodings, face_encoding)
            best_match_index = int(np.argmin(face_distances))

            if matches[best_match_index]:
                name = known_faces.names[best_match_index]

        face_names.append(name)

    return face_locations, face_names
