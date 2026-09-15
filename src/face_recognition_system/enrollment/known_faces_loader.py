"""Loads known/enrolled faces and computes their embeddings.

Original behaviour (face_recognition_orginall.ipynb): each known person
was enrolled by manually calling `fr.load_image_file(...)` and
`fr.face_encodings(...)` once per image file, with the name typed into
a parallel `known_face_names` list by hand.

This module preserves that exact encoding logic (face_recognition,
model="hog") but replaces the manual per-file/per-name editing with a
directory scan: one sub-folder per person under `known_faces/`, e.g.

    known_faces/
        alice/
            alice_1.jpg
        bob/
            bob_front.jpg
            bob_side.jpg

This is a refactor of *how enrollment data is provided*, not a change
to the recognition algorithm itself. "Enrollment" in this project means
adding an image to the right folder - there is no separate enrollment
UI/script in the source material, and this project does not claim one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import face_recognition as fr
import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".jfif", ".bmp"}


@dataclass
class KnownFaces:
    encodings: list
    names: list

    def __len__(self) -> int:
        return len(self.names)


def load_known_faces(known_faces_dir: Path, detector_model: str = "hog") -> KnownFaces:
    """Scan `known_faces_dir` and compute one embedding per image.

    Each immediate sub-directory of `known_faces_dir` is treated as one
    person's name. Every supported image file inside it is encoded with
    `face_recognition.face_encodings`. If a person has multiple photos,
    each photo contributes a separate (encoding, name) entry, matching
    the original notebook's approach of comparing against a flat list
    of encodings/names.

    Images where no face can be encoded are skipped with a warning
    rather than raising, so one bad photo does not break enrollment
    for everyone else.
    """
    known_faces_dir = Path(known_faces_dir)
    encodings: list = []
    names: list = []

    if not known_faces_dir.exists():
        raise FileNotFoundError(
            f"known_faces directory not found: {known_faces_dir}. "
            "Create it and add one sub-folder per person."
        )

    person_dirs = sorted(p for p in known_faces_dir.iterdir() if p.is_dir())

    if not person_dirs:
        logger.warning("No person sub-folders found under %s", known_faces_dir)

    for person_dir in person_dirs:
        person_name = person_dir.name
        image_paths = sorted(
            p for p in person_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not image_paths:
            logger.warning("No images found for '%s' in %s", person_name, person_dir)
            continue

        for image_path in image_paths:
            image = fr.load_image_file(str(image_path))
            face_encodings = fr.face_encodings(image, model=detector_model)

            if not face_encodings:
                logger.warning(
                    "No face detected in %s - skipping this image.", image_path
                )
                continue

            encodings.append(face_encodings[0])
            names.append(person_name)

    if not encodings:
        raise ValueError(
            f"No usable face encodings found under {known_faces_dir}. "
            "Add at least one clear, front-facing photo per person."
        )

    logger.info("Loaded %d face encoding(s) for %d image(s).", len(encodings), len(encodings))
    return KnownFaces(encodings=encodings, names=names)
