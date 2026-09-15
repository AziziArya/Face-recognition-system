"""Haar-cascade face + eye detection (OpenCV), no identity recognition.

Refactored from the last working cell of opencv_camera.ipynb. This is
a separate, simpler detection-only demo - it does NOT identify who a
face belongs to (that's what recognition/recognizer.py does with the
face_recognition library). It is kept as its own module because the
original notebooks never combined Haar-cascade detection with
face_recognition-based identification; merging them was not part of
the verified source behaviour.

Note: this module detects and draws boxes around faces/eyes only. It
has no relation to eye-closure / drowsiness detection - it does not
compute any eye-aspect-ratio or blink state.
"""

from __future__ import annotations

import logging

import cv2 as cv

from face_recognition_system.config.settings import HaarDetectionConfig

logger = logging.getLogger(__name__)


class HaarFaceEyeDetector:
    """Detects faces and, within each face region, eyes - using Haar cascades."""

    def __init__(self, config: HaarDetectionConfig | None = None):
        self.config = config or HaarDetectionConfig()

        self.face_cascade = cv.CascadeClassifier(str(self.config.face_cascade_path))
        self.eye_cascade = cv.CascadeClassifier(str(self.config.eye_cascade_path))

        if self.face_cascade.empty():
            raise IOError(f"Could not load face cascade: {self.config.face_cascade_path}")
        if self.eye_cascade.empty():
            raise IOError(f"Could not load eye cascade: {self.config.eye_cascade_path}")

    def detect(self, frame):
        """Detect faces and eyes in a single BGR frame.

        Returns the frame annotated with rectangles (face: green,
        eyes: red) and does not mutate the input frame.
        """
        annotated = frame.copy()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.face_scale_factor,
            minNeighbors=self.config.face_min_neighbors,
            minSize=self.config.face_min_size,
        )

        for (x, y, w, h) in faces:
            cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                face_roi,
                scaleFactor=self.config.eye_scale_factor,
                minNeighbors=self.config.eye_min_neighbors,
                minSize=self.config.eye_min_size,
            )

            for i, (ex, ey, ew, eh) in enumerate(eyes):
                if i >= self.config.max_eyes_per_face:
                    break
                cv.rectangle(
                    annotated, (x + ex, y + ey), (x + ex + ew, y + ey + eh),
                    (0, 0, 255), 2,
                )

        return annotated, faces


def run_webcam_demo(config: HaarDetectionConfig | None = None) -> None:
    """Runs a live webcam window showing Haar-cascade face/eye detection."""
    config = config or HaarDetectionConfig()
    detector = HaarFaceEyeDetector(config)

    cam = cv.VideoCapture(config.camera_index)
    cam.set(cv.CAP_PROP_FPS, config.target_fps)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, config.frame_width)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if not cam.isOpened():
        raise RuntimeError(f"Could not open camera at index {config.camera_index}.")

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                logger.warning("Failed to read frame from camera; stopping.")
                break

            annotated, _ = detector.detect(frame)
            cv.imshow("Haar Face/Eye Detection", annotated)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_webcam_demo()
