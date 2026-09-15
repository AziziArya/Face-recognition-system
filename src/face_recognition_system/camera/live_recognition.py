"""Live webcam face recognition.

Refactored from the main loop in face_recognition_orginall.ipynb
(cell 1). Behaviour preserved:

- A background thread continuously detects/encodes/matches faces on
  the latest available frame (so recognition doesn't block frame
  capture).
- The main thread reads frames from the camera, draws the most recent
  known face locations/names on top, and displays the video.

Bug fix vs. the original notebook: the original main loop called
`cv.imshow(...)` but never called `cv.waitKey(...)`, so the OpenCV
window never processed UI events and there was no way to exit the
loop or release the camera cleanly. This version adds a `waitKey`
call with a quit key ('q') and releases the camera / destroys windows
on exit.
"""

from __future__ import annotations

import logging
import threading

import cv2 as cv

from face_recognition_system.config.settings import RecognitionConfig
from face_recognition_system.enrollment.known_faces_loader import KnownFaces, load_known_faces
from face_recognition_system.recognition.recognizer import recognize_faces_in_frame

logger = logging.getLogger(__name__)


class LiveRecognizer:
    """Runs webcam capture + background recognition until 'q' is pressed."""

    def __init__(self, config: RecognitionConfig | None = None):
        self.config = config or RecognitionConfig()
        self.known_faces: KnownFaces = load_known_faces(
            self.config.known_faces_dir, detector_model=self.config.detector_model
        )

        self._frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        self._face_locations = []
        self._face_names = []

    def _recognition_worker(self) -> None:
        scale = self.config.frame_resize_scale
        inv_scale = 1.0 / scale

        while not self._stop_event.is_set():
            with self._lock:
                if self._frame is None:
                    continue
                small_frame = cv.resize(self._frame, (0, 0), fx=scale, fy=scale)
                rgb_small_frame = small_frame[:, :, ::-1]  # BGR -> RGB

            locations, names = recognize_faces_in_frame(
                rgb_small_frame,
                self.known_faces,
                detector_model=self.config.detector_model,
                tolerance=self.config.match_tolerance,
            )

            self._face_locations = [
                tuple(int(coord * inv_scale) for coord in loc) for loc in locations
            ]
            self._face_names = names

    def run(self) -> None:
        cap = cv.VideoCapture(self.config.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera at index {self.config.camera_index}."
            )

        worker = threading.Thread(target=self._recognition_worker, daemon=True)
        worker.start()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame from camera; stopping.")
                    break

                with self._lock:
                    self._frame = frame.copy()
                    display_frame = frame.copy()

                for (top, right, bottom, left), name in zip(
                    self._face_locations, self._face_names
                ):
                    cv.rectangle(display_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv.rectangle(
                        display_frame, (left, bottom - 35), (right, bottom),
                        (0, 0, 255), cv.FILLED,
                    )
                    cv.putText(
                        display_frame, name, (left + 6, bottom - 6),
                        cv.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1,
                    )

                cv.imshow("Face Recognition", display_frame)

                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self._stop_event.set()
            worker.join(timeout=1.0)
            cap.release()
            cv.destroyAllWindows()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    LiveRecognizer().run()


if __name__ == "__main__":
    main()
