# Known Limitations

## Functional

- **No persisted embeddings.** Every run re-computes encodings from
  the images in `known_faces/`. There is no embedding database/cache.
- **No attendance system.** This is not an attendance/check-in
  system. There is no timestamp logging, CSV/database storage, or
  duplicate-entry prevention. If you need that, it would be a new
  feature built on top of this recognition pipeline, not something
  this repository currently does.
- **No GPU-accelerated detection.** Only the CPU-based "hog" detector
  from `face_recognition` was used in the source code; the more
  accurate "cnn" model was never used or verified.
- **Single embedding per photo.** If multiple faces appear in an
  enrollment photo, only the first detected face is used.
- **No liveness/anti-spoofing.** The system matches faces against
  static embeddings; it does not attempt to detect photos, videos, or
  masks presented to the camera.

## Recognition quality

- Sensitive to lighting - poor or uneven lighting reduces detection
  and match reliability.
- Sensitive to pose - the HOG detector performs worse on strongly
  angled or partially occluded faces.
- `match_tolerance` (default `0.45`) is the value used in the original
  notebook; it was not tuned against a labeled evaluation set. No
  accuracy/precision/recall numbers are available for this
  implementation.
- Camera resolution/quality directly affects detection and encoding
  quality.

## Testing

- The full recognition pipeline (`enrollment/`, `recognition/`,
  `camera/`) depends on `dlib`, which was **not** built in the
  development/verification sandbox used to prepare this repository
  (no CMake available, single CPU core, and a from-source `dlib`
  build was impractical in the available time). Those modules were
  verified by:
  - `py_compile` syntax checking (passed for all files), and
  - direct code review against the original notebook logic.
  - Their unit tests (`tests/test_known_faces_loader.py`) are written
    but will **skip automatically** (via `pytest.importorskip`) on any
    machine without `face_recognition`/`dlib` installed.
- The Haar-cascade detection module (`detection/haar_detector.py`),
  which only depends on OpenCV, **was** installed and its tests **were
  run successfully**: 6 passed, 1 skipped.
- No end-to-end test with a real webcam or real face photo was
  performed - that would require camera access and real biometric
  data, neither of which is appropriate for an automated test suite.

## Data / privacy

- No dataset is bundled with this repository. `known_faces/` ships
  with a placeholder folder only.
- `shape_predictor_68_face_landmarks.dat`, used elsewhere for facial
  landmark work, is intentionally **not** included - it was only used
  for eye/drowsiness-related code in the source material, which is
  out of scope for this project.
