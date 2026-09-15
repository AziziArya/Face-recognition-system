# Changelog

## [1.0.0] - Initial Release

Reconstructed and refactored from original experimentation notebooks
into a structured Python package.

### Included

- Live webcam face detection + recognition pipeline (`face_recognition`/dlib,
  HOG detector, 128-d embeddings, tolerance-based matching).
- Directory-based face enrollment (`known_faces/<name>/`).
- Supplementary Haar-cascade face/eye detection demo (OpenCV), kept
  as a separate, non-identity-recognition module.
- Centralized configuration (`config/settings.py`).
- Unit tests for the Haar detection module and configuration defaults
  (executed and passing); unit tests for the enrollment loader
  (written, auto-skip without `dlib`).
- README, architecture, setup, and limitations documentation.
- MIT License.

### Changed vs. original notebooks

- Enrollment changed from manually editing hardcoded
  filename/name lists in code to scanning a `known_faces/<name>/`
  directory. The underlying encoding algorithm is unchanged.
- Fixed a bug in the original webcam loop: it called `cv2.imshow()`
  without `cv2.waitKey()`, so the window never processed events and
  the camera/resources were never released. The refactored loop adds
  a `waitKey` call with a quit key (`q`) and releases resources on
  exit.
- Configuration values that were hardcoded inline (camera index,
  resize scale, match tolerance) were moved into
  `config/settings.py`.

### Excluded

- Driver drowsiness / eye-closure detection code (separate project,
  out of scope - see project instructions).
- Attendance/check-in logging - not present in the source material.
- `shape_predictor_68_face_landmarks.dat` - only used by the excluded
  drowsiness code.
