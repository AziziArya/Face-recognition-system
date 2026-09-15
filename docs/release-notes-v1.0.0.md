# v1.0.0 — Initial Release

## Highlights

- Verified face detection (HOG model, `face_recognition`/dlib).
- Verified face recognition (128-d embedding comparison with tolerance-based
  matching and nearest-match tie-breaking).
- Verified embedding pipeline (computed at runtime from enrolled photos;
  not persisted to disk).
- Verified directory-based enrollment (`known_faces/<name>/`).
- Supplementary Haar-cascade face + eye detection demo (OpenCV), kept
  as a separate module with no identity recognition.
- **No attendance functionality** — not present in the source project,
  not included here.

## Technical Stack

- Python
- face_recognition (dlib)
- OpenCV (`opencv-python`)
- NumPy

## Documentation

- `README.md` — overview, features, usage
- `docs/architecture.md` — verified pipeline details
- `docs/setup.md` — installation and run instructions
- `docs/limitations.md` — honest limitations, including what was and
  wasn't executed/tested
- `CHANGELOG.md` — what changed vs. the original notebooks

## Known Limitations

- No persisted embeddings; recomputed every run.
- No attendance/check-in logging.
- Match tolerance (0.45) is inherited from the original code and was
  not tuned against a labeled evaluation set — no accuracy numbers
  are available.
- Recognition quality is sensitive to lighting, pose, and camera
  quality.
- The `dlib`-dependent modules were verified by syntax check and code
  review, not by an executed test run, in the environment used to
  prepare this release (see `docs/limitations.md` for why).
