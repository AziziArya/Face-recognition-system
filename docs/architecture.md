# Architecture

## Pipeline (verified from source code)

```
Webcam frame (BGR, OpenCV)
        ↓
Resize (default 20%) + convert BGR → RGB
        ↓
Face Detection  — face_recognition.face_locations() (HOG model)
        ↓
Face Embedding  — face_recognition.face_encodings() (128-d vector, dlib ResNet)
        ↓
Face Matching   — face_recognition.compare_faces() (tolerance=0.45)
                  + face_recognition.face_distance() (closest-match tie-break)
        ↓
Identity label ("unknown" if no match) → drawn on the live video frame
```

Recognition runs in a background thread on the most recent captured
frame while the main thread keeps grabbing frames and rendering, so
frame capture is not blocked by the (slower) recognition step. This
matches the design of the original notebook.

## Supplementary module: Haar cascade detection

`detection/haar_detector.py` is a **separate, simpler** detection-only
demo (OpenCV Haar cascades for faces and eyes). It does **not** perform
identity recognition. It exists in this repository because it was part
of the original source material (`opencv_camera.ipynb`), but the two
pipelines were never combined in the original code, so this refactor
keeps them as two independent entry points rather than inventing an
integration that wasn't there.

```
Webcam frame
    ↓
Grayscale + Gaussian blur
    ↓
Face detection — haarcascade_frontalface_default.xml
    ↓
Eye detection (within each face box) — haarcascade_eye.xml
    ↓
Annotated frame (boxes only, no identity)
```

## What this project does NOT do

- No embeddings are persisted to disk. Every run re-encodes the photos
  in `known_faces/` from scratch.
- No attendance / check-in logging. There is no timestamp storage,
  CSV/database writing, or duplicate-entry prevention anywhere in the
  source material.
- No web UI, REST API, or database.
- No GPU-accelerated ("cnn") face detection model - the original code
  used the CPU-based HOG model only.
