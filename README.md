# Face Recognition System

A webcam-based face detection and recognition pipeline built with the
`face_recognition` (dlib) library and OpenCV, plus a supplementary
Haar-cascade face/eye detection demo.

## Overview

This project takes video from a webcam, detects faces, computes a face
embedding for each detected face, and compares it against a small set
of enrolled ("known") faces to label who is visible in the frame. It
was originally developed as an experimentation project and has been
refactored here into a structured, documented Python package.

## Features

Verified against the source implementation:

- Real-time face detection from a webcam feed (HOG model via
  `face_recognition`).
- Face embedding generation (128-d vectors).
- Face recognition by comparing embeddings against enrolled faces,
  using a distance tolerance plus nearest-match tie-breaking.
- Background-threaded recognition so the video feed doesn't freeze
  while a match is computed.
- Directory-based enrollment: add a photo to `known_faces/<name>/` to
  register a person.
- A separate Haar-cascade face + eye **detection-only** demo (no
  identity recognition) built with OpenCV.

**Not included** (not present in the source material - see
[docs/limitations.md](docs/limitations.md)):

- Attendance / check-in logging (no timestamps, no CSV/database).
- Persisted embeddings (encodings are recomputed each run).
- A GUI, web app, or REST API.

## Architecture

```
Webcam
  ↓
Face Detection (HOG, face_recognition)
  ↓
Face Embedding (128-d, face_recognition)
  ↓
Face Recognition (compare_faces + face_distance, tolerance=0.45)
  ↓
Identity label ("unknown" if no match)
```

Full details, including the separate Haar-cascade detection module,
are in [docs/architecture.md](docs/architecture.md).

## Technology Stack

- Python
- [face_recognition](https://github.com/ageitgey/face_recognition) (built on dlib)
- OpenCV (`opencv-python`)
- NumPy

## Project Structure

```
face-recognition-system/
├── src/face_recognition_system/
│   ├── detection/     # Haar-cascade face/eye detection (no identity)
│   ├── recognition/   # Embedding comparison / matching logic
│   ├── enrollment/    # Loads known_faces/ into encodings + names
│   ├── camera/        # Live webcam recognition loop
│   └── config/        # Centralized settings
├── known_faces/        # Enrollment photos (placeholder only, see README inside)
├── models/haarcascades/ # Haar cascade XML files
├── scripts/            # Runnable entry points
├── tests/               # Automated tests
└── docs/                # Architecture, setup, limitations
```

## Installation

```bash
git clone <this-repo-url>
cd face-recognition-system
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Full instructions (including `dlib` build prerequisites) in
[docs/setup.md](docs/setup.md).

## Usage

```bash
# 1. Enroll at least one person
mkdir -p known_faces/your_name
cp /path/to/photo.jpg known_faces/your_name/your_image.jpg

# 2. Run live recognition (press 'q' to quit)
python scripts/run_face_recognition.py

# Optional: Haar-cascade detection demo, no identity recognition
python scripts/run_haar_detection_demo.py
```

## Enrollment

Enrollment is directory-based: create a sub-folder under
`known_faces/` named after the person, and place one or more clear,
front-facing photos inside it. On the next run, every photo is
encoded and used for matching. See
[known_faces/README.md](known_faces/README.md).

There is no dedicated enrollment UI or script in this project - adding
a photo to the folder *is* the enrollment step.

## Recognition

For each detected face, the system compares its embedding against all
enrolled embeddings using `face_recognition.compare_faces()` with a
distance tolerance of `0.45`, then uses `face_recognition.face_distance()`
to pick the closest match among any faces within tolerance. If no
enrolled face is within tolerance, the face is labeled `"unknown"`.

## Configuration

All tunables live in `src/face_recognition_system/config/settings.py`:
camera index, resize scale, match tolerance, and (for the Haar demo)
cascade parameters. Edit that file directly - there is no environment
variable / `.env` mechanism in this project, since none of the
original code read configuration from the environment.

## Limitations

See [docs/limitations.md](docs/limitations.md) for the full, honest
list, including: no attendance functionality, no persisted embeddings,
lighting/pose sensitivity, an untuned match tolerance, and which parts
of the code were/weren't executed during preparation of this
repository.

## Privacy / Biometric Considerations

This project processes facial images and computes biometric face
embeddings. Facial recognition data is sensitive personal information
in many jurisdictions. If you deploy or extend this project:

- Obtain clear, informed consent from anyone you enroll.
- Store enrollment photos and any derived data securely and only for
  as long as needed.
- Check applicable local/regional privacy and biometric data laws
  before using this on real people beyond personal experimentation.

This repository makes no legal claims and provides no compliance
guarantees - it is a technical implementation only.

## License

MIT - see [LICENSE](LICENSE).
