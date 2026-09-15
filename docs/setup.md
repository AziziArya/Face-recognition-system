# Setup

## Requirements

- Python 3.9+
- A webcam
- CMake + a C++ compiler (required to build `dlib`, a dependency of
  `face_recognition`). On Ubuntu/Debian: `sudo apt install cmake build-essential`.
  On Windows, install "Desktop development with C++" via the Visual
  Studio Build Tools. On macOS: `xcode-select --install`.

## Install

```bash
git clone <this-repo-url>
cd face-recognition-system

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

`dlib` is the slowest part of this install - it compiles from source
and can take several minutes depending on your machine.

## Enroll faces

```bash
mkdir -p known_faces/your_name
# copy a clear, front-facing photo into that folder
cp /path/to/photo.jpg known_faces/your_name/your_image.jpg
```

See `known_faces/README.md` for details.

## Run

```bash
# Live face recognition (press 'q' to quit)
python scripts/run_face_recognition.py

# Haar-cascade face/eye detection demo, no identity recognition (press 'q' to quit)
python scripts/run_haar_detection_demo.py
```

## Run tests

```bash
pip install -r requirements.txt   # includes pytest
pytest -v
```

Note: tests that depend on `face_recognition`/`dlib` are automatically
skipped if that package isn't installed (see `docs/limitations.md`).
