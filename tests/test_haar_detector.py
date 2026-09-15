import numpy as np
import pytest

from face_recognition_system.config.settings import HaarDetectionConfig
from face_recognition_system.detection.haar_detector import HaarFaceEyeDetector


@pytest.fixture
def detector():
    return HaarFaceEyeDetector(HaarDetectionConfig())


def test_cascades_load_successfully(detector):
    assert not detector.face_cascade.empty()
    assert not detector.eye_cascade.empty()


def test_detect_on_blank_frame_returns_no_faces(detector):
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated, faces = detector.detect(blank_frame)

    assert annotated.shape == blank_frame.shape
    assert len(faces) == 0


def test_detect_does_not_mutate_input_frame(detector):
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = blank_frame.copy()

    detector.detect(blank_frame)

    assert np.array_equal(blank_frame, original)


def test_missing_cascade_file_raises(tmp_path):
    bad_config = HaarDetectionConfig(
        face_cascade_path=tmp_path / "does_not_exist.xml",
        eye_cascade_path=tmp_path / "also_missing.xml",
    )
    with pytest.raises(IOError):
        HaarFaceEyeDetector(bad_config)
