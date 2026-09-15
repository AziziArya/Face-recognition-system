import pytest

fr = pytest.importorskip(
    "face_recognition",
    reason="face_recognition/dlib not installed in this environment",
)

from face_recognition_system.enrollment.known_faces_loader import load_known_faces


def test_missing_directory_raises(tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        load_known_faces(missing_dir)


def test_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "known_faces"
    empty_dir.mkdir()
    (empty_dir / "someone").mkdir()
    with pytest.raises(ValueError):
        load_known_faces(empty_dir)
