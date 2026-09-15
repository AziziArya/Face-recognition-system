from face_recognition_system.config.settings import HaarDetectionConfig, RecognitionConfig


def test_recognition_config_defaults():
    cfg = RecognitionConfig()
    assert cfg.camera_index == 0
    assert cfg.detector_model == "hog"
    assert 0 < cfg.match_tolerance < 1
    assert 0 < cfg.frame_resize_scale <= 1
    assert cfg.process_every_n_frames >= 1


def test_haar_config_defaults():
    cfg = HaarDetectionConfig()
    assert cfg.face_cascade_path.name == "haarcascade_frontalface_default.xml"
    assert cfg.eye_cascade_path.name == "haarcascade_eye.xml"
    assert cfg.max_eyes_per_face == 2
