import mediapipe as mp

MODEL_ASSET_PATH = "./models/hand_landmarker.task"
RUNNING_MODE = mp.tasks.vision.RunningMode.VIDEO
NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3


def initialize_mp_options() -> mp.tasks.vision.HandLandmarkerOptions:
    base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_ASSET_PATH)

    return mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RUNNING_MODE,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
