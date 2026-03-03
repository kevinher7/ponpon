import dataclasses

import mediapipe as mp

MODEL_ASSET_PATH = "./models/hand_landmarker.task"
RUNNING_MODE = mp.tasks.vision.RunningMode.VIDEO
NUM_HANDS = 2
MIN_HAND_DETECTION_CONFIDENCE = 0.3
MIN_HAND_PRESENCE_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3


@dataclasses.dataclass
class INDEX_FINGER_COORDINATES:
    tip: tuple[int, int]
    base: tuple[int, int]


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


def extract_index_finger_coordinates(
    hand_landmarks, cam_height: int, cam_width: int
) -> tuple[INDEX_FINGER_COORDINATES, ...]:
    player_coordinates: list[INDEX_FINGER_COORDINATES] = []

    for hand in hand_landmarks:
        index_tip = hand[8]
        index_base = hand[5]

        tip_coordinates = int(index_tip.x * cam_width), int(index_tip.y * cam_height)
        base_coordinates = int(index_base.x * cam_width), int(index_base.y * cam_height)

        player_coordinates.append(
            INDEX_FINGER_COORDINATES(
                tip=tip_coordinates,
                base=base_coordinates,
            )
        )

    return tuple(player_coordinates)
