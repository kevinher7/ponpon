import time

import cv2
import mediapipe as mp
import pygame

from ponpon.infra.config import SCREEN_SIZE
from ponpon.tracking.mediapipe_hands import extract_index_finger_coordinates
from ponpon.tracking.mediapipe_hands import initialize_mp_options


def main() -> None:
    pygame.init()
    pygame.display.set_caption("PonPon")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    HandLandmarker = mp.tasks.vision.HandLandmarker

    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()

    running = True

    with HandLandmarker.create_from_options(initialize_mp_options()) as landmarker:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = time.monotonic_ns() // 1_000_000
            result = landmarker.detect_for_video(image, timestamp_ms)

            cam_height, cam_width = frame_bgr.shape[:2]

            player_coordinates = extract_index_finger_coordinates(
                result.hand_landmarks,
                cam_height,
                cam_width,
            )

            for player_cord in player_coordinates:
                cv2.circle(
                    frame_bgr,
                    (player_cord.tip[0], player_cord.tip[1]),
                    10,
                    (255, 0, 0),
                    -1,
                )
                cv2.circle(
                    frame_bgr,
                    (player_cord.base[0], player_cord.base[1]),
                    10,
                    (255, 0, 0),
                    -1,
                )

            frame_bgr = cv2.resize(frame_bgr, SCREEN_SIZE)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.image.frombuffer(
                frame_rgb.tobytes(),
                (frame_rgb.shape[1], frame_rgb.shape[0]),
                "RGB",
            )

            screen.blit(frame_surface, (0, 0))
            pygame.display.flip()

            clock.tick(60)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
