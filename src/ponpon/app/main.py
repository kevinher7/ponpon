import time

import mediapipe as mp
import numpy as np
import pygame
import pygame.camera

from ponpon.infra.config import SCREEN_SIZE
from ponpon.tracking.camera import init_camera
from ponpon.tracking.mediapipe_hands import initialize_mp_options


def main() -> None:
    pygame.init()
    pygame.display.set_caption("PonPon")

    cam = init_camera()

    if not cam:
        raise RuntimeError("Failed to find camera")

    cam.start()

    HandLandmarker = mp.tasks.vision.HandLandmarker
    mp_options = initialize_mp_options()

    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()

    running = True

    with HandLandmarker.create_from_options(mp_options) as landmarker:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if cam.query_image():
                surface = cam.get_image()
                frame_rgb = np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))
                frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=frame_rgb,
                )

                timestamp_ms = time.monotonic_ns() // 1_000_000
                hand_landmarker_result = landmarker.detect_for_video(image, timestamp_ms)
                hands_detected = len(hand_landmarker_result.hand_landmarks)

                print(hands_detected)

                screen.blit(surface, (0, 0))
                pygame.display.flip()

            clock.tick(60)

    cam.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
