import time

import cv2
import mediapipe as mp
import pygame

from ponpon.infra.config import SCREEN_SIZE
from ponpon.tracking.mediapipe_hands import initialize_mp_options


def main() -> None:
    pygame.init()
    pygame.display.set_caption("PonPon")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera")

    HandLandmarker = mp.tasks.vision.HandLandmarker
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

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

            h, w = frame_bgr.shape[:2]
            for hand in result.hand_landmarks:
                points: list[tuple[int, int]] = []
                for landmark in hand:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    points.append((x, y))
                    cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1)

                for conn in connections:
                    if hasattr(conn, "start"):
                        s, e = conn.start, conn.end
                    else:
                        s, e = conn
                    cv2.line(frame_bgr, points[s], points[e], (0, 200, 255), 2)

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
