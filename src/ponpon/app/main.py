import pygame
import pygame.camera

from ponpon.tracking.camera import init_camera

SCREEN_SIZE = (1280, 720)


def main() -> None:
    pygame.init()
    pygame.display.set_caption("PonPon")

    cam = init_camera()

    if not cam:
        raise RuntimeError("Failed to find camera")

    cam.start()

    screen = pygame.display.set_mode(SCREEN_SIZE)
    clock = pygame.time.Clock()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if cam.query_image():
            snapshot = cam.get_image()

            screen.blit(snapshot, (0, 0))
            pygame.display.flip()

        # Fill the screen with a color to wipe away anything from last frame
        # screen.fill("black")

        # RENDER YOUR GAME HERE

        clock.tick(60)

    cam.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
