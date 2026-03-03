import pygame.camera

from ponpon.infra.config import SCREEN_SIZE


def init_camera() -> pygame.camera.Camera | None:
    pygame.camera.init()

    camlist = pygame.camera.list_cameras()

    if not camlist:
        print("No cameras detected")
        return None

    return pygame.camera.Camera(camlist[0], SCREEN_SIZE)
