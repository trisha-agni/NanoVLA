# external imports
import sys
import pygame
# internal imports
from environment import NanoVLASimulator, WINDOW_SZ


def main():
    # initialize pygame and create the screen
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SZ, WINDOW_SZ))
    pygame.display.set_caption("NanoVLA Data Collection Sandbox")

    # initialize the simulator
    sim = NanoVLASimulator()

    # core game loop
    running = True
    while running:
        # check for inputs or window events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        sim.draw(screen)
        pygame.display.flip()  # refresh screen
        # clean up and exit smoothly when running is False
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
