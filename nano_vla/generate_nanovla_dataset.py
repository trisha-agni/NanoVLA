# external imports
import sys
import pygame
# internal imports
from nano_vla.environment import NanoVLASimulator, WINDOW_SZ


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
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    # pass the key down event to the simulator
                    action = sim.update(event.key)
                    if action:
                        print(
                            f"action verified: {action['text']} "
                            f"(Token ID: {action['id']})"
                        )
        # render frame refresh
        sim.draw(screen)
        pygame.display.flip()
        # clean up and exit smoothly when running is False
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
