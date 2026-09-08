# external imports
import sys
import pygame
# internal imports
from nano_vla.config import WINDOW_SZ
from nano_vla.dataset import save_dataset_step, get_next_step_index
from nano_vla.environment import NanoVLASimulator


def main():
    # initialize pygame and create the screen
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SZ, WINDOW_SZ))
    pygame.display.set_caption("NanoVLA Data Collection Sandbox")

    # initialize the simulator
    sim = NanoVLASimulator()
    step_counter = 0
    clock = pygame.time.Clock()
    step_counter = get_next_step_index()
    if step_counter > 0:
        print('found existing dataset! '
              f'resuming collection from step {step_counter}')

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
                        sim.draw(screen)
                        pygame.display.flip()

                        # export our newly recorded state data point
                        save_dataset_step(screen, step_counter, action, sim.robot_pos)
                        step_counter += 1
        # render frame refresh
        sim.draw(screen)
        pygame.display.flip()
        clock.tick(60)  # caps engine speed at 60 fps to keep inputs smooth

    # clean up and exit smoothly when running is False
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
