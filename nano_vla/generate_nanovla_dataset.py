# external imports
import json
import os
import sys
import pygame
# internal imports
from nano_vla.config import DATASET_DIR, MANIFEST_PATH, WINDOW_SZ, IMAGES_DIR
from nano_vla.environment import NanoVLASimulator

LANGUAGE_INSTRUCTION = "Navigate to the red target box avoiding obstacles"


def save_dataset_step(screen, step_num, action_data, robot_pos):
    """Captures the current screen pixels and updates the manifest log."""
    # save the visual frame matrix as a png image file
    img_filename = f'frame_{step_num:05d}.png'
    img_path = os.path.join(IMAGES_DIR, img_filename)
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    assert os.path.exists(DATASET_DIR)
    assert os.path.exists(IMAGES_DIR)
    pygame.image.save(screen, img_path)

    # structure the multimodal training sample metadata
    log_entry = {
        'step': step_num,
        'image_path': img_path,
        'instruction': LANGUAGE_INSTRUCTION,
        'robot_state': list(robot_pos.to_tuple()),
        'action_token_id': action_data['id'],
        'action_token_text': action_data['text'],
    }

    logs = []
    # read the existing logs array or initialize a clean one
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    logs.append(log_entry)

    with open(MANIFEST_PATH, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f'recorded step {step_num:04d} | action taken: {action_data['text']}')


def main():
    # initialize pygame and create the screen
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SZ, WINDOW_SZ))
    pygame.display.set_caption("NanoVLA Data Collection Sandbox")

    # initialize the simulator
    sim = NanoVLASimulator()
    step_counter = 0
    clock = pygame.time.Clock()

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
