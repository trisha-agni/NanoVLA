# external imports
from PIL import Image
import pygame
import sys
import time
import torch
# internal imports
from nano_vla.config import WINDOW_SZ
from nano_vla.dataset import NanoVLADataset, LANGUAGE_INSTRUCTION
from nano_vla.environment import NanoVLASimulator
from nano_vla.model import NanoVLAModel
from nano_vla.robot_position import ACTION_SPACE

MAX_STEPS = 30
MODEL_WEIGHTS_FILE = 'nanovla_checkpoint.pt'
# various sleep times
INTER_STEP_CLOCK_TICK = 60
INTER_STEP_SLEEP_TIME = 0.3
MAX_STEPS_SLEEP_TIME = 2.0
START_SLEEP_TIME = 1.0
SUCCESS_SLEEP_TIME = 2.0


def get_model_prediction(model, pixel_values, text_inputs, device):
    """
    Handles model inference and isolates the best allowable movement
    action key based on model score calculations.
    """
    # execute the multi-modal forward inference pass
    with torch.no_grad():
        # pass our vision image data array down to the model graph layers
        action_logits = model(
            pixel_values=pixel_values,
            input_ids=text_inputs
        )
    best_key = None
    best_score = -float('inf')

    # score filtering logic: find the highest scoring action within ACTION_SPACE
    for key_code, action_info in ACTION_SPACE.items():
        token_id = action_info['id']
        score = action_logits[0, token_id].item()
        if score > best_score:
            best_score = score
            best_key = key_code
    return best_key


def run_autopilot(max_steps=MAX_STEPS):
    print("🤖 Booting NanoVLA closed-loop autopilot system...")

    # initialize hardware and game window context
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SZ, WINDOW_SZ))
    pygame.display.set_caption("NanoVLA Autopilot Eval Mode")

    # instantiate the simulator
    sim = NanoVLASimulator()

    # load trained model weights from disk
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available()
                          else 'cpu')
    model = NanoVLAModel()
    try:
        # Load and copy weights on CPU before moving the model to MPS.
        checkpoint = torch.load(MODEL_WEIGHTS_FILE, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded weights from '{MODEL_WEIGHTS_FILE}'")
    except FileNotFoundError:
        print(f"❌ Error: '{MODEL_WEIGHTS_FILE}' not found.")
        return

    model.to(device)
    model.eval()  # set nn to explicit eval inference mode

    eval_dataset = NanoVLADataset()
    img_transform = eval_dataset.img_transform
    tok = eval_dataset.tokenizer
    text_inputs = tok(
        LANGUAGE_INSTRUCTION,
        return_tensors='pt'
    )['input_ids'].to(device)

    print('\n Autopilot engaged! Press ESC to interrupt.')
    running = True
    steps_taken = 0
    clock = pygame.time.Clock()

    # initial display render update step
    sim.draw(screen)
    pygame.display.flip()
    # give the human a brief window to watch the start state
    time.sleep(START_SLEEP_TIME)

    while running and steps_taken < max_steps:
        # check for window exit commands manually
        for event in pygame.event.get():
            if (event.type == pygame.QUIT or
                (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)):
                running = False

        if not running:
            break

        # capture real-time screen pixels
        # convert the active pygame screen buffer back into a PIL image array
        screen_bytes = pygame.image.tobytes(screen, 'RGB')
        pil_img = Image.frombytes("RGB",
                                  (WINDOW_SZ, WINDOW_SZ),
                                  screen_bytes)
        # shape: (1, 3, 224, 224)
        pixel_values = img_transform(pil_img).unsqueeze(0).to(device)

        # inference and score filtering
        best_key = get_model_prediction(model, pixel_values, text_inputs, device)

        # execute the decision closed-loop
        if best_key:
            action_executed = sim.update(best_key)
            if action_executed:
                print(f'Step {steps_taken+1:02d} | '
                      f'Model predicted: {action_executed['text']}')
            else:
                print(
                    f'Step {steps_taken+1:02d} | '
                    f'Model tried an invalid move ({ACTION_SPACE[best_key]['text']})'
                )

        # re-render and evaluate reward check
        sim.draw(screen)
        pygame.display.flip()
        steps_taken += 1

        if sim.robot_pos == sim.target_pos:
            print('\n🎉 Mission success! '
                  'The robot reached the target box on full autopilot.')
            time.sleep(SUCCESS_SLEEP_TIME)
            break

        time.sleep(INTER_STEP_SLEEP_TIME)
        clock.tick(INTER_STEP_CLOCK_TICK)

    if steps_taken >= max_steps:
        print('\nOut of steps! '
              'The model wandered or timed out before hitting the target')
        time.sleep(MAX_STEPS_SLEEP_TIME)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    run_autopilot()
