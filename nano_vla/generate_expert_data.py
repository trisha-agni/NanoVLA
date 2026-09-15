# external imports
import pygame
import random
# internal imports
from nano_vla.config import WINDOW_SZ, GRID_SZ
from nano_vla.dataset import save_dataset_step, get_next_step_index
from nano_vla.environment import NanoVLASimulator
from nano_vla.robot_position import RobotPosition, ACTION_SPACE

MAX_STEPS = 25
NUM_EPISODES = 50
Pos = RobotPosition
RANDOM_SEED = 42


def compute_expert_action(cur_pos, target_pos, obstacles):
    """
    Algorithmic teacher: simple greedy pathfinder with wall avoidance mechanism.
    Randomizes between horizontal and vertical
    movement preferences on every frame step to eliminate pathing bias.
    """
    dx = target_pos.x - cur_pos.x
    dy = target_pos.y - cur_pos.y

    h_moves = []
    v_moves = []

    if dx > 0:
        h_moves.append(pygame.K_RIGHT)
    elif dx < 0:
        h_moves.append(pygame.K_LEFT)

    if dy > 0:
        v_moves.append(pygame.K_DOWN)
    elif dy < 0:
        v_moves.append(pygame.K_UP)

    move_choices = []
    if h_moves:
        move_choices.append(h_moves)
    if v_moves:
        move_choices.append(v_moves)

    # shuffle the list items randomly
    random.shuffle(move_choices)

    # flatten the shuffled groups down into a final list of candidate keys
    pref_moves = [move for sublist in move_choices for move in sublist]

    # pick the first move that doesn't hit a wall obstacle
    for move in pref_moves:
        next_pos = cur_pos.get_next(move)
        if next_pos.is_valid(obstacles):
            return move

    # back-up recovery search scan: try any valid move if shortest paths are blocked
    all_keys = list(ACTION_SPACE.keys())
    random.shuffle(all_keys)
    for move in all_keys:
        if cur_pos.get_next(move).is_valid(obstacles):
            return move

    return None  # fully trapped


def generate_synthetic_data(num_episodes=NUM_EPISODES):
    """
    Master data factory: automatically plays the game, randomizs layouts,
    captures pixel frames and saves the expert trajectory logs out to disk.
    """
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SZ, WINDOW_SZ))
    pygame.display.set_caption("NanoVLA Synthetic Expert Factory")

    sim = NanoVLASimulator()
    step_counter = get_next_step_index()
    print(f'Synthetic generation engine booted. Resuming at index: {step_counter}')

    for ep in range(num_episodes):
        while True:
            rx, ry = random.randint(0, GRID_SZ - 1), random.randint(0, GRID_SZ - 1)
            tx, ty = random.randint(0, GRID_SZ - 1), random.randint(0, GRID_SZ - 1)

            sim.robot_pos = Pos(rx, ry)
            sim.target_pos = Pos(tx, ty)

            if (sim.robot_pos not in sim.obstacles and
                sim.target_pos not in sim.obstacles and
                sim.robot_pos != sim.target_pos):
                break

        # episode execution loop
        steps_in_episode = 0
        while steps_in_episode < MAX_STEPS:
            sim.draw(screen)
            pygame.display.flip()

            # query the randomized algorithmic teacher pathfinder
            best_key = compute_expert_action(sim.robot_pos,
                                             sim.target_pos,
                                             sim.obstacles)
            if best_key is None:
                break  # termiate episode early if fully trapped

            action_data = Pos.parse_action(best_key)

            save_dataset_step(screen, step_counter, action_data, sim.robot_pos)

            sim.update(best_key)
            step_counter += 1
            steps_in_episode += 1

            if sim.robot_pos == sim.target_pos:
                print(f"Episode {ep+1:02d}/{num_episodes:02d} completed successfully!")
                break

    pygame.quit()
    print(f"Generation complete. Total active database entries logged: {step_counter}")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    generate_synthetic_data()
