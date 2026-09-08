# external imports
import json
import os
import pygame
import random
# internal imports
from nano_vla.config import EXPERT_DATA_DIR, WINDOW_SZ, GRID_SZ
from nano_vla.dataset import save_dataset_step, get_next_step_index
from nano_vla.environment import NanoVLASimulator
from nano_vla.robot_position import RobotPosition, ACTION_SPACE

Pos = RobotPosition


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

    return None
