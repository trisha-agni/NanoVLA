# external imports
import pygame
# internal imports
from nano_vla.config import GRID_SZ


# discretized action space mapping for NanoVLA
ACTION_SPACE = {
    pygame.K_UP:    {"id": 0, "text": "move_up",    "delta": (0, -1)},
    pygame.K_DOWN:  {"id": 1, "text": "move_down",  "delta": (0, 1)},
    pygame.K_LEFT:  {"id": 2, "text": "move_left",  "delta": (-1, 0)},
    pygame.K_RIGHT: {"id": 3, "text": "move_right", "delta": (1, 0)},
}


class RobotPosition:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def parse_action(key_event):
        """Looks up a raw keyboard integer and returns its metadata packet safely."""
        return ACTION_SPACE.get(key_event, None)

    def get_next(self, key_event):
        """Computes a speculative position directly from a raw input event key."""
        action = ACTION_SPACE.get(key_event)
        if not action:
            return self  # Return unchanged if key is unrecognized

        dx, dy = action["delta"]
        return RobotPosition(self.x + dx, self.y + dy)

    def is_valid(self, obstacles_list):
        """Self-validates if this coordinate is inside the board and free of walls."""
        if not (0 <= self.x < GRID_SZ and 0 <= self.y < GRID_SZ):
            return False
        if self in obstacles_list:
            return False
        return True

    def __eq__(self, other):
        if isinstance(other, RobotPosition):
            return self.x == other.x and self.y == other.y
        return False

    def to_tuple(self):
        return (self.x, self.y)
