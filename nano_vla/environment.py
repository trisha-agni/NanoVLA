# external imports
import pygame
# internal imports
from nano_vla.robot_position import RobotPosition, GRID_SZ

# define window dimensions
WINDOW_SZ = 400
CELL_SZ = WINDOW_SZ // GRID_SZ

# colors
COLOR_BG = (240, 240, 240)        # light grey background
COLOR_GRID = (210, 210, 210)      # subtle dark grey for lines
COLOR_ROBOT = (50, 200, 50)       # green
COLOR_TARGET = (250, 50, 50)      # red
COLOR_OBSTACLE = (100, 100, 100)  # dark grey wallss


class NanoVLASimulator:
    def __init__(self):
        """Initialize game state and environment layout."""
        self.robot_pos = RobotPosition(0, 0)
        self.target_pos = RobotPosition(9, 9)
        self.obstacles = [
            RobotPosition(3, 3),
            RobotPosition(3, 4),
            RobotPosition(3, 5),
            RobotPosition(6, 4),
            RobotPosition(6, 5),
            RobotPosition(6, 6)
        ]

    def draw(self, screen):
        self._draw_grid(screen)
        self._draw_env(screen)
        self._draw_robot(screen)

    def update(self, key_event):
        """Processes events blindly without knowing anything about key maps or deltas."""
        # ask our smart vector to evaluate the raw keyboard event key
        next_pos = self.robot_pos.get_next(key_event)

        # check if the resulting target spot hits a barrier wall
        if not next_pos.is_valid(self.obstacles):
            return None

        # success: Commit movement tracking state
        self.robot_pos = next_pos

        # return the parsed meta metadata packet back up to our data loop logger
        return RobotPosition.parse_action(key_event)

    def _draw_grid(self, screen):
        """Clear screen and overlay the 10x10 matrix map lines."""
        # fill the screen background color
        screen.fill(COLOR_BG)

        # draw the grid lines
        # loop through every grid intersection coordinates
        for i in range(1, GRID_SZ):
            pos = i * CELL_SZ  # 40, 80, 120, etc.
            # draw vertical lines: (screen, color, start_pos, end_pos)
            pygame.draw.line(screen, COLOR_GRID, (pos, 0), (pos, WINDOW_SZ))
            # draw horizontal lines: (screen, color, start_pos, end_pos)
            pygame.draw.line(screen, COLOR_GRID, (0, pos), (WINDOW_SZ, pos))

    def _draw_env(self, screen):
        """Render static targets and obstacles."""
        self._draw_target(screen)
        self._draw_obstacles(screen)

    def _draw_target(self, screen):
        """Render static target."""
        rect = pygame.Rect(
            self.target_pos.x * CELL_SZ,
            self.target_pos.y * CELL_SZ,
            CELL_SZ,
            CELL_SZ
        )
        pygame.draw.rect(screen, COLOR_TARGET, rect)

    def _draw_obstacles(self, screen):
        """Render static obstacles."""
        for obs in self.obstacles:
            # multiply matrix coordinates by CELL_SZ to get screen pos
            rect = pygame.Rect(
                obs.x * CELL_SZ,
                obs.y * CELL_SZ,
                CELL_SZ,
                CELL_SZ
            )
            pygame.draw.rect(screen, COLOR_OBSTACLE, rect)

    def _draw_robot(self, screen):
        """Handles detailed rendering for the stylized robot agent."""
        rx = self.robot_pos.x * CELL_SZ
        ry = self.robot_pos.y * CELL_SZ

        # metal body
        robot_rect = pygame.Rect(rx + 4, ry + 8, CELL_SZ - 8, CELL_SZ - 12)
        pygame.draw.rect(screen, COLOR_ROBOT, robot_rect, border_radius=6)

        # antenna node
        pygame.draw.line(
            screen,
            (80, 80, 80),
            (rx + CELL_SZ//2, ry + 8),
            (rx + CELL_SZ//2, ry + 2), 3
        )
        pygame.draw.circle(screen, (250, 50, 50), (rx + CELL_SZ//2, ry + 2), 3)

        # expressive eyes
        eye_y = ry + 16
        pygame.draw.circle(screen, (255, 255, 255), (rx + 14, eye_y), 5)
        pygame.draw.circle(screen, (255, 255, 255), (rx + 26, eye_y), 5)
        pygame.draw.circle(screen, (0, 0, 0), (rx + 14, eye_y), 2)
        pygame.draw.circle(screen, (0, 0, 0), (rx + 26, eye_y), 2)
