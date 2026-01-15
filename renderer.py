import pygame
import numpy as np
from typing import Optional

from World import (
    World,
    EMPTY, WALL, SHELF, DROP, CHARGING
)


class WarehouseRenderer:
    """
    Pygame-based renderer for warehouse environment.

    Features:
    - Charging station visualization
    - Color-coded robots
    - Battery bar
    - Metrics overlay
    - Multi-robot legend
    """

    # ---------------- COLORS ---------------- #
    COLORS = {
        'background': (240, 240, 240),
        'grid': (200, 200, 200),
        'wall': (60, 60, 60),
        'shelf': (139, 69, 19),
        'drop': (34, 139, 34),
        'charging': (30, 144, 255),   # BLUE for charging stations
        'robot_empty': (70, 130, 180),
        'robot_carrying': (255, 140, 0),
        'battery_bg': (220, 220, 220),
        'battery_fg': (50, 205, 50),
        'text': (0, 0, 0),
        'text_bg': (255, 255, 255, 220),
    }

    ROBOT_COLORS = [
        (70, 130, 180), (220, 20, 60), (255, 215, 0),
        (147, 112, 219), (0, 191, 255), (255, 105, 180),
        (50, 205, 50), (255, 69, 0), (138, 43, 226),
        (0, 206, 209),
    ]

    def __init__(
            self,
            world: 'World',
            cell_size: int = 50,
            fps: int = 10,
            show_grid: bool = True,
            show_metrics: bool = True
    ):

        pygame.init()
        pygame.font.init()

        self.world = world
        self.cell_size = cell_size
        self.fps = fps
        self.show_grid = show_grid
        self.show_metrics = show_metrics

        self.width = world.width * cell_size
        self.height = world.height * cell_size
        self.metrics_height = 150 if show_metrics else 0

        self.screen = pygame.display.set_mode(
            (self.width, self.height + self.metrics_height)
        )
        pygame.display.set_caption("Scalable Multi-Agent Warehouse")

        self.font = pygame.font.SysFont('Arial', 16)
        self.font_small = pygame.font.SysFont('Arial', 12)
        self.font_large = pygame.font.SysFont('Arial', 24, bold=True)

        self.clock = pygame.time.Clock()

        # Assign unique colors
        self.robot_colors = {}
        for i, robot_id in enumerate(world.robots.keys()):
            self.robot_colors[robot_id] = self.ROBOT_COLORS[i % len(self.ROBOT_COLORS)]

    # -----------------------------------------------------

    def render(self, pause_ms: Optional[int] = None) -> bool:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key == pygame.K_s:
                    self.save_screenshot()

        self.screen.fill(self.COLORS['background'])

        self._draw_grid()
        self._draw_static_entities()
        self._draw_robots()

        if self.show_metrics:
            self._draw_metrics()

        pygame.display.flip()

        if pause_ms:
            pygame.time.wait(pause_ms)
        else:
            self.clock.tick(self.fps)

        return True

    # -----------------------------------------------------

    def _draw_grid(self):

        if not self.show_grid:
            return

        for x in range(0, self.width + 1, self.cell_size):
            pygame.draw.line(
                self.screen, self.COLORS['grid'],
                (x, 0), (x, self.height), 1
            )

        for y in range(0, self.height + 1, self.cell_size):
            pygame.draw.line(
                self.screen, self.COLORS['grid'],
                (0, y), (self.width, y), 1
            )

    # -----------------------------------------------------

    def _draw_static_entities(self):

        for x in range(self.world.height):
            for y in range(self.world.width):

                cell_type = self.world.grid[x, y]

                rect = pygame.Rect(
                    y * self.cell_size,
                    x * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )

                if cell_type == WALL:
                    pygame.draw.rect(self.screen, self.COLORS['wall'], rect)

                elif cell_type == SHELF:
                    pygame.draw.rect(self.screen, self.COLORS['shelf'], rect)
                    self._draw_center_text(rect, "S")

                elif cell_type == DROP:
                    pygame.draw.rect(self.screen, self.COLORS['drop'], rect)
                    self._draw_center_text(rect, "D")

                elif cell_type == CHARGING:
                    pygame.draw.rect(self.screen, self.COLORS['charging'], rect)
                    self._draw_center_text(rect, "C")  # Changed from emoji to "C"

    # -----------------------------------------------------

    def _draw_robots(self):

        for robot_id, (x, y) in self.world.robots.items():

            base_color = self.robot_colors[robot_id]
            color = (
                self.COLORS['robot_carrying']
                if self.world.carrying[robot_id]
                else base_color
            )

            rect = pygame.Rect(
                y * self.cell_size + 5,
                x * self.cell_size + 5,
                self.cell_size - 10,
                self.cell_size - 10
            )

            center = rect.center
            radius = (self.cell_size - 10) // 2

            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(
                self.screen,
                tuple(max(0, c - 50) for c in color),
                center, radius, 2
            )

            robot_num = robot_id.split("_")[1]
            text = self.font_small.render(robot_num, True, (255, 255, 255))
            self.screen.blit(text, text.get_rect(center=center))

            # Carrying box
            if self.world.carrying[robot_id]:
                box = pygame.Rect(center[0]-5, rect.top-8, 10, 8)
                pygame.draw.rect(self.screen, (139, 69, 19), box)
                pygame.draw.rect(self.screen, (0, 0, 0), box, 1)

            # Battery bar
            self._draw_battery_bar(robot_id, rect)

    # -----------------------------------------------------

    def _draw_battery_bar(self, robot_id, rect):

        battery = self.world.battery[robot_id] / 100.0

        bar_w = rect.width
        bar_h = 5

        bg = pygame.Rect(rect.left, rect.bottom + 2, bar_w, bar_h)
        fg = pygame.Rect(rect.left, rect.bottom + 2,
                         int(bar_w * battery), bar_h)

        pygame.draw.rect(self.screen, self.COLORS['battery_bg'], bg)
        pygame.draw.rect(self.screen, self.COLORS['battery_fg'], fg)

    # -----------------------------------------------------

    def _draw_metrics(self):

        surf = pygame.Surface((self.width, self.metrics_height))
        surf.fill(self.COLORS['text_bg'][:3])
        surf.set_alpha(self.COLORS['text_bg'][3])
        self.screen.blit(surf, (0, self.height))

        m = self.world.get_metrics()

        lines = [
            f"Step: {m['step_count']}",
            f"Deliveries: {m['total_deliveries']}",
            f"Urgent: {m['urgent_deliveries']}",
            f"Collisions: {m['collision_count']}",
            f"Efficiency: {m['deliveries_per_step']:.3f}",
            f"Avg Battery: {m['avg_battery']:.1f}%",
        ]

        y = self.height + 10
        for line in lines:
            txt = self.font.render(line, True, self.COLORS['text'])
            self.screen.blit(txt, (10, y))
            y += 22

        # -------- LEGEND -------- #
        lx = self.width - 220
        ly = self.height + 10

        title = self.font.render("Robots:", True, self.COLORS['text'])
        self.screen.blit(title, (lx, ly))
        ly += 25

        for rid, col in self.robot_colors.items():
            num = rid.split("_")[1]
            carrying = self.world.carrying[rid]

            pygame.draw.circle(
                self.screen,
                self.COLORS['robot_carrying'] if carrying else col,
                (lx + 10, ly + 8),
                8
            )

            label = self.font_small.render(
                f"Robot {num}" + (" (load)" if carrying else ""),
                True, self.COLORS['text']
            )
            self.screen.blit(label, (lx + 25, ly))
            ly += 18

    # -----------------------------------------------------

    def _draw_center_text(self, rect, text):

        txt = self.font.render(text, True, (255, 255, 255))
        self.screen.blit(txt, txt.get_rect(center=rect.center))

    # -----------------------------------------------------

    def save_screenshot(self, filename: Optional[str] = None):

        if filename is None:
            import time
            filename = f"warehouse_{int(time.time())}.png"

        pygame.image.save(self.screen, filename)
        print(f"Screenshot saved -> {filename}")

    # -----------------------------------------------------

    def close(self):
        pygame.quit()