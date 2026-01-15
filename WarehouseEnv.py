import numpy as np
from typing import Dict, Optional
from pettingzoo import ParallelEnv
from gymnasium import spaces

from World import World


class WarehouseEnv(ParallelEnv):

    metadata = {
        "name": "warehouse_v1",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }

    def __init__(
        self,
        height: int = 17,
        width: int = 17,
        n_robots: int = 10,
        max_steps: int = 800,
        view_size: int = 5,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.height = height
        self.width = width
        self.n_robots = n_robots
        self.max_steps = max_steps
        self.view_size = view_size
        self.render_mode = render_mode

        # -------- WORLD -------- #
        self.world = World(height, width, n_robots)

        # Agents
        self.possible_agents = [f"robot_{i}" for i in range(n_robots)]
        self.agents = self.possible_agents[:]

        # -------- ACTION SPACE -------- #
        # 0-7 : STAY, UP, DOWN, LEFT, RIGHT, PICK, DROP, CHARGE
        self._action_spaces = {
            agent: spaces.Discrete(8) for agent in self.possible_agents
        }

        # -------- OBS SPACE -------- #
        # 7 channels (as per new world)
        self._observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=1,
                shape=(7, view_size, view_size),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        self.current_step = 0

        # -------- RENDERER -------- #
        self.renderer = None
        if render_mode in ["human", "rgb_array"]:
            try:
                from renderer import WarehouseRenderer
                self.renderer = WarehouseRenderer(
                    self.world,
                    cell_size=50,
                    fps=10,
                    show_grid=True,
                    show_metrics=True
                )
            except Exception as e:
                print("⚠ pygame not available, ASCII fallback")
                print(e)

    # -------------------------------------------------

    @property
    def observation_space(self):
        return self._observation_spaces[self.agents[0]]

    @property
    def action_space(self):
        return self._action_spaces[self.agents[0]]

    def observation_spaces(self, agent):
        return self._observation_spaces[agent]

    def action_spaces(self, agent):
        return self._action_spaces[agent]

    # -------------------------------------------------

    def reset(self, seed=None, options=None):

        self.world.reset()
        self.agents = self.possible_agents[:]
        self.current_step = 0

        observations = {
            agent: self.world.get_local_observation(agent, self.view_size)
            for agent in self.agents
        }

        infos = {agent: {} for agent in self.agents}

        return observations, infos

    # -------------------------------------------------

    def step(self, actions: Dict[str, int]):

        self.current_step += 1

        rewards, infos = self.world.step(actions)

        observations = {
            agent: self.world.get_local_observation(agent, self.view_size)
            for agent in self.agents
        }

        terminations = {agent: False for agent in self.agents}

        truncated = self.current_step >= self.max_steps
        truncations = {agent: truncated for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    # -------------------------------------------------

    def render(self):

        if self.render_mode == "human":

            if self.renderer:
                self.renderer.render()
            else:
                print(self.world.render_ascii())

        elif self.render_mode == "rgb_array":

            if self.renderer:
                import pygame
                self.renderer.render()
                frame = pygame.surfarray.array3d(self.renderer.screen)
                return np.transpose(frame, (1, 0, 2))

        return None

    # -------------------------------------------------

    def close(self):

        if self.renderer:
            self.renderer.close()
            self.renderer = None

    # -------------------------------------------------

    def get_global_state(self):
        return self.world.get_global_state()
