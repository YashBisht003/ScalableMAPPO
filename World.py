import numpy as np
from typing import Dict, Tuple, List, Optional

# ===============================
# Cell types
# ===============================
EMPTY = 0
WALL = 1
SHELF = 2
DROP = 3
CHARGING = 4  # NEW: Charging stations

# ===============================
# Actions
# ===============================
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
PICK = 5
DROP_ACT = 6
CHARGE = 7  # NEW: Charge action

# ===============================
# Priority Levels for Tasks
# ===============================
PRIORITY_LOW = 1
PRIORITY_NORMAL = 2
PRIORITY_HIGH = 3
PRIORITY_URGENT = 4


class World:
    """
    Enhanced 17x17 warehouse with 10 robots and advanced features:
    - Battery management system
    - Task priority queue
    - Shelf capacity tracking
    - Communication between agents
    - Performance analytics
    """

    def __init__(self, height=17, width=17, n_robots=10):

        self.height = height
        self.width = width
        self.n_robots = n_robots

        self.grid = np.zeros((height, width), dtype=int)

        # Robot state
        self.robots = {}
        self.carrying = {}
        self.battery = {}  # NEW: Battery levels (0-100)
        self.robot_tasks = {}  # NEW: Assigned tasks per robot

        # Infrastructure
        self.drop_points = []
        self.shelves = []
        self.charging_stations = []  # NEW
        self.shelf_inventory = {}
        self.shelf_capacity = {}  # NEW: Max capacity per shelf

        # Task management
        self.task_queue = []  # NEW: Priority queue of tasks
        self.completed_tasks = []  # NEW: Task history

        # Metrics
        self.total_deliveries = 0
        self.collision_count = 0
        self.step_count = 0
        self.total_battery_consumed = 0  # NEW
        self.urgent_deliveries = 0  # NEW
        self.overtime_tasks = 0  # NEW: Tasks that took too long

        # Communication (NEW)
        self.robot_messages = {}  # Robots can share info

        self._build_fixed_world()

    # =================================================
    # ENHANCED 17x17 WAREHOUSE LAYOUT
    # =================================================

    def _build_fixed_world(self):
        """
        Professional warehouse design:
        - 3 main zones: Storage (shelves), Processing (drops), Charging
        - Wide corridors for robot movement
        - Optimized shelf placement
        - Strategic charging station locations
        """
        self._add_walls()
        self._add_internal_structure()
        self._place_shelves()
        self._place_drops()
        self._place_charging_stations()
        self._spawn_robots()
        self._initialize_tasks()

    def _add_walls(self):
        """Perimeter walls"""
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

    def _add_internal_structure(self):
        """
        Create optimized corridor system with internal walls
        Main corridors allow efficient robot navigation
        """
        # Add internal walls to create zones
        # Horizontal dividers
        for c in range(2, 7):
            if c != 4:  # Leave passage at column 4
                self.grid[3, c] = WALL
                self.grid[9, c] = WALL
                self.grid[13, c] = WALL

        for c in range(10, 15):
            if c != 12:  # Leave passage at column 12
                self.grid[3, c] = WALL
                self.grid[9, c] = WALL
                self.grid[13, c] = WALL

        # Vertical dividers
        for r in range(6, 11):
            if r != 8:  # Leave passage at row 8
                self.grid[r, 4] = WALL
                self.grid[r, 12] = WALL

    def _place_shelves(self):
        """
        Strategic shelf placement in storage zones
        16 shelves with varying capacities
        """
        # CORRECTED: Ensure shelves don't overlap with walls
        shelf_positions = [
            # Top-left zone
            (2, 2), (2, 3), (2, 5), (2, 6),
            (4, 2), (4, 3), (4, 5), (4, 6),

            # Top-right zone
            (2, 10), (2, 11), (2, 13), (2, 14),
            (4, 10), (4, 11), (4, 13), (4, 14),

            # Middle zones (below row 6, avoiding walls)
            (10, 2), (10, 3), (10, 5), (10, 6),
            (10, 10), (10, 11), (10, 13), (10, 14),
        ]

        self.shelves = shelf_positions[:16]  # Take 16 shelves

        for i, s in enumerate(self.shelves):
            # Check if position is valid (not a wall)
            if self.grid[s] != WALL:
                self.grid[s] = SHELF
                # Varying inventory and capacity
                base_inventory = 40
                variance = np.random.randint(-10, 20)
                self.shelf_inventory[s] = max(10, base_inventory + variance)
                self.shelf_capacity[s] = 60  # Max capacity
            else:
                print(f"Warning: Shelf position {s} conflicts with wall")

    def _place_drops(self):
        """
        4 drop-off points strategically placed in open areas
        """
        self.drop_points = [
            (15, 3),  # Bottom-left
            (15, 8),  # Bottom-center
            (15, 13),  # Bottom-right
            (12, 8),  # Middle-center
        ]

        for d in self.drop_points:
            if self.grid[d] == EMPTY:  # Only place if empty
                self.grid[d] = DROP

    def _place_charging_stations(self):
        """
        3 charging stations for battery management
        """
        self.charging_stations = [
            (1, 8),  # Top-center
            (8, 1),  # Middle-left
            (8, 15),  # Middle-right
        ]

        for c in self.charging_stations:
            if self.grid[c] == EMPTY:  # Only place if empty
                self.grid[c] = CHARGING

    def _spawn_robots(self):
        """
        Spawn 10 robots in strategic starting positions in open areas
        """
        start_positions = [
            (6, 8), (6, 9),  # Center top
            (11, 7), (11, 9),  # Center bottom (corrected from 10 to avoid walls)
            (8, 6), (8, 10),  # Center sides
            (12, 6), (12, 10),  # Lower area
            (14, 7), (14, 9),  # Bottom area
        ]

        for i in range(self.n_robots):
            rid = f"robot_{i}"
            pos = start_positions[i]

            # Ensure spawn position is valid
            if self.grid[pos] == EMPTY or self.grid[pos] == CHARGING:
                self.robots[rid] = pos
                self.carrying[rid] = False
                self.battery[rid] = 100.0  # Start with full battery
                self.robot_tasks[rid] = None
                self.robot_messages[rid] = []
            else:
                # Fallback: find nearest empty cell
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        alt_pos = (pos[0] + dx, pos[1] + dy)
                        if (0 < alt_pos[0] < self.height - 1 and
                                0 < alt_pos[1] < self.width - 1 and
                                self.grid[alt_pos] == EMPTY):
                            self.robots[rid] = alt_pos
                            self.carrying[rid] = False
                            self.battery[rid] = 100.0
                            self.robot_tasks[rid] = None
                            self.robot_messages[rid] = []
                            break
                    else:
                        continue
                    break

    def _initialize_tasks(self):
        """
        Create initial task queue with priorities
        """
        # Start with some initial tasks
        for _ in range(5):
            self._generate_task()

    def _generate_task(self):
        """
        Generate a new delivery task with priority
        """
        # Randomly select a shelf with inventory and drop point
        available_shelves = [s for s in self.shelves if self.shelf_inventory.get(s, 0) > 0]

        if not available_shelves:
            return  # No shelves with inventory

        shelf = available_shelves[np.random.randint(0, len(available_shelves))]
        drop = self.drop_points[np.random.randint(0, len(self.drop_points))]

        # Assign random priority
        priority = np.random.choice(
            [PRIORITY_LOW, PRIORITY_NORMAL, PRIORITY_HIGH, PRIORITY_URGENT],
            p=[0.2, 0.5, 0.2, 0.1]
        )

        task = {
            'id': len(self.completed_tasks) + len(self.task_queue),
            'shelf': shelf,
            'drop': drop,
            'priority': priority,
            'created_step': self.step_count,
            'deadline': self.step_count + (50 if priority == PRIORITY_URGENT else 100)
        }

        self.task_queue.append(task)
        # Sort by priority (higher priority first)
        self.task_queue.sort(key=lambda t: -t['priority'])

    # =================================================
    # ENHANCED STEP FUNCTION
    # =================================================

    def step(self, actions: Dict[str, int]):
        """
        Enhanced step with battery management and task tracking
        """
        self.step_count += 1

        # Generate new tasks periodically
        if self.step_count % 20 == 0 and len(self.task_queue) < 10:
            self._generate_task()

        proposed = {}
        battery_consumed = {}

        # Phase 1: Propose movements and consume battery
        for rid, act in actions.items():
            x, y = self.robots[rid]

            # Battery consumption
            battery_cost = self._get_battery_cost(act, self.carrying[rid])
            self.battery[rid] = max(0, self.battery[rid] - battery_cost)
            battery_consumed[rid] = battery_cost
            self.total_battery_consumed += battery_cost

            # Movement (only if battery > 5%)
            if self.battery[rid] > 5:
                if act == UP:
                    nx, ny = x - 1, y
                elif act == DOWN:
                    nx, ny = x + 1, y
                elif act == LEFT:
                    nx, ny = x, y - 1
                elif act == RIGHT:
                    nx, ny = x, y + 1
                else:
                    nx, ny = x, y

                if self._valid(nx, ny):
                    proposed[rid] = (nx, ny)
                else:
                    proposed[rid] = (x, y)
            else:
                # Low battery - can't move
                proposed[rid] = (x, y)

        # Phase 2: Collision resolution
        final = {}
        collisions = set()

        for rid, pos in proposed.items():
            if list(proposed.values()).count(pos) > 1:
                collisions.add(rid)
                final[rid] = self.robots[rid]
                self.collision_count += 1
            else:
                final[rid] = pos

        self.robots = final

        # Phase 3: Process actions and calculate rewards
        rewards = {}
        info = {}

        for rid in self.robots:
            r = -0.01  # Small living cost
            x, y = self.robots[rid]

            picked = False
            dropped = False
            charged = False

            # PICK action
            if actions[rid] == PICK:
                shelf = self._adjacent_shelf((x, y))

                if shelf and not self.carrying[rid] and self.shelf_inventory[shelf] > 0:
                    self.carrying[rid] = True
                    self.shelf_inventory[shelf] -= 1
                    r += 2.0
                    picked = True
                else:
                    r -= 0.2

            # DROP action
            if actions[rid] == DROP_ACT:
                if (x, y) in self.drop_points and self.carrying[rid]:
                    self.carrying[rid] = False
                    self.total_deliveries += 1

                    # Bonus for task completion
                    task_bonus = self._check_task_completion(rid, (x, y))
                    r += 15.0 + task_bonus

                    dropped = True
                else:
                    r -= 0.2

            # CHARGE action (NEW)
            if actions[rid] == CHARGE:
                if (x, y) in self.charging_stations:
                    charge_amount = min(20.0, 100.0 - self.battery[rid])
                    self.battery[rid] += charge_amount
                    r += 0.5  # Small reward for charging
                    charged = True
                else:
                    r -= 0.2

            # Penalties
            if rid in collisions:
                r -= 1.5

            # Low battery penalty
            if self.battery[rid] < 20:
                r -= 0.5

            # Efficiency bonus (carrying while moving)
            if self.carrying[rid] and actions[rid] in [UP, DOWN, LEFT, RIGHT]:
                r += 0.05

            rewards[rid] = r
            info[rid] = {
                "picked": picked,
                "dropped": dropped,
                "charged": charged,
                "collision": rid in collisions,
                "battery": self.battery[rid],
                "battery_consumed": battery_consumed.get(rid, 0)
            }

        return rewards, info

    def _get_battery_cost(self, action: int, carrying: bool) -> float:
        """Calculate battery consumption for action"""
        base_costs = {
            STAY: 0.05,
            UP: 0.3, DOWN: 0.3, LEFT: 0.3, RIGHT: 0.3,
            PICK: 0.5,
            DROP_ACT: 0.5,
            CHARGE: 0.0
        }

        cost = base_costs.get(action, 0.1)

        # Carrying increases cost
        if carrying and action in [UP, DOWN, LEFT, RIGHT]:
            cost *= 1.5

        return cost

    def _check_task_completion(self, rid: str, drop_pos: Tuple) -> float:
        """Check if robot completed a task and give bonus"""
        bonus = 0.0

        # Check if any task matches this drop
        for task in self.task_queue[:]:
            if task['drop'] == drop_pos:
                # Check priority bonus
                if task['priority'] == PRIORITY_URGENT:
                    bonus = 10.0
                    self.urgent_deliveries += 1
                elif task['priority'] == PRIORITY_HIGH:
                    bonus = 5.0
                elif task['priority'] == PRIORITY_NORMAL:
                    bonus = 2.0
                else:
                    bonus = 1.0

                # Deadline check
                if self.step_count > task['deadline']:
                    bonus *= 0.5  # Reduced reward for late
                    self.overtime_tasks += 1

                self.completed_tasks.append(task)
                self.task_queue.remove(task)
                break

        return bonus

    # =================================================
    # ENHANCED OBSERVATION
    # =================================================

    def get_local_observation(self, rid: str, view_size=5):
        """
        Enhanced observation with 7 channels:
        0: Walls/Shelves
        1: Drop points
        2: Other robots
        3: Self position
        4: Carrying status
        5: Battery level (normalized)
        6: Charging stations
        """
        x, y = self.robots[rid]
        half = view_size // 2

        obs = np.zeros((7, view_size, view_size), dtype=np.float32)

        for i in range(view_size):
            for j in range(view_size):
                wx = x - half + i
                wy = y - half + j

                if wx < 0 or wx >= self.height or wy < 0 or wy >= self.width:
                    obs[0, i, j] = 1
                    continue

                cell = self.grid[wx, wy]

                if cell in [WALL, SHELF]:
                    obs[0, i, j] = 1
                if cell == DROP:
                    obs[1, i, j] = 1
                if cell == CHARGING:
                    obs[6, i, j] = 1

                # Other robots
                for oid, (ox, oy) in self.robots.items():
                    if oid != rid and (ox, oy) == (wx, wy):
                        obs[2, i, j] = 1

        # Self position
        obs[3, half, half] = 1

        # Carrying status
        if self.carrying[rid]:
            obs[4, :, :] = 1

        # Battery level (normalized)
        obs[5, :, :] = self.battery[rid] / 100.0

        return obs

    def get_global_state(self):
        """Enhanced global state with battery and task info"""
        state = []

        # Robot states (10 robots * 4 features = 40)
        for i in range(self.n_robots):
            rid = f"robot_{i}"
            x, y = self.robots[rid]
            state += [
                x / self.height,
                y / self.width,
                float(self.carrying[rid]),
                self.battery[rid] / 100.0
            ]

        # Shelf states (16 shelves * 3 features = 48)
        for s in self.shelves:
            state += [
                s[0] / self.height,
                s[1] / self.width,
                self.shelf_inventory.get(s, 0) / self.shelf_capacity.get(s, 60)
            ]

        # Drop points (4 drops * 2 features = 8)
        for d in self.drop_points:
            state += [d[0] / self.height, d[1] / self.width]

        # Charging stations (3 stations * 2 features = 6)
        for c in self.charging_stations:
            state += [c[0] / self.height, c[1] / self.width]

        # Task queue info (top 3 tasks * 5 features = 15)
        for i in range(3):
            if i < len(self.task_queue):
                task = self.task_queue[i]
                state += [
                    task['shelf'][0] / self.height,
                    task['shelf'][1] / self.width,
                    task['drop'][0] / self.height,
                    task['drop'][1] / self.width,
                    task['priority'] / 4.0
                ]
            else:
                state += [0, 0, 0, 0, 0]

        # Total: 40 + 48 + 8 + 6 + 15 = 117 features
        return np.array(state, dtype=np.float32)

    # =================================================
    # HELPER METHODS
    # =================================================

    def _adjacent_shelf(self, pos):
        x, y = pos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            p = (x + dx, y + dy)
            if p in self.shelves and self.shelf_inventory.get(p, 0) > 0:
                return p
        return None

    def _valid(self, x, y):
        if x < 0 or x >= self.height or y < 0 or y >= self.width:
            return False
        return self.grid[x, y] not in [WALL, SHELF]

    def reset(self):
        self.__init__(self.height, self.width, self.n_robots)

    def get_metrics(self):
        """Enhanced metrics"""
        avg_battery = np.mean([self.battery[f"robot_{i}"] for i in range(self.n_robots)])

        return {
            "total_deliveries": self.total_deliveries,
            "urgent_deliveries": self.urgent_deliveries,
            "collision_count": self.collision_count,
            "step_count": self.step_count,
            "deliveries_per_step": self.total_deliveries / max(1, self.step_count),
            "avg_battery": avg_battery,
            "total_battery_consumed": self.total_battery_consumed,
            "tasks_pending": len(self.task_queue),
            "tasks_completed": len(self.completed_tasks),
            "overtime_tasks": self.overtime_tasks
        }

    def render_ascii(self):
        """ASCII rendering"""
        disp = np.copy(self.grid).astype(object)
        disp[disp == 0] = '.'
        disp[disp == 1] = '#'
        disp[disp == 2] = 'S'
        disp[disp == 3] = 'D'
        disp[disp == 4] = 'C'

        for i, (rid, (x, y)) in enumerate(self.robots.items()):
            disp[x, y] = str(i)

        return "\n".join("".join(str(c) for c in r) for r in disp)