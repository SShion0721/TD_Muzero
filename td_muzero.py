import numpy as np
from engine import TDEngine, TowerType

class MuZeroConfig:
    def __init__(self):
        # Basics
        self.seed = 0
        self.max_num_gpus = None
        
        # Game
        self.observation_shape = (5, 11, 11)  # Channels: [Obstacles, Start/End, Enemies, Base HP, Money]
        self.action_space = list(range(11 * 11 * 6 + 1))  # GRID_W * GRID_H * 6 (4 build, 1 upgrade, 1 sell) + 1 Wait
        self.players = list(range(1))  # Single player
        self.stacked_observations = 0
        
        # Evaluate
        self.muzero_player = 0
        self.opponent = None
        self.num_actions = len(self.action_space)

        # Optimization
        self.batch_size = 64
        self.value_loss_weight = 1.0
        self.training_steps = 100000
        
        # MCTS
        self.num_simulations = 50
        self.discount = 0.99
        self.temperature_threshold = None
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

    def visit_softmax_temperature_fn(self, trained_steps):
        return 1.0

class Game:
    """
    Game wrapper for muzero-general
    """
    def __init__(self, seed=None):
        self.env = TDEngine(width=11, height=11)

    def reset(self):
        self.env.reset()
        return self.get_observation()

    def step(self, action):
        reward = 0
        w = self.env.width
        h = self.env.height
        
        # Action space mapping:
        # 0 to w*h*4 - 1: Build tower (types 0, 1, 2, 3)
        # w*h*4 to w*h*5 - 1: Upgrade tower
        # w*h*5 to w*h*6 - 1: Sell tower
        # w*h*6: Wait
        
        if action < w * h * 4:
            tower_type = action // (w * h)
            rem = action % (w * h)
            y = rem // w
            x = rem % w
            
            if self.env.can_place_tower(x, y, tower_type):
                self.env.place_tower(x, y, tower_type)
            else:
                reward -= 5
        elif action < w * h * 5:
            rem = action - (w * h * 4)
            y = rem // w
            x = rem % w
            if not self.env.upgrade_tower(x, y):
                reward -= 5
        elif action < w * h * 6:
            rem = action - (w * h * 5)
            y = rem // w
            x = rem % w
            if not self.env.sell_tower(x, y):
                reward -= 5
                
        # Advance the simulation by 1 second for each "action step" of MuZero
        step_reward, done = self.env.step(dt=1.0)
        reward += step_reward
        
        return self.get_observation(), reward, done

    def legal_actions(self):
        w = self.env.width
        h = self.env.height
        legal = []
        
        # Build actions
        for tower_type in [TowerType.BASIC, TowerType.SNIPER, TowerType.AOE, TowerType.SLOW]:
            for y in range(h):
                for x in range(w):
                    if self.env.can_place_tower(x, y, tower_type):
                        action = tower_type * (w * h) + y * w + x
                        legal.append(action)
                        
        # Upgrade & Sell actions
        for tower in self.env.towers:
            x, y = tower.x, tower.y
            
            # Upgrade
            if self.env.money >= tower.upgrade_cost:
                legal.append((4 * w * h) + y * w + x)
                
            # Sell
            legal.append((5 * w * h) + y * w + x)
        
        # Wait action is always legal
        legal.append(w * h * 6)
        return legal

    def get_observation(self):
        state = self.env.get_state()
        w, h = self.env.width, self.env.height
        obs = np.zeros((5, h, w), dtype=np.float32)
        
        # Channel 0: Towers (Obstacles)
        for row in range(h):
            for col in range(w):
                if state['grid'][row][col] == 1:
                    obs[0][row][col] = 1.0
                    
        # Channel 1: Start and End map
        obs[1][self.env.spawn_y][self.env.spawn_x] = 1.0
        obs[1][self.env.base_y][self.env.base_x] = 1.0
        
        # Channel 2: Enemies HP map
        for enemy in state['enemies']:
            grid_x, grid_y = int(enemy['x']), int(enemy['y'])
            if 0 <= grid_x < w and 0 <= grid_y < h:
                # Add HP to cell (in case multiple enemies are in the same cell, bound by 1)
                obs[2][grid_y][grid_x] = min(1.0, obs[2][grid_y][grid_x] + (enemy['hp'] / max(1, enemy['max_hp'])))
                
        # Channel 3: Base HP (smeared to entire channel)
        obs[3].fill(state['base_hp'] / 100.0)
        
        # Channel 4: Money (normalized arbitrarily, e.g. 1000 is 1.0)
        obs[4].fill(min(state['money'] / 1000.0, 1.0))
        
        return obs

    def render(self):
        # We have a separate gui.py for human playing/rendering.
        # But can print basic state here if needed.
        pass

    def action_to_string(self, action_number):
        w = self.env.width
        h = self.env.height
        if action_number == w * h * 6:
            return "Wait 1 step"
            
        action_type = action_number // (w * h)
        rem = action_number % (w * h)
        y = rem // w
        x = rem % w
        
        if action_type < 4:
            t_str = {0: "BASIC", 1: "SNIPER", 2: "AOE", 3: "SLOW"}.get(action_type, "UNKNOWN")
            return f"Build {t_str} at ({x}, {y})"
        elif action_type == 4:
            return f"Upgrade Tower at ({x}, {y})"
        else:
            return f"Sell Tower at ({x}, {y})"
