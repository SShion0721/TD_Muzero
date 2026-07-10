"""
Watch a trained MuZero model play Tower Defense with Pygame visualization.

Usage:
    python play_ai.py <checkpoint_dir>

Example:
    python play_ai.py E:\\Desktop\\TD_MuZero\\muzero-general\\results\\td_muzero\\2024-02-15--06-09-25
"""

import sys
import pathlib
import math
import time

import numpy as np
import torch
import pygame

# --- Add muzero-general to path so we can import its modules ---
MUZERO_DIR = pathlib.Path(__file__).resolve().parent / "muzero-general"
sys.path.insert(0, str(MUZERO_DIR))

import models
from games.engine import TDEngine, TowerType

# ──────────────────────────  Minimal MCTS (copied from self_play.py)  ──────

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        return 0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def expand(self, actions, reward, policy_logits, hidden_state):
        self.reward = reward
        self.hidden_state = hidden_state
        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        for i, a in enumerate(actions):
            self.children[a] = Node(policy_values[i])


class MinMaxStats:
    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def run_mcts(model, observation, legal_actions, config):
    """Run MCTS and return the root node."""
    root = Node(0)
    obs_tensor = (
        torch.tensor(observation).float().unsqueeze(0).to(next(model.parameters()).device)
    )
    value, reward, policy_logits, hidden_state = model.initial_inference(obs_tensor)
    root.expand(legal_actions, 
                models.support_to_scalar(reward, config.support_size).item(),
                policy_logits, hidden_state)

    min_max_stats = MinMaxStats()
    for _ in range(config.num_simulations):
        node = root
        search_path = [node]
        while node.expanded():
            # UCB selection
            max_ucb = -float("inf")
            best_action = None
            for action, child in node.children.items():
                pb_c = (
                    math.log((node.visit_count + config.pb_c_base + 1) / config.pb_c_base)
                    + config.pb_c_init
                )
                pb_c *= math.sqrt(node.visit_count) / (child.visit_count + 1)
                prior_score = pb_c * child.prior
                if child.visit_count > 0:
                    value_score = min_max_stats.normalize(
                        child.reward + config.discount * child.value()
                    )
                else:
                    value_score = 0
                ucb = prior_score + value_score
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_action = action
            node = node.children[best_action]
            search_path.append(node)

        parent = search_path[-2]
        value, reward, policy_logits, hidden_state = model.recurrent_inference(
            parent.hidden_state,
            torch.tensor([[best_action]]).to(parent.hidden_state.device),
        )
        value = models.support_to_scalar(value, config.support_size).item()
        reward = models.support_to_scalar(reward, config.support_size).item()
        node.expand(config.action_space, reward, policy_logits, hidden_state)

        # Backpropagate
        for n in reversed(search_path):
            n.value_sum += value
            n.visit_count += 1
            min_max_stats.update(n.reward + config.discount * n.value())
            value = n.reward + config.discount * value

    return root


def select_action(root):
    """Select the action with highest visit count (greedy)."""
    visit_counts = np.array(
        [child.visit_count for child in root.children.values()], dtype="int32"
    )
    actions = list(root.children.keys())
    return actions[np.argmax(visit_counts)]


# ──────────────────────────  Game Wrapper  ──────────────────────────────────

class GameWrapper:
    """Same logic as games/td_muzero.py Game class but without relative imports."""
    def __init__(self):
        self.env = TDEngine(width=11, height=11)

    def reset(self):
        self.env.reset()
        return self.get_observation()

    def step(self, action):
        reward = 0
        w, h = self.env.width, self.env.height
        if action < w * h * 4:
            tower_type = action // (w * h)
            rem = action % (w * h)
            y, x = rem // w, rem % w
            if self.env.can_place_tower(x, y, tower_type):
                self.env.place_tower(x, y, tower_type)
        elif action < w * h * 5:
            rem = action - (w * h * 4)
            y, x = rem // w, rem % w
            self.env.upgrade_tower(x, y)
        elif action < w * h * 6:
            rem = action - (w * h * 5)
            y, x = rem // w, rem % w
            self.env.sell_tower(x, y)

        step_reward, done = self.env.step(dt=1.0)
        reward += step_reward
        return self.get_observation(), reward, done

    def legal_actions(self):
        w, h = self.env.width, self.env.height
        legal = []
        for tower_type in [TowerType.BASIC, TowerType.SNIPER, TowerType.AOE, TowerType.SLOW]:
            for y in range(h):
                for x in range(w):
                    if self.env.can_place_tower(x, y, tower_type):
                        legal.append(tower_type * (w * h) + y * w + x)
        for tower in self.env.towers:
            x, y = tower.x, tower.y
            if self.env.money >= tower.upgrade_cost:
                legal.append((4 * w * h) + y * w + x)
            legal.append((5 * w * h) + y * w + x)
        legal.append(w * h * 6)  # Wait
        return legal

    def get_observation(self):
        state = self.env.get_state()
        w, h = self.env.width, self.env.height
        obs = np.zeros((5, h, w), dtype=np.float32)
        for row in range(h):
            for col in range(w):
                if state['grid'][row][col] == 1:
                    obs[0][row][col] = 1.0
        obs[1][self.env.spawn_y][self.env.spawn_x] = 1.0
        obs[1][self.env.base_y][self.env.base_x] = 1.0
        for enemy in state['enemies']:
            gx, gy = int(enemy['x']), int(enemy['y'])
            if 0 <= gx < w and 0 <= gy < h:
                obs[2][gy][gx] = min(1.0, obs[2][gy][gx] + enemy['hp'] / max(1, enemy['max_hp']))
        obs[3].fill(state['base_hp'] / 100.0)
        obs[4].fill(min(state['money'] / 1000.0, 1.0))
        return obs

    def action_to_string(self, action):
        w, h = self.env.width, self.env.height
        if action == w * h * 6:
            return "Wait"
        t = action // (w * h)
        rem = action % (w * h)
        y, x = rem // w, rem % w
        if t < 4:
            t_str = {0: "BASIC", 1: "SNIPER", 2: "AOE", 3: "SLOW"}.get(t, "?")
            return f"Build {t_str} at ({x},{y})"
        elif t == 4:
            return f"Upgrade ({x},{y})"
        else:
            return f"Sell ({x},{y})"


# ──────────────────────────  Pygame Rendering  ─────────────────────────────

CELL_SIZE = 50
GRID_W, GRID_H = 11, 11
WIDTH = GRID_W * CELL_SIZE
HEIGHT = GRID_H * CELL_SIZE + 120
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (200, 200, 0)
GRAY = (100, 100, 100)


def draw_frame(screen, font, engine, action_str, step, total_reward, root_value):
    screen.fill(BLACK)

    # Grid
    for y in range(GRID_H):
        for x in range(GRID_W):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if (x, y) == (engine.spawn_x, engine.spawn_y):
                pygame.draw.rect(screen, BLUE, rect)
                screen.blit(font.render("SPAWN", True, WHITE), (x * CELL_SIZE, y * CELL_SIZE + 15))
            elif (x, y) == (engine.base_x, engine.base_y):
                pygame.draw.rect(screen, GREEN, rect)
                screen.blit(font.render("BASE", True, WHITE), (x * CELL_SIZE + 5, y * CELL_SIZE + 15))
            elif engine.grid[y][x] == 1:
                pygame.draw.rect(screen, GRAY, rect)
            else:
                pygame.draw.rect(screen, (40, 40, 40), rect, 1)

    # Towers
    for tower in engine.towers:
        cx = tower.x * CELL_SIZE + CELL_SIZE // 2
        cy = tower.y * CELL_SIZE + CELL_SIZE // 2
        color = {
            TowerType.BASIC: YELLOW,
            TowerType.SNIPER: (255, 165, 0),
            TowerType.AOE: (200, 0, 200),
            TowerType.SLOW: (0, 200, 200),
        }.get(tower.type, WHITE)
        pygame.draw.circle(screen, color, (cx, cy), CELL_SIZE // 2 - 4)
        screen.blit(font.render(str(tower.level), True, WHITE), (cx - 5, cy - 8))

    # Enemies
    for enemy in engine.enemies:
        cx = int(enemy.x * CELL_SIZE + CELL_SIZE // 2)
        cy = int(enemy.y * CELL_SIZE + CELL_SIZE // 2)
        pygame.draw.circle(screen, RED, (cx, cy), CELL_SIZE // 3)
        hp_ratio = max(0, enemy.hp / enemy.max_hp)
        bar_w = CELL_SIZE * 0.8
        bar = pygame.Rect(cx - bar_w / 2, cy - CELL_SIZE // 2 - 5, bar_w, 4)
        pygame.draw.rect(screen, RED, bar)
        bar.width = bar_w * hp_ratio
        pygame.draw.rect(screen, GREEN, bar)

    # UI Panel
    ui_y = GRID_H * CELL_SIZE
    pygame.draw.rect(screen, (30, 30, 30), (0, ui_y, WIDTH, 120))

    state = engine.get_state()
    screen.blit(font.render(f"Money: ${state['money']}", True, WHITE), (10, ui_y + 5))
    screen.blit(font.render(f"Base HP: {state['base_hp']}", True, WHITE), (10, ui_y + 30))
    screen.blit(font.render(f"Wave: {state['wave']}", True, WHITE), (10, ui_y + 55))
    screen.blit(font.render(f"Step: {step}  Total Reward: {total_reward:.0f}", True, YELLOW), (10, ui_y + 80))
    screen.blit(font.render(f"AI Action: {action_str}", True, (150, 255, 150)), (200, ui_y + 5))
    screen.blit(font.render(f"Root Value: {root_value:.2f}", True, (150, 150, 255)), (200, ui_y + 30))
    screen.blit(font.render("SPACE=pause  ESC=quit  R=restart", True, (150, 150, 150)), (200, ui_y + 55))

    if engine.game_over:
        big_font = pygame.font.SysFont(None, 64)
        screen.blit(big_font.render("GAME OVER", True, RED), (WIDTH // 2 - 130, HEIGHT // 2 - 40))

    pygame.display.flip()


# ──────────────────────────  Main  ─────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python play_ai.py <checkpoint_dir>")
        print("Example: python play_ai.py results/td_muzero/2024-02-15--06-09-25")
        sys.exit(1)

    checkpoint_dir = pathlib.Path(sys.argv[1])
    checkpoint_path = checkpoint_dir / "model.checkpoint"
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found!")
        sys.exit(1)

    # Load config and model
    from games.td_muzero import MuZeroConfig
    config = MuZeroConfig()

    print("Loading model checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = models.MuZeroNetwork(config)
    model.set_weights(checkpoint["weights"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}. Training step: {checkpoint.get('training_step', '?')}")

    # Init pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MuZero Tower Defense - AI Playing")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 22)

    # Game loop
    game = GameWrapper()
    obs = game.reset()
    total_reward = 0
    step = 0
    done = False
    paused = False
    action_str = "..."
    root_value = 0.0

    print("\nAI is playing! Press SPACE to pause/resume, ESC to quit, R to restart.\n")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    game = GameWrapper()
                    obs = game.reset()
                    total_reward = 0
                    step = 0
                    done = False
                    action_str = "..."
                    root_value = 0.0

        if not done and not paused:
            with torch.no_grad():
                legal = game.legal_actions()
                root = run_mcts(model, obs, legal, config)
                action = select_action(root)
                root_value = root.value()
                action_str = game.action_to_string(action)

                obs, reward, done = game.step(action)
                total_reward += reward
                step += 1

                if step % 5 == 0 or done:
                    print(f"Step {step:4d} | Action: {action_str:30s} | Reward: {reward:+.0f} | Total: {total_reward:.0f} | Value: {root_value:.2f}")

        draw_frame(screen, font, game.env, action_str, step, total_reward, root_value)
        clock.tick(10 if not paused else 30)  # ~10 FPS for AI play (readable speed)

    pygame.quit()
    print(f"\nGame ended at step {step}. Total reward: {total_reward:.0f}")


if __name__ == "__main__":
    main()
