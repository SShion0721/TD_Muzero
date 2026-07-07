import heapq
import math
import numpy as np

class NumpyRNG:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        
    def seed(self, seed):
        self.rng = np.random.RandomState(seed)
        
    def getrandbits(self, k):
        if k <= 0 or k > 32: raise ValueError
        # numpy randint high is exclusive, so 2**32
        r = int(self.rng.randint(0, 2**32, dtype=np.uint32))
        if k < 32:
            return r >> (32 - k)
        return r

    def random(self):
        a = self.getrandbits(32) >> 5
        b = self.getrandbits(32) >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
        
    def randrange(self, stop):
        if stop <= 0: raise ValueError
        n = stop - 1
        k = n.bit_length()
        if k == 0: return 0
        while True:
            r = self.getrandbits(k)
            if r < stop:
                return r

    def choice(self, seq):
        if len(seq) == 0: raise IndexError
        return seq[self.randrange(len(seq))]
        
    def shuffle(self, seq):
        for i in reversed(range(1, len(seq))):
            j = self.randrange(i + 1)
            seq[i], seq[j] = seq[j], seq[i]

class TowerType:
    BASIC = 0
    SNIPER = 1
    AOE = 2
    SLOW = 3 # New slow/support tower

    @staticmethod
    def get_stats(tower_type):
        if tower_type == TowerType.BASIC:
            return {'cost': 50, 'damage': 10, 'range': 2.5, 'cooldown': 1.0, 'aoe_radius': 0.0, 'slow_factor': 1.0, 'slow_duration': 0.0}
        elif tower_type == TowerType.SNIPER:
            return {'cost': 100, 'damage': 50, 'range': 5.5, 'cooldown': 3.0, 'aoe_radius': 0.0, 'slow_factor': 1.0, 'slow_duration': 0.0}
        elif tower_type == TowerType.AOE:
            return {'cost': 150, 'damage': 15, 'range': 2.0, 'cooldown': 1.5, 'aoe_radius': 2.0, 'slow_factor': 1.0, 'slow_duration': 0.0}
        elif tower_type == TowerType.SLOW:
            return {'cost': 75, 'damage': 2, 'range': 3.0, 'cooldown': 1.0, 'aoe_radius': 0.0, 'slow_factor': 0.4, 'slow_duration': 2.0}
        return {}

class Tower:
    def __init__(self, x, y, tower_type):
        self.x = x
        self.y = y
        self.type = tower_type
        stats = TowerType.get_stats(tower_type)
        self.cost = stats['cost']
        self.damage = stats['damage']
        self.range = stats['range']
        self.cooldown_max = stats['cooldown']
        self.aoe_radius = stats['aoe_radius']
        self.slow_factor = stats.get('slow_factor', 1.0)
        self.slow_duration = stats.get('slow_duration', 0.0)
        
        self.cooldown = 0.0
        self.level = 1
        self.total_spent = self.cost
        self.upgrade_cost = int(self.cost * 1.5)

    def upgrade(self):
        self.level += 1
        self.total_spent += self.upgrade_cost
        # Level up bonuses (general stats)
        self.damage *= 1.5
        self.range *= 1.1
        self.cooldown_max = max(0.1, self.cooldown_max * 0.9)
        self.upgrade_cost = int(self.upgrade_cost * 1.5)

    def step(self, dt):
        if self.cooldown > 0:
            self.cooldown -= dt

    def can_shoot(self):
        return self.cooldown <= 0

    def shoot(self):
        self.cooldown = self.cooldown_max

class Enemy:
    def __init__(self, id, x, y, hp, speed, reward):
        self.id = id
        self.x = x # Float coordinates for smooth movement
        self.y = y
        self.target_x = x
        self.target_y = y
        self.hp = hp
        self.max_hp = hp
        self.speed = speed
        self.base_speed = speed # For recovering from slow
        self.slow_timer = 0.0
        self.reward = reward
        self.path = [] # List of (x, y) tuples

    def apply_slow(self, factor, duration):
        if factor < 1.0: # Only applied if it's a slow effect
            self.speed = self.base_speed * factor
            self.slow_timer = max(self.slow_timer, duration)

    def set_path(self, path):
        # path includes current position, we assume it's moving towards path[0] or we pop the first.
        self.path = path.copy()
        if len(self.path) > 0:
            self.target_x, self.target_y = self.path[0]

    def step(self, dt):
        if self.slow_timer > 0:
            self.slow_timer -= dt
            if self.slow_timer <= 0:
                self.speed = self.base_speed
                
        # Move towards target
        if len(self.path) == 0:
            return
        
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        move_dist = self.speed * dt

        if dist <= move_dist:
            # Reached target waypoint
            self.x = self.target_x
            self.y = self.target_y
            self.path.pop(0)
            if len(self.path) > 0:
                self.target_x, self.target_y = self.path[0]
        else:
            self.x += (dx / dist) * move_dist
            self.y += (dy / dist) * move_dist

class TDEngine:
    def __init__(self, width=11, height=11, seed=0):
        self.width = width
        self.height = height
        self.spawn_x, self.spawn_y = 0, height // 2
        self.base_x, self.base_y = width - 1, height // 2
        
        self.seed = seed
        self.rng = NumpyRNG(self.seed)
        
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            self.rng = NumpyRNG(self.seed)
            
        self.money = 200
        self.base_hp = 100
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)] # 0 = empty, 1 = tower
        self.towers = []
        self.enemies = []
        self.enemy_id_counter = 0
        self.wave = 1
        self.enemies_to_spawn = self._get_wave_enemies()
        self.spawn_timer = 0
        self.time = 0.0
        self.game_over = False
        return self.get_state()

    def _get_wave_enemies(self):
        # define how many and what enemies to spawn based on wave
        enemies_format = []
        base_hp = 20 + self.wave * 15 + (self.wave ** 2) * 2 # Exponential HP scaling
        
        # Swarm enemies (fast, low hp, less reward)
        swarm_count = self.wave * 2
        enemies_format.extend([{'hp': base_hp * 0.3, 'speed': 2.8, 'reward': 5} for _ in range(swarm_count)])
        
        # Regular enemies
        reg_count = 5 + self.wave * 2
        enemies_format.extend([{'hp': base_hp, 'speed': 1.5, 'reward': 10} for _ in range(reg_count)])
        
        # Tank enemies (slow, high hp, high reward) starting from wave 3
        if self.wave >= 3:
            tank_count = self.wave
            enemies_format.extend([{'hp': base_hp * 3.5, 'speed': 0.8, 'reward': 30} for _ in range(tank_count)])
            
        # Boss enemies starting wave 5
        if self.wave >= 5 and self.wave % 2 == 1:
            boss_count = 1 + (self.wave // 10)
            enemies_format.extend([{'hp': base_hp * 10.0, 'speed': 0.6, 'reward': 100} for _ in range(boss_count)])
            
        self.rng.shuffle(enemies_format) # Mix them up
        return enemies_format

    def _manhattan(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def find_path(self, start_x, start_y, end_x, end_y):
        """A* Search to find shortest path. Returns list of (x, y) coordinates."""
        # Using a simple A* implementation
        open_set = []
        heapq.heappush(open_set, (0, (start_x, start_y)))
        
        came_from = {}
        g_score = {(x, y): float('inf') for x in range(self.width) for y in range(self.height)}
        g_score[(start_x, start_y)] = 0
        
        f_score = {(x, y): float('inf') for x in range(self.width) for y in range(self.height)}
        f_score[(start_x, start_y)] = self._manhattan(start_x, start_y, end_x, end_y)

        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current == (end_x, end_y):
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            cx, cy = current
            # Check neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.width and 0 <= ny < self.height and self.grid[ny][nx] == 0:
                    tentative_g_score = g_score[current] + 1
                    if tentative_g_score < g_score.get((nx, ny), float('inf')):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g_score
                        f = tentative_g_score + self._manhattan(nx, ny, end_x, end_y)
                        f_score[(nx, ny)] = f
                        heapq.heappush(open_set, (f, (nx, ny)))

        return None # Path completely blocked

    def can_place_tower(self, x, y, tower_type):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if self.grid[y][x] != 0:
            return False
        if (x, y) == (self.spawn_x, self.spawn_y) or (x, y) == (self.base_x, self.base_y):
            return False
        
        stats = TowerType.get_stats(tower_type)
        if self.money < stats['cost']:
            return False
            
        # Simulate placing the tower and check if path still exists
        self.grid[y][x] = 1
        path = self.find_path(self.spawn_x, self.spawn_y, self.base_x, self.base_y)
        self.grid[y][x] = 0 # Revert
        
        if path is None:
            return False
        return True

    def place_tower(self, x, y, tower_type):
        if self.can_place_tower(x, y, tower_type):
            stats = TowerType.get_stats(tower_type)
            self.money -= stats['cost']
            self.grid[y][x] = 1
            self.towers.append(Tower(x, y, tower_type))
            
            # Recalculate paths for all existing enemies
            for enemy in self.enemies:
                # Find path from enemy's NEXT grid cell to avoid getting stuck
                curr_grid_x, curr_grid_y = int(round(enemy.x)), int(round(enemy.y))
                path = self.find_path(curr_grid_x, curr_grid_y, self.base_x, self.base_y)
                if path:
                    enemy.set_path(path)
            return True
        return False
        
    def upgrade_tower(self, x, y):
        # Find tower at x,y
        for tower in self.towers:
            if tower.x == x and tower.y == y:
                if self.money >= tower.upgrade_cost:
                    self.money -= tower.upgrade_cost
                    tower.upgrade()
                    return True
                break
        return False
        
    def sell_tower(self, x, y):
        # Find tower at x,y
        for tower in self.towers:
            if tower.x == x and tower.y == y:
                # 80% refund of total spent (base cost + upgrades)
                refund = int(tower.total_spent * 0.8)
                self.money += refund
                self.towers.remove(tower)
                self.grid[y][x] = 0
                
                # Recalculate paths since a path opened up
                for enemy in self.enemies:
                    curr_grid_x, curr_grid_y = int(round(enemy.x)), int(round(enemy.y))
                    path = self.find_path(curr_grid_x, curr_grid_y, self.base_x, self.base_y)
                    if path:
                        enemy.set_path(path)
                return True
        return False

    def step(self, dt=1.0):
        if self.game_over:
            return 0, True # reward, done

        step_reward = 0

        # 1. Spawn enemies
        self.spawn_timer -= dt
        if self.spawn_timer <= 0 and len(self.enemies_to_spawn) > 0:
            e_data = self.enemies_to_spawn.pop(0)
            enemy = Enemy(self.enemy_id_counter, float(self.spawn_x), float(self.spawn_y), e_data['hp'], e_data['speed'], e_data['reward'])
            self.enemy_id_counter += 1
            path = self.find_path(self.spawn_x, self.spawn_y, self.base_x, self.base_y)
            enemy.set_path(path)
            self.enemies.append(enemy)
            self.spawn_timer = 1.0 # 1 second between spawns

        # 2. Move enemies
        enemies_reached_base = []
        for enemy in self.enemies:
            enemy.step(dt)
            if math.hypot(enemy.x - self.base_x, enemy.y - self.base_y) < 0.2:
                enemies_reached_base.append(enemy)

        # Handle base reaching
        for enemy in enemies_reached_base:
            self.base_hp -= enemy.max_hp // 10  # Base takes damage based on enemy max HP
            self.enemies.remove(enemy)
            step_reward -= 50 # Penalty for base taking damage

        if self.base_hp <= 0:
            self.game_over = True
            step_reward -= 1000 # Big penalty for losing
            return step_reward, True

        # 3. Towers attack
        for tower in self.towers:
            tower.step(dt)
            if tower.can_shoot():
                # Find closest enemy in range
                best_target = None
                best_dist = float('inf')
                for enemy in self.enemies:
                    dist = math.hypot(enemy.x - tower.x, enemy.y - tower.y)
                    if dist <= tower.range and dist < best_dist:
                        best_dist = dist
                        best_target = enemy
                
                if best_target is not None:
                    if tower.aoe_radius > 0.0:
                        # Splash damage
                        for e in self.enemies:
                            if math.hypot(e.x - best_target.x, e.y - best_target.y) <= tower.aoe_radius:
                                e.hp = float(e.hp - tower.damage)
                                e.apply_slow(tower.slow_factor, tower.slow_duration)
                    else:
                        # Single target damage
                        best_target.hp = float(best_target.hp - tower.damage)
                        best_target.apply_slow(tower.slow_factor, tower.slow_duration)
                    tower.shoot()

        # 4. Handle enemy deaths
        dead_enemies = [e for e in self.enemies if e.hp <= 0]
        for enemy in dead_enemies:
            self.money += enemy.reward
            step_reward += enemy.reward
            self.enemies.remove(enemy)

        # 5. Check wave end
        if len(self.enemies) == 0 and len(self.enemies_to_spawn) == 0:
            self.wave += 1
            self.enemies_to_spawn = self._get_wave_enemies()
            self.spawn_timer = 3.0 # Wait 3 seconds before next wave
            step_reward += 100 # Reward for clearing wave

        self.time += dt
        return step_reward, False

    def get_state(self):
        """Returns a snapshot of the game state useful for RL/UI"""
        return {
            'money': self.money,
            'base_hp': self.base_hp,
            'wave': self.wave,
            'grid': [row[:] for row in self.grid],
            'towers': [{'x': t.x, 'y': t.y, 'type': t.type, 'cooldown': t.cooldown} for t in self.towers],
            'enemies': [{'id': e.id, 'x': e.x, 'y': e.y, 'hp': e.hp, 'max_hp': e.max_hp} for e in self.enemies]
        }
