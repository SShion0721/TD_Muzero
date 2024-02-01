import pygame
import sys
from engine import TDEngine, TowerType

# Constants
CELL_SIZE = 50
GRID_WIDTH = 11
GRID_HEIGHT = 11
WIDTH = GRID_WIDTH * CELL_SIZE
HEIGHT = GRID_HEIGHT * CELL_SIZE + 100 # Extra space for UI
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (200, 200, 0)
GRAY = (100, 100, 100)

class TDGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Tower Defense Engine Test")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)
        
        self.engine = TDEngine(GRID_WIDTH, GRID_HEIGHT)
        self.selected_tower = TowerType.BASIC
        
    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Draw cell background
                if (x, y) == (self.engine.spawn_x, self.engine.spawn_y):
                    pygame.draw.rect(self.screen, BLUE, rect)
                    # Label spawn
                    text = self.font.render("SPAWN", True, WHITE)
                    self.screen.blit(text, (x * CELL_SIZE, y * CELL_SIZE + 15))
                elif (x, y) == (self.engine.base_x, self.engine.base_y):
                    pygame.draw.rect(self.screen, GREEN, rect)
                    # Label base
                    text = self.font.render("BASE", True, WHITE)
                    self.screen.blit(text, (x * CELL_SIZE + 5, y * CELL_SIZE + 15))
                elif self.engine.grid[y][x] == 1:
                    pygame.draw.rect(self.screen, GRAY, rect)
                else:
                    pygame.draw.rect(self.screen, (40, 40, 40), rect, 1)

    def draw_entities(self):
        # Draw path of the first enemy (to show the shortest path)
        if len(self.engine.enemies) > 0:
            path = self.engine.enemies[0].path
            for (px, py) in path:
                center = (px * CELL_SIZE + CELL_SIZE // 2, py * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.circle(self.screen, (200, 200, 255), center, 4)

        # Draw towers
        for tower in self.engine.towers:
            center = (tower.x * CELL_SIZE + CELL_SIZE // 2, tower.y * CELL_SIZE + CELL_SIZE // 2)
            if tower.type == TowerType.BASIC:
                color = YELLOW
            elif tower.type == TowerType.SNIPER:
                color = (255, 165, 0) # Orange
            elif tower.type == TowerType.AOE:
                color = (200, 0, 200) # Purple for AOE
            else:
                color = (0, 200, 200) # Cyan for SLOW

            pygame.draw.circle(self.screen, color, center, CELL_SIZE // 2 - 4)
            # Level indicator
            lvl_text = self.font.render(str(tower.level), True, WHITE)
            self.screen.blit(lvl_text, (center[0] - 5, center[1] - 8))
            # draw range ring (optional)
            pygame.draw.circle(self.screen, (60, 60, 60), center, int(tower.range * CELL_SIZE), 1)

        # Draw enemies
        for enemy in self.engine.enemies:
            center = (int(enemy.x * CELL_SIZE + CELL_SIZE // 2), int(enemy.y * CELL_SIZE + CELL_SIZE // 2))
            pygame.draw.circle(self.screen, RED, center, CELL_SIZE // 3)
            # Health bar
            hp_ratio = max(0, enemy.hp / enemy.max_hp)
            hp_w = CELL_SIZE * 0.8
            hp_rect = pygame.Rect(center[0] - hp_w/2, center[1] - CELL_SIZE//2 - 5, hp_w, 4)
            pygame.draw.rect(self.screen, (255, 0, 0), hp_rect)
            hp_rect.width = hp_w * hp_ratio
            pygame.draw.rect(self.screen, (0, 255, 0), hp_rect)

    def draw_ui(self):
        ui_rect = pygame.Rect(0, GRID_HEIGHT * CELL_SIZE, WIDTH, 100)
        pygame.draw.rect(self.screen, (30, 30, 30), ui_rect)
        
        state = self.engine.get_state()
        text_money = self.font.render(f"Money: ${state['money']}", True, WHITE)
        text_hp = self.font.render(f"Base HP: {state['base_hp']}", True, WHITE)
        text_wave = self.font.render(f"Wave: {state['wave']}", True, WHITE)
        
        # Build mode selection text
        basic_color = YELLOW if self.selected_tower == TowerType.BASIC else GRAY
        sniper_color = (255, 165, 0) if self.selected_tower == TowerType.SNIPER else GRAY
        aoe_color = (200, 0, 200) if self.selected_tower == TowerType.AOE else GRAY
        slow_color = (0, 200, 200) if self.selected_tower == TowerType.SLOW else GRAY
        
        text_basic = self.font.render("1: BASIC ($50)", True, basic_color)
        text_sniper = self.font.render("2: SNIPER ($100)", True, sniper_color)
        text_aoe = self.font.render("3: AOE ($150)", True, aoe_color)
        text_slow = self.font.render("4: SLOW ($75)", True, slow_color)
        text_help = self.font.render("L-Click: Build | R-Click: Upgrade | Middle-Click: Sell", True, (150, 150, 150))

        self.screen.blit(text_money, (10, GRID_HEIGHT * CELL_SIZE + 10))
        self.screen.blit(text_hp, (10, GRID_HEIGHT * CELL_SIZE + 40))
        self.screen.blit(text_wave, (10, GRID_HEIGHT * CELL_SIZE + 70))
        
        self.screen.blit(text_basic, (200, GRID_HEIGHT * CELL_SIZE + 10))
        self.screen.blit(text_sniper, (200, GRID_HEIGHT * CELL_SIZE + 40))
        self.screen.blit(text_aoe, (400, GRID_HEIGHT * CELL_SIZE + 10))
        self.screen.blit(text_slow, (400, GRID_HEIGHT * CELL_SIZE + 40))
        self.screen.blit(text_help, (200, GRID_HEIGHT * CELL_SIZE + 70))

        if self.engine.game_over:
            text_over = pygame.font.SysFont(None, 48).render("GAME OVER", True, RED)
            self.screen.blit(text_over, (WIDTH//2 - 100, HEIGHT//2 - 24))

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0 # DT in seconds
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mp_x, mp_y = pygame.mouse.get_pos()
                    grid_x = mp_x // CELL_SIZE
                    grid_y = mp_y // CELL_SIZE
                        
                    if grid_y < GRID_HEIGHT: # Clicked on grid
                        if event.button == 1: # Left click: Build
                            self.engine.place_tower(grid_x, grid_y, self.selected_tower)
                        elif event.button == 3: # Right click: Upgrade
                            self.engine.upgrade_tower(grid_x, grid_y)
                        elif event.button == 2: # Middle click: Sell
                            self.engine.sell_tower(grid_x, grid_y)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.selected_tower = TowerType.BASIC
                    elif event.key == pygame.K_2:
                        self.selected_tower = TowerType.SNIPER
                    elif event.key == pygame.K_3:
                        self.selected_tower = TowerType.AOE
                    elif event.key == pygame.K_4:
                        self.selected_tower = TowerType.SLOW
                    elif event.key == pygame.K_r and self.engine.game_over:
                        self.engine.reset()

            if not self.engine.game_over:
                self.engine.step(dt)

            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_entities()
            self.draw_ui()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    gui = TDGUI()
    gui.run()
