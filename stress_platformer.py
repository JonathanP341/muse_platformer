import pygame
import sys
import random
import threading
import time
import numpy as np

# Import your existing class
from eeg_receiver import EEGReceiver

# --- CONFIGURATION ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
GREEN = (50, 255, 50)
BLUE = (50, 50, 255)
GRAY = (100, 100, 100)

# Game Physics
GRAVITY = 0.5
FRICTION = 0.8
MAX_JUMP_POWER = 15
CHARGE_SPEED = 0.3

# Stress Mechanics
TILT_THRESHOLD = 0.5  # Score above this triggers stress effects
MAX_SHAKE = 10        # Max pixels to shake screen at full tilt

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((30, 30))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.center = (100, SCREEN_HEIGHT - 50)
        
        self.vel_x = 0
        self.vel_y = 0
        self.on_ground = False
        self.charging_jump = False
        self.jump_power = 0

    def update(self, platforms, stress_level):
        # Apply Gravity
        self.vel_y += GRAVITY
        
        # Move X
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.vel_x = -5
        elif keys[pygame.K_RIGHT]:
            self.vel_x = 5
        else:
            self.vel_x *= FRICTION # Slide to stop

        # Apply Movement
        self.rect.x += int(self.vel_x)
        self.collide(platforms, 'x')
        self.rect.y += int(self.vel_y)
        self.collide(platforms, 'y')

        # Jump Charging Logic
        if self.charging_jump:
            # Base charge speed
            increase = CHARGE_SPEED
            
            # --- STRESS MECHANIC 1: Jittery Charge ---
            # If stressed, the charge bar becomes unstable/jittery
            if stress_level > TILT_THRESHOLD:
                # Add random noise to the charge rate based on stress
                jitter = random.uniform(-0.2, 0.4) * stress_level
                increase += jitter

            self.jump_power += increase
            if self.jump_power > MAX_JUMP_POWER:
                self.jump_power = MAX_JUMP_POWER

    def collide(self, platforms, direction):
        hits = pygame.sprite.spritecollide(self, platforms, False)
        if hits:
            if direction == 'x':
                if self.vel_x > 0: self.rect.right = hits[0].rect.left
                if self.vel_x < 0: self.rect.left = hits[0].rect.right
                self.vel_x = 0
            if direction == 'y':
                if self.vel_y > 0: 
                    self.rect.bottom = hits[0].rect.top
                    self.vel_y = 0
                    self.on_ground = True
                if self.vel_y < 0: 
                    self.rect.top = hits[0].rect.bottom
                    self.vel_y = 0

    def jump(self):
        if self.on_ground:
            self.vel_y = -self.jump_power
            self.on_ground = False
        self.jump_power = 0
        self.charging_jump = False

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.image = pygame.Surface((w, h))
        self.image.fill(GRAY)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

def draw_ui(screen, tilt_score, player):
    # Draw Tilt Score
    font = pygame.font.Font(None, 36)
    
    # Color changes based on stress
    color = GREEN if tilt_score < TILT_THRESHOLD else RED
    text = font.render(f"Tilt Score: {tilt_score:.2f}", True, color)
    screen.blit(text, (10, 10))

    # Draw Jump Charge Bar
    bar_width = 100
    bar_height = 10
    fill = (player.jump_power / MAX_JUMP_POWER) * bar_width
    outline_rect = pygame.Rect(player.rect.centerx - bar_width//2, player.rect.top - 20, bar_width, bar_height)
    fill_rect = pygame.Rect(player.rect.centerx - bar_width//2, player.rect.top - 20, fill, bar_height)
    
    pygame.draw.rect(screen, WHITE, outline_rect, 1)
    pygame.draw.rect(screen, color, fill_rect)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Neuro-Feedback Platformer")
    clock = pygame.time.Clock()

    # --- EEG SETUP ---
    print("Initializing EEG Receiver...")
    eeg = EEGReceiver()
    eeg.start() # Starts the OSC server in a background thread
    while len(eeg.AF7Buffer) < 1:
        time.sleep(0.5)

    # --- CALIBRATION STATE ---
    # We need to run calibration without freezing the game loop
    calibrating = True
    calibration_thread = threading.Thread(target=eeg.find_baseline)
    calibration_thread.start()

    font = pygame.font.Font(None, 50)

    while calibrating:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        screen.fill(BLACK)
        
        if calibration_thread.is_alive():
            msg = font.render("Calibrating... Relax.", True, WHITE)
            sub = font.render("Do not move.", True, GRAY)
            screen.blit(msg, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 - 50))
            screen.blit(sub, (SCREEN_WIDTH//2 - 100, SCREEN_HEIGHT//2 + 10))
        else:
            calibrating = False # Calibration done

        pygame.display.flip()
        clock.tick(30)

    # --- GAME LOOP SETUP ---
    player = Player()
    all_sprites = pygame.sprite.Group()
    platforms = pygame.sprite.Group()

    all_sprites.add(player)

    # Create Level (Simple "Jump King" style vertical climbing)
    plat_coords = [
        (0, SCREEN_HEIGHT - 20, SCREEN_WIDTH, 20), # Floor
        (200, 450, 200, 20),
        (500, 350, 150, 20),
        (100, 250, 200, 20),
        (400, 150, 200, 20),
        (300, 50, 100, 20)
    ]
    
    for coord in plat_coords:
        p = Platform(*coord)
        platforms.add(p)
        all_sprites.add(p)

    running = True
    current_tilt = 0.0

    while running:
        # 1. Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and player.on_ground:
                    player.charging_jump = True
            
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    player.jump()

        # 2. Get Data from EEG
        # We fetch the score calculated by your EEGReceiver class
        try:
            new_tilt = eeg.get_tilt_score()
            if new_tilt is not None:
                current_tilt = new_tilt
        except Exception as e:
            print(f"EEG Error: {e}")

        # 3. Update Game Logic
        player.update(platforms, current_tilt)
        
        # 4. Drawing
        screen.fill(BLACK)

        # --- STRESS MECHANIC 2: Screen Shake ---
        # If tilt is high, we offset the drawing surface randomly
        shake_x = 0
        shake_y = 0
        if current_tilt > TILT_THRESHOLD:
            intensity = (current_tilt - TILT_THRESHOLD) / (1.0 - TILT_THRESHOLD)
            intensity = min(intensity, 1.0) # Clamp
            shake_x = random.randint(-MAX_SHAKE, MAX_SHAKE) * intensity
            shake_y = random.randint(-MAX_SHAKE, MAX_SHAKE) * intensity
        
        # Draw everything with the shake offset
        for entity in all_sprites:
            screen.blit(entity.image, (entity.rect.x + shake_x, entity.rect.y + shake_y))

        draw_ui(screen, current_tilt, player)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()