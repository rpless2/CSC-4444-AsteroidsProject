import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

WIDTH, HEIGHT = 800, 600


class AsteroidEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()

        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)

        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Asteroids Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    def reset(self, seed=None, options=None):

        self.ship_x = WIDTH // 2
        self.ship_y = HEIGHT // 2
        self.ship_angle = 0
        self.vel_x = 0
        self.vel_y = 0
        self.alive_steps = 0


        self.asteroids = []
        for _ in range(3):
            while True:
                x = random.randint(50, WIDTH - 50)
                y = random.randint(50, HEIGHT - 50)

                if math.hypot(x - self.ship_x, y - self.ship_y) > 100:
                    break
            self.asteroids.append({
                'x': x,
                'y': y,
                'radius': random.randint(15, 30),
                'vel_x': random.uniform(-1, 1),
                'vel_y': random.uniform(-1, 1)
            })

        self.bullets = []
        self.done = False
        self.score = 0

        return self._get_obs(), {}

    def _get_obs(self):
        if not self.asteroids:
            closest_dist = WIDTH
        else:
            closest_dist = min([math.hypot(self.ship_x - a['x'], self.ship_y - a['y']) for a in self.asteroids])

        return np.array([
            self.ship_x / WIDTH,
            self.ship_y / HEIGHT,
            self.vel_x / 10,
            self.vel_y / 10,
            (self.ship_angle % 360) / 360,
            min(closest_dist / WIDTH, 1.0),
            len(self.bullets) / 10,
            self.alive_steps / 1000
        ], dtype=np.float32)

    def _distance_to_closest_asteroid(self):
        if not self.asteroids:
            return WIDTH
        return min([math.hypot(self.ship_x - a['x'], self.ship_y - a['y']) for a in self.asteroids])

    def step(self, action):
        reward = 0
        truncated = False


        if action == 1:
            self.ship_angle += 10
        elif action == 2:
            self.ship_angle -= 10
        elif action == 3:

            angle_rad = math.radians(self.ship_angle)
            self.vel_x += math.cos(angle_rad) * 0.5
            self.vel_y += math.sin(angle_rad) * 0.5


            speed = math.hypot(self.vel_x, self.vel_y)
            if speed > 8:
                self.vel_x = self.vel_x / speed * 8
                self.vel_y = self.vel_y / speed * 8

            reward -= 0.02

        elif action == 4:

            if len(self.bullets) < 5:
                angle_rad = math.radians(self.ship_angle)
                bullet_dx = math.cos(angle_rad) * 10
                bullet_dy = math.sin(angle_rad) * 10
                self.bullets.append({
                    'x': self.ship_x,
                    'y': self.ship_y,
                    'vel_x': bullet_dx + self.vel_x,
                    'vel_y': bullet_dy + self.vel_y,
                    'life': 60
                })
                reward -= 0.05


        self.ship_x += self.vel_x
        self.ship_y += self.vel_y


        self.ship_x %= WIDTH
        self.ship_y %= HEIGHT


        for asteroid in self.asteroids:
            asteroid['x'] += asteroid['vel_x']
            asteroid['y'] += asteroid['vel_y']
            asteroid['x'] %= WIDTH
            asteroid['y'] %= HEIGHT


        for bullet in self.bullets[:]:
            bullet['x'] += bullet['vel_x']
            bullet['y'] += bullet['vel_y']
            bullet['life'] -= 1


            if (bullet['x'] < 0 or bullet['x'] > WIDTH or
                    bullet['y'] < 0 or bullet['y'] > HEIGHT or
                    bullet['life'] <= 0):
                self.bullets.remove(bullet)


        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                dist = math.hypot(bullet['x'] - asteroid['x'], bullet['y'] - asteroid['y'])
                if dist < asteroid['radius']:

                    if bullet in self.bullets:
                        self.bullets.remove(bullet)
                    if asteroid in self.asteroids:
                        self.asteroids.remove(asteroid)
                    reward += 100
                    self.score += 100

                    self.asteroids.append({
                        'x': random.randint(0, WIDTH),
                        'y': random.randint(0, HEIGHT),
                        'radius': random.randint(15, 30),
                        'vel_x': random.uniform(-1, 1),
                        'vel_y': random.uniform(-1, 1)
                    })
                    break


        if self._distance_to_closest_asteroid() < 18:  # Ship radius ~15
            reward -= 200
            self.done = True


        self.alive_steps += 1
        reward += 0.05


        if action != 4 and len(self.bullets) == 0:
            reward -= 0.01

        return self._get_obs(), reward, self.done, truncated, {}

    def render(self):
        if not self.render_mode:
            return


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True


        self.screen.fill((0, 0, 0))


        for _ in range(100):
            x = random.randint(0, WIDTH)
            y = random.randint(0, HEIGHT)
            self.screen.set_at((x, y), (100, 100, 100))


        for asteroid in self.asteroids:
            pygame.draw.circle(self.screen, (139, 69, 19),
                               (int(asteroid['x']), int(asteroid['y'])),
                               asteroid['radius'])

            pygame.draw.circle(self.screen, (101, 67, 33),
                               (int(asteroid['x']), int(asteroid['y'])),
                               asteroid['radius'] - 3, 2)


        angle_rad = math.radians(self.ship_angle)
        nose_x = self.ship_x + math.cos(angle_rad) * 15
        nose_y = self.ship_y + math.sin(angle_rad) * 15
        left_x = self.ship_x + math.cos(angle_rad + 2.2) * 10
        left_y = self.ship_y + math.sin(angle_rad + 2.2) * 10
        right_x = self.ship_x + math.cos(angle_rad - 2.2) * 10
        right_y = self.ship_y + math.sin(angle_rad - 2.2) * 10


        pygame.draw.polygon(self.screen, (255, 165, 0),
                            [(self.ship_x - math.cos(angle_rad) * 8, self.ship_y - math.sin(angle_rad) * 8),
                             (self.ship_x - math.cos(angle_rad + 0.5) * 12,
                              self.ship_y - math.sin(angle_rad + 0.5) * 12),
                             (self.ship_x - math.cos(angle_rad - 0.5) * 12,
                              self.ship_y - math.sin(angle_rad - 0.5) * 12)])

        pygame.draw.polygon(self.screen, (255, 255, 255),
                            [(nose_x, nose_y), (left_x, left_y), (right_x, right_y)])


        for bullet in self.bullets:
            pygame.draw.circle(self.screen, (0, 255, 0), (int(bullet['x']), int(bullet['y'])), 4)


        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))


        steps_text = self.font.render(f"Steps: {self.alive_steps}", True, (255, 255, 255))
        self.screen.blit(steps_text, (10, 50))

        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS

    def close(self):
        if self.render_mode:
            pygame.quit()