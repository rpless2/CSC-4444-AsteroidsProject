import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

WIDTH, HEIGHT = 600, 600


class AsteroidEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()

        self.render_mode = render_mode
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=1, high=1, shape=(8,), dtype=np.float32
                                            )

        pygame.init()
        if render_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

        self.reset()

    def reset(self, seed=None, options=None):
        self.ship_x = WIDTH // 2
        self.ship_y = HEIGHT // 2
        self.ship_angle = 0
        self.vel_x = 0
        self.vel_y = 0

        self.asteroid_x = random.randint(0, WIDTH)
        self.asteroid_y = random.randint(0, HEIGHT)

        self.done = False

        self.bullets = []
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([self.ship_x / WIDTH, self.ship_y / HEIGHT, self.vel_x / 5,
                         self.vel_y / 5, self.ship_angle / 360, self.asteroid_x / WIDTH, self.asteroid_y / HEIGHT,
                         self._distance_to_asteroid() / WIDTH], dtype=np.float32)

    def _distance_to_asteroid(self):
        return math.hypot(self.ship_x - self.asteroid_x, self.ship_y - self.asteroid_y)

    def step(self, action):
        reward = 0

        if action == 1:
            self.ship_angle -= 10
        elif action == 2:
            self.ship_angle += 10
        elif action == 3:
            self.vel_x += math.cos(math.radians(self.ship_angle)) * 0.5
            self.vel_y += math.sin(math.radians(self.ship_angle)) * 0.5
        elif action == 4:
            bullet_dx = math.cos(math.radians(self.ship_angle)) * 8
            bullet_dy = math.sin(math.radians(self.ship_angle)) * 8

            self.bullets.append([self.ship_x, self.ship_y, bullet_dx, bullet_dy])
            reward -= 0.2

            self.ship_x += self.vel_x
            self.ship_y += self.vel_y

            self.ship_x%=WIDTH
            self.ship_y%=HEIGHT

            for bullet in self.bullets:
                bullet[0] += bullet[2]
                bullet[1] += bullet[3]

                self.bullets = [
                    b for b in self.bullets
                    if 0 <= WIDTH and 0 <= b[1] <= HEIGHT
                ]

                hit = False
                for bullet in self.bullets:
                    dist = math.hypot(bullet[0]-self.asteroid_x, bullet[1]-self.asteroid_y)

                    if dist < 20:
                        hit = True
                        self.bullets.remove(bullet)
                        break

                    if hit:
                        reward += 20

                        self.asteroid_x = random.randint(0, WIDTH)
                        self.asteroid_y = random.randint(0, HEIGHT)

        if self._distance_to_asteroid() < 20:
            reward -= 100
            self.done = True

        reward += 1

        return self._get_obs(), reward, self.done, False, {}

    def render(self):
        if not self.render_mode:
            return

        self.screen.fill((0, 0, 0))

        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ship_x), int(self.ship_y)), 10)

        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.asteroid_x), int(self.asteroid_y)), 10)

        pygame.display.flip()

        for bullet in self.bullets:
            pygame.draw.circle(self.screen, (0, 255, 0), (int(bullet[0]), int(bullet[1])), 3)

