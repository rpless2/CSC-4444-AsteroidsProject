import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import random

WIDTH, HEIGHT = 800, 600

# Main difficulty / environment settings
NUM_ASTEROIDS = 10
OBSERVED_ASTEROIDS = 3
SHIP_RADIUS = 15
MAX_ASTEROID_SPEED = 1.5
MAX_SHIP_SPEED = 8
MAX_BULLETS = 5


class AsteroidEnv(gym.Env):
    def __init__(self, render_mode=False):
        super().__init__()

        self.render_mode = render_mode

        # Actions agent can do:
        # 0 = do nothing
        # 1 = turn right
        # 2 = turn left
        # 3 = thrust
        # 4 = shoot
        self.action_space = spaces.Discrete(5)

        # Things to keep track of:
        # 2 ship position values (x,y)
        # 2 ship velocity values (x,y)
        # 2 ship direction values using cos/sin (angle of direction)
        # 15 asteroid values: 3 asteroids * 5 values each (x_pos,y_pos,size of asteroid,x_vel,y_vel)
        # 1 bullet count value (shot or not)
        # Total = 22
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(22,),
            dtype=np.float32
        )

        if render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Asteroids Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

        self.reset()

    # When game starts or resets, this sets everything to the default values
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            random.seed(seed)

        self.ship_x = WIDTH // 2
        self.ship_y = HEIGHT // 2
        self.ship_angle = 0
        self.vel_x = 0
        self.vel_y = 0
        self.alive_steps = 0

        self.asteroids = []
        for _ in range(NUM_ASTEROIDS):
            self.asteroids.append(self._spawn_asteroid())

        self.bullets = []
        self.done = False
        self.score = 0

        return self._get_obs(), {}

    def _wrapped_delta(self, target_x, target_y):
        """
        Finds the shortest x/y distance from the ship to a target.
        This matters because the screen wraps around.
        """
        dx = target_x - self.ship_x
        dy = target_y - self.ship_y

        if dx > WIDTH / 2:
            dx -= WIDTH
        elif dx < -WIDTH / 2:
            dx += WIDTH

        if dy > HEIGHT / 2:
            dy -= HEIGHT
        elif dy < -HEIGHT / 2:
            dy += HEIGHT

        return dx, dy

    def _spawn_asteroid(self):
        """
        Creates one asteroid away from the ship so the agent does not
        instantly die when the game resets or when a new asteroid appears.
        """
        while True:
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - 1)

            dx, dy = self._wrapped_delta(x, y)
            distance_from_ship = math.hypot(dx, dy)

            if distance_from_ship > 180:
                break

        return {
            "x": x,
            "y": y,
            "radius": random.randint(15, 30),
            "vel_x": random.uniform(-MAX_ASTEROID_SPEED, MAX_ASTEROID_SPEED),
            "vel_y": random.uniform(-MAX_ASTEROID_SPEED, MAX_ASTEROID_SPEED)
        }

    def _get_sorted_asteroids(self):
        """
        Returns asteroids sorted from closest to farthest.
        The agent will only observe the 3 closest asteroids.
        """
        asteroid_info = []

        for asteroid in self.asteroids:
            dx, dy = self._wrapped_delta(asteroid["x"], asteroid["y"])
            dist = math.hypot(dx, dy)
            asteroid_info.append((asteroid, dx, dy, dist))

        asteroid_info.sort(key=lambda item: item[3])
        return asteroid_info

    def _get_closest_asteroid_info(self):
        sorted_asteroids = self._get_sorted_asteroids()

        if not sorted_asteroids:
            return None, 0, 0, WIDTH

        return sorted_asteroids[0]

    def _distance_to_closest_asteroid(self):
        _, _, _, dist = self._get_closest_asteroid_info()
        return dist

    def _aim_alignment(self):
        """
        Returns how well the ship is facing the closest asteroid.
        1.0 means directly facing it.
        0.0 means sideways.
        -1.0 means facing away from it.
        This helps determine for every step the angle of the ship.
        """
        closest, dx, dy, dist = self._get_closest_asteroid_info()

        if closest is None or dist == 0:
            return 0

        angle_rad = math.radians(self.ship_angle)

        ship_dir_x = math.cos(angle_rad)
        ship_dir_y = math.sin(angle_rad)

        asteroid_dir_x = dx / dist
        asteroid_dir_y = dy / dist

        return ship_dir_x * asteroid_dir_x + ship_dir_y * asteroid_dir_y

    def _check_ship_collision(self):
        """
        Checks collision using the ship radius plus asteroid radius.
        This is better than using one fixed distance for every asteroid.
        """
        for asteroid in self.asteroids:
            dx, dy = self._wrapped_delta(asteroid["x"], asteroid["y"])
            dist = math.hypot(dx, dy)

            if dist < asteroid["radius"] + SHIP_RADIUS:
                return True

        return False

    def _get_obs(self):
        angle_rad = math.radians(self.ship_angle)
        sorted_asteroids = self._get_sorted_asteroids()

        obs = [
            self.ship_x / WIDTH,
            self.ship_y / HEIGHT,
            self.vel_x / MAX_SHIP_SPEED,
            self.vel_y / MAX_SHIP_SPEED,
            math.cos(angle_rad),
            math.sin(angle_rad)
        ]

        max_dist = math.hypot(WIDTH / 2, HEIGHT / 2)

        for i in range(OBSERVED_ASTEROIDS):
            if i < len(sorted_asteroids):
                asteroid, dx, dy, dist = sorted_asteroids[i]

                obs.extend([
                    dx / (WIDTH / 2),
                    dy / (HEIGHT / 2),
                    asteroid["vel_x"] / MAX_ASTEROID_SPEED,
                    asteroid["vel_y"] / MAX_ASTEROID_SPEED,
                    min(dist / max_dist, 1.0)
                ])
            else:
                obs.extend([0, 0, 0, 0, 1])

        obs.append(len(self.bullets) / MAX_BULLETS)

        return np.array(obs, dtype=np.float32)

    # Each step/iteration that the agent goes through is done in the code block below
    # All actions that the agent can take are labled below along with the rewards associated
    # Actions of every environemnt entity (asteroids, bullets) are detailed below just movement and collision etc.
    # Some personal choice reward decisions are done below too (- reward for shooting too much) can be changed.
    def step(self, action):
        action = int(action)

        reward = 0
        truncated = False

        old_dist = self._distance_to_closest_asteroid()
        old_alignment = self._aim_alignment()
        hit_this_step = False

        # Action 1: turn right
        if action == 1:
            self.ship_angle += 10

        # Action 2: turn left
        elif action == 2:
            self.ship_angle -= 10

        # Action 3: thrust
        elif action == 3:
            angle_rad = math.radians(self.ship_angle)

            self.vel_x += math.cos(angle_rad) * 0.5
            self.vel_y += math.sin(angle_rad) * 0.5

            speed = math.hypot(self.vel_x, self.vel_y)

            if speed > MAX_SHIP_SPEED:
                self.vel_x = self.vel_x / speed * MAX_SHIP_SPEED
                self.vel_y = self.vel_y / speed * MAX_SHIP_SPEED

            reward -= 0.01

        # Action 4: shoot
        elif action == 4:
            if len(self.bullets) < MAX_BULLETS:
                angle_rad = math.radians(self.ship_angle)

                bullet_dx = math.cos(angle_rad) * 10
                bullet_dy = math.sin(angle_rad) * 10

                self.bullets.append({
                    "x": self.ship_x,
                    "y": self.ship_y,
                    "vel_x": bullet_dx + self.vel_x,
                    "vel_y": bullet_dy + self.vel_y,
                    "life": 60
                })

                # Small shooting penalty so it does not spam bullets forever
                reward -= 0.04

                # Reward good aiming before shooting
                if old_alignment > 0.90:
                    reward += 0.5
                else:
                    reward -= 0.1
            else:
                reward -= 0.1

        # Action 0: do nothing
        elif action == 0:
            reward -= 0.002

        # Move ship
        self.ship_x += self.vel_x
        self.ship_y += self.vel_y

        self.ship_x %= WIDTH
        self.ship_y %= HEIGHT

        # Move asteroids
        for asteroid in self.asteroids:
            asteroid["x"] += asteroid["vel_x"]
            asteroid["y"] += asteroid["vel_y"]

            asteroid["x"] %= WIDTH
            asteroid["y"] %= HEIGHT

        # Move bullets
        for bullet in self.bullets[:]:
            bullet["x"] += bullet["vel_x"]
            bullet["y"] += bullet["vel_y"]
            bullet["life"] -= 1

            if (
                bullet["x"] < 0 or bullet["x"] > WIDTH
                or bullet["y"] < 0 or bullet["y"] > HEIGHT
                or bullet["life"] <= 0
            ):
                self.bullets.remove(bullet)

        # Bullet / asteroid collision
        for bullet in self.bullets[:]:
            for asteroid in self.asteroids[:]:
                dist = math.hypot(
                    bullet["x"] - asteroid["x"],
                    bullet["y"] - asteroid["y"]
                )

                if dist < asteroid["radius"]:
                    if bullet in self.bullets:
                        self.bullets.remove(bullet)

                    if asteroid in self.asteroids:
                        self.asteroids.remove(asteroid)

                    reward += 100
                    self.score += 100
                    hit_this_step = True

                    self.asteroids.append(self._spawn_asteroid())
                    break

        new_dist = self._distance_to_closest_asteroid()

        # Reward the agent for moving away from danger
        if not hit_this_step:
            if new_dist < 180:
                if new_dist > old_dist:
                    reward += 0.05
                else:
                    reward -= 0.05

            if new_dist < 90:
                reward -= 0.1

        # Collision with asteroid
        if self._check_ship_collision():
            reward -= 200
            self.done = True

        self.alive_steps += 1

        # Survival reward
        reward += 0.05

        # Bonus reward for surviving longer
        if self.alive_steps % 500 == 0:
            reward += 10

        return self._get_obs(), reward, self.done, truncated, {}

    # Rendering in a interesting background (stars, asteroids, ships, flame, UI)
    # Stars and flame are nothing but looks but helps add something more intersting to the game for us to look at
    def render(self):
        if not self.render_mode:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        self.screen.fill((0, 0, 0))

        # Stars
        for _ in range(100):
            x = random.randint(0, WIDTH - 1)
            y = random.randint(0, HEIGHT - 1)
            self.screen.set_at((x, y), (100, 100, 100))

        # Asteroids
        for asteroid in self.asteroids:
            pygame.draw.circle(
                self.screen,
                (139, 69, 19),
                (int(asteroid["x"]), int(asteroid["y"])),
                asteroid["radius"]
            )

            pygame.draw.circle(
                self.screen,
                (101, 67, 33),
                (int(asteroid["x"]), int(asteroid["y"])),
                asteroid["radius"] - 3,
                2
            )

        # Ship
        angle_rad = math.radians(self.ship_angle)

        nose_x = self.ship_x + math.cos(angle_rad) * 15
        nose_y = self.ship_y + math.sin(angle_rad) * 15

        left_x = self.ship_x + math.cos(angle_rad + 2.2) * 10
        left_y = self.ship_y + math.sin(angle_rad + 2.2) * 10

        right_x = self.ship_x + math.cos(angle_rad - 2.2) * 10
        right_y = self.ship_y + math.sin(angle_rad - 2.2) * 10

        # Flame
        pygame.draw.polygon(
            self.screen,
            (255, 165, 0),
            [
                (
                    self.ship_x - math.cos(angle_rad) * 8,
                    self.ship_y - math.sin(angle_rad) * 8
                ),
                (
                    self.ship_x - math.cos(angle_rad + 0.5) * 12,
                    self.ship_y - math.sin(angle_rad + 0.5) * 12
                ),
                (
                    self.ship_x - math.cos(angle_rad - 0.5) * 12,
                    self.ship_y - math.sin(angle_rad - 0.5) * 12
                )
            ]
        )

        # Ship body
        pygame.draw.polygon(
            self.screen,
            (255, 255, 255),
            [
                (nose_x, nose_y),
                (left_x, left_y),
                (right_x, right_y)
            ]
        )

        # Bullets
        for bullet in self.bullets:
            pygame.draw.circle(
                self.screen,
                (0, 255, 0),
                (int(bullet["x"]), int(bullet["y"])),
                4
            )

        # UI
        score_text = self.font.render(
            f"Score: {self.score}",
            True,
            (255, 255, 255)
        )
        self.screen.blit(score_text, (10, 10))

        steps_text = self.font.render(
            f"Steps: {self.alive_steps}",
            True,
            (255, 255, 255)
        )
        self.screen.blit(steps_text, (10, 50))

        asteroid_text = self.font.render(
            f"Asteroids: {len(self.asteroids)}",
            True,
            (255, 255, 255)
        )
        self.screen.blit(asteroid_text, (10, 90))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.render_mode:
            pygame.quit()