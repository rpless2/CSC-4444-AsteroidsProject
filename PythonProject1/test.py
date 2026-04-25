from stable_baselines3 import PPO
from asteroids_env import AsteroidEnv
import time

env = AsteroidEnv(render_mode=True)
model = PPO.load("asteroids_model")

obs, _ = env.reset()
total_reward = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

    env.render()
    time.sleep(0.03)

    if done:
        print(f"Game Over! Total Reward: {total_reward:.2f}, Steps survived: {env.alive_steps}")
        time.sleep(1)
        obs, _ = env.reset()
        total_reward = 0
