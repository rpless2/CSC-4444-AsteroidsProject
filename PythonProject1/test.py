from stable_baselines3 import PPO
from asteroids_env import AsteroidEnv
import time

env = AsteroidEnv(render_mode=True)
model = PPO.load("asteroids_model")

obs, _ = env.reset()

while True:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    env.render()
    time.sleep(0.03)

    if done:
        obs, _ = env.reset()
