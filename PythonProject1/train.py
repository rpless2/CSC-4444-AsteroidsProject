from stable_baselines3 import PPO
from asteroids_env import AsteroidEnv

env = AsteroidEnv(render_mode=False)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("asteroids_model")
env.close()