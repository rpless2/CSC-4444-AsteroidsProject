from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from asteroids_env import AsteroidEnv

train_env = AsteroidEnv(render_mode=False)
eval_env = AsteroidEnv(render_mode=False)

eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5000,
                             deterministic=True, render=False)



model = PPO("MlpPolicy", train_env, verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01)

print("Training model...")
model.learn(total_timesteps=200000, callback=eval_callback)
model.save("asteroids_model")
print("Training done!")
#env = AsteroidEnv(render_mode=False)

train_env.close()
eval_env.close()