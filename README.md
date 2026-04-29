# PPO Asteroid Agent

## How to setup and run Environment/Code
- Clone the repo to IDE of choice `git clone https://github.com/rpless2/CSC-4444-AsteroidsProject`
- cd into the proper directory `cd PythonProject1`
- Install any dependency if needed `pip install stable-baselines3 gymnasium pygame numpy torch`
- Check if you have any trained data labeled asteroids_model_xxx.zip
- If no data run `python train.py`
- Once you have data run `python test.py`
- If you want to run certain data modify the code in test.py line 6 to `model = PPO.load("asteroids_model_xxx")`

## Overview
The model for this Asteroid Project is a reinforcement learning (RL) agent that is trained to play Asteroids on its own.
The agent uses PPO or Proximal Policy Optimization to learn which actions will help it survive longer to achieve a higher score/reward.
At every step the agent will look at the game state including the ship's position, velocity, direction, three nearby asteroids, nearby asteroid movement, and bullet count.
From that information it will have five actions to choose from: doing nothing, turning left or right, thrusting, and shooting.
The agent will recieve rewards for surviving, destroying asteroids, aiming properly before shooting, and moving away from danger.
The agent will recieve penalties for crashing, shooting randomly, or moving closer to nearby asteroids.
Over the course of training the model improved it self noticeably by adjusting its behavior in how it moved around and what asteroids it focused on.
The more data trained versions of the agent survived longer and showed a more effective way to play the game compared to the low data trained agents.
<img width="998" height="786" alt="Screenshot 2026-04-28 235429" src="https://github.com/user-attachments/assets/e63e3eb9-80c5-4f65-8000-2c0650408472" /><img width="996" height="778" alt="Screenshot 2026-04-28 235515" src="https://github.com/user-attachments/assets/1fcf6ac2-9fd4-4cee-9b9d-e5b50456bffa" />
