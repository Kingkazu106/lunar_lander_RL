import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create log dir
log_dir = "lunar_lander_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

# Create the vectorized environment
# LunarLander is a simple environment, so we can parallelize it
env = make_vec_env('LunarLander-v3', n_envs=16)

# PPO model
# The default MlpPolicy is suitable for this environment.
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1, 
    tensorboard_log=log_dir,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    learning_rate=0.0001
)

# Train the model
# 500,000 timesteps for better performance.
model.learn(total_timesteps=500000)

# Save the model
model.save("ppo_lunar_lander")

# Close the environment
env.close()

print("Training finished and model saved as ppo_lunar_lander.zip")
