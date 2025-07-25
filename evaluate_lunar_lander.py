import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create the environment
env = gym.make('LunarLander-v3', render_mode='rgb_array')

# Wrap the environment to record a video
video_folder = "lunar_lander_videos/"
os.makedirs(video_folder, exist_ok=True)
# Record a video every 5 episodes
env = gym.wrappers.RecordVideo(env, video_folder, name_prefix="lunar-lander-ppo", episode_trigger=lambda e: e % 5 == 0)

# Load the trained model
try:
    model = PPO.load("ppo_lunar_lander", env=env)
except FileNotFoundError:
    print("Error: ppo_lunar_lander.zip not found. Please run train_lunar_lander.py first.")
    exit()

# Evaluate the agent for 10 episodes
for episode in range(10):
    obs, _ = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    while not terminated and not truncated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Close the environment
env.close()

print(f"Evaluation finished. Videos saved in {video_folder}")
