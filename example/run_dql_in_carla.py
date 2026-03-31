# -*- coding: utf-8 -*-
"""
Author: SilverWings
GitHub: https://github.com/silverwingsbot

Simple example: Run a trained Diffusion_QL model in easycarla
"""

#import gym
import gymnasium as gym
import easycarla
import numpy as np
import torch
import os
from agents.ql_diffusion import Diffusion_QL

# ===================== Helper Functions =====================
def convert_obs_dict_to_vector(obs_dict):
    """Convert observation dictionary to a flattened state vector."""
    return np.concatenate([
        obs_dict['ego_state'],        # 9 dimensions
        obs_dict['lane_info'],        # 2 dimensions
        obs_dict['lidar'],             # 240 dimensions
        obs_dict['nearby_vehicles'],   # 20 dimensions
        obs_dict['waypoints']          # 36 dimensions
    ]).astype(np.float32)

# ===================== Environment Configuration =====================
carla_params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'dt': 0.1,  # time interval between two frames
    'ego_vehicle_filter': 'vehicle.tesla.model3',  # filter for defining ego vehicle
    'surrounding_vehicle_spawned_randomly': True, # Whether surrounding vehicles are spawned randomly (True) or set manually (False)
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypoints': 12,  # maximum number of waypoints
    'visualize_waypoints': True,  # Whether to visualize waypoints (default: True)
    'desired_speed': 8,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'view_mode' : 'top',  # 'top' for bird's-eye view, 'follow' for third-person view
    'traffic': 'off',  # 'on' for normal traffic lights, 'off' for always green and frozen
    'lidar_max_range': 50.0,  # Maximum LIDAR perception range (meters)
    'max_nearby_vehicles': 5,  # Maximum number of nearby vehicles to observe
}

# ===================== Initialize Environment =====================
env = gym.make('carla-v0', params=carla_params)

# ===================== Initialize Model =====================
state_dim = 307
action_dim = 3
max_action = 1.0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Diffusion_QL(
    state_dim=state_dim,
    action_dim=action_dim,
    max_action=max_action,
    device=device,
    discount=0.99,
    tau=0.005,
    eta=0.01,
    beta_schedule='vp',
    n_timesteps=5
)

# ===================== Load Pretrained Model =====================
model_id = 200  # Model checkpoint ID to load
save_path = './params_dql'  # Model checkpoint directory
model.load_model(save_path, id=model_id)
print(f"Successfully loaded model ID {model_id}")

# ===================== Run One Episode =====================
obs = env.reset()
done = False
step = 0
episode_reward = 0.0

while not done:
    obs_vec = convert_obs_dict_to_vector(obs)
    action = model.sample_action(obs_vec)

    try:
        next_obs, reward, cost, done, info = env.step(action)
    except Exception as e:
        print(f"[Error] Carla step failed: {e}")
        obs = env.reset()
        continue

    obs = next_obs
    episode_reward += reward
    step += 1

    # Optional: add a delay for better visualization
    # time.sleep(0.05)

print(f"Episode finished. Total reward: {episode_reward:.2f}, Total steps: {step}")