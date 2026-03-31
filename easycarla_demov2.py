"""
Author: SilverWings
GitHub: https://github.com/silverwingsbot

This script provides a minimal demo to interact with the EasyCarla-RL environment.
It follows the standard Gym interface (reset, step) and demonstrates basic environment usage.

"""
import gymnasium as gym
from gymnasium import spaces
import easycarla
import carla
import random
import numpy as np
import logging
import sys

# --- 1. 配置 logging ---
def setup_logger(log_file_path="demo.log"):
    """
    设置日志记录器，将日志同时输出到文件和控制台。
    """
    # 创建一个顶级的日志记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置最低日志级别

    # 防止重复添加处理器
    if logger.handlers:
        logger.handlers.clear()

    # 创建一个处理器，用于写入日志文件
    file_handler = logging.FileHandler(log_file_path, mode='w') # 'w' 会覆盖旧文件，'a' 会追加
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 创建一个处理器，用于输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 将处理器添加到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 创建日志记录器实例
logger = setup_logger("easycarla_demo.log")

# --- 2. 将 print 替换为 logger.info ---
def get_action(env, obs):
    """Randomly choose either a simple manual action or an autopilot action."""
    p = random.random()
    if p < 0.5:
        # Use autopilot (Expert mode)
        env.ego.set_autopilot(True)
        control = env.ego.get_control()
        action = [control.throttle, control.steer, control.brake]
        logger.debug("Using autopilot action") # 可以记录更多信息
    else:
        # Use random action (Novice mode)
        env.ego.set_autopilot(False)
        throttle = random.uniform(0.0, 1.0)
        steer = random.uniform(-0.6, 0.6)
        brake = random.uniform(0.0, 0.3)
        action = [throttle, steer, brake]
        logger.debug("Using random action") # 可以记录更多信息
    return action

# Configure environment parameters
params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'dt': 0.1,
    'ego_vehicle_filter': 'vehicle.tesla.model3',
    'surrounding_vehicle_spawned_randomly': True,
    'port': 2000,
    'town': 'Town03',
    'max_time_episode': 1000,
    'max_waypoints': 12,
    'visualize_waypoints': True,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'view_mode' : 'top',
    'traffic': 'off',
    'lidar_max_range': 50.0,
    'max_nearby_vehicles': 5,
    'sync_mode': True,
    'delta_seconds': 0.1,
    'max_steps': 1000, # 已添加
}

logger.info("Starting EasyCarla Demo with Camera Support...")

# Create the environment
logger.info("Creating the environment with camera support...")
env = gym.make('carla-v0', params=params)

# Interact with the environment
try:
    for episode in range(5):  # Run 5 episodes
        logger.info(f"--- Starting Episode {episode + 1} ---")
        obs, _ = env.reset()
        done = False
        total_reward = 0

        step_count = 0
        while not done and step_count < 100:  # Limit steps for demo
            action = get_action(env, obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 使用 logger 记录信息，而不是 print
            logger.info(f"Step: {env.time_step}, Reward: {reward:.2f}, Done: {done}, Info: {info}")

            # Access and log camera data from observation
            front_cam = obs['front_camera']  # Shape: (600, 800, 3)
            bev_cam = obs['bev_camera']      # Shape: (600, 800, 3)
            
            if step_count % 20 == 0:  # Log camera info every 20 steps
                logger.info(f"Front camera shape: {front_cam.shape}, BEV camera shape: {bev_cam.shape}")
                logger.info(f"Front camera sample pixel (0,0): {front_cam[0, 0, :]}")
                logger.info(f"BEV camera sample pixel (0,0): {bev_cam[0, 0, :]}")

            step_count += 1
            total_reward += reward

        logger.info(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}")
        logger.info(f"Collision: {info.get('is_collision', 'N/A')}, Off Road: {info.get('is_off_road', 'N/A')}")

except KeyboardInterrupt:
    logger.warning("Demo interrupted by user.")
except Exception as e:
    logger.error(f"An error occurred: {e}", exc_info=True) # exc_info=True 会打印完整的堆栈跟踪
finally:
    env.close()
    logger.info("Environment closed.")
    logger.info("Demo finished.")