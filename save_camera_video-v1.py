"""
Script to save camera data from EasyCarla-RL environment as videos
"""
import os
import gymnasium as gym
import numpy as np
import cv2
import easycarla
import carla
import random
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_camera_data_as_video(params, video_filename_prefix="carla_cameras", max_steps=1000):
    """
    Save camera data from EasyCarla-RL environment as videos
    """
    # Create output directory if it doesn't exist
    output_dir = "/data2/zk/EasyCarla-RL/pictures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the environment
    env = gym.make('carla-v0', params=params)
    
    obs = None  # Initialize obs variable
    try:
        # Initialize video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        front_video_path = os.path.join(output_dir, f"{video_filename_prefix}_front.mp4")
        bev_video_path = os.path.join(output_dir, f"{video_filename_prefix}_bev.mp4")
        
        # Get frame dimensions from camera data (after environment is initialized)
        obs, _ = env.reset()
        frame_height, frame_width = obs['front_camera'].shape[:2]
        
        # Initialize video writers
        front_out = cv2.VideoWriter(front_video_path, fourcc, 10.0, (frame_width, frame_height))
        bev_out = cv2.VideoWriter(bev_video_path, fourcc, 10.0, (frame_width, frame_height))
        
        logger.info(f"Created video files: {front_video_path}, {bev_video_path}")
        logger.info(f"Frame size: {frame_width}x{frame_height}")
        
        episode_count = 0
        max_episodes = 1  # Just record one episode for demonstration
        
        while episode_count < max_episodes:
            obs, _ = env.reset()
            done = False
            step_count = 0
            
            logger.info(f"Starting episode {episode_count + 1}")
            
            while not done and step_count < max_steps:
                # Simple random action for demonstration
                action = [random.uniform(0.0, 1.0), random.uniform(-0.6, 0.6), random.uniform(0.0, 0.3)]
                
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Extract camera images
                front_cam = obs['front_camera']  # Shape: (600, 800, 3)
                bev_cam = obs['bev_camera']      # Shape: (600, 800, 3)
                
                # Convert RGB to BGR for OpenCV
                front_cam_bgr = cv2.cvtColor(front_cam, cv2.COLOR_RGB2BGR)
                bev_cam_bgr = cv2.cvtColor(bev_cam, cv2.COLOR_RGB2BGR)
                
                # Write frames to video
                front_out.write(front_cam_bgr)
                bev_out.write(bev_cam_bgr)
                
                if step_count % 100 == 0:
                    logger.info(f"Recorded {step_count} steps, camera shapes: Front={front_cam.shape}, BEV={bev_cam.shape}")
                
                step_count += 1
            
            logger.info(f"Finished episode {episode_count + 1}, recorded {step_count} steps")
            episode_count += 1
    
    except Exception as e:
        logger.error(f"An error occurred during video recording: {e}")
        # Ensure we still release the video writers even if an error occurs
        if 'front_out' in locals():
            front_out.release()
        if 'bev_out' in locals():
            bev_out.release()
        raise e  # Re-raise the exception to stop execution properly
    finally:
        # Release everything
        if 'front_out' in locals() and front_out:
            front_out.release()
        if 'bev_out' in locals() and bev_out:
            bev_out.release()
        if env:
            env.close()
        
        logger.info(f"Videos saved to: {output_dir}")
        logger.info(f"Front camera video: {front_video_path}")
        logger.info(f"BEV camera video: {bev_video_path}")

def test_camera_data_retrieval(params, num_steps=10):
    """
    Test function to verify camera data is being retrieved correctly
    """
    env = gym.make('carla-v0', params=params)
    
    try:
        obs, _ = env.reset()
        logger.info(f"Initial observation keys: {list(obs.keys())}")
        logger.info(f"Front camera shape: {obs['front_camera'].shape}")
        logger.info(f"BEV camera shape: {obs['bev_camera'].shape}")
        
        for step in range(num_steps):
            action = [random.uniform(0.0, 1.0), random.uniform(-0.6, 0.6), random.uniform(0.0, 0.3)]
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Verify camera data still present
            front_shape = obs['front_camera'].shape
            bev_shape = obs['bev_camera'].shape
            
            if step % 5 == 0:
                logger.info(f"Step {step}: Front shape={front_shape}, BEV shape={bev_shape}")
                
                # Show sample pixel values
                logger.info(f"Sample front pixel [0,0]: {obs['front_camera'][0,0,:]}")
                logger.info(f"Sample BEV pixel [0,0]: {obs['bev_camera'][0,0,:]}")
                
            if done:
                logger.info(f"Episode ended at step {step}")
                break
                
    finally:
        env.close()

if __name__ == "__main__":
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
        'max_steps': 1000,
    }
    
    # First, test if camera data is being retrieved correctly
    logger.info("Testing camera data retrieval...")
    test_camera_data_retrieval(params, num_steps=20)
    
    # Then, save camera data as videos
    logger.info("Saving camera data as videos...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_camera_data_as_video(params, video_filename_prefix=f"carla_record_{timestamp}", max_steps=500)
    
    logger.info("Done!")