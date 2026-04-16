"""
Script to save camera data from EasyCarla-RL environment as GIF animations
Improved version: Collect images first, then create GIFs
With autonomous driving for ego vehicle
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
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def collect_images_for_gif(params, max_steps=500, max_episodes=1):
    """
    Collect images from the environment without creating GIFs in real-time
    With ego vehicle in autonomous mode
    """
    # Create output directory if it doesn't exist
    output_dir = "/data2/zk/EasyCarla-RL/pictures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temporary directory for image frames
    temp_dir = os.path.join(output_dir, "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create the environment
    env = gym.make('carla-v0', params=params)
    
    collected_frames = []
    
    try:
        episode_count = 0
        total_frame_count = 0
        
        while episode_count < max_episodes:
            obs, _ = env.reset()
            # Enable autopilot for ego vehicle to drive autonomously
            env.unwrapped.ego.set_autopilot(True)
            done = False
            step_count = 0
            
            logger.info(f"Collecting frames for episode {episode_count + 1}")
            
            while not done and step_count < max_steps:
                # Since ego is in autopilot mode, we don't need to provide actions
                # But we still need to call step to advance the simulation and get new observations
                obs, reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0])  # Dummy action since autopilot is enabled
                done = terminated or truncated
                
                # Extract camera images
                front_cam = obs['front_camera']  # Shape: (600, 800, 3)
                bev_cam = obs['bev_camera']      # Shape: (600, 800, 3)
                
                # Save frames temporarily
                front_frame_path = os.path.join(temp_dir, f"front_{total_frame_count:06d}.jpg")
                bev_frame_path = os.path.join(temp_dir, f"bev_{total_frame_count:06d}.jpg")
                
                # Convert RGB to BGR for OpenCV
                front_cam_bgr = cv2.cvtColor(front_cam, cv2.COLOR_RGB2BGR)
                bev_cam_bgr = cv2.cvtColor(bev_cam, cv2.COLOR_RGB2BGR)
                
                # Save individual frames
                cv2.imwrite(front_frame_path, front_cam_bgr)
                cv2.imwrite(bev_frame_path, bev_cam_bgr)
                
                collected_frames.append((front_frame_path, bev_frame_path))
                
                if step_count % 100 == 0:
                    logger.info(f"Collected {step_count} frames, camera shapes: Front={front_cam.shape}, BEV={bev_cam.shape}")
                
                step_count += 1
                total_frame_count += 1
            
            logger.info(f"Finished collecting frames for episode {episode_count + 1}, collected {step_count} frames")
            episode_count += 1
    
    finally:
        env.close()
    
    logger.info(f"Total collected frames: {len(collected_frames)}")
    return collected_frames, temp_dir, output_dir

def create_gifs_from_collected_frames(collected_frames, output_dir, gif_filename_prefix="carla_cameras"):
    """
    Create GIFs from collected image frames
    """
    if not collected_frames:
        logger.warning("No frames collected, skipping GIF creation")
        return

    # Get frame dimensions from the first frame
    first_front_frame = cv2.imread(collected_frames[0][0])
    if first_front_frame is None:
        logger.error("Could not read first frame, aborting GIF creation")
        return
    
    frame_height, frame_width = first_front_frame.shape[:2]
    
    # Prepare lists to hold PIL Images for GIF creation
    front_frames = []
    bev_frames = []
    
    # Load all frames into memory
    for i, (front_path, bev_path) in enumerate(collected_frames):
        front_img = cv2.imread(front_path)
        bev_img = cv2.imread(bev_path)
        
        if front_img is not None and bev_img is not None:
            # Convert BGR to RGB for PIL
            front_rgb = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
            bev_rgb = cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Images
            front_pil = Image.fromarray(front_rgb)
            bev_pil = Image.fromarray(bev_rgb)
            
            front_frames.append(front_pil)
            bev_frames.append(bev_pil)
            
            if i % 100 == 0:
                logger.info(f"Loaded {i+1}/{len(collected_frames)} frames for GIF creation")
        else:
            logger.warning(f"Could not read frame {i}, skipping")
    
    # Create GIFs
    gif_duration = 100  # milliseconds between frames (adjust as needed)
    
    front_gif_path = os.path.join(output_dir, f"{gif_filename_prefix}_front.gif")
    bev_gif_path = os.path.join(output_dir, f"{gif_filename_prefix}_bev.gif")
    
    if front_frames:
        # Create front camera GIF
        front_frames[0].save(
            front_gif_path,
            save_all=True,
            append_images=front_frames[1:],
            duration=gif_duration,
            loop=0  # 0 means infinite loop
        )
        logger.info(f"Front camera GIF created: {front_gif_path}")
    
    if bev_frames:
        # Create BEV camera GIF
        bev_frames[0].save(
            bev_gif_path,
            save_all=True,
            append_images=bev_frames[1:],
            duration=gif_duration,
            loop=0  # 0 means infinite loop
        )
        logger.info(f"BEV camera GIF created: {bev_gif_path}")

def cleanup_temp_frames(temp_dir):
    """
    Clean up temporary frame files
    """
    import shutil
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Temporary frames directory cleaned up: {temp_dir}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary frames directory: {e}")

def test_camera_data_retrieval(params, num_steps=10):
    """
    Test function to verify camera data is being retrieved correctly
    """
    env = gym.make('carla-v0', params=params)
    
    try:
        obs, _ = env.reset()
        # Enable autopilot for testing
        env.unwrapped.ego.set_autopilot(True)
        logger.info(f"Initial observation keys: {list(obs.keys())}")
        logger.info(f"Front camera shape: {obs['front_camera'].shape}")
        logger.info(f"BEV camera shape: {obs['bev_camera'].shape}")
        
        for step in range(num_steps):
            # Dummy action since autopilot is enabled
            obs, reward, terminated, truncated, info = env.step([0.0, 0.0, 0.0])
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
        'max_time_episode': 2000,
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
    
    # Then, collect images and create GIFs separately
    logger.info("Collecting camera data as images...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collected_frames, temp_dir, output_dir = collect_images_for_gif(
        params, 
        max_steps=500
    )
    
    if collected_frames:
        logger.info("Creating GIFs from collected images...")
        create_gifs_from_collected_frames(
            collected_frames, 
            output_dir, 
            gif_filename_prefix=f"carla_record_{timestamp}"
        )
        
        # Optionally clean up temporary frames (comment out if you want to keep them)
        logger.info("Cleaning up temporary frames...")
        cleanup_temp_frames(temp_dir)
    
    logger.info("Done!")