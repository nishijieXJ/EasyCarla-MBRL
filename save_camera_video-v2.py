"""
Script to save camera data from EasyCarla-RL environment as videos
Improved version: Collect images first, then create videos
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

def collect_images_for_video(params, max_steps=500, max_episodes=1):
    """
    Collect images from the environment without creating videos in real-time
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
            done = False
            step_count = 0
            
            logger.info(f"Collecting frames for episode {episode_count + 1}")
            
            while not done and step_count < max_steps:
                # Simple random action for demonstration
                action = [random.uniform(0.0, 1.0), random.uniform(-0.6, 0.6), random.uniform(0.0, 0.3)]
                
                obs, reward, terminated, truncated, info = env.step(action)
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

def create_videos_from_collected_frames(collected_frames, output_dir, video_filename_prefix="carla_cameras"):
    """
    Create videos from collected image frames
    """
    if not collected_frames:
        logger.warning("No frames collected, skipping video creation")
        return

    # Get frame dimensions from the first frame
    first_front_frame = cv2.imread(collected_frames[0][0])
    if first_front_frame is None:
        logger.error("Could not read first frame, aborting video creation")
        return
    
    frame_height, frame_width = first_front_frame.shape[:2]
    
    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    front_video_path = os.path.join(output_dir, f"{video_filename_prefix}_front.mp4")
    bev_video_path = os.path.join(output_dir, f"{video_filename_prefix}_bev.mp4")
    
    front_out = cv2.VideoWriter(front_video_path, fourcc, 10.0, (frame_width, frame_height))
    bev_out = cv2.VideoWriter(bev_video_path, fourcc, 10.0, (frame_width, frame_height))
    
    logger.info(f"Creating videos: {front_video_path}, {bev_video_path}")
    
    try:
        # Write frames to videos
        for i, (front_path, bev_path) in enumerate(collected_frames):
            front_frame = cv2.imread(front_path)
            bev_frame = cv2.imread(bev_path)
            
            if front_frame is not None and bev_frame is not None:
                front_out.write(front_frame)
                bev_out.write(bev_frame)
                
                if i % 100 == 0:
                    logger.info(f"Wrote {i+1}/{len(collected_frames)} frames to videos")
            else:
                logger.warning(f"Could not read frame {i}, skipping")
    
    finally:
        # Release video writers
        front_out.release()
        bev_out.release()
    
    logger.info(f"Videos created successfully:")
    logger.info(f"Front camera video: {front_video_path}")
    logger.info(f"BEV camera video: {bev_video_path}")

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
    
    # Then, collect images and create videos separately
    logger.info("Collecting camera data as images...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collected_frames, temp_dir, output_dir = collect_images_for_video(
        params, 
        max_steps=500
    )
    
    if collected_frames:
        logger.info("Creating videos from collected images...")
        create_videos_from_collected_frames(
            collected_frames, 
            output_dir, 
            video_filename_prefix=f"carla_record_{timestamp}"
        )
        
        # Optionally clean up temporary frames (comment out if you want to keep them)
        logger.info("Cleaning up temporary frames...")
        cleanup_temp_frames(temp_dir)
    
    logger.info("Done!")