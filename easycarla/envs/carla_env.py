# -*- coding: utf-8 -*-
"""Author: SilverWings
GitHub: https://github.com/silverwingsbot
"""

from __future__ import division
import numpy as np
import random
import time
import gymnasium as gym
from gymnasium import spaces
import random
import numpy as np
import carla
import math
import collections
import weakref # 使用弱引用来管理 Actor，有助于垃圾回收
import re # Add this import if it doesn't exist yet
import os
from datetime import datetime

# --- ADD THIS FUNCTION TO carla_env.py AT THE TOP LEVEL ---
def get_actor_blueprints(world, filter, generation):
    """
    Helper function to get actor blueprints, copied from manual_control script.
    """
    bps = world.get_blueprint_library().filter(filter)
    if generation.lower() == "all":
        return bps
    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps
    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print(" Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print(" Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

# --- END OF FUNCTION TO ADD ---

def get_current_speed(vehicle):
    """Calculate the current speed of the vehicle in km/h."""
    vel = vehicle.get_velocity()
    return 3.6 * np.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def get_lane_type(waypoint):
    """Get the type of the lane at the given waypoint."""
    lane_type = waypoint.lane_type
    if lane_type == carla.LaneType.Driving:
        return 1
    else:
        return 0

def get_lane_direction(waypoint, next_waypoints):
    """Get the direction of the lane change."""
    if len(next_waypoints) < 2:
        return 0  # No direction can be determined

    current_location = waypoint.transform.location
    next_location = next_waypoints[1].transform.location

    direction_vector = next_location - current_location
    vehicle_forward = waypoint.transform.get_forward_vector()

    dot_product = direction_vector.x * vehicle_forward.x + direction_vector.y * vehicle_forward.y
    if dot_product > 0:
        return 1  # Forward
    else:
        return -1  # Backward (or reverse turn)

def get_lane_width(waypoint):
    """Get the width of the lane at the given waypoint."""
    return waypoint.lane_width

def get_distance_along_route(route, current_location):
    """Calculate the distance along the route from the start."""
    distance = 0.0
    prev_location = route[0][0].transform.location
    for wp, _ in route:
        location = wp.transform.location
        distance += location.distance(prev_location)
        if location.distance(current_location) < 2.0:  # Assuming we are on this segment
            break
        prev_location = location
    return distance

def get_distance_from_lane_center(waypoint, current_location):
    """Get the signed distance from the center of the lane."""
    # Get the right vector of the lane (perpendicular to forward vector, pointing right)
    right_vector = waypoint.transform.get_right_vector()
    # Vector from lane center to vehicle
    center_to_vehicle = current_location - waypoint.transform.location
    # Project the vector onto the right vector to get the signed distance
    signed_distance = center_to_vehicle.x * right_vector.x + center_to_vehicle.y * right_vector.y
    lane_width = waypoint.lane_width
    # Normalize by half lane width
    if lane_width > 0:
        normalized_distance = abs(signed_distance) / (lane_width / 2)
    else:
        normalized_distance = 0.0
    # Return positive if on the right side, negative if on the left side
    return signed_distance, normalized_distance

def get_waypoint_list(ego, world_map, max_waypoints=20):
    """Get a list of waypoints ahead of the ego vehicle."""
    current_waypoint = world_map.get_waypoint(ego.get_location())
    next_waypoints = current_waypoint.next(2.0)  # Start with immediate next waypoint
    
    if not next_waypoints:
        return []

    route = [current_waypoint]
    
    # Follow the lane until we have enough waypoints or run out of road
    while len(route) < max_waypoints:
        current_wp = route[-1]
        next_wps = current_wp.next(2.0)  # Get next waypoint 2m ahead
        
        if not next_wps:
            break  # End of road or no further path
        
        # Choose the best next waypoint (usually the first one)
        closest_next_wp = min(next_wps, key=lambda x: x.transform.location.distance(current_wp.transform.location))
        
        # Check for lane changes or junctions if needed, for now just follow the closest
        route.append(closest_next_wp)
        
    return route

def get_lane_info(ego, world_map):
    """Get information about the current lane."""
    current_waypoint = world_map.get_waypoint(ego.get_location())
    
    # Lane type
    lane_type = get_lane_type(current_waypoint)
    
    # Lane direction
    next_waypoints = current_waypoint.next(5.0)
    lane_direction = get_lane_direction(current_waypoint, next_waypoints)
    
    # Distance from lane center
    current_location = ego.get_location()
    _, dist_from_center_norm = get_distance_from_lane_center(current_waypoint, current_location)
    
    # Lane width
    lane_width = get_lane_width(current_waypoint)
    
    return lane_type, lane_direction, dist_from_center_norm, lane_width

class CarlaEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, params):
        # --- Store params ---
        self.params = params
        self.max_steps = params.get("max_steps", 1000)
        self.number_of_vehicles = params['number_of_vehicles']
        self.number_of_walkers = params['number_of_walkers']
        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.max_waypoints = params['max_waypoints']
        self.visualize_waypoints = params['visualize_waypoints']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.view_mode = params['view_mode']
        self.traffic = params['traffic']
        self.lidar_max_range = params['lidar_max_range']
        self.max_nearby_vehicles = params['max_nearby_vehicles']
        self.surrounding_vehicle_spawned_randomly = params['surrounding_vehicle_spawned_randomly']
        # Keep backward compatibility: trailer is enabled by default.
        self.enable_trailer = params.get('enable_trailer', True)
        # Spawn diagnostics are non-intrusive and enabled by default for troubleshooting.
        self.spawn_diagnostics = params.get('spawn_diagnostics', True)
        # Debug image dump on each reset for quick visual verification.
        self.dump_reset_images = params.get('dump_reset_images', False)
        self.dump_image_dir = params.get('dump_image_dir', '/data2/zk/EasyCarla-RL/pictures')
        # Task mode:
        # - legacy: keep current random-driving behavior
        # - multitask: enable lightweight task-conditioned training
        self.task_mode = params.get('task_mode', 'legacy')
        self.default_task_type = params.get('default_task_type', 'navigation')
        self.default_task_difficulty = params.get('default_task_difficulty', 'easy')
        self.current_task = {
            'task_type': self.default_task_type,
            'difficulty': self.default_task_difficulty
        }
        self._task_success = False
        self._task_failure_reason = ''
        self._episode_start_location = None
        self._episode_start_yaw = 0.0
        self._target_distance_m = 30.0

        # --- Observation and Action Spaces ---
        self.observation_space = spaces.Dict({
                'lidar': spaces.Box(low=0.0, high=1.0, shape=(240,), dtype=np.float32),
                'ego_state': spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
                'nearby_vehicles': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.max_nearby_vehicles * 4,), dtype=np.float32),
                'waypoints': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.max_waypoints * 3,), dtype=np.float32),
                'lane_info': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                'front_camera': spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),  # Front camera image
                'bev_camera': spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),    # BEV camera image
            })
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )

        # --- CARLA Connection ---
        print('Connecting to Carla server...')
        self.client = carla.Client('localhost', params['port'])
        self.client.set_timeout(10.0)

        # 兼容两种模式：
        # 1) town/umap 模式（保留原有 Town01~Town10 调用方式）
        # 2) xodr 模式（用于 Roundabouts 等仅 OpenDRIVE 快速验证）
        self.map_mode = params.get('map_mode', 'town')  # 'town' | 'xodr'
        if self.map_mode == 'xodr':
            xodr_path = params.get('xodr_path', None)
            if xodr_path is None:
                xodr_root = params.get('xodr_root', '/data1/zk/carla/Unreal/CarlaUE4/Content/Carla/Maps/Roundabouts')
                map_id = params.get('map_id', '20m')
                xodr_path = f"{xodr_root}/{map_id}/OpenDrive/{map_id}.xodr"

            xodr_abs_path = os.path.abspath(xodr_path)
            xodr_exists = os.path.isfile(xodr_abs_path)
            print(f"[MAP] mode=xodr, xodr_path={xodr_path}")
            print(f"[MAP] resolved_xodr_abs_path={xodr_abs_path}, exists={xodr_exists}")
            if not xodr_exists:
                raise FileNotFoundError(f"XODR file not found: {xodr_abs_path}")

            with open(xodr_abs_path, 'r', encoding='utf-8') as f:
                odr_data = f.read()
            self.world = self.client.generate_opendrive_world(odr_data)
        else:
            town_name = params.get('town', 'Town03')
            print(f"[MAP] mode=town, town={town_name}")
            self.world = self.client.load_world(town_name)

        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        self.map = self.world.get_map()
        print(f"Connection established! current_map={self.map.name}")

        # --- Initialize Variables ---
        self._setup_world_and_actors_lists()
        self._setup_blueprints_and_sensors()
        self._setup_internal_state()
        # Inside __init__ method of CarlaEnv class
        # ... other initializations ...
        # Store the actor generation setting (copied from manual_control logic)
        self._actor_generation = "2"  # Or get it from an argument if you want flexibility, e.g., args.generation
        self.ego = None
        self.attached_trailer = None 


    def _setup_world_and_actors_lists(self):
        """Initialize lists to hold strong references to actors."""
        self.spawned_vehicles = []  # List of surrounding vehicles
        self.spawned_walkers = []   # List of walkers
        self.all_sensors = []       # List of all sensors (collision, lidar, etc.)
        self.all_actors = []        # General list for any other actors that need cleanup
        # Store spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

    def _setup_blueprints_and_sensors(self):
        """Setup blueprints and sensor attributes."""
        self.ego_bp = self._create_vehicle_blueprint(self.params['ego_vehicle_filter'], color='255,0,0')
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Setup LiDAR
        self.lidar_height = 0.8
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_bp.set_attribute('channels', '1')
        self.lidar_bp.set_attribute('range', '50')
        self.lidar_bp.set_attribute('rotation_frequency', '10')
        self.lidar_bp.set_attribute('points_per_second', '10000')
        self.lidar_bp.set_attribute('upper_fov', '0')
        self.lidar_bp.set_attribute('lower_fov', '0')# ... existing code ...
        
        # Setup RGB Camera
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '800')  # Width in pixels
        self.camera_bp.set_attribute('image_size_y', '600')  # Height in pixels
        self.camera_bp.set_attribute('fov', '110')  # Field of view in degrees
        # Front-facing camera transform
        # Move camera slightly forward and upward to reduce ego body occlusion.
        front_camera_x = self.params.get('front_camera_x', 2.8)
        front_camera_z = self.params.get('front_camera_z', 1.9)
        front_camera_pitch = self.params.get('front_camera_pitch', -2.0)
        self.front_camera_trans = carla.Transform(
            carla.Location(x=front_camera_x, z=front_camera_z),
            carla.Rotation(pitch=front_camera_pitch)
        )
        
        # BEV (Bird's Eye View) camera transform
        self.bev_camera_trans = carla.Transform(
            carla.Location(x=0.0, z=40.0),  # High above the vehicle
            carla.Rotation(pitch=-90)       # Pointing downward
        )

    def _camera_handler(self, image, sensor_type):
        """Generic handler for camera images."""
        try:
            # Convert CARLA image to numpy array
            img_array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img_array = np.reshape(img_array, (image.height, image.width, 4))  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel, keep RGB

            # Store image based on sensor type
            if sensor_type == 'front':
                self.front_camera_image = img_array
            elif sensor_type == 'bev':
                self.bev_camera_image = img_array
        except Exception as e:
            print(f"Error processing camera image for {sensor_type} camera: {e}")

    def _setup_internal_state(self):
        """Reset internal state variables."""
        self.ego = None
        self.collision_sensor = None
        self.lidar_sensor = None
        self.front_camera_sensor = None  # New front camera sensor
        self.bev_camera_sensor = None    # New BEV camera sensor
        self.lidar_data = None
        self.front_camera_image = None   # New front camera image data
        self.bev_camera_image = None     # New BEV camera image data
        self._is_collision = False
        self._is_off_road = False
        self.off_road_counter = 0
        self.collision_hist = []
        self.collision_hist_l = 1
        self.vehicle_polygons = []
        self.walker_polygons = []
        self.time_step = 0
        self.total_step = 0
        self.reset_step = 0
        self.sync_mode = self.params.get('sync_mode', False)
        self.delta_seconds = self.params.get('delta_seconds', self.dt)
        self._task_success = False
        self._task_failure_reason = ''
        self._episode_start_location = None
        self._episode_start_yaw = 0.0
        self._target_distance_m = 30.0

    def _normalize_yaw_delta(self, yaw_now, yaw_ref):
        """Return yaw delta in degrees within [-180, 180]."""
        delta = (yaw_now - yaw_ref + 180.0) % 360.0 - 180.0
        return delta

    def _resolve_task_request(self, options):
        """
        Resolve task request from reset options.
        Falls back to defaults and keeps backward compatibility.
        """
        if self.task_mode != 'multitask':
            self.current_task = {
                'task_type': self.default_task_type,
                'difficulty': self.default_task_difficulty
            }
            return self.current_task

        req = options if isinstance(options, dict) else {}
        task_type = str(req.get('task_id', req.get('task_type', self.default_task_type))).lower()
        difficulty = str(req.get('difficulty', self.default_task_difficulty)).lower()

        valid_task_types = {'left_turn', 'right_turn', 'navigation'}
        valid_difficulties = {'easy', 'medium', 'hard'}
        if task_type not in valid_task_types:
            task_type = self.default_task_type
        if difficulty not in valid_difficulties:
            difficulty = self.default_task_difficulty

        self.current_task = {'task_type': task_type, 'difficulty': difficulty}
        return self.current_task

    def _apply_task_profile(self):
        """
        Apply lightweight task profile to environment knobs.
        This is intentionally minimal and non-invasive.
        """
        task_type = self.current_task.get('task_type', 'navigation')
        difficulty = self.current_task.get('difficulty', 'easy')

        # Difficulty controls
        if difficulty == 'easy':
            self.number_of_vehicles = int(self.params.get('task_easy_num_vehicles', 30))
            self.max_time_episode = int(self.params.get('task_easy_max_time_episode', 800))
            self._target_distance_m = float(self.params.get('task_easy_target_distance_m', 25.0))
        elif difficulty == 'medium':
            self.number_of_vehicles = int(self.params.get('task_medium_num_vehicles', 60))
            self.max_time_episode = int(self.params.get('task_medium_max_time_episode', 1000))
            self._target_distance_m = float(self.params.get('task_medium_target_distance_m', 40.0))
        else:
            self.number_of_vehicles = int(self.params.get('task_hard_num_vehicles', 100))
            self.max_time_episode = int(self.params.get('task_hard_max_time_episode', 1200))
            self._target_distance_m = float(self.params.get('task_hard_target_distance_m', 60.0))

        # Task-specific target distance adjustment
        if task_type in ('left_turn', 'right_turn'):
            self._target_distance_m = max(15.0, 0.7 * self._target_distance_m)

    def _dump_reset_images(self, obs):
        """Dump one front/bev frame on each reset for map visual diagnostics."""
        if not self.dump_reset_images:
            return

        try:
            os.makedirs(self.dump_image_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            prefix = f"reset_{self.reset_step:05d}_{timestamp}"

            front = obs.get('front_camera', None)
            bev = obs.get('bev_camera', None)

            if front is not None and isinstance(front, np.ndarray) and front.ndim == 3:
                front_path = os.path.join(self.dump_image_dir, f"{prefix}_front.png")
                try:
                    from PIL import Image
                    Image.fromarray(front).save(front_path)
                    print(f">>> [RESET_IMG] saved front_camera: {front_path}")
                except Exception as e:
                    print(f">>> [RESET_IMG] failed to save front_camera: {e}")

            if bev is not None and isinstance(bev, np.ndarray) and bev.ndim == 3:
                bev_path = os.path.join(self.dump_image_dir, f"{prefix}_bev.png")
                try:
                    from PIL import Image
                    Image.fromarray(bev).save(bev_path)
                    print(f">>> [RESET_IMG] saved bev_camera: {bev_path}")
                except Exception as e:
                    print(f">>> [RESET_IMG] failed to save bev_camera: {e}")

        except Exception as e:
            print(f">>> [RESET_IMG] dump failed: {e}")

    def _clear_all_actors(self):
        """Clear all actors managed by this environment instance."""
        # Stop and destroy sensors first
        for sensor in self.all_sensors:
            try:
                if sensor and sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except Exception as e:
                print(f"Warning: Error destroying sensor {sensor.id if sensor else 'Unknown'}: {e}")
        self.all_sensors.clear()

        # Destroy ego vehicle
        if self.ego and self.ego.is_alive:
            try:
                self.ego.set_autopilot(False)
                #self.ego.destroy()
                if self.ego:
                    self.ego.destroy()
                    self.ego = None
                if self.attached_trailer: # Add these lines
                    self.attached_trailer.destroy()
                    self.attached_trailer = None
            except Exception as e:
                print(f"Warning: Error destroying ego vehicle {self.ego.id if self.ego else 'Unknown'}: {e}")
        self.ego = None # Ensure reference is cleared

        # Destroy surrounding vehicles
        for vehicle in self.spawned_vehicles:
            if vehicle and vehicle.is_alive:
                try:
                    vehicle.destroy()
                except Exception as e:
                    print(f"Warning: Error destroying vehicle {vehicle.id}: {e}")
        self.spawned_vehicles.clear()

        # Destroy walkers and their controllers
        for walker_controller, walker in self.spawned_walkers:
            if walker and walker.is_alive:
                try:
                    walker.destroy()
                except Exception as e:
                    print(f"Warning: Error destroying walker {walker.id}: {e}")
            if walker_controller and walker_controller.is_alive:
                try:
                    walker_controller.stop()
                    walker_controller.destroy()
                except Exception as e:
                    print(f"Warning: Error destroying walker controller {walker_controller.id}: {e}")
        self.spawned_walkers.clear()

        # Destroy any other general actors if needed
        for actor in self.all_actors:
            if actor and actor.is_alive:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Warning: Error destroying general actor {actor.id}: {e}")
        self.all_actors.clear()

    def reset(self, *, seed=None, options=None):
        print("\n>>> [RESET] Starting comprehensive reset...")
        resolved_task = self._resolve_task_request(options)
        self._apply_task_profile()
        print(f">>> [TASK] mode={self.task_mode}, task={resolved_task}")

        # 1. Thoroughly clean up ALL previous actors
        print(">>> [RESET] step1: Cleaning up old actors...")
        self._clear_all_actors()
        # Tick once to ensure destruction is processed by CARLA
        self.world.tick()
        print(">>> [RESET] step1: Cleanup done.")

        # 2. Re-initialize internal state
        print(">>> [RESET] step2: Resetting internal state...")
        self._setup_internal_state()
        # Re-fetch spawn points as they might have changed (though unlikely in static maps)
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        self.walker_spawn_points = []
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)
        print(">>> [RESET] step2: Internal state reset.")

        # 3. Spawn surrounding vehicles
        print(">>> [RESET] step3: Spawning surrounding vehicles...")
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        self.spawned_vehicles = []
        self.used_spawn_points = []
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
                if vehicle:
                    self.spawned_vehicles.append(vehicle)
                    self.used_spawn_points.append(spawn_point)
                    count -= 1
                    if count <= 0:
                        break
        print(f">>> [RESET] step3: Spawned {len(self.spawned_vehicles)} surrounding vehicles.")

        # 4. Spawn walkers
        print(">>> [RESET] step4: Spawning walkers...")
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        if count > 0:
            for spawn_point in self.walker_spawn_points:
                if self._try_spawn_random_walker_at(spawn_point):
                    count -= 1
                    if count <= 0:
                        break
            while count > 0: # Fallback for any remaining
                if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                    count -= 1
        print(f">>> [RESET] step4: Spawned {len(self.spawned_walkers)} walkers.")

        # 5. Update actor polygons BEFORE spawning ego
        print(">>> [RESET] step5: Updating actor polygons...")
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons = [vehicle_poly_dict]
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons = [walker_poly_dict]
        print(">>> [RESET] step5: Polygons updated.")

        # 6. Spawn ego vehicle
        # print(">>> [RESET] step6: Spawning ego vehicle...") #生成随机ego vehicle
        # ego_spawned = False
        # for attempt in range(self.max_ego_spawn_times + 1):
        #     print(f">>> [RESET] Attempt {attempt + 1}/{self.max_ego_spawn_times + 1} for ego spawn.")
        #     available_spawn_points = [
        #         sp for sp in self.vehicle_spawn_points if sp not in self.used_spawn_points
        #     ]
        #     if not available_spawn_points:
        #         available_spawn_points = self.vehicle_spawn_points # fallback

        #     transform = random.choice(available_spawn_points)
        #     if self._try_spawn_ego_vehicle_at(transform):
        #         print(">>> [RESET] Ego vehicle spawned successfully!")
        #         # Important: Tick once to ensure ego is fully initialized in the world
        #         # before attaching sensors and proceeding
        #         self.world.tick()
        #         ego_spawned = True
        #         break
        #     else:
        #         print(f">>> [RESET] Attempt {attempt + 1} failed.")

        # if not ego_spawned:
        #     raise RuntimeError("Failed to spawn ego vehicle after all attempts!")
        # 6. Spawn ego vehicle (Now using the multi-point retry logic again)
        # 6. Spawn ego vehicle (with debug info)
        print(">>> [RESET] step6: Spawning ego vehicle (semi-trailer)...")
        print(f">>> [DEBUG] Total available spawn points in map: {len(self.vehicle_spawn_points)}") # Add this line

        ego_spawned = False
        for attempt in range(self.max_ego_spawn_times + 1):
            print(f">>> [RESET] Attempt {attempt + 1}/{self.max_ego_spawn_times + 1} for ego spawn.")
            
            available_spawn_points = [
                sp for sp in self.vehicle_spawn_points if sp not in self.used_spawn_points
            ]
            print(f">>> [DEBUG] Available spawn points for this attempt: {len(available_spawn_points)}") # Add this line

            if not available_spawn_points:
                print(">>> [DEBUG] Out of spawn points. Reusing all spawn points.")
                available_spawn_points = self.vehicle_spawn_points # fallback

            if not available_spawn_points:
                print("CRITICAL ERROR: No spawn points available at all. Check map loading.")
                return self._get_obs(), {
                    'task_mode': self.task_mode,
                    'task_type': self.current_task.get('task_type', 'navigation'),
                    'task_difficulty': self.current_task.get('difficulty', 'easy'),
                    'task_success': False,
                    'task_failure_reason': 'no_spawn_points',
                }

            transform = random.choice(available_spawn_points)
            print(f">>> [DEBUG] Selected spawn transform: {transform}") # Add this line

            if self._try_spawn_ego_vehicle_at(transform):
                print(">>> [RESET] Ego vehicle (semi-trailer) spawned successfully!")
                self.world.tick()
                ego_spawned = True
                break
            else:
                print(f">>> [RESET] Attempt {attempt + 1} failed.")

        if not ego_spawned:
            print("CRITICAL ERROR: Could not spawn ego vehicle (semi-trailer) after all attempts!")
            return self._get_obs(), {
                'task_mode': self.task_mode,
                'task_type': self.current_task.get('task_type', 'navigation'),
                'task_difficulty': self.current_task.get('difficulty', 'easy'),
                'task_success': False,
                'task_failure_reason': 'ego_spawn_failed',
            }

        # Record episode reference pose for task progress/success checks
        ego_tf = self.ego.get_transform()
        self._episode_start_location = ego_tf.location
        self._episode_start_yaw = ego_tf.rotation.yaw

        # Spawn stability diagnostics (non-intrusive)
        if self.spawn_diagnostics and self.ego is not None:
            try:
                pre_tick_loc = self.ego.get_location()
                pre_tick_vel = self.ego.get_velocity()
                pre_wp = self.map.get_waypoint(pre_tick_loc, project_to_road=False)
                print(
                    f">>> [SPAWN_DIAG][PRE_TICK] ego_loc=({pre_tick_loc.x:.3f}, {pre_tick_loc.y:.3f}, {pre_tick_loc.z:.3f}), "
                    f"ego_speed={np.sqrt(pre_tick_vel.x**2 + pre_tick_vel.y**2 + pre_tick_vel.z**2):.3f}, "
                    f"wp_found={pre_wp is not None}, enable_trailer={self.enable_trailer}"
                )
                if self.attached_trailer is not None:
                    tr_loc = self.attached_trailer.get_location()
                    print(f">>> [SPAWN_DIAG][PRE_TICK] trailer_loc=({tr_loc.x:.3f}, {tr_loc.y:.3f}, {tr_loc.z:.3f})")

                # One diagnostic tick to observe immediate settling/fall behavior
                self.world.tick()
                post_tick_loc = self.ego.get_location()
                post_tick_vel = self.ego.get_velocity()
                post_wp = self.map.get_waypoint(post_tick_loc, project_to_road=False)
                print(
                    f">>> [SPAWN_DIAG][POST_TICK] ego_loc=({post_tick_loc.x:.3f}, {post_tick_loc.y:.3f}, {post_tick_loc.z:.3f}), "
                    f"ego_speed={np.sqrt(post_tick_vel.x**2 + post_tick_vel.y**2 + post_tick_vel.z**2):.3f}, "
                    f"wp_found={post_wp is not None}, dz={(post_tick_loc.z - pre_tick_loc.z):.3f}"
                )
                if self.attached_trailer is not None:
                    tr_loc2 = self.attached_trailer.get_location()
                    print(f">>> [SPAWN_DIAG][POST_TICK] trailer_loc=({tr_loc2.x:.3f}, {tr_loc2.y:.3f}, {tr_loc2.z:.3f})")
            except Exception as e:
                print(f">>> [SPAWN_DIAG] warning: failed to collect spawn diagnostics: {e}")

        # 7. Set traffic lights
        print(">>> [RESET] step7: Setting traffic lights...")
        if self.traffic == 'off':
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.set_state(carla.TrafficLightState.Green)
                actor.freeze(True)
        elif self.traffic == 'on':
            for actor in self.world.get_actors().filter('traffic.traffic_light*'):
                actor.freeze(False)

        # 8. Attach sensors to ego
        print(">>> [RESET] step8: Attaching sensors to ego...")
        self.collision_sensor = self.world.spawn_actor(
            self.collision_bp, carla.Transform(), attach_to=self.ego )
        self.all_sensors.append(self.collision_sensor) # Add to list for management
        self.collision_hist = []
        self.collision_sensor.listen(self._collision_handler)

        self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
        self.all_sensors.append(self.lidar_sensor) # Add to list for management
        self.lidar_sensor.listen(self._lidar_handler)
        
        # Attach front camera
        try:
            self.front_camera_sensor = self.world.spawn_actor(
                self.camera_bp, self.front_camera_trans, attach_to=self.ego)
            self.all_sensors.append(self.front_camera_sensor)
            # Pass 'front' as an argument to distinguish between camera types
            self.front_camera_sensor.listen(lambda image: self._camera_handler(image, 'front'))
            print(">>> [RESET] Front camera attached.")
        except Exception as e:
            print(f">>> [RESET] Failed to attach front camera: {e}")
            self.front_camera_image = np.zeros((600, 800, 3), dtype=np.uint8)  # Default image
            
        # Attach BEV camera
        try:
            self.bev_camera_sensor = self.world.spawn_actor(
                self.camera_bp, self.bev_camera_trans, attach_to=self.ego )
            self.all_sensors.append(self.bev_camera_sensor)
            # Pass 'bev' as an argument to distinguish between camera types
            self.bev_camera_sensor.listen(lambda image: self._camera_handler(image, 'bev'))
            print(">>> [RESET] BEV camera attached.")
        except Exception as e:
            print(f">>> [RESET] Failed to attach BEV camera: {e}")
            self.bev_camera_image = np.zeros((600, 800, 3), dtype=np.uint8)  # Default image
        
        print(">>> [RESET] step8: Sensors attached.")

        # 9. Set autopilot for surrounding vehicles
        print(">>> [RESET] step9: Setting autopilot for surroundings...")
        for vehicle in self.spawned_vehicles:
            vehicle.set_autopilot(True)
        print(">>> [RESET] step9: Autopilot set.")

        # 10. Apply world settings
        print(">>> [RESET] step10: Applying world settings...")
        settings = self.world.get_settings()
        settings.synchronous_mode = self.sync_mode
        settings.fixed_delta_seconds = self.delta_seconds
        self.world.apply_settings(settings)
        print(">>> [RESET] step10: Settings applied.")

        # 11. Finalize and return observation
        self.time_step = 1
        self.reset_step += 1
        obs, _ = self._get_obs(), {}

        # Optional: dump one frame per reset for visual diagnostics.
        self._dump_reset_images(obs)
        
        # Debug prints (as in original code)
        print("Lidar shape:", obs['lidar'].shape)
        print("Ego shape:", obs['ego_state'].shape)
        print("Nearby shape:", obs['nearby_vehicles'].shape)
        print("Waypoints shape:", obs['waypoints'].shape)
        print("Lane shape:", obs['lane_info'].shape)
        print("--- DEBUG OBS SHAPES ---")
        for k, v in obs.items():
            print(f" {k}: {v.shape if hasattr(v, 'shape') else type(v)} -> {type(v)}")
        print("-----------------------")
        print(">>> [RESET] step11: Reset completed successfully.\n")

        reset_info = {
            'task_mode': self.task_mode,
            'task_type': self.current_task.get('task_type', 'navigation'),
            'task_difficulty': self.current_task.get('difficulty', 'easy'),
            'task_success': False,
            'task_failure_reason': '',
        }
        return obs, reset_info

    def step(self, action):
        # New safety check: Ensure ego exists and is alive before proceeding
        if not self.ego or not self.ego.is_alive:
            raise RuntimeError("Ego vehicle is dead or missing. Environment needs reset.")

        throttle = float(np.clip(action[0], 0.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego.apply_control(control)
        self.world.tick() # Simulation tick

        # Spectator view update
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        if self.view_mode == 'top':
            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=40),
                    carla.Rotation(pitch=-90)
                )
            )
        elif self.view_mode == 'follow':
            cam_location = transform.transform(carla.Location(x=-6.0, z=3.0))
            cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw, roll=0)
            spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Update timesteps and get obs/reward/done/info
        self.time_step += 1
        self.total_step += 1
        obs = self._get_obs()
        done = bool(self._terminal())
        reward = self._get_reward(obs, done)
        cost = self._get_cost(obs)
        info = {
            'is_collision': bool(self._is_collision),
            'is_off_road': bool(self._is_off_road),
            'cost': cost,
            'task_mode': self.task_mode,
            'task_type': self.current_task.get('task_type', 'navigation'),
            'task_difficulty': self.current_task.get('difficulty', 'easy'),
            'task_success': bool(self._task_success),
            'task_failure_reason': self._task_failure_reason,
        }

        terminated = bool(done)
        truncated = bool(self.time_step >= self.max_steps)
        reward = float(reward)

        return obs, reward, terminated, truncated, info

    # --- Helper Methods (Adjusted) ---

    def _create_vehicle_blueprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library += [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        if self.surrounding_vehicle_spawned_randomly:
            blueprint = self._create_vehicle_blueprint('vehicle.*', number_of_wheels=number_of_wheels)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
        else:
            blueprint = self._create_vehicle_blueprint('vehicle.tesla.model3', color='0,0,255', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            self.spawned_vehicles.append(vehicle) # Add to list immediately upon success
        return vehicle

    def _try_spawn_random_walker_at(self, transform):
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.pedestrian.*'))
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            walker_controller_actor.start()
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            walker_controller_actor.set_max_speed(1 + random.random())
            # Store both controller and walker
            self.spawned_walkers.append((walker_controller_actor, walker_actor))
            return True
        return False
    #随机调用任意车辆作为ego vehicle
    # def _try_spawn_ego_vehicle_at(self, transform):
    #     # Check for overlap with current world state (vehicles & walkers)
    #     # This check uses the polygons calculated *before* this ego spawn attempt
    #     for poly_dict in self.vehicle_polygons + self.walker_polygons:
    #         for idx, poly in poly_dict.items():
    #             poly_center = np.mean(poly, axis=0)
    #             ego_center = np.array([transform.location.x, transform.location.y])
    #             dis = np.linalg.norm(poly_center - ego_center)
    #             if dis <= 8: # Threshold for considering overlap
    #                 return False # Overlap found, cannot spawn here

    #     vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
    #     if vehicle is not None:
    #         self.ego = vehicle # Assign to instance variable
    #         return True
    #     return False # Spawn failed due to collision or other CARLA reasons
    #调用半拖挂卡车dafxf
    # def _try_spawn_ego_vehicle_at(self, transform):
    #     """
    #     Attempts to spawn the ego vehicle (semi-trailer tractor) at the given transform.
    #     This version fetches the 'dafxf' blueprint dynamically.
    #     """
    #     # --- 1. Fetch the 'dafxf' tractor blueprint ---
    #     # Use the same filter and generation as in manual_control
    #     tractor_blueprint_list = get_actor_blueprints(self.world, "dafxf", self._actor_generation)
    #     if not tractor_blueprint_list:
    #         print("Warning: No 'dafxf' tractor blueprint found. Cannot spawn ego vehicle.")
    #         return False

    #     # Randomly select a blueprint from the available ones
    #     blueprint = random.choice(tractor_blueprint_list)
    #     blueprint.set_attribute('role_name', 'ego_vehicle_truck')

    #     # --- 2. Attempt to spawn the tractor at the provided transform ---
    #     # This transform comes from the reset loop which selects from spawn points
    #     vehicle = self.world.try_spawn_actor(blueprint, transform)
        
    #     if vehicle is not None:
    #         self.ego = vehicle # Assign the spawned tractor to the environment's ego variable
    #         print(f"Ego tractor vehicle spawned: {self.ego.type_id} at {self.ego.get_transform().location}")
    #         return True
    #     else:
    #         # Spawn failed at this location (likely due to collision with static geometry)
    #         # The reset loop will try another spawn point.
    #         return False
    def _try_spawn_ego_vehicle_at(self, transform):
        """
        Attempts to spawn ego vehicle at the given transform.
        - If enable_trailer=True, spawn tractor + trailer pair (original behavior).
        - If enable_trailer=False, spawn only tractor for stability diagnostics.
        """
        # --- 1. Fetch the 'dafxf' tractor blueprint ---
        tractor_blueprint_list = get_actor_blueprints(self.world, "dafxf", self._actor_generation)
        if not tractor_blueprint_list:
            print("Warning: No 'dafxf' tractor blueprint found. Cannot spawn ego vehicle.")
            return False

        tractor_blueprint = random.choice(tractor_blueprint_list)
        tractor_blueprint.set_attribute('role_name', 'ego_vehicle_truck')

        # Fast path: tractor only mode (for compatibility/stability testing)
        if not self.enable_trailer:
            vehicle = self.world.try_spawn_actor(tractor_blueprint, transform)
            if vehicle is not None:
                self.ego = vehicle
                self.attached_trailer = None
                print(f"Ego tractor successfully placed (no trailer mode): {self.ego.type_id} at {self.ego.get_transform().location}")
                return True
            return False

        # --- 2. Fetch the 'trailer' blueprint ---
        trailer_blueprint_list = get_actor_blueprints(self.world, "trailer", self._actor_generation)
        if not trailer_blueprint_list:
            print("Warning: No 'trailer' blueprint found. Cannot spawn trailer.")
            return False

        trailer_blueprint = random.choice(trailer_blueprint_list)
        trailer_blueprint.set_attribute('role_name', 'attached_trailer')

        # --- 3. Calculate the spawn point for the trailer based on the tractor's transform ---
        forward_vector = transform.get_forward_vector()
        offset_distance = 5.2

        # Create a COPY of the transform to avoid modifying the original
        spawn_point_for_trailer = carla.Transform(transform.location, transform.rotation)
        spawn_point_for_trailer.location -= (forward_vector * offset_distance)

        # --- 4. Try to spawn the TRAILER first ---
        trailer = self.world.try_spawn_actor(trailer_blueprint, spawn_point_for_trailer)
        if trailer is None:
            return False
        else:
            print(f"Trailer successfully placed for trial: {trailer.type_id} at {trailer.get_transform().location}")

        # --- 5. If trailer spawned, try to spawn the TRACTOR ---
        vehicle = self.world.try_spawn_actor(tractor_blueprint, transform)
        if vehicle is not None:
            self.ego = vehicle
            self.attached_trailer = trailer
            print(f"Ego tractor successfully placed: {self.ego.type_id} at {self.ego.get_transform().location}")
            return True
        else:
            print("Tractor spawn failed, cleaning up previously spawned trailer.")
            trailer.destroy()
            return False
    
    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    # --- Core RL Environment Logic ---

    def _collision_handler(self, event):
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)
        if len(self.collision_hist) > self.collision_hist_l:
            self.collision_hist.pop(0)

    def _lidar_handler(self, point_cloud):
        """Store the latest LiDAR data."""
        # Process point cloud into a fixed-size array
        # This is a simplified version, you might want to implement a more sophisticated one
        points = np.frombuffer(point_cloud.raw_data, dtype=np.float32)
        # Structure: (x, y, z, intensity) for each point
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        
        # Group points into bins based on angle for a 360-degree sweep
        angles = np.arctan2(points[:, 1], points[:, 0]) # Y is forward, X is right in LiDAR coord
        distances = np.linalg.norm(points[:, :2], axis=1)
        
        # Create bins for 240 degrees (e.g., 240 bins for 360 deg FOV)
        num_bins = 240
        bin_edges = np.linspace(-np.pi, np.pi, num_bins + 1)
        hist, _ = np.histogram(angles, bins=bin_edges, weights=distances, density=False)
        counts, _ = np.histogram(angles, bins=bin_edges)
        
        # Normalize histogram to get minimum distance per bin
        # Treat empty bins as maximum range
        lidar_processed = np.full(num_bins, self.lidar_max_range, dtype=np.float32)
        mask = counts > 0
        lidar_processed[mask] = hist[mask] / counts[mask]
        
        # Normalize to [0, 1]
        self.lidar_data = np.clip(lidar_processed / self.lidar_max_range, 0.0, 1.0)

    def _get_obs(self):
        """Get the current observation."""
        if self.ego is None:
            # Return a default/zero observation if ego doesn't exist
            obs = {
                'lidar': np.zeros(240, dtype=np.float32),
                'ego_state': np.zeros(9, dtype=np.float32),
                'nearby_vehicles': np.zeros(self.max_nearby_vehicles * 4, dtype=np.float32),
                'waypoints': np.zeros(self.max_waypoints * 3, dtype=np.float32),
                'lane_info': np.zeros(2, dtype=np.float32),
                'front_camera': np.zeros((600, 800, 3), dtype=np.uint8),  # Default camera image
                'bev_camera': np.zeros((600, 800, 3), dtype=np.uint8),    # Default camera image
            }
            return obs

        # 1. Get LiDAR data
        lidar = self.lidar_data if self.lidar_data is not None else np.ones(240, dtype=np.float32)

        # 2. Get ego state
        velocity = self.ego.get_velocity()
        acceleration = self.ego.get_acceleration()
        transform = self.ego.get_transform()
        control = self.ego.get_control()

        v = get_current_speed(self.ego)
        compass = math.degrees(transform.rotation.yaw) % 360
        if compass > 180:
            compass -= 360

        ego_state = [
            v / 30.0,  # Normalized speed
            (acceleration.x**2 + acceleration.y**2 + acceleration.z**2)**0.5 / 10.0,  # Norm of acceleration
            compass / 180.0,  # Normalized compass
            transform.location.x / 100.0,  # Normalized position X
            transform.location.y / 100.0,  # Normalized position Y
            velocity.x / 10.0,  # Velocity X
            velocity.y / 10.0,  # Velocity Y
            control.throttle,
            control.brake
        ]
        ego_state = np.array(ego_state, dtype=np.float32)

        # 3. Get nearby vehicles
        ego_location = self.ego.get_location()
        nearby_vehicles = []
        for vehicle in self.world.get_actors().filter('vehicle.*'):
            if vehicle.id != self.ego.id:  # Exclude ego itself
                distance = ego_location.distance(vehicle.get_location())
                if distance < 50:  # Only consider vehicles within 50m
                    rel_pos = vehicle.get_transform().location - ego_location
                    rel_vel = vehicle.get_velocity()
                    # Relative position and velocity features
                    nearby_vehicles.append([
                        rel_pos.x / 50.0,  # Normalized relative X
                        rel_pos.y / 50.0,  # Normalized relative Y
                        rel_vel.x / 10.0,  # Normalized relative velocity X
                        rel_vel.y / 10.0,  # Normalized relative velocity Y
                    ])
        
        # Pad or truncate the list to a fixed size
        if len(nearby_vehicles) > self.max_nearby_vehicles:
            nearby_vehicles = nearby_vehicles[:self.max_nearby_vehicles]
        while len(nearby_vehicles) < self.max_nearby_vehicles:
            nearby_vehicles.append([0.0, 0.0, 0.0, 0.0]) # Padding with zeros

        nearby_vehicles = np.concatenate(nearby_vehicles).astype(np.float32)

        # 4. Get waypoints
        route = get_waypoint_list(self.ego, self.map, self.max_waypoints)
        waypoints = []
        for i in range(self.max_waypoints):
            if i < len(route):
                wp_transform = route[i].transform
                # Waypoint relative to ego's current location
                rel_x = wp_transform.location.x - ego_location.x
                rel_y = wp_transform.location.y - ego_location.y
                waypoints.extend([rel_x / 100.0, rel_y / 100.0, wp_transform.rotation.yaw / 180.0])
            else:
                # Pad with zeros if we run out of waypoints
                waypoints.extend([0.0, 0.0, 0.0])

        waypoints = np.array(waypoints, dtype=np.float32)

        # 5. Get lane info
        lane_type, _, dist_from_center_norm, _ = get_lane_info(self.ego, self.map)
        lane_info = np.array([lane_type, dist_from_center_norm], dtype=np.float32)

        # 6. Get camera images
        front_camera_img = self.front_camera_image if self.front_camera_image is not None else np.zeros((600, 800, 3), dtype=np.uint8)
        bev_camera_img = self.bev_camera_image if self.bev_camera_image is not None else np.zeros((600, 800, 3), dtype=np.uint8)

        observation = {
            'lidar': lidar,
            'ego_state': ego_state,
            'nearby_vehicles': nearby_vehicles,
            'waypoints': waypoints,
            'lane_info': lane_info,
            'front_camera': front_camera_img,
            'bev_camera': bev_camera_img,
        }
        return observation


    def _terminal(self):
        """Check if the episode is terminal."""
        # Collision check
        if len(self.collision_hist) > 0 and max(self.collision_hist) > 0:
            self._is_collision = True
            self._task_failure_reason = 'collision'
            return True

        # Off-road check (based on distance from center)
        current_waypoint = self.map.get_waypoint(self.ego.get_location())
        if current_waypoint.lane_type != carla.LaneType.Driving:
            self._is_off_road = True
            self.off_road_counter += 1
            if self.off_road_counter > 50: # Allow some tolerance
                self._task_failure_reason = 'off_road'
                return True
        else:
            self.off_road_counter = 0 # Reset counter when back on road

        # Timeout check
        if self.time_step > self.max_time_episode:
            self._task_failure_reason = 'timeout'
            return True

        # Multitask success conditions
        if self.task_mode == 'multitask' and self.ego is not None and self._episode_start_location is not None:
            cur_tf = self.ego.get_transform()
            cur_loc = cur_tf.location
            distance = cur_loc.distance(self._episode_start_location)
            yaw_delta = self._normalize_yaw_delta(cur_tf.rotation.yaw, self._episode_start_yaw)
            task_type = self.current_task.get('task_type', 'navigation')

            if task_type == 'navigation':
                if distance >= self._target_distance_m:
                    self._task_success = True
                    return True
            elif task_type == 'left_turn':
                if distance >= self._target_distance_m and yaw_delta >= 35.0:
                    self._task_success = True
                    return True
            elif task_type == 'right_turn':
                if distance >= self._target_distance_m and yaw_delta <= -35.0:
                    self._task_success = True
                    return True

        return False

    def _get_reward(self, obs, done):
        """Calculate the reward."""
        if self.ego is None:
            return 0.0

        v = get_current_speed(self.ego)
        current_waypoint = self.map.get_waypoint(self.ego.get_location())

        # Reward for desired speed
        reward_speed = max(0.0, (v - 5.0) / 10.0) 

        # Penalty for being off road
        penalty_off_road = -1.0 if self._is_off_road else 0.0

        # Penalty for collision
        penalty_collision = -10.0 if self._is_collision else 0.0

        # Reward for following waypoints (progress)
        # This is a simple proxy using forward speed in the general direction
        transform = self.ego.get_transform()
        vel_vec = self.ego.get_velocity()
        forward_vec = transform.get_forward_vector()
        # Dot product to see if moving forward
        speed_factor = np.dot([vel_vec.x, vel_vec.y], [forward_vec.x, forward_vec.y])
        reward_progress = speed_factor / 10.0

        # Combine rewards
        reward = reward_speed + reward_progress + penalty_off_road + penalty_collision

        # Lightweight task-shaped reward for multitask mode
        if self.task_mode == 'multitask' and self.ego is not None and self._episode_start_location is not None:
            cur_tf = self.ego.get_transform()
            distance = cur_tf.location.distance(self._episode_start_location)
            yaw_delta = self._normalize_yaw_delta(cur_tf.rotation.yaw, self._episode_start_yaw)
            task_type = self.current_task.get('task_type', 'navigation')

            if task_type == 'navigation':
                reward += 0.02 * min(distance, self._target_distance_m)
            elif task_type == 'left_turn':
                reward += 0.01 * max(0.0, yaw_delta)
            elif task_type == 'right_turn':
                reward += 0.01 * max(0.0, -yaw_delta)
        
        if done:
            if self._task_success:
                reward += 50.0
            if self._is_collision:
                reward -= 100.0
            elif self._is_off_road:
                reward -= 50.0
            else:
                reward += 10.0  # Bonus for surviving

        return reward

    def _get_cost(self, obs):
        """Calculate the cost (e.g., for constrained RL)."""
        # Example: Cost could be related to risk, energy, etc.
        # For now, let's define it as a function of collision risk or deviation from lane
        current_waypoint = self.map.get_waypoint(self.ego.get_location())
        _, _, dist_from_center_norm, _ = get_lane_info(self.ego, self.map)
        
        # Higher cost for being far from lane center
        cost_lane_deviation = max(0.0, dist_from_center_norm - 0.5)
        
        # Higher cost near collisions (using LiDAR data)
        lidar_distances = obs['lidar']
        # Cost based on how many bins have very close obstacles
        critical_threshold = 0.1 # 10% of max range
        cost_proximity = np.sum(lidar_distances < critical_threshold) / len(lidar_distances)
        
        total_cost = cost_lane_deviation + cost_proximity
        return total_cost


    def close(self):
        print("\n>>> [CLOSE] Shutting down Carla environment...")
        # Use the same comprehensive cleanup as in reset
        self._clear_all_actors()
        
        # Restore world settings
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        
        # Tick once more to ensure settings take effect
        self.world.tick()
        print(">>> [CLOSE] Environment closed successfully.")