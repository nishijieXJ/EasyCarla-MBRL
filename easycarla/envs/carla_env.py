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
import sys
from datetime import datetime

# Try to locate CARLA PythonAPI so `agents.navigation` can be imported.
def _try_add_carla_pythonapi_path():
    candidates = [
        os.environ.get('CARLA_PYTHONAPI', ''),
        os.environ.get('CARLA_PYTHON_API', ''),
        os.environ.get('CARLA_ROOT', ''),
        '/opt/carla-simulator/PythonAPI/carla',
        '/opt/carla/PythonAPI/carla',
        '/data1/zk/carla/PythonAPI/carla',
        '/data2/zk/carla/PythonAPI/carla',
    ]

    for p in candidates:
        if not p:
            continue
        p = os.path.abspath(p)

        # If given CARLA_ROOT, append PythonAPI/carla.
        if os.path.isdir(os.path.join(p, 'PythonAPI', 'carla')):
            candidate = os.path.join(p, 'PythonAPI', 'carla')
        else:
            candidate = p

        nav_file = os.path.join(candidate, 'agents', 'navigation', 'global_route_planner.py')
        if os.path.isfile(nav_file):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate

    return None

_CARLA_PYTHONAPI_PATH = _try_add_carla_pythonapi_path()
if _CARLA_PYTHONAPI_PATH is not None:
    print(f"[GUIDANCE_ROUTE] using CARLA PythonAPI path: {_CARLA_PYTHONAPI_PATH}")
else:
    print("[GUIDANCE_ROUTE] warning: CARLA PythonAPI path not found (agents.navigation may be unavailable).")

# CARLA navigation API compatibility across versions:
# - Newer: GlobalRoutePlanner(map, sampling_resolution)
# - Older: GlobalRoutePlanner(dao) + grp.setup(), with GlobalRoutePlannerDAO
_GRP_IMPORT_ERROR = None
_GRP_DAO_IMPORT_ERROR = None
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except Exception as e:
    GlobalRoutePlanner = None
    _GRP_IMPORT_ERROR = repr(e)

try:
    from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
except Exception as e:
    GlobalRoutePlannerDAO = None
    _GRP_DAO_IMPORT_ERROR = repr(e)

if GlobalRoutePlanner is None:
    print(f"[GUIDANCE_ROUTE] import GlobalRoutePlanner failed: {_GRP_IMPORT_ERROR}")
if GlobalRoutePlannerDAO is None:
    print(f"[GUIDANCE_ROUTE] import GlobalRoutePlannerDAO unavailable: {_GRP_DAO_IMPORT_ERROR}")

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
        self.task_split = params.get('task_split', 'train')
        self.current_task = {
            'task_type': self.default_task_type,
            'difficulty': self.default_task_difficulty
        }
        self._last_spawn_source = 'random'
        self._task_success = False
        self._task_failure_reason = ''
        self._episode_start_location = None
        self._episode_start_yaw = 0.0
        self._target_distance_m = 30.0
        # Reset per-episode trailer spawn diagnostics
        self._trailer_spawn_attempts = 0
        self._trailer_spawn_failures = 0
        self._trailer_spawn_fail_reasons = collections.Counter()
        self._goal_location = None
        self._task_goal_tolerance_m = float(params.get('task_goal_tolerance_m', 1.0))
        # Town12/LargeMap compatibility switches (minimal-invasive, only enabled on Town12)
        self.town12_hero_mode = params.get('town12_hero_mode', True)
        self.town12_spawn_warmup_ticks = int(params.get('town12_spawn_warmup_ticks', 80))
        # Optional task catalog for explicit start/goal per map/task/difficulty
        # Structure example:
        # {
        #   "Town03": {
        #     "left_turn": {
        #       "easy": {
        #         "spawn": {"x": 10.0, "y": 20.0, "z": 0.5, "yaw": 90.0},
        #         "goal": {"x": 30.0, "y": 40.0, "z": 0.5}
        #       }
        #     }
        #   }
        # }
        self.task_catalog = params.get('task_catalog', {})

        # Guidance mode switch (for A/B/A+B experiments):
        # - off: disable both A(debug draw) and B(goal-relative obs)
        # - A: enable debug draw only
        # - B: enable goal-relative observation only
        # - AB / A+B / both: enable both
        self.guidance_mode = str(params.get('guidance_mode', 'AB')).strip().upper()
        if self.guidance_mode in ('A+B', 'B+A', 'BOTH'):
            self.guidance_mode = 'AB'
        self.enable_guidance_draw = self.guidance_mode in ('A', 'AB')
        self.enable_goal_relative_obs = self.guidance_mode in ('B', 'AB')
        # Spawn alignment diagnostics
        self.spawn_alignment_log = bool(params.get('spawn_alignment_log', True))
        # Debug draw tuning
        self.guidance_draw_life_time = float(params.get('guidance_draw_life_time', 8.0))
        self.guidance_draw_thickness = float(params.get('guidance_draw_thickness', 0.15))

        # Route-style guidance (GlobalRoutePlanner) tuning
        self.enable_route_guidance = bool(params.get('enable_route_guidance', True))
        self.route_sample_resolution = float(params.get('route_sample_resolution', 2.0))
        self.route_draw_every_n_points = max(1, int(params.get('route_draw_every_n_points', 2)))
        self.route_draw_point_size = float(params.get('route_draw_point_size', 0.08))
        self.route_guidance_color = params.get('route_guidance_color', (0, 0, 255))  # Blue lane-center style
        # route_draw_style: line_points | points_only | line_only
        self.route_draw_style = str(params.get('route_draw_style', 'points_only')).strip().lower()
        if self.route_draw_style not in ('line_points', 'points_only', 'line_only'):
            self.route_draw_style = 'points_only'
        self._grp = None
        self._current_route_waypoints = []

        # --- Reward/Cost tunable parameters (for sweep experiments) ---
        self.collision_intensity_threshold = float(params.get('collision_intensity_threshold', 50.0))
        self.collision_soft_penalty = float(params.get('collision_soft_penalty', -10.0))
        self.collision_hard_penalty = float(params.get('collision_hard_penalty', -100.0))
        self.offroad_terminal_steps = int(params.get('offroad_terminal_steps', 50))
        self.timeout_terminal_bonus = float(params.get('timeout_terminal_bonus', 0.0))

        # Desired-speed tracking reward (ablation-friendly)
        self.enable_reward_speed_tracking = bool(params.get('enable_reward_speed_tracking', True))
        self.reward_speed_tracking_weight = float(params.get('reward_speed_tracking_weight', 1.0))
        self.reward_speed_tolerance_kmh = float(params.get('reward_speed_tolerance_kmh', 2.0))
        self.reward_speed_scale_kmh = float(params.get('reward_speed_scale_kmh', 10.0))

        # Optional speeding cost (only penalize above desired_speed + margin)
        self.enable_cost_speeding = bool(params.get('enable_cost_speeding', False))
        self.cost_speeding_weight = float(params.get('cost_speeding_weight', 1.0))
        self.cost_speeding_margin_kmh = float(params.get('cost_speeding_margin_kmh', 3.0))
        self.cost_speeding_scale_kmh = float(params.get('cost_speeding_scale_kmh', 10.0))

        # Semi-trailer risk cost knobs
        self.cost_lane_deviation_weight = float(params.get('cost_lane_deviation_weight', 1.0))
        self.cost_proximity_weight = float(params.get('cost_proximity_weight', 1.0))
        self.cost_articulation_weight = float(params.get('cost_articulation_weight', 0.8))
        self.cost_jackknife_weight = float(params.get('cost_jackknife_weight', 2.0))
        self.cost_trailer_offroad_weight = float(params.get('cost_trailer_offroad_weight', 2.0))
        self.cost_ttc_weight = float(params.get('cost_ttc_weight', 1.5))
        self.jackknife_angle_deg = float(params.get('jackknife_angle_deg', 55.0))
        self.articulation_safe_angle_deg = float(params.get('articulation_safe_angle_deg', 20.0))
        self.ttc_threshold_s = float(params.get('ttc_threshold_s', 3.0))
        self.ttc_max_consider_distance_m = float(params.get('ttc_max_consider_distance_m', 50.0))

        # Component-level ablation switches
        self.enable_cost_lane_deviation = bool(params.get('enable_cost_lane_deviation', True))
        self.enable_cost_proximity = bool(params.get('enable_cost_proximity', True))
        self.enable_cost_articulation = bool(params.get('enable_cost_articulation', True))
        self.enable_cost_jackknife = bool(params.get('enable_cost_jackknife', True))
        self.enable_cost_trailer_offroad = bool(params.get('enable_cost_trailer_offroad', True))
        self.enable_cost_ttc = bool(params.get('enable_cost_ttc', True))
        # Optional reward ablation switch (keep base behavior by default)
        self.enable_reward_task_shaping = bool(params.get('enable_reward_task_shaping', True))

        # Cache latest truck/risk metrics for logging
        self._last_risk_metrics = {
            'collision_intensity': 0.0,
            'articulation_deg': 0.0,
            'jackknife': 0.0,
            'trailer_offroad': 0.0,
            'min_ttc_s': -1.0,
            'cost_speeding': 0.0,
        }

        # --- Expert warm-start + mixed exploration controls ---
        # control_mode: 'rl' | 'expert' | 'mixed'
        self.control_mode = str(params.get('control_mode', 'rl')).strip().lower()
        if self.control_mode not in ('rl', 'expert', 'mixed'):
            self.control_mode = 'rl'
        # Plan A: run pure expert for first N env steps, then switch to RL.
        self.expert_warmup_steps = int(params.get('expert_warmup_steps', 0))
        # Plan B: mixed exploration with annealed expert probability.
        self.expert_prob_init = float(params.get('expert_prob_init', 1.0))
        self.expert_prob_final = float(params.get('expert_prob_final', 0.0))
        self.expert_prob_decay_steps = int(params.get('expert_prob_decay_steps', 100000))
        self._expert_prob_runtime = self.expert_prob_init
        self._expert_prob_fixed = False
        self._global_env_step = 0
        self._last_action_source = 'rl'
        self._last_applied_control = np.zeros(3, dtype=np.float32)

        # --- Observation and Action Spaces ---
        self.observation_space = spaces.Dict({
                'lidar': spaces.Box(low=0.0, high=1.0, shape=(240,), dtype=np.float32),
                'ego_state': spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),
                'nearby_vehicles': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.max_nearby_vehicles * 4,), dtype=np.float32),
                'waypoints': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.max_waypoints * 3,), dtype=np.float32),
                'lane_info': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                'goal_relative': spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
                'front_camera': spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),  # Front camera image
                'bev_camera': spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8),    # BEV camera image
            })
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )

        # --- CARLA Connection ---
        print('Connecting to Carla server...')
        print(f"[CARLA] client timeout={params.get('carla_timeout', 60.0)}s")
        self.client = carla.Client('localhost', params['port'])
        self.carla_timeout = float(params.get('carla_timeout', 60.0))
        self.client.set_timeout(self.carla_timeout)

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
        print(
            f"[INIT] enable_trailer={self.enable_trailer}, "
            f"safe_startup={self.params.get('safe_startup', None)}, "
            f"map_mode={self.map_mode}, town={self.params.get('town', None)}"
        )
        # Inside __init__ method of CarlaEnv class
        # ... other initializations ...
        # Store the actor generation setting (copied from manual_control logic)
        self._actor_generation = "2"  # Or get it from an argument if you want flexibility, e.g., args.generation
        self.ego = None
        self.attached_trailer = None
        # Lifecycle guards for idempotent cleanup and callback isolation
        self._is_resetting = False
        self._is_closing = False
        self._is_closed = False
        # Trailer spawn diagnostics
        self._trailer_spawn_attempts = 0
        self._trailer_spawn_failures = 0
        self._trailer_spawn_fail_reasons = collections.Counter()


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
        # NOTE:
        # During reset we still need camera callbacks to fill first frames,
        # otherwise dumped reset images become all-black default arrays.
        if self._is_closing or self._is_closed or self.ego is None:
            return
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

    def _set_runtime_control_profile(self, options):
        """Apply runtime control profile from reset options."""
        opts = options if isinstance(options, dict) else {}

        requested_mode = str(opts.get('control_mode', self.control_mode)).strip().lower()
        if requested_mode in ('rl', 'expert', 'mixed'):
            self.control_mode = requested_mode

        if 'expert_warmup_steps' in opts:
            try:
                self.expert_warmup_steps = max(0, int(opts.get('expert_warmup_steps', self.expert_warmup_steps)))
            except Exception:
                pass

        if 'expert_prob_init' in opts:
            try:
                self.expert_prob_init = float(opts.get('expert_prob_init', self.expert_prob_init))
            except Exception:
                pass

        if 'expert_prob_final' in opts:
            try:
                self.expert_prob_final = float(opts.get('expert_prob_final', self.expert_prob_final))
            except Exception:
                pass

        if 'expert_prob_decay_steps' in opts:
            try:
                self.expert_prob_decay_steps = max(1, int(opts.get('expert_prob_decay_steps', self.expert_prob_decay_steps)))
            except Exception:
                pass

        self.expert_prob_init = float(np.clip(self.expert_prob_init, 0.0, 1.0))
        self.expert_prob_final = float(np.clip(self.expert_prob_final, 0.0, 1.0))

        # Allow explicit per-episode override in mixed mode.
        if 'expert_prob' in opts:
            try:
                p = float(opts.get('expert_prob'))
                self._expert_prob_runtime = float(np.clip(p, 0.0, 1.0))
                self._expert_prob_fixed = True
            except Exception:
                self._expert_prob_runtime = float(np.clip(self.expert_prob_init, 0.0, 1.0))
                self._expert_prob_fixed = False
        else:
            self._expert_prob_runtime = float(np.clip(self.expert_prob_init, 0.0, 1.0))
            self._expert_prob_fixed = False

    def _get_annealed_expert_prob(self):
        """Linear annealing from expert_prob_init -> expert_prob_final by global step."""
        if self.expert_prob_decay_steps <= 0:
            return float(self.expert_prob_final)
        frac = min(1.0, float(self._global_env_step) / float(self.expert_prob_decay_steps))
        p = self.expert_prob_init + (self.expert_prob_final - self.expert_prob_init) * frac
        return float(np.clip(p, 0.0, 1.0))

    def _resolve_control_source(self):
        """Choose action source for this step: expert or rl."""
        # Plan A: warm-start expert for first N steps regardless of mode.
        if self._global_env_step < self.expert_warmup_steps:
            return 'expert'

        if self.control_mode == 'expert':
            return 'expert'
        if self.control_mode == 'rl':
            return 'rl'

        # mixed mode
        p = self._expert_prob_runtime if self._expert_prob_fixed else self._get_annealed_expert_prob()
        return 'expert' if random.random() < float(np.clip(p, 0.0, 1.0)) else 'rl'

    def _normalize_yaw_delta(self, yaw_now, yaw_ref):
        """Return yaw delta in degrees within [-180, 180]."""
        delta = (yaw_now - yaw_ref + 180.0) % 360.0 - 180.0
        return delta

    def _is_town12_map(self):
        """Best-effort Town12 detection without affecting other towns."""
        try:
            if self.map is None:
                return False
            map_name = self.map.name if hasattr(self.map, 'name') else ''
            return 'Town12' in str(map_name)
        except Exception:
            return False

    def _cleanup_existing_hero_vehicles(self):
        """Remove stale hero vehicles before new ego spawn (Town12 guard)."""
        removed = 0
        try:
            for v in self.world.get_actors().filter('vehicle.*'):
                role = v.attributes.get('role_name', '') if hasattr(v, 'attributes') else ''
                if role == 'hero' and (self.ego is None or v.id != self.ego.id):
                    try:
                        if v.is_alive:
                            v.destroy()
                            removed += 1
                    except Exception:
                        pass
        except Exception as e:
            print(f">>> [RESET] hero cleanup warning: {e}")
        if removed > 0:
            print(f">>> [RESET] cleaned stale hero vehicles: {removed}")

    def _town12_pre_spawn_warmup(self):
        """Extra warmup ticks for Town12 large-map streaming before ego spawn."""
        if not self._is_town12_map():
            return
        ticks = max(0, int(self.town12_spawn_warmup_ticks))
        if ticks <= 0:
            return
        print(f">>> [RESET][Town12] pre-spawn warmup ticks: {ticks}")
        for _ in range(ticks):
            self.world.tick()

    def _resolve_task_request(self, options):
        """
        Resolve task request from reset options.
        Supports task_id/scenario_type/case_index while preserving backward compatibility.
        """
        if self.task_mode != 'multitask':
            self.current_task = {
                'task_type': self.default_task_type,
                'difficulty': self.default_task_difficulty,
                'scenario_type': '',
                'case_index': None,
            }
            return self.current_task

        req = options if isinstance(options, dict) else {}
        map_name = self._get_current_map_name()

        requested_task = req.get('task_id', req.get('task_type', self.default_task_type))
        task_type = str(requested_task).lower() if requested_task is not None else str(self.default_task_type).lower()
        scenario_type = str(req.get('scenario_type', '')).strip()

        difficulty = str(req.get('difficulty', self.default_task_difficulty)).lower()
        valid_difficulties = {'easy', 'medium', 'hard'}
        if difficulty not in valid_difficulties:
            difficulty = self.default_task_difficulty

        # Allow dynamic task keys from catalog in addition to legacy task types.
        valid_task_types = {'left_turn', 'right_turn', 'navigation'}
        if isinstance(self.task_catalog, dict):
            mcfg = self.task_catalog.get(map_name, None)
            if isinstance(mcfg, dict):
                valid_task_types = set(valid_task_types).union(set(mcfg.keys()))

        if task_type not in valid_task_types:
            task_type = self.default_task_type

        case_index = req.get('case_index', None)
        normalized_case_index = None
        if case_index is not None:
            try:
                normalized_case_index = int(case_index)
            except Exception:
                normalized_case_index = None

        self.current_task = {
            'task_type': task_type,
            'difficulty': difficulty,
            'scenario_type': scenario_type,
            'case_index': normalized_case_index,
        }
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

    def _get_current_map_name(self):
        """Best-effort map name extraction used by task catalog lookup."""
        try:
            map_name = self.map.name if hasattr(self.map, 'name') else ''
            # e.g. /Game/Carla/Maps/Town03 -> Town03
            return map_name.split('/')[-1] if map_name else ''
        except Exception:
            return ''

    def _resolve_task_pose_from_catalog(self):
        """
        Resolve optional explicit spawn/goal from task catalog.
        Supports optional scenario_type and case_index selection.
        Returns (spawn_transform_or_none, goal_location_or_none).
        """
        if not isinstance(self.task_catalog, dict) or len(self.task_catalog) == 0:
            return None, None

        map_name = self._get_current_map_name()
        task_type = self.current_task.get('task_type', 'navigation')
        difficulty = self.current_task.get('difficulty', 'easy')
        scenario_type = str(self.current_task.get('scenario_type', '')).strip()
        case_index = self.current_task.get('case_index', None)

        mcfg = self.task_catalog.get(map_name, None)
        if not isinstance(mcfg, dict):
            return None, None
        tcfg = mcfg.get(task_type, None)
        if not isinstance(tcfg, dict):
            return None, None
        dcfg = tcfg.get(difficulty, None)
        if not isinstance(dcfg, dict):
            return None, None

        # Support either single case format or case list format
        selected_case = None
        if isinstance(dcfg.get('cases', None), list) and len(dcfg['cases']) > 0:
            candidates = dcfg['cases']
            if scenario_type:
                filtered = [c for c in candidates if str(c.get('scenario_type', '')).strip() == scenario_type]
                if len(filtered) > 0:
                    candidates = filtered

            if len(candidates) > 0:
                if isinstance(case_index, int):
                    idx = case_index
                    if idx < 0:
                        idx = len(candidates) + idx
                    if 0 <= idx < len(candidates):
                        selected_case = candidates[idx]
                    else:
                        selected_case = random.choice(candidates)
                else:
                    selected_case = random.choice(candidates)

            if isinstance(selected_case, dict):
                spawn_cfg = selected_case.get('spawn', None)
                goal_cfg = selected_case.get('goal', None)
            else:
                spawn_cfg = None
                goal_cfg = None
        else:
            spawn_cfg = dcfg.get('spawn', None)
            goal_cfg = dcfg.get('goal', None)

        # Optional per-case/per-difficulty overrides
        cfg_src = selected_case if isinstance(selected_case, dict) else dcfg
        if isinstance(cfg_src, dict):
            if 'goal_tolerance_m' in cfg_src:
                try:
                    self._task_goal_tolerance_m = float(cfg_src.get('goal_tolerance_m'))
                except Exception:
                    pass
            if 'max_time_episode' in cfg_src:
                try:
                    self.max_time_episode = int(cfg_src.get('max_time_episode'))
                except Exception:
                    pass
            if 'num_vehicles' in cfg_src:
                try:
                    self.number_of_vehicles = int(cfg_src.get('num_vehicles'))
                except Exception:
                    pass

        spawn_transform = None
        if isinstance(spawn_cfg, dict):
            try:
                sx = float(spawn_cfg['x'])
                sy = float(spawn_cfg['y'])
                sz = float(spawn_cfg.get('z', 0.5))
                syaw = float(spawn_cfg.get('yaw', 0.0))
                spawn_transform = carla.Transform(
                    carla.Location(x=sx, y=sy, z=sz),
                    carla.Rotation(yaw=syaw)
                )
            except Exception:
                spawn_transform = None

        goal_location = None
        if isinstance(goal_cfg, dict):
            try:
                gx = float(goal_cfg['x'])
                gy = float(goal_cfg['y'])
                gz = float(goal_cfg.get('z', 0.5))
                goal_location = carla.Location(x=gx, y=gy, z=gz)
            except Exception:
                goal_location = None

        return spawn_transform, goal_location

    def _log_spawn_alignment(self, source='random'):
        """Log ego spawn alignment against nearest driving waypoint heading."""
        if not self.spawn_alignment_log or self.ego is None:
            return
        try:
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            ego_yaw = ego_tf.rotation.yaw
            wp = self.map.get_waypoint(ego_loc, project_to_road=False)
            wp_yaw = None
            lane_type = 'None'
            road_id = -1
            lane_id = -1
            lane_center_dist = -1.0
            if wp is not None:
                wp_yaw = wp.transform.rotation.yaw
                lane_type = str(wp.lane_type)
                road_id = int(getattr(wp, 'road_id', -1))
                lane_id = int(getattr(wp, 'lane_id', -1))
                lane_center_dist = float(ego_loc.distance(wp.transform.location))

            if wp_yaw is None:
                yaw_delta = 999.0
            else:
                yaw_delta = abs(self._normalize_yaw_delta(ego_yaw, wp_yaw))

            print(
                f">>> [SPAWN_ALIGN] source={source}, map={self._get_current_map_name()}, "
                f"ego=({ego_loc.x:.3f},{ego_loc.y:.3f},{ego_loc.z:.3f}), "
                f"ego_yaw={ego_yaw:.2f}, wp_yaw={wp_yaw if wp_yaw is not None else 'None'}, "
                f"delta_yaw={yaw_delta:.2f}, lane_type={lane_type}, road_id={road_id}, lane_id={lane_id}, "
                f"dist_to_lane_center={lane_center_dist:.3f}"
            )
        except Exception as e:
            print(f">>> [SPAWN_ALIGN] warning: failed to log spawn alignment: {e}")

    def _get_route_guidance_color(self):
        """Parse route guidance color from tuple/list or fallback to blue."""
        try:
            c = self.route_guidance_color
            if isinstance(c, (list, tuple)) and len(c) >= 3:
                r = int(np.clip(c[0], 0, 255))
                g = int(np.clip(c[1], 0, 255))
                b = int(np.clip(c[2], 0, 255))
                return carla.Color(r, g, b)
        except Exception:
            pass
        return carla.Color(0, 0, 255)

    def _build_route_guidance(self, start_loc, goal_loc):
        """Build lane-center route by GlobalRoutePlanner and cache waypoint locations."""
        self._current_route_waypoints = []
        if not self.enable_route_guidance:
            return
        if start_loc is None or goal_loc is None:
            return

        if GlobalRoutePlanner is None:
            print(
                f">>> [GUIDANCE_ROUTE] GlobalRoutePlanner unavailable "
                f"(pythonapi_path={_CARLA_PYTHONAPI_PATH}, import_error={_GRP_IMPORT_ERROR}), "
                "fallback to straight line draw."
            )
            return

        try:
            if self._grp is None:
                built = False

                # API style A: GlobalRoutePlanner(map, sampling_resolution)
                try:
                    self._grp = GlobalRoutePlanner(self.map, self.route_sample_resolution)
                    built = True
                    print(">>> [GUIDANCE_ROUTE] planner initialized via map+resolution API.")
                except Exception:
                    self._grp = None

                # API style B: GlobalRoutePlanner(GlobalRoutePlannerDAO(...)); grp.setup()
                if (not built) and (GlobalRoutePlannerDAO is not None):
                    try:
                        dao = GlobalRoutePlannerDAO(self.map, self.route_sample_resolution)
                        self._grp = GlobalRoutePlanner(dao)
                        if hasattr(self._grp, 'setup'):
                            self._grp.setup()
                        built = True
                        print(">>> [GUIDANCE_ROUTE] planner initialized via DAO API.")
                    except Exception:
                        self._grp = None

                if not built or self._grp is None:
                    print(">>> [GUIDANCE_ROUTE] planner init failed, fallback to straight line draw.")
                    return

            route = self._grp.trace_route(start_loc, goal_loc)
            if not isinstance(route, list) or len(route) == 0:
                print(">>> [GUIDANCE_ROUTE] trace_route returned empty route.")
                return

            points = []
            for item in route:
                try:
                    wp = item[0]
                    if wp is not None:
                        points.append(wp.transform.location)
                except Exception:
                    continue

            self._current_route_waypoints = points
            print(f">>> [GUIDANCE_ROUTE] route points={len(self._current_route_waypoints)}")
        except Exception as e:
            print(f">>> [GUIDANCE_ROUTE] build failed: {e}")
            self._current_route_waypoints = []

    def _draw_task_guidance(self):
        """Draw start/goal/guide line in CARLA debug layer (方案A)."""
        if not self.enable_guidance_draw:
            return
        if self.ego is None:
            return
        try:
            dbg = self.world.debug
            life_time = self.guidance_draw_life_time
            thickness = self.guidance_draw_thickness
            start_loc = self._episode_start_location if self._episode_start_location is not None else self.ego.get_location()
            ego_loc = self.ego.get_location()

            # Draw start marker
            dbg.draw_point(start_loc, size=0.2, color=carla.Color(0, 255, 0), life_time=life_time, persistent_lines=False)
            dbg.draw_string(start_loc + carla.Location(z=1.2), 'START', False, carla.Color(0, 255, 0), life_time)

            # Draw ego marker
            dbg.draw_point(ego_loc, size=0.2, color=carla.Color(0, 120, 255), life_time=life_time, persistent_lines=False)

            # Draw goal marker and route guidance when explicit or fallback goal exists
            if self._goal_location is not None:
                goal_loc = self._goal_location

                # Keep GOAL marker always visible for easy verification
                dbg.draw_point(goal_loc, size=0.35, color=carla.Color(255, 0, 255), life_time=0.0, persistent_lines=True)
                dbg.draw_string(goal_loc + carla.Location(z=1.5), 'GOAL', False, carla.Color(255, 0, 255), 0.0)

                # Prefer lane-center route guidance generated by GlobalRoutePlanner.
                route_pts = self._current_route_waypoints if isinstance(self._current_route_waypoints, list) else []
                if len(route_pts) >= 2:
                    route_color = self._get_route_guidance_color()
                    step_n = max(1, int(self.route_draw_every_n_points))
                    psize = max(0.04, float(self.route_draw_point_size))
                    line_thickness = max(0.05, min(float(thickness), 0.12))  # make route line visibly thinner
                    draw_points = self.route_draw_style in ('line_points', 'points_only')
                    draw_lines = self.route_draw_style in ('line_points', 'line_only')
                    for i in range(0, len(route_pts) - 1, step_n):
                        p0 = route_pts[i]
                        p1 = route_pts[min(i + step_n, len(route_pts) - 1)]
                        p0z = carla.Location(x=p0.x, y=p0.y, z=p0.z + 0.2)
                        p1z = carla.Location(x=p1.x, y=p1.y, z=p1.z + 0.2)
                        if draw_lines:
                            dbg.draw_line(p0z, p1z, thickness=line_thickness, color=route_color, life_time=life_time, persistent_lines=False)
                        if draw_points:
                            dbg.draw_point(p0z, size=psize, color=route_color, life_time=life_time, persistent_lines=False)
                else:
                    # Fallback to straight line if no route is available.
                    dbg.draw_line(start_loc, goal_loc, thickness=max(0.3, thickness), color=carla.Color(255, 0, 255), life_time=0.0, persistent_lines=True)

                # Ego->Goal helper line keeps orange for dynamic heading cue
                dbg.draw_line(ego_loc, goal_loc, thickness=max(0.2, thickness), color=carla.Color(255, 120, 0), life_time=life_time, persistent_lines=False)
            else:
                print(">>> [GUIDANCE_DRAW] no goal_location, only START marker is drawn.")
        except Exception as e:
            print(f">>> [GUIDANCE_DRAW] warning: draw failed: {e}")

    def _get_goal_relative_obs(self):
        """Goal-relative compact vector (方案B): [dx_local, dy_local, distance, yaw_error]."""
        if not self.enable_goal_relative_obs:
            return np.zeros(4, dtype=np.float32)
        if self.ego is None or self._goal_location is None:
            return np.zeros(4, dtype=np.float32)

        try:
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            ego_yaw_rad = math.radians(ego_tf.rotation.yaw)

            dx_world = float(self._goal_location.x - ego_loc.x)
            dy_world = float(self._goal_location.y - ego_loc.y)

            # Rotate world delta into ego local frame
            cos_y = math.cos(ego_yaw_rad)
            sin_y = math.sin(ego_yaw_rad)
            dx_local = cos_y * dx_world + sin_y * dy_world
            dy_local = -sin_y * dx_world + cos_y * dy_world

            dist_goal = math.sqrt(dx_world * dx_world + dy_world * dy_world)
            goal_heading_deg = math.degrees(math.atan2(dy_world, dx_world))
            yaw_error = self._normalize_yaw_delta(goal_heading_deg, ego_tf.rotation.yaw)

            return np.array([
                dx_local / 100.0,
                dy_local / 100.0,
                dist_goal / 100.0,
                yaw_error / 180.0,
            ], dtype=np.float32)
        except Exception:
            return np.zeros(4, dtype=np.float32)

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
        """Clear all actors managed by this environment instance (idempotent, de-duplicated, callback-safe)."""

        def _safe_actor_id(actor):
            try:
                return int(actor.id)
            except Exception:
                return None

        def _is_alive(actor):
            try:
                return actor is not None and bool(actor.is_alive)
            except Exception:
                return False

        # Prevent callbacks from touching stale state while we are cleaning up.
        self._is_resetting = True

        # 1) Stop sensors/listeners first to isolate callbacks.
        for sensor in list(self.all_sensors):
            try:
                if _is_alive(sensor):
                    sensor.stop()
            except Exception:
                pass

        # 2) Build a unique actor-id set (avoid double destroy).
        actor_ids = set()

        def _collect(actor):
            aid = _safe_actor_id(actor)
            if aid is not None:
                actor_ids.add(aid)

        _collect(self.ego)
        _collect(self.attached_trailer)

        for sensor in list(self.all_sensors):
            _collect(sensor)

        for vehicle in list(self.spawned_vehicles):
            _collect(vehicle)

        for pair in list(self.spawned_walkers):
            try:
                walker_controller, walker = pair
                _collect(walker_controller)
                _collect(walker)
            except Exception:
                continue

        for actor in list(self.all_actors):
            _collect(actor)

        # 3) Disable ego autopilot if possible (best-effort).
        try:
            if _is_alive(self.ego):
                self.ego.set_autopilot(False)
        except Exception:
            pass

        # 4) Prefer server-side batch destroy to reduce race conditions.
        if len(actor_ids) > 0:
            try:
                cmds = [carla.command.DestroyActor(aid) for aid in actor_ids]
                responses = self.client.apply_batch_sync(cmds, True)
                # Non-fatal: actor may already be gone.
                for r in responses:
                    try:
                        if getattr(r, 'error', None):
                            err = str(r.error)
                            if 'not found' not in err.lower() and 'already' not in err.lower() and 'destroyed' not in err.lower():
                                print(f"Warning: batch destroy actor failed: {err}")
                    except Exception:
                        pass
            except Exception as e:
                print(f"Warning: batch destroy failed, falling back to per-actor destroy: {e}")
                for aid in actor_ids:
                    try:
                        actor = self.world.get_actor(aid)
                        if actor is not None and _is_alive(actor):
                            actor.destroy()
                    except RuntimeError as re:
                        if 'destroyed actor' not in str(re):
                            print(f"Warning: destroy actor {aid} runtime error: {re}")
                    except Exception as ex:
                        print(f"Warning: destroy actor {aid} error: {ex}")

        # 5) Always clear all references to avoid stale handles across resets.
        self.all_sensors.clear()
        self.spawned_vehicles.clear()
        self.spawned_walkers.clear()
        self.all_actors.clear()

        self.ego = None
        self.attached_trailer = None
        self.collision_sensor = None
        self.lidar_sensor = None
        self.front_camera_sensor = None
        self.bev_camera_sensor = None

        self.lidar_data = None
        self.front_camera_image = None
        self.bev_camera_image = None
        self.collision_hist = []

    def reset(self, *, seed=None, options=None):
        print("\n>>> [RESET] Starting comprehensive reset...")
        self._is_closed = False
        self._is_closing = False
        self._is_resetting = True
        self._set_runtime_control_profile(options)
        resolved_task = self._resolve_task_request(options)
        self._apply_task_profile()
        print(f">>> [TASK] mode={self.task_mode}, task={resolved_task}")
        print(
            f">>> [CONTROL] mode={self.control_mode}, warmup_steps={self.expert_warmup_steps}, "
            f"expert_prob_init={self.expert_prob_init:.3f}, expert_prob_final={self.expert_prob_final:.3f}, "
            f"expert_prob_decay_steps={self.expert_prob_decay_steps}, expert_prob_fixed={self._expert_prob_fixed}"
        )

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

        # Optional explicit spawn/goal from task catalog
        task_spawn_transform, task_goal_location = self._resolve_task_pose_from_catalog()
        self._goal_location = task_goal_location

        # Fallback: if catalog has no explicit goal, synthesize one ahead of ego route
        # so guidance overlay can still be verified visually.
        if self._goal_location is None:
            try:
                fallback_wp = None
                for sp in self.vehicle_spawn_points:
                    w0 = self.map.get_waypoint(sp.location, project_to_road=True)
                    if w0 is None:
                        continue
                    nxt = w0.next(max(20.0, float(getattr(self, '_target_distance_m', 30.0))))
                    if isinstance(nxt, list) and len(nxt) > 0:
                        fallback_wp = random.choice(nxt)
                        break
                if fallback_wp is not None:
                    self._goal_location = fallback_wp.transform.location
                    print(f">>> [TASK] fallback goal generated at ({self._goal_location.x:.2f}, {self._goal_location.y:.2f}, {self._goal_location.z:.2f})")
                else:
                    print(">>> [TASK] warning: no explicit goal and failed to generate fallback goal.")
            except Exception as e:
                print(f">>> [TASK] warning: fallback goal generation failed: {e}")

        # Town12 minimal-invasive guards: clear stale hero and wait for large-map stabilization.
        if self._is_town12_map() and self.town12_hero_mode:
            self._cleanup_existing_hero_vehicles()
            for _ in range(10):
                self.world.tick()
            self._town12_pre_spawn_warmup()

        ego_spawned = False
        self._last_spawn_source = 'random'

        # If task catalog provides spawn, try it first.
        if task_spawn_transform is not None:
            print(f">>> [TASK] trying catalog spawn first: {task_spawn_transform}")
            if self._try_spawn_ego_vehicle_at(task_spawn_transform):
                print(">>> [RESET] Ego vehicle spawned from task catalog.")
                self.world.tick()
                ego_spawned = True
                self._last_spawn_source = 'catalog'
            else:
                print(">>> [TASK] catalog spawn failed, fallback to random spawn points.")

        for attempt in range(self.max_ego_spawn_times + 1):
            if ego_spawned:
                break
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
                print(
                    f">>> [SPAWN_SUMMARY] attempts={self._trailer_spawn_attempts}, "
                    f"failures={self._trailer_spawn_failures}, "
                    f"by_reason={dict(self._trailer_spawn_fail_reasons)}"
                )
                self._is_resetting = False
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
            print(
                f">>> [SPAWN_SUMMARY] attempts={self._trailer_spawn_attempts}, "
                f"failures={self._trailer_spawn_failures}, "
                f"by_reason={dict(self._trailer_spawn_fail_reasons)}"
            )
            self._is_resetting = False
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

        # Spawn alignment diagnostics (Q1)
        self._log_spawn_alignment(source=self._last_spawn_source)

        # Build lane-center route guidance (GlobalRoutePlanner) if goal exists.
        if self._goal_location is not None:
            self._build_route_guidance(self._episode_start_location, self._goal_location)

        # Optional visual guidance overlay (Q2A)
        self._draw_task_guidance()

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

        # Wait a few ticks for sensor callbacks to populate first valid frames.
        warmup_ticks = int(self.params.get('camera_warmup_ticks', 6))
        for _ in range(max(0, warmup_ticks)):
            self.world.tick()

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
        self._is_resetting = False
        print(
            f">>> [SPAWN_SUMMARY] attempts={self._trailer_spawn_attempts}, "
            f"failures={self._trailer_spawn_failures}, "
            f"by_reason={dict(self._trailer_spawn_fail_reasons)}"
        )
        print(">>> [RESET] step11: Reset completed successfully.\n")

        reset_info = {
            'task_mode': self.task_mode,
            'task_type': self.current_task.get('task_type', 'navigation'),
            'task_difficulty': self.current_task.get('difficulty', 'easy'),
            'task_success': False,
            'task_failure_reason': '',
            'control_mode': self.control_mode,
            'expert_warmup_steps': int(self.expert_warmup_steps),
            'expert_prob': float(self._expert_prob_runtime if self._expert_prob_fixed else self._get_annealed_expert_prob()),
            'expert_prob_fixed': bool(self._expert_prob_fixed),
        }
        return obs, reset_info

    def step(self, action):
        # New safety check: Ensure ego exists and is alive before proceeding
        if not self.ego or not self.ego.is_alive:
            raise RuntimeError("Ego vehicle is dead or missing. Environment needs reset.")

        rl_action = np.asarray(action, dtype=np.float32)
        throttle = float(np.clip(rl_action[0], 0.0, 1.0))
        steer = float(np.clip(rl_action[1], -1.0, 1.0))
        brake = float(np.clip(rl_action[2], 0.0, 1.0))

        action_source = self._resolve_control_source()
        if action_source == 'expert':
            # Use CARLA autopilot as expert policy.
            self.ego.set_autopilot(True)
            applied = self.ego.get_control()
            self._last_applied_control = np.array([
                float(getattr(applied, 'throttle', 0.0)),
                float(getattr(applied, 'steer', 0.0)),
                float(getattr(applied, 'brake', 0.0)),
            ], dtype=np.float32)
        else:
            # Use RL action directly.
            self.ego.set_autopilot(False)
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
            self.ego.apply_control(control)
            self._last_applied_control = np.array([throttle, steer, brake], dtype=np.float32)

        self._last_action_source = action_source
        self.world.tick() # Simulation tick

        # Keep guidance overlay refreshed during rollout (especially useful for non-persistent elements)
        self._draw_task_guidance()

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
            # Expert/mixed exploration diagnostics
            'control_mode': self.control_mode,
            'action_source': self._last_action_source,
            'expert_warmup_steps': int(self.expert_warmup_steps),
            'expert_prob': float(self._expert_prob_runtime if self._expert_prob_fixed else self._get_annealed_expert_prob()),
            'expert_prob_fixed': bool(self._expert_prob_fixed),
            'rl_action': rl_action.astype(np.float32),
            'applied_action': self._last_applied_control.astype(np.float32),
            # Enhanced semi-trailer risk diagnostics
            'collision_intensity': float(self._last_risk_metrics.get('collision_intensity', 0.0)),
            'articulation_deg': float(self._last_risk_metrics.get('articulation_deg', 0.0)),
            'jackknife': float(self._last_risk_metrics.get('jackknife', 0.0)),
            'trailer_offroad': float(self._last_risk_metrics.get('trailer_offroad', 0.0)),
            'min_ttc_s': float(self._last_risk_metrics.get('min_ttc_s', -1.0)),
            'cost_speeding': float(self._last_risk_metrics.get('cost_speeding', 0.0)),
            # Ablation switch states
            'enable_cost_lane_deviation': bool(self.enable_cost_lane_deviation),
            'enable_cost_proximity': bool(self.enable_cost_proximity),
            'enable_cost_articulation': bool(self.enable_cost_articulation),
            'enable_cost_jackknife': bool(self.enable_cost_jackknife),
            'enable_cost_trailer_offroad': bool(self.enable_cost_trailer_offroad),
            'enable_cost_ttc': bool(self.enable_cost_ttc),
            'enable_cost_speeding': bool(self.enable_cost_speeding),
            'enable_reward_task_shaping': bool(self.enable_reward_task_shaping),
            'enable_reward_speed_tracking': bool(self.enable_reward_speed_tracking),
        }

        self._global_env_step += 1
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
        self._trailer_spawn_attempts += 1
        print(
            f">>> [SPAWN_FORCE] ego_spawn_try={self._trailer_spawn_attempts}, "
            f"enable_trailer={self.enable_trailer}, "
            f"trailer_failures={self._trailer_spawn_failures}, "
            f"fail_reasons={dict(self._trailer_spawn_fail_reasons)}"
        )

        # --- 1. Fetch the 'dafxf' tractor blueprint ---
        tractor_blueprint_list = get_actor_blueprints(self.world, "dafxf", self._actor_generation)
        if not tractor_blueprint_list:
            self._trailer_spawn_failures += 1
            self._trailer_spawn_fail_reasons['no_tractor_blueprint'] += 1
            print("Warning: No 'dafxf' tractor blueprint found. Cannot spawn ego vehicle.")
            return False

        tractor_blueprint = random.choice(tractor_blueprint_list)
        # Keep default role for non-Town12; force hero on Town12 to satisfy LargeMapManager checks.
        if self._is_town12_map() and self.town12_hero_mode and tractor_blueprint.has_attribute('role_name'):
            tractor_blueprint.set_attribute('role_name', 'hero')
        else:
            tractor_blueprint.set_attribute('role_name', 'ego_vehicle_truck')

        # Fast path: tractor only mode (for compatibility/stability testing)
        if not self.enable_trailer:
            vehicle = self.world.try_spawn_actor(tractor_blueprint, transform)
            if vehicle is not None:
                self.ego = vehicle
                self.attached_trailer = None
                print(f"Ego tractor successfully placed (no trailer mode): {self.ego.type_id} at {self.ego.get_transform().location}")
                return True
            self._trailer_spawn_failures += 1
            self._trailer_spawn_fail_reasons['tractor_spawn_failed_no_trailer_mode'] += 1
            return False

        # --- 2. Fetch the 'trailer' blueprint ---
        trailer_blueprint_list = get_actor_blueprints(self.world, "trailer", self._actor_generation)
        if not trailer_blueprint_list:
            self._trailer_spawn_failures += 1
            self._trailer_spawn_fail_reasons['no_trailer_blueprint'] += 1
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
            self._trailer_spawn_failures += 1
            self._trailer_spawn_fail_reasons['trailer_spawn_collision_or_blocked'] += 1
            print(
                ">>> [SPAWN_FORCE] trailer spawn failed: reason=trailer_spawn_collision_or_blocked, "
                f"trailer_failures={self._trailer_spawn_failures}, "
                f"fail_reasons={dict(self._trailer_spawn_fail_reasons)}"
            )
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
            self._trailer_spawn_failures += 1
            self._trailer_spawn_fail_reasons['tractor_spawn_failed_after_trailer_ok'] += 1
            print(
                ">>> [SPAWN_FORCE] tractor spawn failed after trailer success, cleaning trailer. "
                f"trailer_failures={self._trailer_spawn_failures}, "
                f"fail_reasons={dict(self._trailer_spawn_fail_reasons)}"
            )
            try:
                trailer.destroy()
            except Exception as e:
                print(f">>> [SPAWN_FORCE] warning: failed to destroy temporary trailer: {e}")
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
        if self._is_resetting or self._is_closing or self._is_closed or self.ego is None:
            return
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._last_risk_metrics['collision_intensity'] = float(intensity)
        self.collision_hist.append(intensity)
        if len(self.collision_hist) > self.collision_hist_l:
            self.collision_hist.pop(0)

    def _compute_articulation_deg(self):
        """Absolute yaw difference between tractor and trailer in degrees."""
        if self.ego is None or self.attached_trailer is None:
            return 0.0
        try:
            ego_yaw = self.ego.get_transform().rotation.yaw
            trailer_yaw = self.attached_trailer.get_transform().rotation.yaw
            return abs(self._normalize_yaw_delta(trailer_yaw, ego_yaw))
        except Exception:
            return 0.0

    def _compute_trailer_offroad_flag(self):
        """Return 1.0 if trailer center is outside driving lane, else 0.0."""
        if self.attached_trailer is None:
            return 0.0
        try:
            tr_loc = self.attached_trailer.get_location()
            wp = self.map.get_waypoint(tr_loc, project_to_road=False)
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def _compute_min_ttc(self):
        """Approximate minimum TTC to nearby vehicles in seconds; -1 means no valid threat."""
        if self.ego is None:
            return -1.0
        try:
            ego_tf = self.ego.get_transform()
            ego_loc = ego_tf.location
            ego_vel = self.ego.get_velocity()
            ego_v = np.array([ego_vel.x, ego_vel.y], dtype=np.float32)
            min_ttc = float('inf')

            for vehicle in self.world.get_actors().filter('vehicle.*'):
                if vehicle.id == self.ego.id:
                    continue
                vloc = vehicle.get_location()
                rel_pos = np.array([vloc.x - ego_loc.x, vloc.y - ego_loc.y], dtype=np.float32)
                dist = float(np.linalg.norm(rel_pos))
                if dist < 1e-3 or dist > self.ttc_max_consider_distance_m:
                    continue

                vvel = vehicle.get_velocity()
                other_v = np.array([vvel.x, vvel.y], dtype=np.float32)
                rel_vel = ego_v - other_v
                closing_speed = float(np.dot(rel_pos / max(dist, 1e-6), rel_vel))
                if closing_speed > 1e-3:
                    ttc = dist / closing_speed
                    if ttc < min_ttc:
                        min_ttc = ttc

            if np.isfinite(min_ttc):
                return float(min_ttc)
            return -1.0
        except Exception:
            return -1.0

    def _lidar_handler(self, point_cloud):
        """Store the latest LiDAR data."""
        if self._is_resetting or self._is_closing or self._is_closed or self.ego is None:
            return
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
                'goal_relative': np.zeros(4, dtype=np.float32),
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

        # 6. Goal-relative feature (Q2B)
        goal_relative = self._get_goal_relative_obs()

        # 7. Get camera images
        front_camera_img = self.front_camera_image if self.front_camera_image is not None else np.zeros((600, 800, 3), dtype=np.uint8)
        bev_camera_img = self.bev_camera_image if self.bev_camera_image is not None else np.zeros((600, 800, 3), dtype=np.uint8)

        observation = {
            'lidar': lidar,
            'ego_state': ego_state,
            'nearby_vehicles': nearby_vehicles,
            'waypoints': waypoints,
            'lane_info': lane_info,
            'goal_relative': goal_relative,
            'front_camera': front_camera_img,
            'bev_camera': bev_camera_img,
        }
        return observation


    def _terminal(self):
        """Check if the episode is terminal."""
        # Reset sticky flags per-step; recompute from current state.
        self._is_collision = False
        self._is_off_road = False

        # Collision check with configurable intensity threshold
        collision_intensity = max(self.collision_hist) if len(self.collision_hist) > 0 else 0.0
        self._last_risk_metrics['collision_intensity'] = float(collision_intensity)
        if collision_intensity >= self.collision_intensity_threshold:
            self._is_collision = True
            self._task_failure_reason = 'collision'
            return True

        # Off-road check (based on lane type persistence)
        current_waypoint = self.map.get_waypoint(self.ego.get_location(), project_to_road=False)
        if current_waypoint is None or current_waypoint.lane_type != carla.LaneType.Driving:
            self._is_off_road = True
            self.off_road_counter += 1
            if self.off_road_counter > self.offroad_terminal_steps:
                self._task_failure_reason = 'off_road'
                return True
        else:
            self._is_off_road = False
            self.off_road_counter = 0

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

            # If explicit goal exists, goal radius check takes precedence.
            if self._goal_location is not None:
                dist_to_goal = cur_loc.distance(self._goal_location)
                if dist_to_goal <= float(getattr(self, '_task_goal_tolerance_m', self.params.get('task_goal_tolerance_m', 1.0))):
                    self._task_success = True
                    return True

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

        # Symmetric desired-speed tracking reward (penalize both too slow and too fast)
        if self.enable_reward_speed_tracking:
            speed_err = abs(v - float(self.desired_speed))
            tol = max(0.0, float(self.reward_speed_tolerance_kmh))
            scale = max(1e-6, float(self.reward_speed_scale_kmh))
            reward_speed = self.reward_speed_tracking_weight * (1.0 - max(0.0, speed_err - tol) / scale)
        else:
            reward_speed = 0.0

        # Penalty for being off road
        penalty_off_road = -1.0 if self._is_off_road else 0.0

        # Collision penalty is configurable and tied to terminal collision state
        penalty_collision = self.collision_soft_penalty if self._is_collision else 0.0

        # Reward for following waypoints (progress)
        transform = self.ego.get_transform()
        vel_vec = self.ego.get_velocity()
        forward_vec = transform.get_forward_vector()
        speed_factor = np.dot([vel_vec.x, vel_vec.y], [forward_vec.x, forward_vec.y])
        reward_progress = speed_factor / 10.0

        # Combine rewards
        reward = reward_speed + reward_progress + penalty_off_road + penalty_collision

        # Lightweight task-shaped reward for multitask mode (ablation switch)
        if (
            self.enable_reward_task_shaping and
            self.task_mode == 'multitask' and
            self.ego is not None and
            self._episode_start_location is not None
        ):
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
                reward += self.collision_hard_penalty
            elif self._is_off_road:
                reward -= 50.0
            elif self._task_failure_reason == 'timeout':
                reward += self.timeout_terminal_bonus
            else:
                reward += 10.0  # Non-timeout, non-failure terminal bonus

        return reward

    def _get_cost(self, obs):
        """Calculate safety/risk cost for constrained RL."""
        _, _, dist_from_center_norm, _ = get_lane_info(self.ego, self.map)

        # 1) Base lane deviation cost
        cost_lane_deviation = max(0.0, dist_from_center_norm - 0.5)

        # 2) Proximity cost from LiDAR occupancy
        lidar_distances = obs['lidar']
        critical_threshold = 0.1  # 10% of max range
        cost_proximity = np.sum(lidar_distances < critical_threshold) / max(1, len(lidar_distances))

        # 3) Semi-trailer articulation & jackknife risk
        articulation_deg = self._compute_articulation_deg()
        articulation_norm = max(0.0, (articulation_deg - self.articulation_safe_angle_deg) / max(1e-6, (90.0 - self.articulation_safe_angle_deg)))
        jackknife_flag = 1.0 if articulation_deg >= self.jackknife_angle_deg else 0.0

        # 4) Trailer off-road/lane-boundary risk
        trailer_offroad = self._compute_trailer_offroad_flag()

        # 5) Time-to-collision risk
        min_ttc = self._compute_min_ttc()
        if min_ttc > 0.0:
            cost_ttc = max(0.0, (self.ttc_threshold_s - min_ttc) / max(1e-6, self.ttc_threshold_s))
        else:
            cost_ttc = 0.0

        # 6) Optional speeding risk (only above desired_speed + margin)
        if self.enable_cost_speeding:
            v_kmh = get_current_speed(self.ego)
            overspeed = v_kmh - (float(self.desired_speed) + float(self.cost_speeding_margin_kmh))
            cost_speeding = max(0.0, overspeed / max(1e-6, float(self.cost_speeding_scale_kmh)))
        else:
            cost_speeding = 0.0

        total_cost = (
            (self.cost_lane_deviation_weight * cost_lane_deviation if self.enable_cost_lane_deviation else 0.0) +
            (self.cost_proximity_weight * cost_proximity if self.enable_cost_proximity else 0.0) +
            (self.cost_articulation_weight * articulation_norm if self.enable_cost_articulation else 0.0) +
            (self.cost_jackknife_weight * jackknife_flag if self.enable_cost_jackknife else 0.0) +
            (self.cost_trailer_offroad_weight * trailer_offroad if self.enable_cost_trailer_offroad else 0.0) +
            (self.cost_ttc_weight * cost_ttc if self.enable_cost_ttc else 0.0) +
            (self.cost_speeding_weight * cost_speeding if self.enable_cost_speeding else 0.0)
        )

        # Cache metrics for info/logging
        self._last_risk_metrics['articulation_deg'] = float(articulation_deg)
        self._last_risk_metrics['jackknife'] = float(jackknife_flag)
        self._last_risk_metrics['trailer_offroad'] = float(trailer_offroad)
        self._last_risk_metrics['min_ttc_s'] = float(min_ttc)
        self._last_risk_metrics['cost_speeding'] = float(cost_speeding)

        return float(total_cost)


    def close(self):
        if self._is_closed:
            return

        print("\n>>> [CLOSE] Shutting down Carla environment...")
        self._is_closing = True
        self._is_resetting = False

        # Use the same comprehensive cleanup as in reset
        self._clear_all_actors()

        # Restore world settings
        try:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            # Tick once more to ensure settings take effect
            self.world.tick()
        except Exception as e:
            print(f">>> [CLOSE] Warning while restoring world settings: {e}")

        self._is_closing = False
        self._is_closed = True
        print(">>> [CLOSE] Environment closed successfully.")
        print(">>> [CLOSE] Environment closed successfully.")