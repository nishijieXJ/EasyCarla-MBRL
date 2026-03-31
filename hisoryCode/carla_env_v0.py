# -*- coding: utf-8 -*-
"""
Author: SilverWings
GitHub: https://github.com/silverwingsbot
"""

from __future__ import division
import numpy as np
import random
import time
#import gym
#from gym import spaces
#from gym.utils import seeding
import gymnasium as gym
from gymnasium import spaces
# 注意：gymnasium 移除了 gym.utils.seeding，改用标准库
import random
import numpy as np
import carla

class CarlaEnv(gym.Env):
    def __init__(self, params):
        # Initialize actor lists and polygons for collision checks
        self.spawned_vehicles = []
        self.spawned_walkers = []
        self.used_spawn_points = []
        # --- 新增这两行 ---
        self.vehicle_polygons = [] # 初始化为空列表
        self.walker_polygons = [] # 初始化为空列表
        # -------------------
        self.ego = None        
        self.collision_sensor = None
        self.lidar_sensor = None
        self._is_collision = False
        self._is_off_road = False
        self.off_road_counter = 0
        # 👇 确保这行存在，从 params 字典中获取 max_steps 并赋值给实例属性
        self.max_steps = params.get("max_steps", 1000) # 1000 是一个默认值
        # 确保 time_step 也在初始化时被定义
        self.time_step = 0
        self.total_step = 0
        self.reset_step = 0
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
        # self.observation_space = spaces.Dict({
        #     'lidar': spaces.Box(low=0.0, high=1.0, shape=(240,), dtype=np.float32),
        #     'ego_state': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        #     'nearby_vehicles': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nearby_vehicles, 6), dtype=np.float32),
        #     'waypoints': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_waypoints, 4), dtype=np.float32),
        #     'lane_info': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        # })
        # 在 __init__ 中替换 observation_space 定义为：
        self.observation_space = spaces.Dict({
            'lidar': spaces.Box(low=0.0, high=1.0, shape=(240,), dtype=np.float32),
            'ego_state': spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32),  # ← 改为 9
            'nearby_vehicles': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_nearby_vehicles * 4,),  # ← flatten to 1D: (5*4=20,)
                dtype=np.float32
            ),
            'waypoints': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_waypoints * 3,),  # ← flatten to 1D: (12*3=36,)
                dtype=np.float32
            ),
            'lane_info': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32)
        )
    
        print('Connecting to Carla server...')
        #client = carla.Client('localhost', params['port'])
        self.client = carla.Client('localhost', params['port'])  # ← self.client
        self.client.set_timeout(10.0)                     # ← self.client
        self.world = self.client.load_world(params['town'])# ← self.client
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        print('Connection established!')
    
        # Get all predefined vehicle spawn points from the map
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
        # Prepare a list to hold spawn points for pedestrians (walkers)
        self.walker_spawn_points = []
        # Randomly generate spawn points for the specified number of pedestrians
        for i in range(self.number_of_walkers):
            spawn_point = carla.Transform()  # Create an empty transform object
            # Try to get a random navigable location in the environment
            loc = self.world.get_random_location_from_navigation()
            # If a valid location is found, use it as a spawn point for a pedestrian
            if loc is not None:
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)
    
    
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='255,0,0')
    
        self.collision_hist = []
        self.collision_hist_l = 1
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
    
        self.lidar_data = None  # Placeholder to store incoming LiDAR data
        self.lidar_height = 0.8  # Height at which the LiDAR is mounted on the vehicle (in meters)
        # Set the position of the LiDAR sensor using a transform (translation only in Z direction)
        self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
        # Get the LiDAR blueprint from Carla's sensor library
        self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        # Set LiDAR attributes
        self.lidar_bp.set_attribute('channels', '1')  # Use 1 channel to perform a flat 360° horizontal scan
        self.lidar_bp.set_attribute('range', '50')  # Maximum LiDAR range in meters
        self.lidar_bp.set_attribute('rotation_frequency', '10')  # How many full 360° rotations per second
        self.lidar_bp.set_attribute('points_per_second', '10000')  # Total number of points generated per second
        self.lidar_bp.set_attribute('upper_fov', '0')  # upper and lower FOV are both 0 for a flat horizontal scan
        self.lidar_bp.set_attribute('lower_fov', '0')  
        # 👇 在这里添加缺失的属性定义
        self.lidar_num_channels = int(float(self.lidar_bp.get_attribute('channels').recommended_values[0]))
        # 对于单线水平扫描，每通道的点数是 points_per_second / rotation_frequency
        points_per_sec_str = self.lidar_bp.get_attribute('points_per_second').recommended_values[0]
        rotation_freq_str = self.lidar_bp.get_attribute('rotation_frequency').recommended_values[0]
        self.lidar_points_per_channel = int(float(points_per_sec_str)) // int(float(rotation_freq_str))
        # self.lidar_bp.set_attribute('channels', '32')      # 改为 32 线
        # self.lidar_bp.set_attribute('upper_fov', '10')     # 上视场角 10°
        # self.lidar_bp.set_attribute('lower_fov', '-30')    # 下视场角 -30°
        # self.lidar_bp.set_attribute('range', '50')
        # self.lidar_bp.set_attribute('points_per_second', '100000')
        # self.lidar_bp.set_attribute('rotation_frequency', '20')
    
    
        self.settings = self.world.get_settings()  # Get the current world settings
        self.settings.fixed_delta_seconds = self.dt  # Set the physics simulation step size (in seconds)
                                                      # This ensures consistent time intervals for simulation updates
    
    
        self.reset_step = 0
        self.sync_mode = params.get('sync_mode', False)  #修改zk 默认 False 兼容旧配置
        self.delta_seconds = params.get('delta_seconds', self.dt) #修改zk
        self.total_step = 0

    # def reset(self):
    #     # Stop and destroy the collision sensor if it exists
    #     if self.collision_sensor is not None:
    #         try:
    #             self.collision_sensor.stop()
    #             self.collision_sensor.destroy()
    #         except:
    #             pass
    #         self.collision_sensor = None
    
    #     # Stop and destroy the LiDAR sensor if it exists
    #     if self.lidar_sensor is not None:
    #         try:
    #             self.lidar_sensor.stop()
    #             self.lidar_sensor.destroy()
    #         except:
    #             pass
    #         self.lidar_sensor = None
    
    #     # Reset collision and off-road status flags
    #     self._is_collision = False
    #     self._is_off_road = False
    
    #     self._set_synchronous_mode(False)  # Switch back to asynchronous mode
    #     self._clear_all_actors([
    #         'sensor.other.collision',
    #         'sensor.lidar.ray_cast',
    #         'sensor.camera.rgb',
    #         'vehicle.*',
    #         'controller.ai.walker',
    #         'walker.*'
    #     ])  # Remove all specified actors from the world

    #     # Spawn surrounding vehicles
    #     random.shuffle(self.vehicle_spawn_points)
    #     count = self.number_of_vehicles
    #     self.spawned_vehicles = []
    #     self.used_spawn_points = []
        
    #     if count > 0:
    #         for spawn_point in self.vehicle_spawn_points:
    #             vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
    #             if vehicle:
    #                 self.spawned_vehicles.append(vehicle)  # Record the spawned vehicle
    #                 self.used_spawn_points.append(spawn_point)  # Mark spawn point as used
    #                 count -= 1
    #             if count <= 0:
    #                 break
    #     # print(f"Surrounding vehicles number is {len(self.spawned_vehicles)}")

    #     # Spawn pedestrians
    #     random.shuffle(self.walker_spawn_points)
    #     count = self.number_of_walkers
        
    #     if count > 0:
    #         for spawn_point in self.walker_spawn_points:
    #             if self._try_spawn_random_walker_at(spawn_point):
    #                 count -= 1
    #             if count <= 0:
    #                 break
        
    #     # Try random spawn points until all pedestrians are spawned
    #     while count > 0:
    #         if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
    #             count -= 1

    #     # Get actors' polygon list
    #     # Calculate and collect the bounding polygons (e.g., four corners) of surrounding vehicles and pedestrians
    #     self.vehicle_polygons = []
    #     vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    #     self.vehicle_polygons.append(vehicle_poly_dict)
        
    #     self.walker_polygons = []
    #     walker_poly_dict = self._get_actor_polygons('walker.*')
    #     self.walker_polygons.append(walker_poly_dict)

    #     # Spawn the ego vehicle
    #     ego_spawn_times = 0
    #     while True:
    #         if ego_spawn_times > self.max_ego_spawn_times:
    #             self.reset()  # If failed too many times, reset the environment
        
    #         # Select a spawn point for the ego vehicle by excluding locations used by nearby vehicles
    #         available_spawn_points = [
    #             sp for sp in self.vehicle_spawn_points if sp not in self.used_spawn_points
    #         ]
            
    #         if len(available_spawn_points) > 0:
    #             transform = random.choice(available_spawn_points)  # Choose a spawn point not used by nearby vehicles
    #         else:
    #             transform = random.choice(self.vehicle_spawn_points)  # Fallback: use any spawn point
        
    #         # Try to spawn the ego vehicle at the selected location
    #         if self._try_spawn_ego_vehicle_at(transform):
    #             break  # Successfully spawned the ego vehicle
    #         else:
    #             ego_spawn_times += 1  # Retry counter
    #             time.sleep(0.1)  # Small delay before retrying

    #     if self.traffic == 'off':
    #         # Set all traffic lights to green and freeze them
    #         for actor in self.world.get_actors().filter('traffic.traffic_light*'):
    #             actor.set_state(carla.TrafficLightState.Green)
    #             actor.freeze(True)
    #     elif self.traffic == 'on':
    #         # Allow traffic lights to work normally
    #         for actor in self.world.get_actors().filter('traffic.traffic_light*'):
    #             actor.freeze(False)

    #     # Add collision sensor
    #     self.collision_sensor = self.world.spawn_actor(
    #         self.collision_bp,
    #         carla.Transform(),  # Attach at the center of the ego vehicle (no offset)
    #         attach_to=self.ego
    #     )
        
    #     # Start listening for collision events
    #     self.collision_sensor.listen(
    #         lambda event: get_collision_hist(event)  # When a collision event happens, pass the event to get_collision_hist()
    #     )

    #     def get_collision_hist(event):
    #         impulse = event.normal_impulse  # Get the collision impulse (a 3D vector)
    #         intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)  # Calculate collision intensity (vector norm)
    #         self.collision_hist.append(intensity)  # Record the collision intensity
    #         if len(self.collision_hist) > self.collision_hist_l:
    #             self.collision_hist.pop(0)  # Keep only the latest collision records (FIFO)
        
    #     # Initialize collision history list
    #     # Clear collision history after each episode because in gym-carla setup,
    #     # a collision typically triggers episode termination and reset.
    #     self.collision_hist = []

    #     # Add lidar sensor
    #     self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
    #     self.lidar_sensor.listen(lambda data: get_lidar_data(data))
    #     def get_lidar_data(data):
    #         self.lidar_data = data

    #     # Update timesteps
    #     self.time_step = 1  # Indicates a new episode has started
    #     self.reset_step += 1  # Count how many resets have occurred
        
    #     # Enable autopilot for all surrounding vehicles
    #     for vehicle in self.spawned_vehicles:
    #         vehicle.set_autopilot()
        
    #     self._set_synchronous_mode(True)  # Switch to synchronous mode for simulation
    #     self.world.tick()  # Advance the simulation by one tick
        
    #     return self._get_obs()  # Return the initial observation after reset
    # def reset(self, *, seed=None, options=None):
    #     print(">>> [RESET] Starting reset...")
        
    #     # 处理随机种子（Gymnasium 要求）
    #     if seed is not None:
    #         random.seed(seed)
    #         np.random.seed(seed)
    #     ego_spawned = False
    #     # 尝试 spawn ego vehicle（最多 self.max_ego_spawn_times 次）
    #     for attempt in range(self.max_ego_spawn_times + 1):
    #         print(f">>> [RESET] Attempt {attempt + 1}/{self.max_ego_spawn_times + 1}")
    #         # ========== 第一步：彻底清理上一局所有 actors ==========
    #         print(">>> [RESET] step1 Cleaning up old actors...")
    #         # 1. 安全地销毁旧的 ego 车辆
    #         if hasattr(self, 'ego') and self.ego is not None:
    #             try:
    #                 # 关闭其 autopilot
    #                 self.ego.set_autopilot(False)
    #                 # 确保车辆停止
    #                 self.ego.apply_control(carla.VehicleControl(brake=1.0))
    #                 # tick 一下让物理引擎处理
    #                 self.world.tick()
    #                 # 销毁
    #                 if self.ego.is_alive:
    #                     self.ego.destroy()
    #                 print(f">>> [RESET] Destroyed old ego vehicle {self.ego.id if self.ego else 'None'}.")
    #                 # 从内存中移除引用
    #                 #self.ego = None
    #             except Exception as e:
    #                 print(f">>> [RESET] Warning: Error destroying old ego: {e}")
    #             finally:
    #                 self.ego = None # 确保引用被清空                    

    #         # 2. 安全地销毁旧的传感器
    #         if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
    #             try:
    #                 self.collision_sensor.stop()
    #                 if self.collision_sensor.is_listening:# 等待几帧确保回调停止
    #                     # 等待几帧确保回调停止
    #                     for _ in range(3):
    #                         self.world.tick()
    #                 if self.collision_sensor.is_alive: # 👈 检查是否还活着
    #                     self.collision_sensor.destroy()
    #                 print(f">>> [RESET] Destroyed old collision sensor {self.collision_sensor.id if self.collision_sensor else 'None'}.")
    #             except Exception as e:
    #                 print(f">>> [RESET] Warning: Error destroying old collision sensor: {e}")
    #             finally:
    #                 self.collision_sensor = None

    #         if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
    #             try:
    #                 self.lidar_sensor.stop()
    #                 if self.lidar_sensor.is_listening: # 等待几帧确保回调停止
    #                     for _ in range(3):
    #                         self.world.tick()
    #                 if self.lidar_sensor.is_alive: # 👈 检查是否还活着
    #                     self.lidar_sensor.destroy()
    #                 print(f">>> [RESET] Destroyed old lidar sensor {self.lidar_sensor.id if self.lidar_sensor else 'None'}.")
    #             except Exception as e:
    #                 print(f">>> [RESET] Warning: Error destroying old lidar sensor: {e}")
    #             finally:
    #                 self.lidar_sensor = None

    #         # 3. 清理其他 actors
    #         try:
    #             self._clear_all_actors([
    #                 'sensor.other.collision',
    #                 'sensor.lidar.ray_cast',
    #                 'sensor.camera.rgb',
    #                 'vehicle.*',
    #                 'controller.ai.walker',
    #                 'walker.*'
    #             ])
    #         except Exception as e:
    #             print(f">>> [RESET] Warning: Error during general cleanup: {e}")
                
    #         print(">>> [RESET] Cleanup done.")            
    #         # ========== 第二步：重置内部状态 ==========
    #         self._is_collision = False
    #         self._is_off_road = False
    #         self.off_road_counter = 0
    #         self.collision_hist = []

    #         # 获取新的 spawn points
    #         self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())
    #         self.walker_spawn_points = []
    #         for i in range(self.number_of_walkers):
    #             spawn_point = carla.Transform()
    #             loc = self.world.get_random_location_from_navigation()
    #             if loc is not None:
    #                 spawn_point.location = loc
    #                 self.walker_spawn_points.append(spawn_point)
    #         print(">>> [RESET] step2 reset inner states done.")

    #         # ========== 第三步：spawn 周围车辆 ==========
    #         print(">>> [RESET] step3 Spawning surrounding vehicles...")
    #         random.shuffle(self.vehicle_spawn_points)
    #         count = self.number_of_vehicles
    #         print(f">>> [RESET] Debug: self.number_of_vehicles = {self.number_of_vehicles}")
    #         self.spawned_vehicles = []
    #         self.used_spawn_points = []
    #         if count > 0:
    #             for spawn_point in self.vehicle_spawn_points:
    #                 vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
    #                 if vehicle:
    #                     self.spawned_vehicles.append(vehicle)
    #                     self.used_spawn_points.append(spawn_point)
    #                     count -= 1
    #                     if count <= 0:
    #                         break
    #         print(f">>> [RESET] Spawned {len(self.spawned_vehicles)} vehicles.")

    #         # ========== 第四步：spawn 行人 ==========
    #         random.shuffle(self.walker_spawn_points)
    #         count = self.number_of_walkers
    #         if count > 0:
    #             for spawn_point in self.walker_spawn_points:
    #                 if self._try_spawn_random_walker_at(spawn_point):
    #                     count -= 1
    #                     if count <= 0:
    #                         break
    #             while count > 0:
    #                 if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
    #                     count -= 1

    #         # # ========== 第五步：获取多边形（用于碰撞检测）==========
    #         # self.vehicle_polygons = []
    #         # vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    #         # self.vehicle_polygons.append(vehicle_poly_dict)
    #         # self.walker_polygons = []
    #         # walker_poly_dict = self._get_actor_polygons('walker.*')
    #         # self.walker_polygons.append(walker_poly_dict)
    #         # ========== 【新增/修改】第五步：获取多边形（用于碰撞检测）==========
    #         # 这一步必须在周围车辆生成之后，ego车辆生成之前！
    #         print(">>> [RESET] step5 Getting actor polygons for collision check...")
    #         # 重新获取当前环境中所有车辆的多边形
    #         vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
    #         # 重新初始化 self.vehicle_polygons 列表，只包含当前帧的数据
    #         self.vehicle_polygons = [vehicle_poly_dict]

    #         # 重新获取当前环境中所有行人的多边形
    #         walker_poly_dict = self._get_actor_polygons('walker.*')
    #         # 重新初始化 self.walker_polygons 列表，只包含当前帧的数据
    #         self.walker_polygons = [walker_poly_dict]

    #         # ========== 第六步：尝试 spawn ego vehicle ==========
    #         print(">>> [RESET] step6 Trying to spawn ego vehicle...")
    #         available_spawn_points = [sp for sp in self.vehicle_spawn_points if sp not in self.used_spawn_points]
    #         if not available_spawn_points:
    #             available_spawn_points = self.vehicle_spawn_points  # fallback

    #         transform = random.choice(available_spawn_points)
    #         if self._try_spawn_ego_vehicle_at(transform):
    #             # 成功 spawn ego，跳出循环
    #             print(">>> [RESET] Ego vehicle spawned successfully!")
    #             # 【新增】关键一步：让 ego 完成初始化
    #             if not self.ego.is_alive:
    #                 raise RuntimeError("Ego is not alive after spawn!")
    #             self.world.tick()  # 👈 必须加在这里！
    #             print(">>> [RESET] Ticked after ego spawn.")
    #             ego_spawned = True
    #             break
    #         # 否则继续下一次尝试
    #         else:                
    #             print(f">>> [RESET] Attempt {attempt + 1} failed to spawn ego.")
    #             #raise RuntimeError(f"Failed to spawn ego vehicle after {self.max_ego_spawn_times} attempts")
    #     if not ego_spawned:
    #         print(">>> [RESET] Failed to spawn ego.")
    #         raise RuntimeError("Failed to spawn ego vehicle after all attempts!")

    #     # ========== 第七步：设置交通灯 ==========
    #     if self.traffic == 'off':
    #         for actor in self.world.get_actors().filter('traffic.traffic_light*'):
    #             actor.set_state(carla.TrafficLightState.Green)
    #             actor.freeze(True)
    #     elif self.traffic == 'on':
    #         for actor in self.world.get_actors().filter('traffic.traffic_light*'):
    #             actor.freeze(False)

    #     # ========== 第八步：挂载传感器 ==========
    #     print(">>> [RESET] step8 Attaching sensors...")
    #     print(">>> [RESET] Attaching collision sensor...")
    #     self.collision_sensor = self.world.spawn_actor(
    #         self.collision_bp, carla.Transform(), attach_to=self.ego
    #     )
    #     self.collision_hist = []
    #     self.collision_sensor.listen(self._collision_handler)  # ← 直接传方法
    #     print(">>> [RESET] Collision sensor attached.")
    #     # def get_collision_hist(event):
    #     #     impulse = event.normal_impulse
    #     #     intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
    #     #     self.collision_hist.append(intensity)
    #     #     if len(self.collision_hist) > self.collision_hist_l:
    #     #         self.collision_hist.pop(0)
    #     # self.collision_sensor.listen(lambda event: get_collision_hist(event))

    #     self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego)
    #     self.lidar_sensor.listen(self._lidar_handler)  # ← 直接传方法
    #     # def get_lidar_data(data):
    #     #     self.lidar_data = data
    #     # self.lidar_sensor.listen(lambda data: get_lidar_data(data))
    #     print(">>> [RESET] Sensors attached.")

    #     # ========== 第九步：启用周围车辆的 autopilot ==========
    #     print(">>> [RESET] step9 Setting autopilot...")
    #     for vehicle in self.spawned_vehicles:
    #         vehicle.set_autopilot(True)
    #     print(">>> [RESET] Autopilot set.")

    #     # ========== 第十步：应用同步模式设置 ==========
    #     print(">>> [RESET] step10 Applying world settings...")
    #     settings = self.world.get_settings()
    #     settings.synchronous_mode = getattr(self, 'sync_mode', False)
    #     settings.fixed_delta_seconds = getattr(self, 'delta_seconds', self.dt)
    #     self.world.apply_settings(settings)
    #     #self.world.tick()  # 确保设置生效
    #     #print(">>> [RESET] World ticked.")

    #     # ========== 第十一步：返回观测 ==========
    #     # Update timesteps
    #     self.time_step = 1  # Indicates a new episode has started
    #     self.reset_step += 1  # Count how many resets have occurred
    #     self.total_step = 0  # 👈 确保这行也存在
    #     obs, _ = self._get_obs(), {}
    #     print("Lidar shape:", obs['lidar'].shape) # Should be (240,)
    #     print("Ego shape:", obs['ego_state'].shape) # Should be (9,)
    #     print("Nearby shape:", obs['nearby_vehicles'].shape) # Should be (20,) if max=5
    #     print("Waypoints shape:", obs['waypoints'].shape) # Should be (36,) if max=12
    #     print("Lane shape:", obs['lane_info'].shape) # Should be (2,)
    #     print(">>> [RESET] step11 Reset completed.")
    #     # Debug: Check shapes before returning
    #     debug_obs, _ = self._get_obs(), {}
    #     print("--- DEBUG OBS SHAPES ---")
    #     for k, v in debug_obs.items():
    #         print(f"  {k}: {v.shape if hasattr(v, 'shape') else type(v)} -> {type(v)}")
    #     print("-----------------------")
    #     return obs, {}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print(">>> [RESET] Starting reset...")

        # ========== 第一步：彻底清理上一局所有 actors ==========
        print(">>> [RESET] step1 Cleaning up old actors...")
        # (保留你已有的清理逻辑，但加上 is_alive 检查)
        # 1. 安全地销毁旧的 ego 车辆
        if hasattr(self, 'ego') and self.ego is not None:
            try:
                self.ego.set_autopilot(False)
                self.ego.apply_control(carla.VehicleControl(brake=1.0))
                self.world.tick()
                if self.ego.is_alive: # 👈 检查是否还活着
                    self.ego.destroy()
                print(f">>> [RESET] Destroyed old ego vehicle {self.ego.id}.")
            except Exception as e:
                print(f">>> [RESET] Warning: Error destroying old ego: {e}")
            finally:
                self.ego = None

        # 2. 安全地销毁旧的传感器
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            try:
                self.collision_sensor.stop()
                # 等待几帧确保回调停止
                for _ in range(3):
                    self.world.tick()
                if self.collision_sensor.is_alive: # 👈 检查是否还活着
                    self.collision_sensor.destroy()
                print(f">>> [RESET] Destroyed old collision sensor {self.collision_sensor.id}.")
            except Exception as e:
                print(f">>> [RESET] Warning: Error destroying old collision sensor: {e}")
            finally:
                self.collision_sensor = None

        if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
            try:
                self.lidar_sensor.stop()
                # 等待几帧确保回调停止
                for _ in range(3):
                    self.world.tick()
                if self.lidar_sensor.is_alive: # 👈 检查是否还活着
                    self.lidar_sensor.destroy()
                print(f">>> [RESET] Destroyed old lidar sensor {self.lidar_sensor.id}.")
            except Exception as e:
                print(f">>> [RESET] Warning: Error destroying old lidar sensor: {e}")
            finally:
                self.lidar_sensor = None

        # 3. 清理其他 actors
        try:
            self._clear_all_actors([
                'sensor.other.collision',
                'sensor.lidar.ray_cast',
                'sensor.camera.rgb',
                'vehicle.*',
                'controller.ai.walker',
                'walker.*'
            ])
        except Exception as e:
            print(f">>> [RESET] Warning: Error during general cleanup: {e}")

        print(">>> [RESET] Cleanup done.")

        # ========== 第二步：重置内部状态 ==========
        print(">>> [RESET] step2 reset inner states done.")
        # 重置计数器和列表
        self.time_step = 0
        self.total_reward = 0
        self.collision_hist = []
        self.lane_invasion_hist = []
        self.waypoint_route = []
        self.success = False
        self.off_road = False
        self.collided = False
        self.spawned_vehicles = []
        self.spawned_walkers = []
        self.used_spawn_points = []
        # --- 重置多边形列表 ---
        self.vehicle_polygons = []
        self.walker_polygons = []
        # -----------------------

        # ！！！【关键修复】！！！
        # 确保在重置 lidar_data 之前，相关参数已被定义
        # 从类的配置中获取参数，而不是假设它们已经被设置了
        # 如果这些参数在 __init__ 中是通过 self.lidar_num_channels = ... 这样的方式设置的，
        # 那么它们应该在 __init__ 结束后就存在。如果不是，则需要在这里重新定义。
        # 通常，这些值在 __init__ 中定义，所以直接使用即可。
        # 为了绝对安全，我们可以先检查一下，如果没有再报错或给出默认值。
        if not hasattr(self, 'lidar_num_channels'):
            raise AttributeError("Attribute 'lidar_num_channels' is missing from CarlaEnv. Please check __init__ method.")
        if not hasattr(self, 'lidar_points_per_channel'):
            raise AttributeError("Attribute 'lidar_points_per_channel' is missing from CarlaEnv. Please check __init__ method.")

        # 重置传感器数据
        self.lidar_data = np.zeros((self.lidar_num_channels, self.lidar_points_per_channel), dtype=np.float32)
        self.collision_hist = []

        # ========== 第三步：生成周围车辆 ==========
        print(">>> [RESET] step3 Spawning surrounding vehicles...")
        random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles # 👈 确保 self.number_of_vehicles 已定义
        print(f">>> [RESET] Debug: self.number_of_vehicles = {self.number_of_vehicles}") # 添加调试信息
        # 注意：如果 vehicle_spawn_points 不足，count 可能会大于 0 但实际生成的车辆少于 count
        for spawn_point in self.vehicle_spawn_points:
            if count <= 0:
                break
            vehicle = self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4])
            if vehicle:
                self.spawned_vehicles.append(vehicle)
                self.used_spawn_points.append(spawn_point)
                count -= 1
        print(f">>> [RESET] Spawned {len(self.spawned_vehicles)} vehicles.")

        # ========== 第四步：生成行人 ==========
        print(">>> [RESET] step4 Spawning walkers...")
        random.shuffle(self.walker_spawn_points)
        count = self.number_of_walkers
        for spawn_point in self.walker_spawn_points:
            if count <= 0:
                break
            walker = self._try_spawn_random_walker_at(spawn_point)
            if walker:
                self.spawned_walkers.append(walker)
                self.used_spawn_points.append(spawn_point)
                count -= 1
        print(f">>> [RESET] Spawned {len(self.spawned_walkers)} walkers.")

        # ========== 第五步：获取 actor 多边形（用于碰撞检查） ==========
        # 这一步必须在周围车辆和行人生成之后，ego车辆生成之前！
        print(">>> [RESET] step5 Getting actor polygons for collision check...")
        # 获取当前环境中所有车辆的多边形
        current_vehicle_polys = self._get_actor_polygons('vehicle.*')
        # 获取当前环境中所有行人的多边形
        current_walker_polys = self._get_actor_polygons('walker.*')

        # 重新初始化列表，只包含当前帧的数据
        self.vehicle_polygons = [current_vehicle_polys]
        self.walker_polygons = [current_walker_polys]

        print(f">>> [RESET] Got polygons for {len(current_vehicle_polys)} vehicles and {len(current_walker_polys)} walkers.")


        # ========== 第六步：尝试生成 ego 车辆 ==========
        print(">>> [RESET] step6 Trying to spawn ego vehicle...")
        ego_spawned = False
        for attempt in range(self.max_ego_spawn_times):
            print(f">>> [RESET] Attempt {attempt + 1}/{self.max_ego_spawn_times}")
            available_spawn_points = [sp for sp in self.vehicle_spawn_points if sp not in self.used_spawn_points]
            if not available_spawn_points:
                available_spawn_points = self.vehicle_spawn_points # fallback if all points were used

            transform = random.choice(available_spawn_points)
            if self._try_spawn_ego_vehicle_at(transform):
                print(">>> [RESET] Ego vehicle spawned successfully!")
                # 【新增】关键一步：让 ego 完成初始化
                if not self.ego.is_alive:
                    raise RuntimeError("Ego is not alive after spawn!")
                self.world.tick() # 👈 必须加在这里！
                print(">>> [RESET] Ticked after ego spawn.")
                ego_spawned = True
                break # 成功则跳出循环
            else:
                print(f">>> [RESET] Attempt {attempt + 1} failed to spawn ego.")
                # 将失败的点加入已用列表，避免重复尝试
                self.used_spawn_points.append(transform)

        if not ego_spawned:
            raise RuntimeError(f"Failed to spawn ego vehicle after {self.max_ego_spawn_times} attempts.")

        # ========== 第八步：附加传感器 ==========
        print(">>> [RESET] step8 Attaching sensors...")
        self._attach_collision_sensor()
        print(">>> [RESET] Attaching collision sensor...")
        print(">>> [RESET] Collision sensor attached.")
        self._attach_lidar_sensor()
        print(">>> [RESET] Attaching lidar sensor...")
        print(">>> [RESET] Lidar sensor attached.")
        print(">>> [RESET] Sensors attached.")

        # ========== 第九步：设置 autopilot ==========
        print(">>> [RESET] step9 Setting autopilot...")
        self.ego.set_autopilot(False) # Start with manual control
        print(">>> [RESET] Autopilot set.")

        # ========== 第十步：应用世界设置 ==========
        print(">>> [RESET] step10 Applying world settings...")
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_seconds
        self.world.apply_settings(settings)
        self.world.tick() # Ensure settings take effect immediately
        print(">>> [RESET] step11 Reset completed.")

        # ========== 第十一步：获取初始观测 ==========
        obs = self._get_observation()
        info = self._get_info()
        print(f"Lidar shape: {obs['lidar'].shape}")
        print(f"Ego shape: {obs['ego_state'].shape}")
        print(f"Nearby shape: {obs['nearby_vehicles'].shape}")
        print(f"Waypoints shape: {obs['waypoints'].shape}")
        print(f"Lane shape: {obs['lane_info'].shape}")

        print("--- DEBUG OBS SHAPES ---")
        for key, value in obs.items():
            print(f"{key}: {value.shape} -> {type(value)}")
        print("-----------------------")

        return obs, info

    def step(self, action):
        # 新增安全检查
        if not hasattr(self, 'ego') or self.ego is None or not self.ego.is_alive:
            raise RuntimeError("Ego vehicle is invalid. Environment needs reset.")
        throttle = float(np.clip(action[0], 0.0, 1.0))
        steer    = float(np.clip(action[1], -1.0, 1.0))
        brake    = float(np.clip(action[2], 0.0, 1.0))

        # Apply control
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.ego.apply_control(control)

        self.world.tick()

        # Set spectator (camera) view
        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        if self.view_mode == 'top':
            # Top-down view (bird's eye)
            spectator.set_transform(
                carla.Transform(
                    transform.location + carla.Location(z=40),
                    carla.Rotation(pitch=-90)
                )
            )
        elif self.view_mode == 'follow':
            # Follow view (behind and above the ego vehicle)
            cam_location = transform.transform(carla.Location(x=-6.0, z=3.0))  # 6 meters behind, 3 meters above
            cam_rotation = carla.Rotation(pitch=-10, yaw=transform.rotation.yaw, roll=0)
            spectator.set_transform(carla.Transform(cam_location, cam_rotation))

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        obs = self._get_obs()
        done = bool(self._terminal())
        reward = self._get_reward(obs, done)
        cost = self._get_cost(obs)

        # state information
        # info = {
        #   'is_collision': bool(self._is_collision),
        #   'is_off_road': bool(self._is_off_road)
        # }
        # return (obs, reward, cost, done, info)
        info = {
            'is_collision': bool(self._is_collision),
            'is_off_road': bool(self._is_off_road),
            'cost': cost, # 将 cost 放到 info 字典里
        }

        # --- ⭐️【关键修改】---
        # 将 done 信号分解为 terminated 和 truncated
        # terminated: 因为任务失败（如碰撞）导致的结束
        # truncated: 因为达到最大步数等限制导致的结束
        terminated = bool(done) # 你的 _terminal() 通常表示失败
        truncated = bool(self.time_step >= self.max_steps) # 超过最大步数

        # 确保 reward 是 float
        reward = float(reward)
        
        # 返回符合 Gymnasium API 标准的元组
        # (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info


    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create a vehicle blueprint based on the given filter and wheel number.

        Args:
            actor_filter (str): Filter string to select vehicle types, e.g., 'vehicle.lincoln*' 
                                ('*' matches a series of models).
            color (str, optional): Desired vehicle color. Randomly chosen if None.
            number_of_wheels (list): A list of acceptable wheel numbers (default is [4]).

        Returns:
            bp (carla.ActorBlueprint): A randomly selected blueprint matching the criteria.
        """
        # Get all blueprints matching the actor filter
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []

        # Further filter blueprints based on the number of wheels
        # Keeping number_of_wheels as a list makes it flexible to match multiple types (e.g., cars, trucks)
        for nw in number_of_wheels:
            blueprint_library += [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]

        # Randomly select one blueprint from the filtered list
        bp = random.choice(blueprint_library)

        # Set the vehicle color
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        return bp

    def _set_synchronous_mode(self, synchronous=True):

        """Enable or disable synchronous mode for the simulation.
        Args:
            synchronous (bool):
                True to enable synchronous mode (server waits for client each frame),
                False to disable and run in asynchronous mode (default is True).
        """
        self.settings.synchronous_mode = synchronous  # Set the synchronous mode
        self.world.apply_settings(self.settings)  # Apply the updated settings to the world

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at a specific transform.
    
        Args:
            transform (carla.Transform): Location and orientation where the vehicle should be spawned.
            number_of_wheels (list): Acceptable number(s) of wheels for the vehicle blueprint.
            random_vehicle (bool): 
                False to use Tesla Model 3 with a blue color,
                True to randomly select a vehicle model and color (default).
    
        Returns:
            carla.Actor or None: Spawned vehicle actor if successful, otherwise None.
        """
        if self.surrounding_vehicle_spawned_randomly:
            # Randomly choose any vehicle blueprint
            blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
        else:
            # Fixed: Tesla Model 3 with blue color
            blueprint = self._create_vehicle_bluepprint('vehicle.tesla.model3', color='0,0,255', number_of_wheels=number_of_wheels)
        
        blueprint.set_attribute('role_name', 'autopilot')  # Set the vehicle to autopilot mode
    
        # Try to spawn the vehicle
        vehicle = self.world.try_spawn_actor(blueprint, transform)
    
        return vehicle if vehicle is not None else None

    def _try_spawn_random_walker_at(self, transform):
        """Try to spawn a walker at a specific transform with a random blueprint.
    
        Args:
            transform (carla.Transform): Location and orientation where the walker should be spawned.
    
        Returns:
            Bool: True if spawn is successful, False otherwise.
        """
        # Randomly select a walker blueprint
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
    
        # Make the walker vulnerable (can be affected by collisions)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
    
        # Try to spawn the walker actor
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
    
        if walker_actor is not None:
            # Spawn a controller for the walker
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
    
            # Start the controller to control the walker
            walker_controller_actor.start()
    
            # Move the walker to a random location
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
    
            # Set a random walking speed between 1 m/s and 2 m/s (default is 1.4 m/s)
            walker_controller_actor.set_max_speed(1 + random.random())
    
            return True  # Spawn and initialization successful
    
        return False  # Failed to spawn

    # def _try_spawn_ego_vehicle_at(self, transform):
    #     """Try to spawn the ego vehicle at a specific transform.
    
    #     Args:
    #         transform (carla.Transform): Target location and orientation.
    
    #     Returns:
    #         Bool: True if spawn is successful, False otherwise.
    #     """
    #     vehicle = None
    #     overlap = False
    
    #     # Check if ego position overlaps with surrounding vehicles
    #     for idx, poly in self.vehicle_polygons[-1].items():  # Use .items() to iterate over dict keys and values
    #         poly_center = np.mean(poly, axis=0)
    #         ego_center = np.array([transform.location.x, transform.location.y])
    #         dis = np.linalg.norm(poly_center - ego_center)
    
    #         if dis > 8:
    #             continue
    #         else:
    #             overlap = True
    #             break
    
    #     # If no overlap, try to spawn the ego vehicle
    #     if not overlap:
    #         vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
    
    #     if vehicle is not None:
    #         self.ego = vehicle
    #         return True
    
    #     return False
    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at a specific transform."""
        vehicle = None
        overlap = False # Check if ego position overlaps with surrounding vehicles

        # 👇 安全地访问 vehicle_polygons
        current_vehicle_polys = self.vehicle_polygons[-1] if self.vehicle_polygons else {}
        for idx, poly in current_vehicle_polys.items(): # 使用 current_vehicle_polys
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8: # Use the same threshold as in collision check
                continue
            else:
                overlap = True
                print(f">>> [DEBUG] Ego spawn blocked due to overlap with vehicle {idx} at distance {dis:.2f}")
                break

        # Also check against walkers
        current_walker_polys = self.walker_polygons[-1] if self.walker_polygons else {}
        if not overlap:
            for idx, poly in current_walker_polys.items(): # 使用 current_walker_polys
                poly_center = np.mean(poly, axis=0)
                ego_center = np.array([transform.location.x, transform.location.y])
                dis = np.linalg.norm(poly_center - ego_center)
                if dis > 3: # Smaller threshold for walkers
                    continue
                else:
                    overlap = True
                    print(f">>> [DEBUG] Ego spawn blocked due to overlap with walker {idx} at distance {dis:.2f}")
                    break


        # If no overlap, try to spawn the ego vehicle
        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
            if vehicle is not None:
                self.ego = vehicle
                print(f">>> [DEBUG] Attempted to spawn ego at {transform.location}")
                return True
            else:
                print(f">>> [DEBUG] try_spawn_actor returned None for ego at {transform.location}")
        else:
            print(f">>> [DEBUG] Overlap detected, skipping spawn at {transform.location}")

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.
    
        Args:
            filt: the filter indicating what type of actors we'll look at.
    
        Returns:
            actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
        """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt): 
            # Get all actors in the current world that meet the filt condition, such as vehicle.* or walker.*
            # Note that self.world.get_actors() retrieves all objects in the current simulation environment (vehicles, pedestrians, traffic lights, etc.).
    
            # Get x, y and yaw of the actor
            trans = actor.get_transform() 
            # Get the actor's global position (location) and heading angle (rotation).
    
            x = trans.location.x 
            # x, y are the actor's global coordinates.
    
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi 
            # yaw is the heading angle, whose unit is degrees, needs to be converted to radians (multiply by pi/180) to facilitate subsequent matrix calculations.
    
            # Get length and width
            bb = actor.bounding_box 
            # Get the "half-length" boundary.
    
            l = bb.extent.x 
            # "Half-length" in the x-direction (the distance from the center to the edge).
    
            w = bb.extent.y
            # "Half-width" in the y-direction (the distance from the center to the edge).
    
            # Get bounding box polygon in the actor's local coordinate
            # Take the vehicle center as the origin, build a local coordinate system, and list four corner points:
            # (l, w): front right corner, (l, -w): rear right corner, (-l, -w): rear left corner, (-l, w): front left corner
            poly_local = np.array([
                [l, w], [l, -w], [-l, -w], [-l, w]
            ]).transpose() 
            # Transpose() here is to facilitate subsequent matrix operations,
            # changing the matrix from (4,2) to (2,4) format.
    
            # Get rotation matrix to transform to global coordinate
            # This is a standard 2D rotation matrix: used to transform points from the local coordinate system to the global coordinate system.
            # Rotation matrix R = [cosθ  -sinθ]
            #                     [sinθ   cosθ]
            R = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw), np.cos(yaw)]
            ])
    
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0) 
            # np.matmul(R, poly_local):
            # Transform the four corners (in the local coordinate system) into the global direction through the rotation matrix.
            # After .transpose(), it becomes (4,2) format (one point per row).
            # + np.repeat([[x,y]],4,axis=0):
            # Add the global position offset of the vehicle/pedestrian to each point
            # to obtain the final polygon coordinates in the global coordinate system.
    
            actor_poly_dict[actor.id] = poly 
            # Store the calculated poly (a 4×2 array, four corner points in global coordinates)
            # with actor.id as the key into actor_poly_dict.
            # After returning, the entire dictionary structure:
            # {
            # actor_id_1: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # actor_id_2: np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]),
            # ...
            # }
    
        return actor_poly_dict

    def _get_obs(self):
        obs = {}
        
# ========================== LIDAR feature extraction (240 dimensions) ==========================
        max_range = self.lidar_max_range  # Set a maximum perception distance
        lidar_features = np.full((240,), max_range, dtype=np.float32)  # Initialize all values to the maximum distance
    
        # Get ego pose
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
        ego_yaw = np.deg2rad(ego_transform.rotation.yaw)
    
        # Traverse all point clouds
        for detection in self.lidar_data:
            x = detection.point.x
            y = detection.point.y
    
            # Rotate back to ego vehicle heading direction (make ego vehicle heading as 0 degrees)
            local_x = np.cos(-ego_yaw) * x - np.sin(-ego_yaw) * y
            local_y = np.sin(-ego_yaw) * x + np.cos(-ego_yaw) * y
    
            distance = np.sqrt(local_x**2 + local_y**2)
            angle = np.arctan2(local_y, local_x)  # Range [-π, π]
            angle_deg = (np.degrees(angle) + 360) % 360  # Map to [0, 360)
            index = int(angle_deg // 1.5)  # Each angular bin has a width of 1.5 degrees, 240 bins in total
        
            if index < 240:
                lidar_features[index] = min(lidar_features[index], distance)
    
        # Normalize to [0, 1]
        lidar_features /= max_range
        # 确保没有 NaN 或 inf
        lidar_features = np.clip(lidar_features, 0.0, 1.0)
    
        # Store into observation
        obs['lidar'] = lidar_features

# ========================== Ego vehicle state extraction =======================================
        velocity = self.ego.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        angular_velocity = self.ego.get_angular_velocity()
        acceleration = self.ego.get_acceleration()
        
        front_vehicle_distance = 0.0
        relative_speed = 0.0
        
        min_front_distance = 20.0  # Search range threshold
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        
        for vehicle in vehicle_list:
            if vehicle.id == self.ego.id:
                continue
        
            transform = vehicle.get_transform()
            rel_x = transform.location.x - ego_x
            rel_y = transform.location.y - ego_y
        
            local_x = np.cos(-ego_yaw) * rel_x - np.sin(-ego_yaw) * rel_y
            local_y = np.sin(-ego_yaw) * rel_x + np.cos(-ego_yaw) * rel_y
        
            if 0 < local_x < min_front_distance and abs(local_y) < 2.5:
                d = np.sqrt(local_x**2 + local_y**2)
                if front_vehicle_distance == 0.0 or d < front_vehicle_distance:
                    front_vehicle_distance = d
                    front_speed = vehicle.get_velocity()
                    front_speed_mag = np.sqrt(front_speed.x**2 + front_speed.y**2 + front_speed.z**2)
                    relative_speed = speed - front_speed_mag
        
        ego_state = np.array([
            ego_x,
            ego_y,
            ego_yaw,
            speed,
            angular_velocity.z,
            acceleration.x,
            acceleration.y,
            front_vehicle_distance,
            relative_speed
        ], dtype=np.float32)
        
        obs['ego_state'] = ego_state

# ================ Nearby vehicles state extraction (up to 5 vehicles, within perception range) ===============
        max_vehicles = self.max_nearby_vehicles
        perception_range = self.lidar_max_range
        vehicle_list = self.world.get_actors().filter('vehicle.*')
        
        vehicle_data = []
        for vehicle in vehicle_list:
            if vehicle.id == self.ego.id:
                continue  # Skip the ego vehicle itself
        
            transform = vehicle.get_transform()
            x = transform.location.x
            y = transform.location.y
            yaw = np.deg2rad(transform.rotation.yaw)
        
            rel_x = x - ego_x
            rel_y = y - ego_y
        
            distance = np.sqrt(rel_x**2 + rel_y**2)
            if distance > perception_range:
                continue  # Ignore vehicles outside the perception range
        
            # Transform to ego-centric local coordinates
            local_x = np.cos(-ego_yaw) * rel_x - np.sin(-ego_yaw) * rel_y
            local_y = np.sin(-ego_yaw) * rel_x + np.cos(-ego_yaw) * rel_y
        
            v = vehicle.get_velocity()
            speed = np.sqrt(v.x**2 + v.y**2 + v.z**2)
        
            vehicle_data.append((distance, [local_x, local_y, yaw - ego_yaw, speed]))
        
        # Sort vehicles by distance and select the nearest max_vehicles
        vehicle_data.sort(key=lambda x: x[0])
        nearby_vehicles = [data for _, data in vehicle_data[:max_vehicles]]
        
        # Pad with zeros if fewer than max_vehicles are detected
        while len(nearby_vehicles) < max_vehicles:
            nearby_vehicles.append([0.0, 0.0, 0.0, 0.0])
        
        obs['nearby_vehicles'] = np.array(nearby_vehicles, dtype=np.float32).flatten()

# ========================== Current reference waypoints (up to N waypoints) ==========================
        max_waypoints = self.max_waypoints
        world_map = self.world.get_map()
        waypoint = world_map.get_waypoint(self.ego.get_location())
        waypoints_array = np.zeros((max_waypoints, 3), dtype=np.float32)
        
        for i in range(max_waypoints):
            if waypoint is None:
                break
        
            loc = waypoint.transform.location
            yaw = waypoint.transform.rotation.yaw
        
            # Transform waypoint location into ego-centric local coordinates
            local_x = np.cos(-ego_yaw) * (loc.x - ego_x) - np.sin(-ego_yaw) * (loc.y - ego_y)
            local_y = np.sin(-ego_yaw) * (loc.x - ego_x) + np.cos(-ego_yaw) * (loc.y - ego_y)
            yaw_relative = np.deg2rad(yaw) - ego_yaw  # Relative heading
        
            waypoints_array[i] = [local_x, local_y, yaw_relative]
        
            # Move to the next waypoint 2.0 meters ahead
            next_waypoints = waypoint.next(2.0)
            waypoint = next_waypoints[0] if next_waypoints else None
        
        obs['waypoints'] = waypoints_array.flatten()

# ============================= Lane boundary information =========================================
        waypoint_center = world_map.get_waypoint(
            self.ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        
        if waypoint_center is None:
            # If no valid driving lane is found
            obs['lane_info'] = np.array([0.0, 0.0], dtype=np.float32)
        else:
            lane_width = waypoint_center.lane_width
            ego_location = self.ego.get_location()
            center_location = waypoint_center.transform.location
        
            # Calculate lateral offset between ego position and lane centerline
            lateral_offset = np.linalg.norm(
                np.array([
                    ego_location.x - center_location.x,
                    ego_location.y - center_location.y
                ])
            )
        
            obs['lane_info'] = np.array([lane_width, lateral_offset], dtype=np.float32)

# =============================== Visualize current reference waypoints ===============================
        if self.visualize_waypoints:
            for i in range(max_waypoints):
                wx, wy, _ = waypoints_array[i]
        
                # Transform from ego-centric local coordinates to global coordinates
                gx = np.cos(ego_yaw) * wx - np.sin(ego_yaw) * wy + ego_x
                gy = np.sin(ego_yaw) * wx + np.cos(ego_yaw) * wy + ego_y
        
                self.world.debug.draw_point(
                    carla.Location(x=gx, y=gy, z=ego_transform.location.z + 1.0),
                    size=0.1,
                    life_time=0.5,
                    color=carla.Color(0, 255, 0)  # Green points
                )
        # 在返回 obs 之前，添加以下代码以确保值在合理范围内
        obs['lidar'] = np.clip(obs['lidar'], 0.0, 1.0) # 确保 LiDAR 在 [0, 1]
        obs['ego_state'] = np.nan_to_num(obs['ego_state'], nan=0.0, posinf=1e3, neginf=-1e3)
        obs['nearby_vehicles'] = np.nan_to_num(obs['nearby_vehicles'], nan=0.0, posinf=1e3, neginf=-1e3)
        obs['waypoints'] = np.nan_to_num(obs['waypoints'], nan=0.0, posinf=1e3, neginf=-1e3)
        obs['lane_info'] = np.nan_to_num(obs['lane_info'], nan=0.0, posinf=1.0, neginf=0.0)
        return obs

    def _get_reward(self, obs, done):
        reward = 0.0
    
        # 1. Forward driving reward (within speed limit and along lane direction)
        speed = obs['ego_state'][3]
        if speed <= self.desired_speed:
            reward += 1.0 * speed
        else:
            reward += -1.0 * (speed - self.desired_speed)
    
        # 2. Lane deviation penalty (penalize offset from lane center)
        lane_width, lateral_offset = obs['lane_info']
        reward += -1.0 * lateral_offset
    
        # 3. Smooth driving penalty (lateral acceleration penalty)
        a_lat = obs['ego_state'][6]
        reward += -0.5 * abs(a_lat)
    
        # 4. Stationary penalty (if no vehicle ahead but ego is barely moving)
        front_distance = obs['ego_state'][7]
        if front_distance > 10.0 and speed < 0.1:
            reward += -1.0
    
        # 5. Collision penalty
        if self._is_collision:
            reward += -100.0
    
        # 6. Off-road penalty
        if self._is_off_road:
            reward += -100.0
    
        # # 7. Sparse terminal reward (for safely reaching the goal)
        # if done:
        #     if not self._is_collision and not self._is_off_road:
        #         reward += 200.0
    
        return reward

    def _get_cost(self, obs):
        """
        Calculate the constraint cost for safe reinforcement learning.
    
        This cost is only used in safe RL settings and does not affect the reward function.
        It penalizes collisions, off-road events, and overspeeding behavior.
        
        Args:
            obs: The current observation dictionary.
    
        Returns:
            cost (float): The accumulated constraint cost.
        """
        cost = 0.0
    
        # 1. Collision cost
        if self._is_collision:
            cost += 20.0
    
        # 2. Off-road cost
        if self._is_off_road:
            cost += 20.0
    
        # 3. Overspeed cost
        speed = obs['ego_state'][3]
        if speed > self.desired_speed:
            cost += (speed - self.desired_speed) / self.desired_speed  # Cost proportional to overspeed percentage
    
        return cost

    def _terminal(self):
        ego_transform = self.ego.get_transform()
        ego_x = ego_transform.location.x
        ego_y = ego_transform.location.y
    
        # 1. Collision termination
        if len(self.collision_hist) > 0:
            self._is_collision = True
            print('Collision occurred')
            return True
    
        # 2. Exceeding maximum allowed timesteps
        if self.time_step > self.max_time_episode:
            print('Exceeded maximum timesteps')
            return True
    
        # # 3. Goal reaching termination (optional)
        # if self.dests is not None:
        #     for dest in self.dests:
        #         if np.sqrt((ego_x - dest[0])**2 + (ego_y - dest[1])**2) < 4:
        #             return True
    
        # 4. Check if the current lane is a drivable lane
        waypoint = self.world.get_map().get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if waypoint is None:
            self._is_off_road = True
            print('Non-drivable lane detected')
            return True
    
        # 5. Check if the vehicle's heading deviates too much from lane direction (> ±90°)
        ego_yaw = self.ego.get_transform().rotation.yaw
        lane_yaw = waypoint.transform.rotation.yaw
        yaw_diff = np.deg2rad(ego_yaw - lane_yaw)
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))  # Normalize to [-π, π]
        if not waypoint.is_intersection:
            if abs(yaw_diff) > np.pi / 2:  # More than 90 degrees deviation (wrong-way driving)
                self._is_off_road = True
                print('Wrong-way driving detected')
                return True
    
        # 6. Deviation too far from lane center
        lane_width, lateral_offset = self._get_obs()['lane_info']
        if not waypoint.is_intersection:
            if lateral_offset > lane_width / 2 + 1.0:
                self._is_off_road = True
                print('Deviated from lane')
                return True
    
        return False

    def _clear_all_actors(self, actor_filters):
        """Clear (destroy) all actors matching the given filter patterns.
    
        Args:
            actor_filters (list): A list of filter strings, e.g., ['vehicle.*', 'walker.*', 'sensor.*'].
        """
        for actor_filter in actor_filters:
            for actor in self.world.get_actors().filter(actor_filter):
                try:
                    # If the actor is a sensor, stop it before destroying
                    if 'sensor' in actor.type_id:
                        actor.stop()
                    actor.destroy()
                except:
                    pass  # Ignore any errors during destruction

    #新增两个方法
    def _collision_handler(self, event):
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(intensity)
        if len(self.collision_hist) > self.collision_hist_l:
            self.collision_hist.pop(0)

    def _lidar_handler(self, data):
        self.lidar_data = data
    
    # def close(self):
    #     """Clean up all actors and properly close the CARLA client."""
    #     print("Shutting down Carla environment...")
        
    #     # Destroy ego vehicle and sensors
    #     if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
    #         try:
    #             self.collision_sensor.stop()
    #             self.collision_sensor.destroy()
    #         except:
    #             pass
    #     if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
    #         try:
    #             self.lidar_sensor.stop()
    #             self.lidar_sensor.destroy()
    #         except:
    #             pass
    #     if hasattr(self, 'ego') and self.ego is not None:
    #         try:
    #             self.ego.destroy()
    #         except:
    #             pass

    #     # Clear other actors
    #     self._clear_all_actors([
    #         'vehicle.*', 'walker.*',
    #         'sensor.other.collision', 'sensor.lidar.ray_cast'
    #     ])

    #     # Restore world settings
    #     if hasattr(self, 'world') and self.world is not None:
    #         settings = self.world.get_settings()
    #         settings.synchronous_mode = False
    #         settings.fixed_delta_seconds = None
    #         self.world.apply_settings(settings)

    #     # 【关键】显式删除 client 引用，触发正常关闭
    #     if hasattr(self, 'client') and self.client is not None:
    #         del self.client  # 或 self.client = None

    #     print("Carla environment closed successfully.")
    # def close(self):
    #     print("Shutting down Carla environment...")
    #     # 1. 停止传感器
    #     if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
    #         try:
    #             self.collision_sensor.stop() # Stop first
    #             # 等待几帧确保回调停止
    #             if self.collision_sensor.is_listening:
    #                 for _ in range(3):
    #                     self.world.tick()
    #             if self.collision_sensor.is_alive: # 👈 检查是否还活着
    #                 self.collision_sensor.destroy()
    #             print(f">>> [CLOSE] Destroyed collision sensor {self.collision_sensor.id if self.collision_sensor else 'None'}.")
    #         except Exception as e:
    #             print(f"Error stopping/destroying collision sensor: {e}")
    #         finally:
    #             self.collision_sensor = None # 确保引用被清空

    #     if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
    #         try:
    #             self.lidar_sensor.stop() # Stop first
    #             # 等待几帧确保回调停止
    #             if self.lidar_sensor.is_listening:
    #                 for _ in range(3):
    #                     self.world.tick()
    #             if self.lidar_sensor.is_alive: # 👈 检查是否还活着
    #                 self.lidar_sensor.destroy()
    #             print(f">>> [CLOSE] Destroyed lidar sensor {self.lidar_sensor.id if self.lidar_sensor else 'None'}.")
    #         except Exception as e:
    #             print(f"Error stopping/destroying lidar sensor: {e}")
    #         finally:
    #             self.lidar_sensor = None # 确保引用被清空

    #     # 2. 销毁 ego 车辆
    #     if hasattr(self, 'ego') and self.ego is not None:
    #         try:
    #             # 确保 ego 车辆不在 autopilot 模式，避免冲突
    #             self.ego.set_autopilot(False)
    #             # 等待一帧，让 CARLA 更新状态
    #             if hasattr(self, 'world') and self.world is not None:
    #                 self.world.tick()
    #             # 销毁
    #             if self.ego.is_alive: # 👈 检查是否还活着
    #                 self.ego.destroy()
    #             print(f">>> [CLOSE] Destroyed ego vehicle {self.ego.id if self.ego else 'None'}.")
    #         except Exception as e:
    #             print(f"Error destroying ego vehicle: {e}")
    #         finally:
    #             self.ego = None # 确保引用被清空

    #     # 3. 清理其他 actor
    #     try:
    #         self._clear_all_actors([
    #             'vehicle.*', 'walker.*', 'sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'controller.ai.walker',
    #         ])
    #         print(">>> [CLOSE] Cleared all remaining actors.")
    #     except Exception as e:
    #         print(f"Error clearing all actors: {e}")

    #     # 4. 恢复世界设置
    #     if hasattr(self, 'world') and self.world is not None:
    #         try:
    #             settings = self.world.get_settings()
    #             settings.synchronous_mode = False
    #             settings.fixed_delta_seconds = None
    #             self.world.apply_settings(settings)
    #             # Tick once more to ensure settings take effect
    #             self.world.tick()
    #             print(">>> [CLOSE] Restored world settings.")
    #         except Exception as e:
    #             print(f"Error restoring world settings: {e}")

    #     # 5. 删除 client 引用（可选，但推荐）
    #     if hasattr(self, 'client'):
    #         try:
    #             # self.client = None # 或者 del self.client
    #             print(">>> [CLOSE] Client reference kept (CARLA handles its own lifecycle).")
    #         except Exception as e:
    #             print(f"Error closing client: {e}")
    #     print("Carla environment closed successfully.")
    def close(self):
        print("Shutting down Carla environment...")
        # 停止传感器
        if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
            try:
                self.collision_sensor.stop() # Stop first
                # 等待几帧确保回调停止
                for _ in range(3):
                    self.world.tick()
                if self.collision_sensor.is_alive: # 👈 检查是否还活着
                    self.collision_sensor.destroy()
                print(f">>> [CLOSE] Destroyed collision sensor {self.collision_sensor.id}.")
            except Exception as e:
                print(f"Error stopping/destroying collision sensor: {e}")
            finally:
                self.collision_sensor = None # 确保引用被清空

        if hasattr(self, 'lidar_sensor') and self.lidar_sensor is not None:
            try:
                self.lidar_sensor.stop() # Stop first
                # 等待几帧确保回调停止
                for _ in range(3):
                    self.world.tick()
                if self.lidar_sensor.is_alive: # 👈 检查是否还活着
                    self.lidar_sensor.destroy()
                print(f">>> [CLOSE] Destroyed lidar sensor {self.lidar_sensor.id}.")
            except Exception as e:
                print(f"Error stopping/destroying lidar sensor: {e}")
            finally:
                self.lidar_sensor = None # 确保引用被清空

        # 销毁 ego 车辆
        if hasattr(self, 'ego') and self.ego is not None:
            try:
                # 确保 ego 车辆不在 autopilot 模式，避免冲突
                self.ego.set_autopilot(False)
                # 等待一帧，让 CARLA 更新状态
                if hasattr(self, 'world') and self.world is not None:
                    self.world.tick()
                # 销毁
                if self.ego.is_alive: # 👈 检查是否还活着
                    self.ego.destroy()
                print(f">>> [CLOSE] Destroyed ego vehicle {self.ego.id}.")
            except Exception as e:
                print(f"Error destroying ego vehicle: {e}")
            finally:
                self.ego = None # 确保引用被清空

        # 清理其他 actor
        try:
            self._clear_all_actors([
                'vehicle.*', 'walker.*', 'sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'controller.ai.walker',
            ])
            print(">>> [CLOSE] Cleared all remaining actors.")
        except Exception as e:
            print(f"Error clearing all actors: {e}")

        # 恢复世界设置
        if hasattr(self, 'world') and self.world is not None:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
                # Tick once more to ensure settings take effect
                self.world.tick()
                print(">>> [CLOSE] Restored world settings.")
            except Exception as e:
                print(f"Error restoring world settings: {e}")

        print("Carla environment closed successfully.")






