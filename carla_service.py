"""
CARLA Environment Service for Drama integration
Running in carla_rl (Python 3.7) environment
"""
import zmq
import pickle
import numpy as np
from easycarla.envs.carla_env import CarlaEnv
import traceback
import time
import carla

def _format_for_drama(obs):
    """Convert EasyCarla-RL observation to Drama expected format"""
    return {
        "image": obs["front_camera"],      # shape: (H, W, C) - front view
        "bev": obs["bev_camera"],          # shape: (H, W, C) - bird eye view
        "lidar": obs["lidar"],             # shape: (240,) - lidar scan
        "ego_state": obs["ego_state"],     # shape: (9,) - ego vehicle state
        "truck_state": _extract_truck_states(obs)  # custom truck-specific states
    }

def _extract_truck_states(obs):
    """Extract semi-trailer specific states"""
    # This would require accessing CARLA vehicle objects directly
    # You'll need to extend your CarlaEnv to expose vehicle objects
    # For now, returning placeholder values
    return np.array([0.0, 0.0, 0.0])  # [articulation_angle, trailer_yaw_rate, off_tracking]

def _get_drama_obs_space():
    """Return Drama-compatible observation space info"""
    return {
        "image": {"shape": (600, 800, 3), "dtype": "uint8"},
        "bev": {"shape": (600, 800, 3), "dtype": "uint8"},
        "lidar": {"shape": (240,), "dtype": "float32"},
        "ego_state": {"shape": (9,), "dtype": "float32"},
        "truck_state": {"shape": (3,), "dtype": "float32"}
    }

def _probe_available_maps(port, timeout=5.0):
    """Probe CARLA server maps for better diagnostics before env init."""
    try:
        client = carla.Client('localhost', port)
        client.set_timeout(timeout)
        maps = client.get_available_maps()
        # Normalize and deduplicate while keeping original order
        seen = set()
        normalized = []
        for m in maps:
            if m not in seen:
                seen.add(m)
                normalized.append(m)
        return normalized
    except Exception as e:
        print(f"[MAP] warning: failed to probe available maps from CARLA: {e}")
        return []


def _validate_town_name(requested_town, available_maps):
    """Validate town name with exact and suffix matching for user-friendly checks."""
    if not requested_town:
        return False, "empty town name", []

    # Exact match first
    if requested_town in available_maps:
        return True, requested_town, [requested_town]

    # Suffix match: allow Town03 to match /Game/Carla/Maps/Town03
    suffix_matches = [m for m in available_maps if m.endswith('/' + requested_town) or m.endswith(requested_town)]
    if len(suffix_matches) == 1:
        return True, suffix_matches[0], suffix_matches
    if len(suffix_matches) > 1:
        return False, f"ambiguous town name '{requested_town}'", suffix_matches

    return False, f"town '{requested_town}' not found", []


def start_carla_service():
    # Initialize ZMQ context
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")
    
    print("Starting CARLA service on tcp://127.0.0.1:5555")
    
    # Environment parameters
    # map_mode:
    # - 'town': 保留原有 Town01~Town10 / 自定义umap路径加载能力
    # - 'xodr': 用 OpenDRIVE 动态生成地图（适合 Roundabouts 快速验证）
    params = {
        'number_of_vehicles': 100,  #100
        'number_of_walkers': 0,
        'dt': 0.1,
        'ego_vehicle_filter': 'vehicle.dafxf.dafxf',  # Your semi-trailer tractor
        'surrounding_vehicle_spawned_randomly': True,
        'port': 2000,

        # ===== Map switch (A/B) =====
        'map_mode': 'town',  # 'town' | 'xodr'

        # town 模式：可填 'Town03'，也可填 '/Game/Carla/Maps/...' 资源路径
        'town': 'Town03',   #Town03  #20m #这里如果使用town模式导入环岛20m地图，会以为缺失很多材质导致很多问题

        # xodr 模式：三张环岛图可选 '16m50m' / '20m' / '32m40m'
        'map_id': '20m',    #使用xodr模式导入环岛20m的地图，能正常导入，但是也会因为确实部分材质导致车道线等看不见
        'xodr_root': '/data1/zk/carla/Unreal/CarlaUE4/Content/map_package/Maps',
        # 可选：若给了 xodr_path，则优先使用该绝对路径
        # 'xodr_path': '/data1/zk/carla/Unreal/CarlaUE4/Content/map_package/Maps/20m/OpenDrive/20m.xodr',
        # =============================

        'max_time_episode': 1000,
        'max_waypoints': 12,
        'visualize_waypoints': True,
        'desired_speed': 8,
        'max_ego_spawn_times': 200,
        'view_mode': 'top',
        'traffic': 'off',
        'lidar_max_range': 50.0,
        'max_nearby_vehicles': 5,

        # Spawn/debug controls (backward-compatible defaults)
        'enable_trailer': True,
        'spawn_diagnostics': True,
        'dump_reset_images': True,
        'dump_image_dir': '/data2/zk/EasyCarla-RL/pictures',
        # Front camera mount (tractor local frame): forward(x), up(z), look-down(pitch)
        'front_camera_x': 2.8,
        'front_camera_z': 1.9,
        'front_camera_pitch': -2.0,
        # Task mode switch: 'legacy' or 'multitask'
        'task_mode': 'legacy',
        'default_task_type': 'navigation',
        'default_task_difficulty': 'easy',
        # Difficulty profiles for multitask mode
        'task_easy_num_vehicles': 30,
        'task_easy_max_time_episode': 800,
        'task_easy_target_distance_m': 25.0,
        'task_medium_num_vehicles': 60,
        'task_medium_max_time_episode': 1000,
        'task_medium_target_distance_m': 40.0,
        'task_hard_num_vehicles': 100,
        'task_hard_max_time_episode': 1200,
        'task_hard_target_distance_m': 60.0,

        'sync_mode': True,
        'delta_seconds': 0.1,
        'max_steps': 1000,
    }

    # Pre-flight map diagnostics for clearer failure messages.
    mode = params.get('map_mode', 'town')
    if mode == 'town':
        requested_town = params.get('town', '')
        available_maps = _probe_available_maps(params['port'])

        print(f"[MAP] mode=town, requested_town={requested_town}")
        if available_maps:
            print(f"[MAP] available_maps_count={len(available_maps)}")
            preview_count = min(10, len(available_maps))
            print(f"[MAP] available_maps_preview(first {preview_count})={available_maps[:preview_count]}")
        else:
            print("[MAP] available_maps: (empty or probe failed)")

        ok, resolved_town_or_reason, candidates = _validate_town_name(requested_town, available_maps)
        if not ok:
            hint = (
                "Use a CARLA map name or UE package path (e.g. 'Town03' or '/Game/.../20m'), "
                "NOT a filesystem path like '/xxx/20m.umap'."
            )
            if candidates:
                raise ValueError(
                    f"[MAP] invalid town config: {resolved_town_or_reason}. "
                    f"Candidates={candidates}. Hint: {hint}"
                )
            raise ValueError(
                f"[MAP] invalid town config: {resolved_town_or_reason}. "
                f"Hint: {hint}. Available maps={available_maps}"
            )

        # Canonicalize to resolved name/path for env loading.
        params['town'] = resolved_town_or_reason
        print(f"[MAP] town validation passed, resolved_town={params['town']}")

    env = CarlaEnv(params)
    
    try:
        while True:
            try:
                # Receive request
                message = pickle.loads(socket.recv())
                cmd, payload = message["cmd"], message["payload"]
                
                print(f"Received command: {cmd}")
                
                if cmd == "reset":
                    reset_options = payload if isinstance(payload, dict) else None
                    if isinstance(reset_options, dict) and hasattr(env, "task_mode"):
                        requested_mode = reset_options.get("task_mode", None)
                        if requested_mode in ("legacy", "multitask"):
                            env.task_mode = requested_mode
                    obs, info = env.reset(options=reset_options)
                    response = {
                        "obs": _format_for_drama(obs),
                        "info": info,
                        "success": True
                    }
                    
                elif cmd == "step":
                    action = payload["action"]
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    response = {
                        "obs": _format_for_drama(obs),
                        "reward": float(reward),
                        "done": done,
                        "info": info,
                        "success": True
                    }
                    
                elif cmd == "get_spaces":
                    response = {
                        "action_space": {
                            "shape": (3,),  # [throttle, steer, brake]
                            "bounds": [[0.0, 1.0], [-1.0, 1.0], [0.0, 1.0]]
                        },
                        "observation_space": _get_drama_obs_space(),
                        "success": True
                    }
                    
                elif cmd == "close":
                    env.close()
                    response = {"success": True, "message": "Environment closed"}
                    socket.send(pickle.dumps(response))
                    break
                    
                else:
                    response = {"error": f"Unknown command: {cmd}", "success": False}
                    
            except Exception as e:
                print(f"Error processing command: {e}")
                print(traceback.format_exc())
                response = {
                    "error": str(e),
                    "success": False,
                    "traceback": traceback.format_exc()
                }
            
            # Send response
            socket.send(pickle.dumps(response))
            
    except KeyboardInterrupt:
        print("Service interrupted by user")
    finally:
        env.close()
        socket.close()
        context.term()

if __name__ == "__main__":
    start_carla_service()