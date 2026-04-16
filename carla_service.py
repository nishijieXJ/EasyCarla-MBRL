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
import json
import os

def _format_for_drama(obs):
    """Convert EasyCarla-RL observation to Drama expected format"""
    return {
        "image": obs["front_camera"],      # shape: (H, W, C) - front view
        "bev": obs["bev_camera"],          # shape: (H, W, C) - bird eye view
        "lidar": obs["lidar"],             # shape: (240,) - lidar scan
        "ego_state": obs["ego_state"],     # shape: (9,) - ego vehicle state
        "goal_relative": obs.get("goal_relative", np.zeros(4, dtype=np.float32)),  # [dx_local, dy_local, dist, yaw_err]
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
        "goal_relative": {"shape": (4,), "dtype": "float32"},
        "truck_state": {"shape": (3,), "dtype": "float32"}
    }

def _probe_available_maps(port, timeout=10.0):
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


def _normalize_town_tag(town_name):
    """Convert '/Game/.../Town05' or 'Town05' -> 'town05'."""
    if not isinstance(town_name, str) or not town_name.strip():
        return ''
    tail = town_name.strip().split('/')[-1]
    return tail.lower()


def _pick_catalog_path(params, split, source):
    """
    Pick catalog path by task source profile.

    For lb10, prefer auto town-aligned path:
      {task_catalog_lb10_dir}/{town_tag}_coarse_{split}_catalog.json
    so that `town` and catalog always stay aligned.
    """
    split = str(split).lower()
    source = str(source).lower()

    # Auto town-specific LB1.0 catalog selection (highest priority for lb10)
    if source == 'lb10':
        town_tag = _normalize_town_tag(params.get('town', ''))
        lb10_dir = params.get('task_catalog_lb10_dir', '/data2/zk/EasyCarla-RL/taskcatalog/lb10')
        if isinstance(lb10_dir, str) and lb10_dir.strip() and town_tag:
            candidate = os.path.join(lb10_dir, f"{town_tag}_coarse_{split}_catalog.json")
            if os.path.isfile(candidate):
                return candidate

            # Backward compatibility for older naming without "_coarse"
            legacy_candidate = os.path.join(lb10_dir, f"{town_tag}_{split}_catalog.json")
            if os.path.isfile(legacy_candidate):
                return legacy_candidate

    profile_key = f"task_catalog_{source}_{split}_path"
    prof_path = params.get(profile_key, '')
    if isinstance(prof_path, str) and prof_path.strip():
        return prof_path

    legacy_split_key = f"task_catalog_{split}_path"
    legacy_split_path = params.get(legacy_split_key, '')
    if isinstance(legacy_split_path, str) and legacy_split_path.strip():
        return legacy_split_path

    fallback = params.get('task_catalog_path', '')
    if isinstance(fallback, str) and fallback.strip():
        return fallback

    return ''


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
        'number_of_vehicles': 20,  #100
        'number_of_walkers': 0,
        'dt': 0.1,
        'ego_vehicle_filter': 'vehicle.dafxf.dafxf',  # Your semi-trailer tractor
        'surrounding_vehicle_spawned_randomly': True,
        'port': 2000,
        # CARLA RPC timeout (seconds), useful for slow first map load
        'carla_timeout': 60.0,
        # Safety bootstrap for unstable maps/scenarios:
        # - lower traffic load
        # - disable trailer spawn chain
        # - disable task catalog spawn points
        'safe_startup': False,

        # ===== Map switch (A/B) =====
        'map_mode': 'town',  # 'town' | 'xodr'

        # town 模式：可填 'Town03'，也可填 '/Game/Carla/Maps/...' 资源路径
        'town': 'Town01',   #需要注意这里的town序号要与后面catelog的编号一致；Town12  #20m #这里如果使用town模式导入环岛20m地图，会以为缺失很多材质导致很多问题

        # xodr 模式：三张环岛图可选 '16m50m' / '20m' / '32m40m'
        #'map_id': '20m',    #使用xodr模式导入环岛20m的地图，能正常导入，但是也会因为确实部分材质导致车道线等看不见
        #'xodr_root': '/data1/zk/carla/Unreal/CarlaUE4/Content/map_package/Maps',
        'map_id': 'Town12',    #使用xodr模式导入环岛20m的地图，能正常导入，但是也会因为确实部分材质导致车道线等看不见
        'xodr_root': '/data1/zk/carla/Unreal/CarlaUE4/Content/Carla/Maps',
        # 可选：若给了 xodr_path，则优先使用该绝对路径
        # 'xodr_path': '/data1/zk/carla/Unreal/CarlaUE4/Content/map_package/Maps/20m/OpenDrive/20m.xodr',
        # =============================

        'max_time_episode': 1000,
        'max_waypoints': 12,
        'visualize_waypoints': True,
        'desired_speed': 5,
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
        'task_mode': 'multitask',

        # ===== Expert warm-start + mixed exploration =====
        # control_mode: 'rl' | 'expert' | 'mixed'
        'control_mode': 'rl',
        # Plan A: first N environment steps forced to expert autopilot, then RL/mixed takes over
        'expert_warmup_steps':500,
        # Plan B: mixed exploration probability schedule (expert usage)
        'expert_prob_init': 1.0,
        'expert_prob_final': 0.0,
        'expert_prob_decay_steps': 1000,

        # Guidance mode switch for optional aids:
        # off | A (debug draw) | B (goal_relative obs) | AB (both)
        'guidance_mode': 'A',
        'spawn_alignment_log': True,
        'guidance_draw_life_time': 8.0,
        'guidance_draw_thickness': 0.20,
        # Route guidance style: line_points | points_only | line_only
        'route_draw_style': 'points_only',
        'route_draw_point_size': 0.08,
        'default_task_type': 'navigation',
        'default_task_difficulty': 'easy',
        # Task split and catalogs
        'task_split': 'train',  # 'train' | 'valid' | 'test'
        # Task source profile switch: lb21 | lb10
        'task_source': 'lb10',
        # Single fallback catalog path (backward-compatible)
        'task_catalog_path': '',
        # Preferred split-specific catalog paths (active profile fallback)
        'task_catalog_train_path': '/data2/zk/EasyCarla-RL/alltowns_train_catalog.json',
        'task_catalog_valid_path': '/data2/zk/EasyCarla-RL/alltowns_valid_catalog.json',
        'task_catalog_test_path': '/data2/zk/EasyCarla-RL/alltowns_test_catalog.json',
        # Leaderboard 2.1 profile
        'task_catalog_lb21_train_path': '/data2/zk/EasyCarla-RL/taskcatalog/lb21/alltowns_train_catalog.json',
        'task_catalog_lb21_valid_path': '/data2/zk/EasyCarla-RL/taskcatalog/lb21/alltowns_valid_catalog.json',
        'task_catalog_lb21_test_path': '/data2/zk/EasyCarla-RL/taskcatalog/lb21/alltowns_test_catalog.json',
        # Leaderboard 1.0 profile (set to generated files as needed)
        'task_catalog_lb10_train_path': '/data2/zk/EasyCarla-RL/taskcatalog/lb10/town01_train_catalog.json',
        'task_catalog_lb10_valid_path': '/data2/zk/EasyCarla-RL/taskcatalog/lb10/town01_valid_catalog.json',
        'task_catalog_lb10_test_path': '/data2/zk/EasyCarla-RL/taskcatalog/lb10/town01_test_catalog.json',
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
        # Goal success radius (explicit goal mode)
        'task_goal_tolerance_m': 1.0,

        'sync_mode': True,
        'delta_seconds': 0.1,
        'max_steps': 1000,

        # ===== Reward/Cost tuning knobs (for sweep) =====
        # Minimal fix knobs
        'collision_intensity_threshold': 50.0,
        'collision_soft_penalty': -10.0,
        'collision_hard_penalty': -100.0,
        'offroad_terminal_steps': 50,
        'timeout_terminal_bonus': 0.0,

        # Desired-speed tracking reward (symmetric around desired_speed)
        'enable_reward_speed_tracking': True,
        'reward_speed_tracking_weight': 1.0,
        'reward_speed_tolerance_kmh': 2.0,
        'reward_speed_scale_kmh': 10.0,

        # Optional speeding cost (active only when enabled)
        'enable_cost_speeding': False,
        'cost_speeding_weight': 1.0,
        'cost_speeding_margin_kmh': 3.0,
        'cost_speeding_scale_kmh': 10.0,

        # Semi-trailer enhanced cost knobs
        'cost_lane_deviation_weight': 1.0,
        'cost_proximity_weight': 1.0,
        'cost_articulation_weight': 0.8,
        'cost_jackknife_weight': 2.0,
        'cost_trailer_offroad_weight': 2.0,
        'cost_ttc_weight': 1.5,
        'articulation_safe_angle_deg': 20.0,
        'jackknife_angle_deg': 55.0,
        'ttc_threshold_s': 3.0,
        'ttc_max_consider_distance_m': 50.0,

        # ===== Ablation switches =====
        'enable_cost_lane_deviation': True,
        'enable_cost_proximity': True,
        'enable_cost_articulation': True,
        'enable_cost_jackknife': True,
        'enable_cost_trailer_offroad': True,
        'enable_cost_ttc': True,
        'enable_reward_task_shaping': True,
    }

    # Optional safe startup profile for first bring-up/debug.
    if params.get('safe_startup', False):
        safe_vehicle_count = int(params.get('safe_num_vehicles', 10))
        params['number_of_vehicles'] = min(params.get('number_of_vehicles', safe_vehicle_count), safe_vehicle_count)
        params['task_easy_num_vehicles'] = min(params.get('task_easy_num_vehicles', safe_vehicle_count), safe_vehicle_count)
        params['task_medium_num_vehicles'] = min(params.get('task_medium_num_vehicles', safe_vehicle_count), safe_vehicle_count)
        params['task_hard_num_vehicles'] = min(params.get('task_hard_num_vehicles', safe_vehicle_count), safe_vehicle_count)
        params['enable_trailer'] = False
        params['task_catalog_path'] = ''
        params['task_catalog_train_path'] = ''
        params['task_catalog_valid_path'] = ''
        params['task_catalog_test_path'] = ''
        params['task_catalog'] = {}
        print(f"[SAFE] safe_startup enabled: carla_timeout={params.get('carla_timeout')}s, "
              f"num_vehicles<={safe_vehicle_count}, trailer_disabled=True, task_catalog_disabled=True")

    # Pre-flight map diagnostics for clearer failure messages.
    mode = params.get('map_mode', 'town')
    if mode == 'town':
        requested_town = params.get('town', '')
        available_maps = _probe_available_maps(params['port'], timeout=float(params.get('carla_timeout', 60.0)))

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

    # Optional: load task catalog json for explicit task start/goal definitions
    requested_split = str(params.get('task_split', 'train')).lower()
    requested_source = str(params.get('task_source', 'lb21')).lower()
    selected_catalog_path = _pick_catalog_path(params, requested_split, requested_source)

    if isinstance(selected_catalog_path, str) and selected_catalog_path.strip():
        abs_path = os.path.abspath(selected_catalog_path)
        if os.path.isfile(abs_path):
            try:
                with open(abs_path, 'r', encoding='utf-8') as f:
                    params['task_catalog'] = json.load(f)
                print(
                    f"[TASK] loaded source={requested_source}, split={requested_split} "
                    f"task catalog: {abs_path}"
                )
            except Exception as e:
                print(f"[TASK] failed to load task catalog '{abs_path}': {e}")
                params['task_catalog'] = {}
        else:
            print(f"[TASK] task catalog path not found: {abs_path}")
            params['task_catalog'] = {}

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

                        requested_source = reset_options.get("task_source", None)
                        if requested_source in ("lb10", "lb21"):
                            params['task_source'] = requested_source
                            env.params['task_source'] = requested_source

                        # Optional runtime split switch: train/valid/test catalogs
                        requested_split = reset_options.get("task_split", None)
                        if requested_split in ("train", "valid", "test"):
                            env.task_split = requested_split

                        # Runtime controls for expert warm-start / mixed exploration
                        if "control_mode" in reset_options:
                            requested_control_mode = str(reset_options.get("control_mode", "")).strip().lower()
                            if requested_control_mode in ("rl", "expert", "mixed"):
                                params['control_mode'] = requested_control_mode
                                env.params['control_mode'] = requested_control_mode

                        runtime_numeric_keys = (
                            "expert_warmup_steps",
                            "expert_prob",
                            "expert_prob_init",
                            "expert_prob_final",
                            "expert_prob_decay_steps",
                        )
                        for k in runtime_numeric_keys:
                            if k in reset_options:
                                env.params[k] = reset_options[k]
                                if k != "expert_prob":
                                    params[k] = reset_options[k]

                        runtime_source = params.get('task_source', 'lb21')
                        runtime_split = env.task_split if hasattr(env, 'task_split') else params.get('task_split', 'train')
                        selected_path = _pick_catalog_path(params, runtime_split, runtime_source)
                        if isinstance(selected_path, str) and selected_path.strip():
                            abs_path = os.path.abspath(selected_path)
                            if os.path.isfile(abs_path):
                                try:
                                    with open(abs_path, 'r', encoding='utf-8') as f:
                                        env.task_catalog = json.load(f)
                                    print(
                                        f"[TASK] switched source={runtime_source}, split={runtime_split}, "
                                        f"loaded catalog={abs_path}"
                                    )
                                except Exception as e:
                                    print(f"[TASK] runtime switch load failed for '{abs_path}': {e}")
                            else:
                                print(f"[TASK] runtime switch catalog not found: {abs_path}")
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