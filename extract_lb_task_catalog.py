#!/usr/bin/env python3
"""
Extract CARLA Leaderboard routes/scenarios into task catalogs for RL multitask training.

Input:
  - Tools/leaderboard/data/routes_training.xml   -> train catalog
  - Tools/leaderboard/data/routes_validation.xml -> valid catalog
  - Tools/leaderboard/data/routes_devtest.xml    -> test catalog

Output:
  - taskcatalog/lb21/town03_train_catalog.json
  - taskcatalog/lb21/town03_valid_catalog.json
  - taskcatalog/lb21/town03_test_catalog.json

Usage:
  python extract_lb_task_catalog.py \
    --leaderboard-data /data2/zk/Tools/leaderboard/data \
    --output-dir /data2/zk/EasyCarla-RL/taskcatalog/lb21 \
    --town Town03
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


LB_COARSE_TASK_TYPE_MAP = {
    "1.0": {
        "SignalizedJunctionLeftTurn": "left_turn",
        "NonSignalizedJunctionLeftTurn": "left_turn",
        "SignalizedJunctionRightTurn": "right_turn",
        "NonSignalizedJunctionRightTurn": "right_turn",
    },
    "2.0": {
        "SignalizedJunctionLeftTurn": "left_turn",
        "NonSignalizedJunctionLeftTurn": "left_turn",
        "SignalizedJunctionRightTurn": "right_turn",
        "NonSignalizedJunctionRightTurn": "right_turn",
    },
    "2.1": {
        "SignalizedJunctionLeftTurn": "left_turn",
        "NonSignalizedJunctionLeftTurn": "left_turn",
        "SignalizedJunctionRightTurn": "right_turn",
        "NonSignalizedJunctionRightTurn": "right_turn",
    },
}


def euclid2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.sqrt(euclid2(a, b))


def infer_yaw(p0: Dict, p1: Dict) -> float:
    dx = float(p1["x"]) - float(p0["x"])
    dy = float(p1["y"]) - float(p0["y"])
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


@dataclass
class DifficultyCfg:
    spawn_backtrack: int
    goal_forward: int
    goal_tolerance_m: float
    max_time_episode: int
    num_vehicles: int


DIFF_CFG: Dict[str, DifficultyCfg] = {
    "easy": DifficultyCfg(spawn_backtrack=2, goal_forward=4, goal_tolerance_m=7.0, max_time_episode=800, num_vehicles=25),
    "medium": DifficultyCfg(spawn_backtrack=4, goal_forward=7, goal_tolerance_m=6.0, max_time_episode=1000, num_vehicles=50),
    "hard": DifficultyCfg(spawn_backtrack=6, goal_forward=10, goal_tolerance_m=5.5, max_time_episode=1200, num_vehicles=80),
}


def nearest_waypoint_index(waypoints: List[Dict], tx: float, ty: float) -> int:
    target = (tx, ty)
    best_i, best_d = 0, float("inf")
    for i, wp in enumerate(waypoints):
        d2 = euclid2((float(wp["x"]), float(wp["y"])), target)
        if d2 < best_d:
            best_d = d2
            best_i = i
    return best_i


def append_case(catalog: Dict, town: str, task_type: str, difficulty: str, case: Dict) -> None:
    catalog.setdefault(town, {})
    catalog[town].setdefault(task_type, {})
    catalog[town][task_type].setdefault(difficulty, {"cases": []})
    catalog[town][task_type][difficulty]["cases"].append(case)


def route_end_case(route_id: str, waypoints: List[Dict], difficulty: str) -> Optional[Dict]:
    if len(waypoints) < 3:
        return None
    cfg = DIFF_CFG[difficulty]
    spawn_i = max(0, min(cfg.spawn_backtrack, len(waypoints) - 2))
    goal_i = len(waypoints) - 1
    spawn_wp = waypoints[spawn_i]
    spawn_next = waypoints[min(spawn_i + 1, len(waypoints) - 1)]
    goal_wp = waypoints[goal_i]
    case = {
        "route_id": route_id,
        "scenario_name": "route_navigation",
        "source": "route_terminal",
        "spawn": {
            "x": float(spawn_wp["x"]),
            "y": float(spawn_wp["y"]),
            "z": float(spawn_wp.get("z", 0.5)),
            "yaw": float(infer_yaw(spawn_wp, spawn_next)),
        },
        "goal": {
            "x": float(goal_wp["x"]),
            "y": float(goal_wp["y"]),
            "z": float(goal_wp.get("z", 0.5)),
        },
        "goal_tolerance_m": cfg.goal_tolerance_m,
        "max_time_episode": cfg.max_time_episode,
        "num_vehicles": cfg.num_vehicles,
        "reward_profile": f"task_navigation_{difficulty}",
    }
    return case


def _normalize_task_key(name: str) -> str:
    s = str(name).strip()
    if not s:
        return "unknown"
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("_")
    key = "".join(out)
    while "__" in key:
        key = key.replace("__", "_")
    return key.strip("_") or "unknown"


def _resolve_task_type(s_type: str, leaderboard_version: str, task_granularity: str) -> Optional[str]:
    if task_granularity == "fine":
        return _normalize_task_key(s_type)

    version_map = LB_COARSE_TASK_TYPE_MAP.get(leaderboard_version, LB_COARSE_TASK_TYPE_MAP["2.1"])
    return version_map.get(s_type, None)


def extract_from_xml(xml_path: Path, town_filter: str, leaderboard_version: str, task_granularity: str) -> Dict:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    catalog = {
        "schema_version": "1.2",
        "leaderboard_version": str(leaderboard_version),
        "catalog_mode": str(task_granularity),
        "source": str(xml_path),
        "notes": "Auto-extracted from Leaderboard routes/scenarios. Cases sampled per task_type/difficulty.",
    }

    for route in root.findall("route"):
        town = route.get("town", "")
        if town_filter and town != town_filter:
            continue
        route_id = route.get("id", "unknown")

        waypoints_node = route.find("waypoints")
        if waypoints_node is None:
            continue
        waypoints = []
        for pos in waypoints_node.findall("position"):
            waypoints.append(
                {
                    "x": float(pos.get("x", "0")),
                    "y": float(pos.get("y", "0")),
                    "z": float(pos.get("z", "0.5")),
                }
            )
        if len(waypoints) < 2:
            continue

        scenarios_node = route.find("scenarios")
        if scenarios_node is not None:
            for scenario in scenarios_node.findall("scenario"):
                s_type = scenario.get("type", "")
                task_type = _resolve_task_type(s_type, leaderboard_version=leaderboard_version, task_granularity=task_granularity)
                if task_type is None:
                    continue

                tp = scenario.find("trigger_point")
                if tp is None:
                    continue
                tx = float(tp.get("x", "0"))
                ty = float(tp.get("y", "0"))
                tz = float(tp.get("z", "0.5"))
                tyaw = float(tp.get("yaw", "0"))

                trigger_idx = nearest_waypoint_index(waypoints, tx, ty)

                for difficulty, cfg in DIFF_CFG.items():
                    spawn_i = max(0, trigger_idx - cfg.spawn_backtrack)
                    goal_i = min(len(waypoints) - 1, trigger_idx + cfg.goal_forward)

                    spawn_wp = waypoints[spawn_i]
                    spawn_next = waypoints[min(spawn_i + 1, len(waypoints) - 1)]
                    goal_wp = waypoints[goal_i]

                    # Use waypoint yaw (stable for vehicle alignment), fall back to trigger yaw
                    yaw_est = infer_yaw(spawn_wp, spawn_next)
                    if abs(yaw_est) < 1e-6:
                        yaw_est = tyaw

                    case = {
                        "route_id": route_id,
                        "scenario_name": scenario.get("name", s_type),
                        "scenario_type": s_type,
                        "trigger": {"x": tx, "y": ty, "z": tz, "yaw": tyaw},
                        "source": "scenario_trigger",
                        "spawn": {
                            "x": float(spawn_wp["x"]),
                            "y": float(spawn_wp["y"]),
                            "z": float(spawn_wp.get("z", 0.5)),
                            "yaw": float(yaw_est),
                        },
                        "goal": {
                            "x": float(goal_wp["x"]),
                            "y": float(goal_wp["y"]),
                            "z": float(goal_wp.get("z", 0.5)),
                        },
                        "goal_tolerance_m": cfg.goal_tolerance_m,
                        "max_time_episode": cfg.max_time_episode,
                        "num_vehicles": cfg.num_vehicles,
                        "reward_profile": f"task_{task_type}_{difficulty}",
                        "reward_weights": {
                            "success": 100.0,
                            "progress": 1.0,
                            "collision": -50.0,
                            "lane_invasion": -8.0,
                            "red_light": -25.0,
                            "timeout": -20.0,
                        },
                    }
                    append_case(catalog, town, task_type, difficulty, case)

        # Add route-terminal navigation cases so navigation always has enough coverage
        for difficulty in DIFF_CFG.keys():
            nav_case = route_end_case(route_id, waypoints, difficulty)
            if nav_case is not None:
                append_case(catalog, town, "navigation", difficulty, nav_case)

    return catalog


def summarize(catalog: Dict) -> Dict:
    out = {}
    meta_keys = {"schema_version", "leaderboard_version", "catalog_mode", "source", "notes"}
    for town, tcfg in catalog.items():
        if town in meta_keys or not isinstance(tcfg, dict):
            continue
        out[town] = {}
        for task, dcfg in tcfg.items():
            if not isinstance(dcfg, dict):
                continue
            out[town][task] = {}
            for diff, payload in dcfg.items():
                if isinstance(payload, dict):
                    out[town][task][diff] = len(payload.get("cases", []))
    return out


def find_xml(base: Path, names: List[str]) -> Path:
    for n in names:
        p = base / n
        if p.exists():
            return p
    raise FileNotFoundError(f"None found under {base}: {names}")


def find_xml_optional(base: Path, names: List[str]) -> Optional[Path]:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


def resolve_data_dir(base: Path) -> Path:
    """
    Support these common inputs:
    - direct xml dir: .../scenario_runner/srunner/data or .../leaderboard/data
    - repo root: .../leaderboard1.0 or .../leaderboard2.1
    - mistaken ".../xxx/data" that has no route xml (auto-fallback to repo-root layouts)
    """
    parent = base.parent
    candidates = [
        base,
        base / "data",
        base / "scenario_runner" / "srunner" / "data",
        base / "leaderboard" / "data",
        parent,
        parent / "scenario_runner" / "srunner" / "data",
        parent / "leaderboard" / "data",
    ]

    seen = set()
    for c in candidates:
        c = c.resolve() if c.exists() else c
        key = str(c)
        if key in seen:
            continue
        seen.add(key)

        if c.exists() and c.is_dir():
            if find_xml_optional(c, ["routes_training.xml", "routes_devtest.xml", "routes_debug.xml"]) is not None:
                return c

    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaderboard-data", type=str, default="/data2/zk/Tools/leaderboard/data")
    parser.add_argument("--output-dir", type=str, default="/data2/zk/EasyCarla-RL/taskcatalog/lb21")
    parser.add_argument("--town", type=str, default="", help="Only keep routes from this town (e.g., Town03). Empty means all towns.")
    parser.add_argument("--leaderboard-version", type=str, default="2.1", choices=["1.0", "2.0", "2.1"])
    parser.add_argument("--task-granularity", type=str, default="coarse", choices=["coarse", "fine"])
    parser.add_argument("--output-prefix", type=str, default="", help="Optional output prefix. Default: {town_tag}.")
    args = parser.parse_args()

    data_dir = resolve_data_dir(Path(args.leaderboard_data))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_xml = find_xml_optional(data_dir, ["routes_training.xml"])
    valid_xml = find_xml_optional(data_dir, ["routes_validation.xml", "routes_valid.xml"])
    test_xml = find_xml_optional(data_dir, ["routes_devtest.xml", "routes_testing.xml"])

    # LB1.0 compatibility: often no explicit validation/testing split.
    if valid_xml is None:
        valid_xml = find_xml_optional(data_dir, ["routes_devtest.xml", "routes_debug.xml"]) or train_xml
    if test_xml is None:
        test_xml = find_xml_optional(data_dir, ["routes_devtest.xml", "routes_debug.xml"]) or train_xml

    if train_xml is None:
        raise FileNotFoundError(
            f"Could not resolve routes_training.xml under '{data_dir}'. "
            f"Please point --leaderboard-data to a directory containing route XML files."
        )

    split2xml = {
        "train": train_xml,
        "valid": valid_xml,
        "test": test_xml,
    }

    town_tag = (args.town.lower() if isinstance(args.town, str) and args.town.strip() else "alltowns")
    prefix = args.output_prefix.strip() if isinstance(args.output_prefix, str) and args.output_prefix.strip() else town_tag

    for split, xml_path in split2xml.items():
        catalog = extract_from_xml(
            xml_path,
            town_filter=args.town,
            leaderboard_version=args.leaderboard_version,
            task_granularity=args.task_granularity,
        )
        out_path = out_dir / f"{prefix}_{split}_catalog.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"[OK] {split}: {xml_path} -> {out_path}")
        print(json.dumps(summarize(catalog), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
