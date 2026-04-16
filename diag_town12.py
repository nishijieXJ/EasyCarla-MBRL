import time
import carla

HOST = "127.0.0.1"
PORT = 2000


def safe_destroy(actor):
    if actor is None:
        return
    try:
        if actor.is_alive:
            actor.destroy()
    except RuntimeError:
        pass


def cleanup_existing_hero(world):
    actors = world.get_actors().filter("vehicle.*")
    removed = 0
    for a in actors:
        try:
            role = a.attributes.get("role_name", "")
            if role == "hero":
                safe_destroy(a)
                removed += 1
        except RuntimeError:
            continue
    if removed > 0:
        print(f"cleaned existing hero vehicles: {removed}")


def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(120.0)  # Town12/large map切图可能超过30s

    print("=== server version ===")
    print(client.get_server_version())
    print("=== client maps ===")
    maps = client.get_available_maps()
    print("\n".join(maps))
    has_town12 = any("Town12" in m for m in maps)
    print(f"Town12 in available maps: {has_town12}")

    print("\n=== load Town12 ===")
    world = client.load_world("Town12")

    # 大地图预热
    for _ in range(100):
        world.tick()
    print("warmup ticks done")

    m = world.get_map()
    spawns = m.get_spawn_points()
    print(f"spawn points: {len(spawns)}")
    if not spawns:
        print("[FAIL] no spawn points -> map/nav likely broken")
        return

    # 检查 large map 参数
    settings = world.get_settings()
    has_tile = hasattr(settings, "tile_stream_distance")
    has_actor = hasattr(settings, "actor_active_distance")
    print(f"has tile_stream_distance: {has_tile}")
    print(f"has actor_active_distance: {has_actor}")
    if has_tile:
        print(f"tile_stream_distance = {settings.tile_stream_distance}")
    if has_actor:
        print(f"actor_active_distance = {settings.actor_active_distance}")

    # 清理旧 hero，避免 ACTIVE VEHICLE DETECTED
    cleanup_existing_hero(world)
    for _ in range(10):
        world.tick()

    # 选车并明确设为 hero
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find("vehicle.tesla.model3")
    if vehicle_bp is None:
        vehicle_bp = bp_lib.filter("vehicle.*")[0]

    if vehicle_bp.has_attribute("role_name"):
        vehicle_bp.set_attribute("role_name", "hero")

    spawn_tf = spawns[0]
    spawn_tf.location.z += 1.0  # 防止贴地/穿地

    ego = world.try_spawn_actor(vehicle_bp, spawn_tf)
    if ego is None:
        print("[FAIL] try_spawn_actor returned None -> spawn transform/collision/active vehicle issue")
        return

    print(f"spawned ego id={ego.id}")

    # 生成后短时稳定性检查，防止刚生成被系统回收
    alive_ok = True
    for _ in range(20):
        world.tick()
        if (not ego.is_alive) or (world.get_actor(ego.id) is None):
            alive_ok = False
            break

    if not alive_ok:
        print("[FAIL] ego removed shortly after spawn -> hero/active vehicle/large map lifecycle issue")
        safe_destroy(ego)
        return

    # 监控 5 秒是否持续下落
    zs = []
    for _ in range(100):
        world.tick()
        if (not ego.is_alive) or (world.get_actor(ego.id) is None):
            print("[FAIL] ego disappeared during z-monitoring")
            break
        zs.append(ego.get_transform().location.z)

    if not zs:
        safe_destroy(ego)
        return

    z_start = zs[0]
    z_min = min(zs)
    z_last = zs[-1]
    drop = z_start - z_min
    print(f"z_start={z_start:.3f}, z_min={z_min:.3f}, z_last={z_last:.3f}, max_drop={drop:.3f}")

    if drop > 5.0:
        print("[FAIL] significant falling detected -> collision/tile streaming/map asset issue")
    else:
        print("[PASS] no major falling -> Town12 basic load+collision likely OK")

    safe_destroy(ego)


if __name__ == "__main__":
    main()
