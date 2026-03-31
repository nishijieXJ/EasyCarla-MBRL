# -*- coding: utf-8 -*-  # 👈 关键声明！必须放在文件最顶部
# 在CARLA服务端（Py3.7）快速测试
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

with open('/data1/zk/carla/Unreal/CarlaUE4/Content/map_package/Maps/20m/OpenDrive/20m.xodr') as f:
    world = client.generate_opendrive_world(f.read(), carla.OpendriveGenerationParameters(
        vertex_distance=1.0,  # 小环岛需精细网格
        max_road_length=50.0,
        wall_height=1.0,
        additional_width=0.5,
        smooth_junctions=True,
        enable_mesh_visibility=True
    ))
print("✅ 20m环岛.xodr加载成功！生成点数量:", len(world.get_map().get_spawn_points()))