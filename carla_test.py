import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)

print("server:", client.get_server_version())
print("client:", client.get_client_version())
print("has Town12:", any("Town12" in m for m in client.get_available_maps()))

w = client.get_world()
s = w.get_settings()
print("has tile_stream_distance:", hasattr(s, "tile_stream_distance"))
print("has actor_active_distance:", hasattr(s, "actor_active_distance"))