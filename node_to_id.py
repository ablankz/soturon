# ノードIDを元の形式に戻す
def map_path_to_original(path, node_id_map):
    return [{v: k for k, v in node_id_map.items()}[node_id] for node_id in path]