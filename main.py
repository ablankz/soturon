import random
import networkx as nx
import osmnx as ox
import pandas as pd
import numpy as np
from decimal import *
from math import *
import time
import matplotlib.pyplot as plt
import statistics
# import japanize_matplotlib

# パラメータの設定
TOTAL_SIMULATIONS = 2000 #人の数
MAX_SIMULATION_TIME = 1500  # シミュレーション最大の時間（秒）
ALPHA = 1.919 #BPRのα 参考論文から1.919 or 1.4118
BETA = 6.9373 #BPRのβ 参考論文から6.9373 or 5.0365
SINK_NODE_1 = 81  # 避難所1のノードID
SINK_NODE_2 = 213  # 避難所2のノードID
HUMAN_SPEED = 1.25  # 人の速度（m/s）

random.seed(91) #ランダムの固定化
start_time = time.time() #実行時間開始

# 地図データを取得し、双方向グラフを作成
original_graph = ox.graph_from_place("Ginza, Tokyo, Japan", network_type="drive")
reversed_graph = original_graph.reverse(copy=True)
full_graph = nx.compose(original_graph, reversed_graph)

# 新しいグラフの初期化
graph_with_time = nx.DiGraph()

# 最短距離から最適な避難所を選択
def select_nearest_sink_node(source_node):
    # 各避難所までの距離を計算し、最短の避難所を返す
    distance_to_sink_1 = nx.shortest_path_length(graph_with_time, source=source_node, target=SINK_NODE_1, weight='length')
    distance_to_sink_2 = nx.shortest_path_length(graph_with_time, source=source_node, target=SINK_NODE_2, weight='length')
    return SINK_NODE_1 if distance_to_sink_1 < distance_to_sink_2 else SINK_NODE_2

# 不要なノードを削除
nodes_to_remove = [(139.760389, 35.6677416), (139.760424, 35.6720689),(139.7705251, 35.6746863), (139.7714512, 35.6741518),(139.767575, 35.675270), (139.764824, 35.674587),(139.765952, 35.675782), (139.758799, 35.669082),(139.764527, 35.6739148), (139.767491, 35.675292),(139.768255, 35.6752416), (139.76476, 35.6677338)]
for lon, lat in nodes_to_remove:
    nearest_node = ox.distance.nearest_nodes(full_graph, lon, lat)
    if full_graph.has_node(nearest_node):
        full_graph.remove_node(nearest_node)

# 不要なエッジを削除
start_node_to_remove = ox.distance.nearest_nodes(full_graph, 139.77018, 35.67295)
end_node_to_remove = ox.distance.nearest_nodes(full_graph, 139.76460, 35.66746)
if full_graph.has_edge(start_node_to_remove, end_node_to_remove):
    full_graph.remove_edge(start_node_to_remove, end_node_to_remove)

# CSVから道路情報を読み込み、重み付きエッジを追加
edges_df = pd.read_csv('dataset/data.csv')
unique_nodes = pd.unique(edges_df[['u', 'v']].values.ravel('K'))
node_id_map = {node: idx + 1 for idx, node in enumerate(unique_nodes)}
edges_df['travel_time'] = (edges_df['length'] / HUMAN_SPEED).round(4)

# 道路タイプごとのキャパシティ係数
capacity_coefficients = {'trunk': 10,'primary': 10,'tertiary': 10,'unclassified': 5,'secondary': 5,'footway': 2,'foot': 2,'residential': 2}
# capacity_coefficients = {'trunk': 10,'primary': 10,'tertiary':10,'unclassified': 10,'secondary': 10,'footway': 10,'foot': 10,'residential': 10}

# 移動時間と容量を持たせたグラフ作成
time_edges = []
for _, row in edges_df.iterrows():
    source_node = node_id_map[row['u']]
    target_node = node_id_map[row['v']]
    length = row['length']
    travel_time = row['travel_time']
    highway_type = row['highway']
    coefficient = capacity_coefficients.get(highway_type, 1)
    capacity = round(coefficient * length, 2)
    time_edges.append((source_node, target_node, {'travel_time': travel_time, 'capacity': capacity}))
graph_with_time.add_weighted_edges_from(time_edges)

# ノードIDを元の形式に戻す
def map_path_to_original(path):
    return [{v: k for k, v in node_id_map.items()}[node_id] for node_id in path]

# 時間付きのグラフを生成
def create_time_expanded_graph():
    time_intervals = np.arange(0.0, MAX_SIMULATION_TIME + 1, 1)
    time_expanded_graph = nx.DiGraph()

    # 各ノードの時間ステップエッジを作成
    for node in graph_with_time.nodes:
        for current_time in time_intervals:
            if current_time + 1 > MAX_SIMULATION_TIME:
                break
            time_expanded_graph.add_edge((node, current_time), (node, current_time + 1), travel_time=0, capacity=10000, traffic=0, adjusted_travel_time=0)

    # 元のエッジに基づいた時間付きエッジを作成(travel_time:移動時間, capacity:容量, traffic:交通量, adjusted_travel_time:交通量を考慮した移動時間)
    for source, neighbors in graph_with_time.adjacency():
        for target, edge_data in neighbors.items():
            for start_time in time_intervals:
                travel_time = edge_data['weight']['travel_time']
                capacity = edge_data['weight']['capacity']
                time_expanded_graph.add_edge((source, start_time), (target, start_time + int(travel_time)), travel_time=travel_time, capacity=capacity, traffic=0, adjusted_travel_time=travel_time)

    return time_expanded_graph

# シミュレーションの実行
def simulate_evacuation(total_evacuations):
    time_expanded_graph = create_time_expanded_graph()
    # nx.write_edgelist(full_graph, "full_graph.edgelist")
    # nx.write_edgelist(graph_with_time, "graph_with_time.edgelist")
    # nx.write_edgelist(time_expanded_graph, "time_expanded.edgelist")
    evacuation_results = []
    for _ in range(total_evacuations):
        while(1):
            try:
                while True:
                    start_node = random.choice(list(graph_with_time.nodes))
                    if start_node not in (SINK_NODE_1, SINK_NODE_2):
                        break
                destination_sink = select_nearest_sink_node(start_node)
                evacuation_path = nx.dijkstra_path(time_expanded_graph, (start_node, 0), (destination_sink, MAX_SIMULATION_TIME - 1), weight='adjusted_travel_time')
                # evacuation_path = nx.dijkstra_path(time_expanded_graph, (start_node, 0), (destination_sink, MAX_SIMULATION_TIME - 1), weight='length')
                total_time = sum(time_expanded_graph[u][v]['adjusted_travel_time'] for u, v in zip(evacuation_path[:-1], evacuation_path[1:]))
                for u, v in zip(evacuation_path[:-1], evacuation_path[1:]):
                    time_expanded_graph[u][v]['traffic'] += 1
                    time_expanded_graph[u][v]['adjusted_travel_time'] *= 1 + ALPHA * ((time_expanded_graph[u][v]['traffic'] / time_expanded_graph[u][v]['capacity']) ** BETA)   
                evacuation_results.append((total_time, dict.fromkeys([step[0] for step in evacuation_path])))
                break
            except Exception as e:
                pass

    evacuation_results.sort(key=lambda x: x[0])
    
    # 結果のプロット
    for idx, (time_taken, path) in enumerate(evacuation_results):
        route_nodes = map_path_to_original(path)
        start_node = route_nodes[0]
        #fig, ax = ox.plot_graph_route(full_graph, route_nodes, route_color='green', route_linewidth=3, node_size=0, show=False, close=False)
        sink = map_path_to_original([SINK_NODE_1,SINK_NODE_2])
        start_node_position = full_graph.nodes[start_node]['x'], full_graph.nodes[start_node]['y']
        # ax.scatter(start_node_position[0],start_node_position[1],c='blue',s=100)
        # for sink_node_position in [(full_graph.nodes[node]['x'], full_graph.nodes[node]['y']) for node in sink]:
        #     ax.scatter(sink_node_position[0],sink_node_position[1],c='red',s=100,)
        # fig.savefig(f"img/{idx}.png")
        # plt.close(fig)
        print(f"避難者 {idx+1}: 避難時間 = {time_taken:.2f} 秒, パス: {' -> '.join(map(str, path))}")
    # グラフのタイトルと軸ラベルの設定
    plt.xlabel('避難人数')
    plt.ylabel('避難時間（秒）')
    evacuation_times = [result[0] for result in evacuation_results]
    evacuees = np.arange(1, total_evacuations+1)
    # CSV保存用コード
    evacuation_data = pd.DataFrame({
    'Evacuees': evacuees,
    'Evacuation_Time': evacuation_times
    })
    csv_filename = '2000_simulate.csv'
    evacuation_data.to_csv(csv_filename, index=False, encoding='utf-8-sig')
    # 折れ線グラフの作成
    plt.plot(evacuees, evacuation_times, marker='o', linestyle='-', color='b')
    # グリッドの表示（オプション）
    plt.grid(True)
    # グラフを保存
    plt.savefig('plot.png')  # PNG形式で保存
    # average_time = sum([result[0] for result in evacuation_results]) / total_evacuations
    # #stddev_time = statistics.pstdev([result[0] for result in evacuation_results])
    # max_time = max([result[0] for result in evacuation_results])
    # min_time = min([result[0] for result in evacuation_results])
    # print(f"平均避難時間: {average_time:.2f}秒, 標準偏差: {stddev_time:.2f}, 最大: {max_time:.2f}, 最小: {min_time:.2f}")

simulate_evacuation(TOTAL_SIMULATIONS)  # シミュレーション開始
end_time = time.time()  # 実行時間計測終了
print(f"計算時間：{end_time - start_time:.2f}秒")


# グラフの表示
plt.show()