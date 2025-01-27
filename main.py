import random
import networkx as nx
import pandas as pd
from decimal import *
from math import *
import time
import numpy as np
import vi
from select_node import select_nearest_sink_node

# パラメータの設定
TOTAL_SIMULATIONS = 3000 #人の数
MAX_SIMULATION_TIME = 1500  # シミュレーション最大の時間（秒）
ALPHA = 1.919 #BPRのα 参考論文から1.919 or 1.4118
BETA = 6.9373 #BPRのβ 参考論文から6.9373 or 5.0365
SINK_NODE_1 = 81  # 避難所1のノードID
SINK_NODE_2 = 213  # 避難所2のノードID
HUMAN_SPEED = 1.25  # 人の速度（m/s）
TIME_INTERVAL = 1  # 時間間隔

random.seed(1) #ランダムの固定化
start_time = time.time() #実行時間開始
# 新しいグラフの初期化
graph_with_time = nx.DiGraph()

# CSVから道路情報を読み込み、重み付きエッジを追加
edges_df = pd.read_csv('dataset/data.csv')
# ノードIDを取得して1次元配列に変換後、重複を削除
unique_nodes = pd.unique(edges_df[['u', 'v']].values.ravel('K'))
node_id_map = {node: idx + 1 for idx, node in enumerate(unique_nodes)}
edges_df['travel_time'] = (edges_df['length'] / HUMAN_SPEED).round(4)

edges_df = edges_df[['u', 'v', 'length', 'travel_time', 'highway']]

# highwayの種類
highway_types = edges_df['highway'].unique()

# 道路タイプごとのキャパシティ係数
capacity_coefficients = {'trunk': 10,'primary': 10,'tertiary': 10,'unclassified': 5,'secondary': 5,'footway': 2,'foot': 2,'residential': 2}

# 移動時間と容量を持たせたグラフ作成
capacity_map = {}
time_map = {}
now_map = {}
g_count_map = {}
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
    key = (source_node, target_node)
    capacity_map[key] = capacity
    time_map[key] = travel_time
    now_map[key] = 0
    g_count_map[key] = 0
graph_with_time.add_weighted_edges_from(time_edges)

n_di_map = {}
for i, row in edges_df.iterrows():
    source_node = node_id_map[row['u']]
    target_node = node_id_map[row['v']]
    key = (source_node, target_node)
    length = row['length']
    n_di_map[key] = length

def dijkstra_with_nx_dijkstra_path(graph) -> tuple[list, int, int]:
    paths = []
    error_count = 0
    goal_select = 0
    i = 0
    while(i != TOTAL_SIMULATIONS):
        try:
            start = random.choice(list(graph_with_time.nodes))
            if start in (SINK_NODE_1, SINK_NODE_2):
                goal_select += 1
                continue
            # `nx.dijkstra_path` を使って最短経路を取得
            path = nx.dijkstra_path(graph, source=start, target=select_nearest_sink_node(graph_with_time, start), weight='length')
            # 距離を計算
            # length = nx.dijkstra_path_length(graph, source=start, target=select_nearest_sink_node(start), weight='length')
            length = 0
            for j in range(len(path)-1):
                key = (path[j], path[j+1])
                length += n_di_map[key]
            # 結果を保存
            paths.append((length, path))
            i += 1
        except nx.NetworkXNoPath:
            # 到達不能な場合はスキップ
            error_count += 1
            continue    

    return paths, error_count, goal_select

paths, no_path_count, from_goal_count = dijkstra_with_nx_dijkstra_path(graph_with_time)

#　距離による並び替え（近い順）
paths.sort(key=lambda x: x[0])
#　距離による並び替え（遠い順）
# paths.sort(key=lambda x: x[0], reverse=True)
#　距離による並び替え（ランダム）
# random.shuffle(paths)


# 並び替え後避難者のスタート位置
current = {}
yet = {}
for i, value in enumerate(paths):
    init_start_node = value[1][0]
    init_target_node = value[1][1]
    key = (init_start_node, init_target_node)
    travel_time = time_map[key]
    g_count_map[key] += 1
    current[i] = {
        "node": init_start_node,
        "index": 0,
        "from_index": 0,
        "need_time": travel_time
    }
    yet[i] = True

# 時間によるシミュレート
time=0

evac_time = {}

# 初期化済みのnow_mapを使う
# シミュレーションの実行
while(len(yet) != 0 or time < MAX_SIMULATION_TIME):
    for i, value in enumerate(paths):
        if i not in yet:
            continue
        current[i]["from_index"] += TIME_INTERVAL
        # 時間が経過したら次のノードに移動
        if current[i]["from_index"] >= current[i]["need_time"]:
            prev_key = (current[i]["node"], value[1][current[i]["index"]+1])
            g_count_map[prev_key] -= 1
            current[i]["node"] = value[1][current[i]["index"]+1]
            current[i]["index"] += 1
            if current[i]["index"] == len(value[1])-1:
                print(f"避難者{i+1}は避難所に到達しました, 残り人数: {len(yet)}")
                evac_time[i] = time
                yet.pop(i)
                continue
            key = (current[i]["node"], value[1][current[i]["index"]+1])
            g_count_map[key] += 1
            current[i]["need_time"] = time_map[key] * (1 + ALPHA * ((g_count_map[key] / capacity_map[key]) ** BETA))
            current[i]["from_index"] = 0
    time += TIME_INTERVAL

for k, v in evac_time.items():
    print(f"避難者{k+1}の避難時間: {v}")

exit()
# 経路選択

res = {}
for i, value in enumerate(paths):
    path = value[1]
    time = 0
    for j in range(len(path)-1):
        key = (path[j], path[j+1])
        capacity = capacity_map[key]
        traffic = g_count_map[key]
        distance = n_di_map[key]
        time += (distance/HUMAN_SPEED) * (1 + ALPHA * ((traffic / capacity) ** BETA))
        print(f"エッジ{key}の容量: {capacity}, 交通量: {traffic}, 距離: {distance}, 時間: {time}")
    res[i] = time
        
        
    # print(f"避難者 {i+1}:{p}")
print(f"到達不能な避難者数: {no_path_count}")
print(f"避難所を選択した避難者数: {from_goal_count}")

for k, v in res.items():
    print(f"避難者{k+1}の避難時間: {v}")


# print(g_count_map)
# print(capacity_map)

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

def simulate_evacuation(total_evacuations):
    time_expanded_graph = create_time_expanded_graph()







# end_time = time.time()  # 実行時間計測終了
# print(f"計算時間：{end_time - start_time:.2f}秒")