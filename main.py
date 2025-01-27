import random
import networkx as nx
import pandas as pd
from decimal import *
from math import *
import time
from select_node import select_nearest_sink_node

# パラメータの設定
TOTAL_SIMULATIONS = 3000 #人の数
MAX_SIMULATION_TIME = 1500  # シミュレーション最大の時間（秒）
MAX_SIMULATION_TIME = inf
ALPHA = 1.919 #BPRのα 参考論文から1.919 or 1.4118
BETA = 6.9373 #BPRのβ 参考論文から6.9373 or 5.0365
SINK_NODE_1 = 81  # 避難所1のノードID
SINK_NODE_2 = 213  # 避難所2のノードID
HUMAN_SPEED = 1.25  # 人の速度（m/s）
TIME_INTERVAL = 0.0001  # 時間間隔

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
init_time_map = {}
current_travel_time_map = {}
now_map = {}
pass_count_map = {}
n_di_map = {}
time_edges = []
for _, row in edges_df.iterrows():
    source_node = node_id_map[row['u']]
    target_node = node_id_map[row['v']]
    length = row['length']
    travel_time = row['travel_time']
    highway_type = row['highway']
    coefficient = capacity_coefficients.get(highway_type, 1)
    capacity = round(coefficient * length, 2)
    # 重み付きエッジを追加
    graph_with_time.add_edge(source_node, target_node, weight=travel_time)
    key = (source_node, target_node)
    # キャパシティと移動時間を保存
    capacity_map[key] = capacity
    init_time_map[key] = travel_time
    current_travel_time_map[key] = travel_time
    now_map[key] = 0
    pass_count_map[key] = 0
    n_di_map[key] = length
    
def dijkstra_with_nx_dijkstra_path(graph):
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
            path = nx.dijkstra_path(graph, source=start, target=select_nearest_sink_node(graph_with_time, start), weight='weight')
            length = 0
            for j in range(len(path)-1):
                key = (path[j], path[j+1])
                length += n_di_map[key]
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

# 避難者の状態
yet = []
# 避難完了した避難者
terminated = {}
# 避難完了した避難者の数
term_count = 0

for i, value in enumerate(paths):
    init_start_node = value[1][0]
    # 現在、最も移動時間が短い経路を選択
    path = nx.dijkstra_path(
        graph_with_time, 
        source=init_start_node,
        target=select_nearest_sink_node(graph_with_time, init_start_node),
        weight='weight'
    )
    # 現在、最も移動時間が短い経路を通る場合の次のノード
    init_target_node = path[1]
    key = (init_start_node, init_target_node)
    travel_time = current_travel_time_map[key]
    current = {
        "node_to_node": key,
        "from_index": 0,
        "need_time": travel_time
    }
    yet.append(current)
    terminated[i] = False   
    # そこの道を通る人数をカウント  
    pass_count_map[key] += 1
    # 移動時間の算出
    updated_travel_time = init_time_map[key] * (1 + ALPHA * ((pass_count_map[key] / capacity_map[key]) ** BETA))
    # グラフ上の移動時間を更新
    nx.set_edge_attributes(graph_with_time, {key: {'weight': updated_travel_time}})
    # 移動時間の更新
    current_travel_time_map[key] = updated_travel_time

# 時間によるシミュレート
ctime=0

evac_time = {}

# 初期化済みのnow_mapを使う
# シミュレーションの実行
while(ctime <= MAX_SIMULATION_TIME and term_count < TOTAL_SIMULATIONS):
    for i, current in enumerate(yet):
        if terminated[i]: # 避難完了した避難者はスキップ
            continue
        # 今のノードからどれだけ時間が経過したか
        current["from_index"] += TIME_INTERVAL
        # 時間が経過したら次のノードに移動
        if current["from_index"] >= current["need_time"]:
            key = current["node_to_node"]
            # 今のノードから出る人数をマイナス
            pass_count_map[key] -= 1
            # 更新後移動時間を計算
            updated_travel_time = init_time_map[key] * (1 + ALPHA * ((pass_count_map[key] / capacity_map[key]) ** BETA))
            # グラフ上の移動時間を更新
            nx.set_edge_attributes(graph_with_time, {key: {'weight': updated_travel_time}})
            # 移動時間の更新
            current_travel_time_map[key] = updated_travel_time
            if key[1] in (SINK_NODE_1, SINK_NODE_2): # 到達ノードがゴールであった場合
                print(f"避難者{i+1}は避難所に到達しました, 残り人数: {TOTAL_SIMULATIONS - term_count} 現在時刻: {ctime}")
                # ゴール到達時間
                evac_time[i] = ctime
                # 避難完了した避難者を更新
                terminated[i] = True
                # 避難完了した避難者の数を更新
                term_count += 1
                continue
            # 最適パスを選択
            path = nx.dijkstra_path(
                graph_with_time, 
                source=key[1], # いま到着したノード
                target=select_nearest_sink_node(graph_with_time, key[1]),
                weight='weight'
            )
            # 次のノードを選択
            next_node = path[1]
            # いま到着したノードから次のノード
            key = (key[1], next_node)
            current["node_to_node"] = key
            # その道を通る人数をカウント
            pass_count_map[key] += 1
            # 自分が含まれていない移動時間
            current["need_time"] = current_travel_time_map[key]
            # 更新後移動時間を計算
            updated_travel_time = init_time_map[key] * (1 + ALPHA * ((pass_count_map[key] / capacity_map[key]) ** BETA))
            # グラフ上の移動時間を更新
            nx.set_edge_attributes(graph_with_time, {key: {'weight': updated_travel_time}})
            # 移動時間の更新
            current_travel_time_map[key] = updated_travel_time
            # 次のノードでの経過時間を0にリセット
            current["from_index"] = 0
    ctime += TIME_INTERVAL

for k, v in evac_time.items():
    print(f"避難者{k+1}の避難時間: {v}")

end_time = time.time()
print(f"実行時間: {end_time - start_time}秒")