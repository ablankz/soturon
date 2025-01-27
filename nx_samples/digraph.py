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

# 有効グラフ
DG = nx.DiGraph()
DG.add_edge(1, 2)  # 1から2への有向エッジ
DG.add_edge(2, 3)

# nx.draw(DG, with_labels=True, node_color='lightblue', edge_color='gray')
# plt.show()

# # 重みつき有向グラフ
# G = nx.DiGraph()
# G.add_edge(1, 2, weight=1)  # 1から2への有向エッジ
# G.add_edge(2, 3, weight=2)

# nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
# plt.show()

# 避難シミュレーション
G = nx.DiGraph()
G.add_weighted_edges_from([(1, 2, 10), (2, 3, 5), (1, 3, 15)])  # 重みは道路の容量など

nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()

# 人数をシミュレーション
people_count = {1: 100, 2: 0, 3: 0}  # 各ノードの初期人数
path = nx.shortest_path(G, source=1, target=3, weight='weight')  # 最短経路

# 経路に従って人数を移動
for i in range(len(path) - 1):
    edge = (path[i], path[i+1])
    capacity = G[edge[0]][edge[1]]['weight']
    move_people = min(people_count[path[i]], capacity)
    people_count[path[i]] -= move_people
    people_count[path[i+1]] += move_people

print("シミュレーション後の人数:", people_count)