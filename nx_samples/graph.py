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

# 無効グラフ
G = nx.Graph()

# ノードの追加
G.add_node(1)  # 単一のノード
G.add_nodes_from([2, 3, 4])  # 複数のノード

# エッジの追加
G.add_edge(1, 2)  # 単一のエッジ
G.add_edges_from([(2, 3), (3, 4)])  # 複数のエッジ

nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()
