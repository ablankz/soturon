import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_fontja
from simulate import simulate
from patterns import *

patterns = [
    # exponential('travel_time', 'near', 2000, 1.25, 0.03, 2.0),
    exponential('travel_time', 'near', 5000, 1.25, 0.05, 2.0),
    exponential('travel_time', 'far', 5000, 1.25, 0.05, 2.0),
    exponential('travel_time', 'random', 5000, 1.25, 0.05, 2.0),
    # exponential('travel_time', 'near', 2000, 1.25, 0.03, 4.0),
    # exponential('travel_time', 'near', 2000, 1.25, 0.03, 5.0),
    # exponential('length', 'near', 2000, 1.25, 0.05, 2.0),
    # bpr('travel_time', 'near', 2000, 1.25, 1.919, 6.9373),
]

res_df = pd.DataFrame()
for i, pt in enumerate(patterns):
    print(f"{pt['title']}")
    p = pt['pattern']
    res = simulate(
        model=p['model'],
        weight=p['weight'],
        order=p['order'],
        simulations=p['simulation'],
        human_speed=p['human_speed'],
        time_interval=p['time_interval'],
        v_min=p['v_min'],
        exponential_alpha=p['exponential_alpha'],
        bpr_alpha=p['bpr_alpha'],
        bpr_beta=p['bpr_beta']
    )
    ev_time = res['evac_time']
    ev_time.index = np.arange(1, len(ev_time)+1)
    res_df[f'{pt["title"]}'] = ev_time

res_df.to_csv('simulations.csv', index=False, encoding='utf-8-sig')

res_df = pd.read_csv('simulations.csv')

plt.figure(figsize=(8, 5))
for column in res_df.columns:
    plt.plot(res_df.index, res_df[column], marker='o', label=column, markersize=1)

# 軸ラベルとタイトル
plt.xlabel('避難人数')
plt.ylabel('避難時間（秒）')
plt.title("シミュレーション結果")
plt.legend()
plt.grid(True)

# グラフを表示
plt.savefig('plot.png') 