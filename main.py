import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib_fontja
from simulate import simulate
from patterns import *
import time

patterns = [
    # # 1 比較手法の概要説明
    # bpr('length', 'near', 1000, [1.25, 1.0, 1.5], 1.919, 6.9373),

    # # BPR関数を用いた比較手法では、避難所までの距離に基づき、最短経路を選択する方式で行う

    # 2 比較手法の悪い点
    # bpr('length', 'near', 1500, [1.25, 1.0, 1.5], 1.919, 6.9373),
    # bpr('length', 'far', 2000, [1.25, 1.0, 1.5], 1.919, 6.9373),
    # bpr('length', 'random', 2200, [1.25, 1.0, 1.5], 1.919, 6.9373),

    # 比較手法では、人数が増えた際に避難時間が大幅に増加してしまう。
    # これは、避難所までの距離に基づき、初めに決定された経路を選択する方式で行うため、
    # 避難者が混んでいる道を避けることができないためである。
    # # BPR関数を用いた比較手法では、避難所までの距離に基づき、最短経路を選択する方式で行う
    # - 爆発しやすさ: near(1500) -> random(2300) -> far(2500)

    # 3 提案手法の提案
    # bpr('length', 'near', 1500, [1.25, 1.0, 1.5], 1.919, 6.9373, '比較手法'),
    # bpr('travel_time', 'near', 1500, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法'),

    # # 提案手法では、避難所までの所要時間に基づき、最短経路を選択する方式で行う.
    # # これは、所要時間は、BPR関数によって計算されており、一定時間間隔おきに経路が再決定されるため、
    # # 計算リソースの消費が大きいが、避難者が混んでいる道を避けることをシミュレートすることができる。

    # # # 4. 提案手法の内部比較(順序による違い)
    bpr('travel_time', 'near', 1000, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法 近'),
    bpr('travel_time', 'far', 1000, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法 遠'),
    bpr('travel_time', 'random', 1000, [1.25, 1.0, 1.5], 1.919, 6.9373, '提案手法 ランダム'),
    # bpr('length', 'random', 1000, [1.25, 1.0, 1.5], 1.919, 6.9373, '比較手法'),

    # 1000: 基本形
    # ~2500: ある程度増えても、避難所付近のキャパシティを超えない限りは、避難時間がほとんど増えない
    # 2500~5000: 順序による差がなくなってくる

    # # 避難を時間間隔おきに行動を開始する避難者の優先順位を変更することで、避難所までの所要時間が変化することがわかる。

    # # 5. 比較手法と提案手法の比較
    # bpr('travel_time', 'near', 1000, [1.25, 1.0, 1.5], 1.919, 6.9373),
    # bpr('length', 'near', 1000, [1.25, 1.0, 1.5], 1.919, 6.9373),

    # bpr('length', 'near', 1500, [1.25, 1.0, 1.5], 1.919, 6.9373),
    # bpr('travel_time', 'near', 1500, [1.25, 1.0, 1.5], 1.919, 6.9373),
]

time_df = pd.DataFrame()
speed_df = pd.DataFrame()
length_df = pd.DataFrame()
res_df = pd.DataFrame()
for i, pt in enumerate(patterns):
    print(f"{pt['title']}")
    p = pt['pattern']
    start_time = time.time()
    res = simulate(
        human_speeds=p['human_speeds'],
        model=p['model'],
        weight=p['weight'],
        order=p['order'],
        simulations=p['simulation'],
        default_human_speed=p['human_speeds'][0],
        time_interval=p['time_interval'],
        v_min=p['v_min'],
        exponential_alpha=p['exponential_alpha'],
        bpr_alpha=p['bpr_alpha'],
        bpr_beta=p['bpr_beta']
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"elapsed_time: {elapsed_time}")
    time_df[f'{pt["title"]}'] = [elapsed_time]

    ev_speed = res['speed']
    ev_speed.index = np.arange(1, len(ev_speed)+1)
    speed_df[f'{pt["title"]}'] = ev_speed

    ev_length = res['length']
    ev_length.index = np.arange(1, len(ev_length)+1)
    length_df[f'{pt["title"]}'] = ev_length
    
    ev_time = res['evac_time']
    ev_time.index = np.arange(1, len(ev_time)+1)
    res_df[f'{pt["title"]}'] = ev_time

res_df.to_csv('outputs/simulations.csv', index=False, encoding='utf-8-sig')
time_df.to_csv('outputs/time.csv', index=False, encoding='utf-8-sig')
speed_df.to_csv('outputs/speed.csv', index=False, encoding='utf-8-sig')
length_df.to_csv('outputs/length.csv', index=False, encoding='utf-8-sig')

res_df = pd.read_csv('outputs/simulations.csv')

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
plt.savefig('plot/plot.png') 

length_df = pd.read_csv('outputs/length.csv')

# indexが到着順位となっている
length_df["到着順位"] = length_df.index

# 順位100区切りの平均を取得
length_df["到着順位"] = length_df["到着順位"].apply(lambda x: x//50 * 50)
length_df = length_df.groupby("到着順位").mean()

plt.figure(figsize=(8, 5))
for column in length_df.columns:
    plt.plot(length_df.index, length_df[column], marker='o', label=column, markersize=1)

# 軸ラベルとタイトル
plt.xlabel('到着順位')
plt.ylabel('避難経路長（m）')
plt.title("避難経路長")
plt.legend()
plt.grid(True)

# グラフを表示
plt.savefig('plot/length.png')

speed_df = pd.read_csv('outputs/speed.csv')

# indexが到着順位となっている
speed_df["到着順位"] = speed_df.index

# 順位100区切りの平均を取得
speed_df["到着順位"] = speed_df["到着順位"].apply(lambda x: x//50 * 50)
speed_df = speed_df.groupby("到着順位").mean()

plt.figure(figsize=(8, 5))
for column in speed_df.columns:
    plt.plot(speed_df.index, speed_df[column], marker='o', label=column, markersize=1)

# 軸ラベルとタイトル
plt.xlabel('到着順位')
plt.ylabel('避難速度（m/s）')
plt.title("避難速度")
plt.legend()
plt.grid(True)

# グラフを表示
plt.savefig('plot/speed.png')