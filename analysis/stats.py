# %%
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')
from network.config import bsPositions

folder = Path('sim_stats/adaptive') / 'B'

bs_stats = pd.read_csv(folder/'bs_stats.csv', index_col=0)
ue_stats = pd.read_csv(folder/'net_stats.csv', index_col=0, header=None)[1]
sleep_stats = dict()

def parse_np_series(s, **kwds):
    return pd.DataFrame(
            np.vstack([np.fromstring(x[1:-1], sep=' ')
                       if type(x) is str and x[0] == '[' else x
                       for x in s]), **kwds)
    
for c in list(bs_stats.columns):
    v = bs_stats[c].iloc[0]
    if type(v) == str and v.startswith('['):
        sleep_stats[c] = parse_np_series(bs_stats.pop(c), index=bs_stats.index)
bs_stats

# %%
bs_idx = [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1]
x, y = bsPositions[bs_idx].T
kpis = [k for k in bs_stats.columns if k.startswith('avg_')]
kpis = ['avg_sum_rate', 'avg_serving_ues', 'avg_num_ants',
        'avg_signal', 'avg_interf', 'avg_sinr', 'avg_pc']

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=20, y=0, z=0),
    eye=dict(x=1.5, y=1.5, z=1.5)
)

for kpi in kpis:
    f = bs_stats.loc[bs_idx, kpi].to_frame()
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=f[kpi], mode='markers+lines', marker_color=f.index))
    fig.update_scenes(xaxis_title='x', yaxis_title='y', zaxis_title=kpi)
    fig.update_layout(scene_camera=camera,
                      margin=dict(l=0, r=0, b=0, t=0))
    # fig.write_image(folder/f'bs_{kpi}.png')
    fig.show()

# %%
sleep_stats = pd.concat(sleep_stats, axis=1)
sleep_stats

# %%
scalars = {c: ue_stats.pop(c) for c, a in ue_stats.iteritems()
           if type(a) is not str or a[0] != '['}
ue_stats = parse_np_series(ue_stats.values, index=ue_stats.index)
ue_stats

# %%
