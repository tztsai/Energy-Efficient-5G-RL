# %%
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import plotly.express as px

folder = Path('sim-mappo') / 'B'

df = pd.read_csv(folder/'bs_stats.csv', index_col=0)
vec_df = dict()
bs_df = defaultdict(dict)

for c in list(df.columns):
    v = df[c].iloc[0]
    if c.startswith('bs_'):
        _, b, k = c.split('_', 2)
        bs_df[int(b)][k] = df.pop(c)
    elif type(v) == str and v.startswith('['):
        vec_df[c] = pd.DataFrame(
            np.vstack([np.fromstring(x[1:-1], sep=' ')
                       for x in df.pop(c)]), index=df.index)
df

# %%
bs_stats = defaultdict(dict)
for b, v in bs_df.items():
    for k in list(v):
        if k.startswith('avg_') or k.startswith('total_'):
            bs_stats[b][k] = v.pop(k).iloc[-1]
bs_stats = pd.DataFrame(bs_stats).T
bs_stats

# %%
bs_df = {b: pd.DataFrame(v) for b, v in bs_df.items()}
bs_df[0]

# %%
vec_df = pd.concat(vec_df.values(), keys=vec_df.keys())
vec_df

# %%
px.scatter_3d(df.loc[df.S == 0],
              x='M', y='K', z='Pld')
# df.plot(kind='scatter', x='M', y='P')
# df.plot(kind='scatter', x='K', y='Pld')
# df.plot(kind='scatter', x='R', y='Pld')
# df.plot(kind='scatter', x='S', y='Pnl')

# %%
px.scatter_3d(df.loc[df.S == 0],
              x='M', y='R', z='Pld')

# %%
# px.scatter_3d(df, x='S', y='M', z='Pnl')
px.scatter(df, x='S', y='Pnl')

# %%
