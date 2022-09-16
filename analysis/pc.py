# %%
import numpy as np
import pandas as pd
import plotly.express as px

stats_file = '../results/pc.txt'
a = np.loadtxt(stats_file)
df = pd.DataFrame(a, columns='M K R S Pnl Pld P'.split())
df

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
