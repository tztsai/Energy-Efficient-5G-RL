# %%
import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv('pc.csv', index_col=0)
df

# %%
px.scatter_3d(df.loc[df.S == 0],
              x='M', y='K', z='Pld',
              color='Pld')
# df.plot(kind='scatter', x='M', y='P')
# df.plot(kind='scatter', x='K', y='Pld')
# df.plot(kind='scatter', x='R', y='Pld')
# df.plot(kind='scatter', x='S', y='Pnl')

# %%
px.scatter_3d(df.loc[df.S == 0],
              x='M', y='R', z='Pld',
              color='Pld')

# %%
# px.scatter_3d(df, x='S', y='M', z='Pnl')
px.scatter_3d(df, x='S', y='M', z='P', color='P')

# %%
