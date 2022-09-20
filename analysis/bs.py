# %%
import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv('bs_stats.csv', index_col=0)
print(df.columns)
df

# %%
df.describe()

# %%
stats_df = df[[c for c in df.keys() if c.startswith('avg_')]].copy()

stats_df.describe()

# %%
stats_df['avg_sum_rate'] /= 1e6
stats_df['avg_req_rate'] /= 1e6
stats_df['avg_cell_data_rate'] /= 1e6
px.scatter(stats_df)

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
