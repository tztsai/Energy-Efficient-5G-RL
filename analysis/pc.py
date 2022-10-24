# %%
import numpy as np
import pandas as pd
import plotly.express as px

df = pd.read_csv('sim_stats/random/B/pc.csv', index_col=0)
df

# %%
fig = px.scatter_3d(df.loc[(df.S == 0) & (df.K <= 20)],
              x='m', y='K', z='P', color='P',
              labels=dict(P='P (W)', m='Antennas', K='Users'))
# fig.update_traces(marker=dict(size=6))
fig.update_layout(margin=dict(l=5, r=5, b=5, t=5),
                  scene_aspectmode='cube',
                  coloraxis_colorbar_x=0.95,)
fig.write_html('P-M-K.html')
fig

# %%
# px.scatter_3d(df.loc[df.S == 0],
#               x='M', y='R', z='P',
#               color='P')

# %%
# px.scatter_3d(df, x='S', y='M', z='Pnl')
fig = px.scatter_3d(df[df.K <= 20], x='S', y='m', z='P', color='K',
                    labels=dict(P='P (W)', m='Antennas', K='Users', S='Sleep level'))
fig.update_layout(scene=dict(xaxis=dict(dtick=1)),
                  scene_aspectmode='cube',
                  margin=dict(l=5, r=5, b=5, t=5),
                  coloraxis_colorbar_x=0.95,)
# fig.update_traces(marker=dict(size=6))
fig.write_html('P-S-M.html')
fig

# %%
