# %%
import numpy as np  
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

df = pd.read_csv('avg_traffic.csv', index_col=[0, 1, 2])
df = pd.DataFrame(df.values.reshape(len(df), 24, 2).sum(axis=-1),
                  index=df.index, columns=df.columns[::2])
df /= 1e6  # convert to Gb
df

# %%
# monthly traffic
df1 = df.sum(axis=1).groupby('day').mean() / 24
df1.plot()

# %%
# weekly traffic
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
df2 = df.groupby('weekday').mean().loc[days].stack()
df2.index = [f'{h} {d}' for d, h in df2.index]
df2 = pd.Series(df2.values.reshape(-1, 2).mean(axis=1), 
                index=df2.index[::2])
df2['24:00 Sun'] = df2['00:00 Mon']
df2.plot()

# %%
df3 = df.mean()
df3['24:00'] = df3['00:00']
df3.plot()

# %%
fig = make_subplots(
    rows=1, cols=3, shared_yaxes=True,
    subplot_titles=['Monthly Average', 'Weekly Average', 'Daily Average'],
    y_title='traffic volume per hour (GB)')
fig.add_trace(px.line(df1).data[0], row=1, col=1)
fig.add_trace(px.line(df2).data[0], row=1, col=2)
fig.add_trace(px.line(df3).data[0], row=1, col=3)
fig.update_layout(showlegend=False,
                  width=1500, height=400,
                  margin=dict(l=80, r=30, t=50, b=30),
                  xaxis=dict(dtick=3, tick0=1, title='day of month'),
                  xaxis2=dict(title='day of week', tickvals=df2.index[6::12], ticktext=days),
                  xaxis3=dict(tickvals=df3.index[::2], ticktext=[f'{h:02}' for h in range(0, 26, 2)], title='hour of day'))
fig.write_image('avg_traffic.pdf', scale=2)
fig

# %%
