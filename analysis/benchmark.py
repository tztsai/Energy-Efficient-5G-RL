# %%
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

scenario = 'A'
files = glob.glob(f'sim_stats/*/{scenario}/trajectory.csv')
frames = [pd.read_csv(f, index_col=0) for f in files]
agents = [f.split('\\')[1] for f in files]
df = pd.concat(frames, keys=agents, names=['policy'])
df = df[~df.index.duplicated(keep='last')]
df

# %%
key_pat = re.compile('(reward|pc_penalty|qos_reward|interference)$')
vars_df = df[list(filter(key_pat.match, df.columns))].copy().rename(columns={'pc_penalty': 'Power Consumption (kW)', 'qos_reward': 'QoS Reward', 'reward': 'Reward', 'interference': 'Interference'})
vars_df

# %%
for key, ser in vars_df.items():
    _df = ser.unstack(level=0).rolling(20).mean()
    # a = _df.values
    # s = np.nanstd(a) * 2
    # m = np.nanmedian(a)
    # ymin, ymax = max(np.nanmin(a) , m - s), min(np.nanmax(a) , m + s)
    fig = px.line(_df, title=key, labels={'value': '', 'time': ''})
    # fig.update_yaxes(range=[ymin, ymax])
    fig.update_layout()
    fig.show()
    # _df.plot(title=key)
    # plt.legend(title='')

# %%
key_pat = re.compile('(arrival_rate|actual_rate|required_rate)$')
rate_df = df[list(filter(key_pat.match, df.columns))].copy()
for agent in rate_df.index.levels[0]:
    _df = rate_df.loc[agent].rolling(60).mean()
    _df.plot(title=agent)
    
# %%
