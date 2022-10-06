# %%
import re
import glob
import numpy as np
import pandas as pd

scenario = 'B'
files = glob.glob(f'sim-*/{scenario}/trajectory.csv')
frames = [pd.read_csv(f, index_col=0) for f in files]
agents = [f.split('-')[1].split('\\')[0] for f in files]
df = pd.concat(frames, keys=agents, names=['agent'])
df = df[~df.index.duplicated(keep='last')]
df

# %%
key_pat = re.compile('(actual_rate|reward|pc_penalty|qos_reward)$')
vars_df = df[list(filter(key_pat.match, df.columns))].copy()
vars_df

# %%
for key, ser in vars_df.items():
    _df = ser.unstack(level=0).rolling(20).mean()
    _df.plot(title=key)

# %%
key_pat = re.compile('(arrival_rate|actual_rate|required_rate)$')
rate_df = df[list(filter(key_pat.match, df.columns))].copy()
for agent in rate_df.index.levels[0]:
    _df = rate_df.loc[agent].rolling(60).mean()
    _df.plot(title=agent)
    
# %%
