# %%
import glob
import os, sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from network.network import MultiCellNetwork as Net
from network.base_station import BaseStation as BS

def get_bs_obs_name(idx):
    i = int(idx)
    if i < BS.public_obs_ndims:
        return f'public_{i}'
    i -= BS.public_obs_ndims
    if i < BS.private_obs_ndims:
        if i < 5:
            return f'wakeup_{i}'
        i -= 5
        if i < BS.bs_stats_dim:
            t, i = divmod(i, BS.buffer_size[1])
            return f'bs_stats_{t}_{i}'
        i -= BS.bs_stats_dim
        return f'ue_stats_{i}'
    i -= BS.private_obs_ndims
    nb_id, i = divmod(i, BS.other_obs_ndims)
    if i < BS.public_obs_ndims:
        return f'other_{nb_id}_public_{i}'
    i -= BS.public_obs_ndims
    return f'other_{nb_id}_shared_{i}'

def get_net_obs_name(idx):
    i = int(idx)
    if i < Net.global_obs_ndims:
        return f'global_{i}'
    i -= Net.global_obs_ndims
    return str(i)
    
mean_stats, std_stats = [pd.read_csv(f) for f in glob.glob('feature_stats_13*.csv')]

mean_stats.columns = mean_stats.columns.map(get_bs_obs_name)
std_stats.columns = std_stats.columns.map(get_bs_obs_name)

# %%
mean_stats

# %%
mean_stats.mean(axis=0).plot(legend=False)

# %%
mean_stats.std(axis=0).plot(legend=False)  # std among agents

# %%
std_stats.mean(axis=0).plot(legend=False)  # std over time

# %%
mean_stats[mean_stats > 20].dropna(axis=1, how='all')

# %%
std_stats[std_stats > 20].dropna(axis=1, how='all')

# %%
const_stats = mean_stats[std_stats < 1e-8].dropna(axis=1, how='any')
const_stats

# ======================================================

# %%
mean_stats, std_stats = [pd.read_csv(f) for f in glob.glob('analysis/feature_stats_4*.csv')]

mean_stats.columns = mean_stats.columns.map(get_net_obs_name)
std_stats.columns = std_stats.columns.map(get_net_obs_name)

mean_stats = mean_stats.iloc[0,:Net.global_obs_ndims]
std_stats = std_stats.iloc[0,:Net.global_obs_ndims]

# %%
mean_stats.plot()

# %%
std_stats.plot()

# %%
mean_stats[mean_stats > 20].dropna(how='all')

# ======================================================

# %%
mean_stats, std_stats = [pd.read_csv(f) for f in glob.glob('analysis/feature_norm_stats_1*.csv')]

mean_stats.columns = mean_stats.columns.map(get_bs_obs_name)
std_stats.columns = std_stats.columns.map(get_bs_obs_name)

# %%
mean_stats

# %%
mean_stats.mean(axis=0).plot(legend=False)

# %%
mean_stats.std(axis=0).plot(legend=False)  # std among agents

# %%
std_stats.mean(axis=0).plot(legend=False)  # std over time

# %%
const_stats = mean_stats[std_stats < 0.01].dropna(axis=1, how='any')
const_stats

# %%
