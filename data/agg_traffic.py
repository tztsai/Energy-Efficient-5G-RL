# %%
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product as cartesian_product

# %%
sql_url = 'sqlite:///cell_traffic.sql'
pred_path = 'prediction.csv'

# %%
preds = pd.read_csv(pred_path, index_col=0).squeeze()
preds -= 1
preds

# %%
cell_ids = pd.read_sql("SELECT name FROM sqlite_schema WHERE type = 'table'", sql_url)['name']

def cell_id_to_site_id(cell_id):
    return int(cell_id) >> 8

cell_clusters = {}

for cell_id in cell_ids:
    site_id = cell_id_to_site_id(cell_id)
    try:
        cell_clusters[cell_id] = preds.loc[site_id]
    except KeyError:
        continue
    
pd.Series(cell_clusters).value_counts().sort_index().plot(kind='pie')

# %%
class Everything:
    def __contains__(self, item):
        return True
    
app_delay_cats = {
    'Delay Stringent':
        {"Gaming", "Real-Time Communication"},
    'Delay Sensitive':
        {"Multimedia Streaming", "Social Networking"},
    'Delay Tolerant':
        Everything()
}

# %%
interval_mins = 20
time_interval = '%dT' % interval_mins
time_splits = [t.strftime('%m-%d %a %H:%M') for t in
               pd.date_range('2021-11-25', '2022-1-26',
                             freq=time_interval, inclusive='left')]
bin_indices = {t: i for i, t in enumerate(time_splits)}
arr_index = list(cartesian_product(app_delay_cats, time_splits))
index_map = {t: i for i, t in enumerate(arr_index)}

# %%
cell_flows = {}

def read_flows(cell_id, min_num_flows=62*24, min_flow_size=128):
    # print('\nReading {}'.format(path))
    df = pd.read_sql_table(cell_id, sql_url)
    if len(df) < min_num_flows: return
    df['start_time'] = pd.to_datetime(df['start_time'])
    bins = cell_flows.setdefault(
        cell_id, np.zeros(len(arr_index)))
    for t in df.itertuples(index=False):
        if t.duration_ms == 0.: continue
        if t.down_octets < min_flow_size: continue
        flows = divide_flow(t.start_time, t.duration_ms, t.down_octets)
        for k, v in app_delay_cats.items():
            if t.app_group in v:
                delay_cat = k
                break
        for time, bps in flows:
            idx = index_map[(delay_cat, time)]
            bins[idx] += bps

def divide_flow(start_time, duration_ms, octets):
    if np.isnan(octets):
        raise ValueError('NaN DPI')
    t0 = start_time.minute

    duration = duration_ms / 60000
    t1 = t0 + duration
    a = int(np.floor(t0 / interval_mins))
    b = int(np.ceil(t1 / interval_mins))
    n_bins = b - a
    rate = octets / duration_ms * 8000  # bps
    ta = a * interval_mins
    k = start_time.strftime('%m-%d %a %H:{:02}').format(ta)
    try:
        bi = bin_indices[k]
    except KeyError:
        return
    if n_bins == 1:
        yield time_splits[bi], rate
        return
    for i in range(n_bins):
        # if i == 0:
        #     dt = ta + interval_mins - t0
        # elif i == n_bins - 1:
        #     dt = t1 % interval_mins or interval_mins
        # else:
        #     dt = interval_mins
        yield time_splits[bi+i], rate # * dt

# %%
try:
    flows_df = pd.read_feather('cell_flows.feather')
except:
    for cell_id in tqdm(cell_clusters):
        read_flows(cell_id)
    flows_df = pd.DataFrame(cell_flows)
    flows_df.to_feather('cell_flows.feather')

# %%
week_time_idx = pd.MultiIndex.from_tuples(
    [t.strftime('%a %H:%M').split() for t in
     pd.date_range('2021-11-29', '2021-12-6',
                   freq=time_interval, inclusive='left')])
week_idx = week_time_idx.get_level_values(0).unique()
time_idx = week_time_idx.get_level_values(1).unique()

time_splits_indices = {t: [] for t in week_time_idx}
for i, t in enumerate(time_splits):
    t = tuple(t.split()[1:])
    time_splits_indices[t].append(i)

clusters = np.arange(preds.nunique())
profiles_index = pd.MultiIndex.from_product([
    clusters, app_delay_cats, week_idx, time_idx
], names=['cluster', 'delay_cat', 'weekday', 'time'])
profiles_index

# %%
cluster_profiles = {k: [] for k in profiles_index}

for cell_id, bins in tqdm(flows_df.items(), total=len(flows_df.columns)):
    if cell_id not in cell_clusters: continue
    cluster = cell_clusters[cell_id]
    bins = bins.values
    for cat, bins in zip(app_delay_cats, np.split(bins, len(app_delay_cats))):
        for t, idx in time_splits_indices.items():
            l = cluster_profiles[(cluster, cat, *t)]
            a = bins[idx]
            l.extend(a[a > 0])

print(pd.Series([len(l) for l in cluster_profiles.values()]).describe())

# %% Oversample
for i in range(len(profiles_index) - 1):
    k1, k2 = profiles_index[i], profiles_index[i+1]
    if k1[:3] != k2[:3]: continue
    cluster_profiles[k1].extend(cluster_profiles[k2])

# %%
def aggregate(nums):
    lb, ub = np.percentile(nums, [0, 80])
    return np.mean([x for x in nums if lb <= x <= ub])
    
for key, vals in list(cluster_profiles.items()):
    if not vals:
        cluster_profiles[key] = np.nan
    else:
        cluster_profiles[key] = aggregate(vals)

# %%
profiles_df = pd.Series(cluster_profiles,
                        name='bps',index=profiles_index)
profiles_df = (profiles_df.unstack(level=1)
               .reindex(profiles_index.droplevel(1)
                        .drop_duplicates()))
profiles_df = profiles_df.loc[profiles_df.index.levels[0]]
profiles_df.fillna(0, inplace=True)
#[
    # ~profiles_df.isna().groupby('cluster').any().any(axis=1)]]
    
# %%
for i in clusters:
    for c in app_delay_cats:
        df = profiles_df.loc[i, c]
        s = df.values
        s = np.append(np.append(s[-1], s), s[0])
        for j in range(2, len(s)-2):
            a = s[j-2:j+3]
            s[j-2:j+3] = np.clip(a, 0, np.median(a) * 3)
        s = np.convolve(s, (0.2, 0.6, 0.2), mode='valid')
        df[:] = s

# %%
profiles_df.to_csv('cluster_traffic_profiles.csv')

# %%
import plotly.express as px

for i in [2]:
    print('\nCluster {}'.format(i))
    for c in app_delay_cats:
        print(c)
        df = profiles_df.loc[i][c].unstack().reindex(week_idx)
        px.imshow(df, aspect='auto', title=f'Cluster {i} {c}').show()

# %%
