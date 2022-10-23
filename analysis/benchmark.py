# %%
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

scenario = 'B'
files = glob.glob(f'sim_stats/*/{scenario}/trajectory.csv')
frames = [pd.read_csv(f, index_col=0).iloc[1:] for f in files]
agents = [tuple(f.split('\\')[1:3]) for f in files]
df = pd.concat(frames, keys=agents, names=['policy', 'scenario'])
# df = df.sort_index(level=0, ascending=False)[~df.index.duplicated(keep='last')]
df = df.rename(index={'fixed': 'always_on'}, level=0)
# df['interference_db'] = 10 * np.log10(df['interference'] + 1e-1000)
df

# %%
print(df.index.levels[0])
policies = 'always_on mappo simple simple1 simple2'.split()
policies = ['always_on', 'simple1', 'mappo_w_qos=5.0']
df = df.loc[policies]

# %%
key_pat = re.compile('(pc_kw|interference|.*antenna|.*drop|actual_rate|arrival_rate)')
vars_df = df[list(filter(key_pat.match, df.columns))].copy().rename(columns={'pc_kw': 'Power Consumption (kW)', 'drop_ratio': 'Drop Ratio', 'reward': 'Reward', 'interference': 'Interference', 'actual_rate': 'Actual Rate', 'arrival_rate': 'Arrival Rate'})
vars_df['Energy Efficiency'] = vars_df['Actual Rate'] / (
    vars_df['Power Consumption (kW)'] + 1e-6)
win_sz = len(df.loc[(policies[0], scenario)]) // 168
vars_df = vars_df.rolling(win_sz).mean().iloc[win_sz-1::win_sz]
vars_df

# %%
for scenario in vars_df.index.levels[1]:
    _sdf = vars_df.xs(scenario, level=1)
    idx = _sdf.index.get_level_values(1).drop_duplicates()
    for key, ser in _sdf.items():
        _df = ser.unstack(level=0).reindex(idx)
        fig = px.line(_df, title=key, labels={'value': '', 'time': ''}, log_y=key=='Interference')
        fig.update_yaxes(exponentformat='power')  # range=[ymin, ymax]
        fig.update_layout()
        fig.show()
        # fig.write_image(f'bm_plots/{scenario}-{key}.png', scale=2)
        # _df.plot(title=key)
        # plt.legend(title='')

    rate_df = _sdf[['Actual Rate', 'Arrival Rate']]
    arr_rates = rate_df['Arrival Rate'].unstack().values
    # assert all(np.allclose(r1, r2) for r1, r2 in zip(arr_rates, arr_rates[1:]))
    req_rates = rate_df['Actual Rate'].copy().unstack(level=0)
    req_rates['arrival'] = rate_df.loc['always_on']['Arrival Rate']
    fig = px.line(req_rates, labels=dict(value='Rate (Mbps)', policy='', time=''))
    # fig.write_image(f'bm_plots/{scenario}-data-rate.png', scale=2)
    fig.show()

# %%
stats = vars_df.drop_duplicates().groupby(level=[0,1]).mean()
stats['Power Consumption'] = stats.pop('Power Consumption (kW)') * 1e3
del stats['avg_antennas']
stats['Actual Sum Rate (Mb/s)'] = stats.pop('Actual Rate')
stats['Arrival Sum Rate (Mb/s)'] = stats.pop('Arrival Rate')
stats['Interference (dB)'] = 10 * np.log10(stats.pop('Interference'))
stats['Drop Ratio (%)'] = 100 * stats.pop('Drop Ratio')
stats

# %%
files = glob.glob(f'sim_stats/*/*/bs_stats.csv')
frames = [pd.read_csv(f, index_col=0).mean() for f in files]
agents = [f.split('\\')[1:3] for f in files]
bs_stats = pd.concat(frames, keys=list(map(tuple, agents))).unstack()
bs_stats

# %%
files = glob.glob(f'sim_stats/*/*/ue_stats.csv')
frames = [pd.read_csv(f) for f in files]
agents = [f.split('\\')[1:3] for f in files]
df = pd.concat(frames, keys=list(map(tuple, agents)), names=['policy', 'scenario'])#.unstack()
df

# %%
df['actual_rate'] = df.done / df.delay
df['req_rate'] = df.demand / df.delay_budget
ue_stats = df.groupby(level=[0,1]).agg(['sum', 'mean'])
ue_stats['drop_ratio'] = ue_stats.dropped['sum'] / ue_stats.demand['sum']
ue_stats.drop('sum', axis=1, level=1, inplace=True)
ue_stats.droplevel(1, axis=1, inplace=True)
ue_stats = ue_stats['']
ue_stats

# %%
ue_stats_per_service = df.groupby(['policy', 'scenario', 'delay_budget']).mean()
ue_stats_per_service

# %%
