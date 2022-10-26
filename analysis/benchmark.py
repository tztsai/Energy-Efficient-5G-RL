# %%
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

scenario = 'A'
files = glob.glob(f'sim_stats/*/{scenario}/trajectory.csv')
df = [pd.read_csv(f, index_col=0).iloc[1:] for f in files]
for f in df:
    f.index = f.index.str.replace(',', '')
agents = [tuple(f.split('\\')[1:3]) for f in files]
df = pd.concat(df, keys=agents, names=['policy', 'scenario'])
# df = df.sort_index(level=0, ascending=False)[~df.index.duplicated(keep='last')]
df = df.rename(index={'fixed': 'always_on'}, level=0)
df

# %%
group = 'wqos'
if group == 'baselines':
    policies = 'always_on simple1 mappo_w_qos=4.0'.split()
elif group == 'wqos':
    policies = 'mappo_w_qos=1.0 mappo_w_qos=2.0 mappo_w_qos=4.0 mappo_w_qos=8.0'.split()
elif group == 'interf':
    policies = 'mappo_w_qos=4.0 mappo_w_qos=4.0_nointerf'
elif group == 'sm':
    policies = 'mappo_w_qos=4.0 mappo_w_qos=4.0_sm1 mappo_w_qos=4.0_sm12'
# policies = 'simple simple1 simple2'.split()
df = df.loc[policies]
# df = df.rename({'mappo_w_qos=4.0': 'MAPPO', 'always_on': 'Always On', 'simple1': 'Auto SM1'})
df = df.rename({'mappo_w_qos=4.0': '4.0',
                'mappo_w_qos=8.0': '8.0',
                'mappo_w_qos=2.0': '2.0',
                'mappo_w_qos=1.0': '1.0',})
if group == 'wqos':
    df.index.set_names({'policy': 'w_qos'}, inplace=True)
eg_policy = df.index.levels[0][0]

# %%
key_pat = re.compile('(pc_kw|interf.*|.*power|.*antenna.*|sm._cnt|.*drop|actual_rate|arrival_rate)')
vars_df = df[list(filter(key_pat.match, df.columns))].copy().rename(columns={
    'pc_kw': 'Power Consumption (kW)', 'drop_ratio': 'Drop Ratio', 'reward': 'Reward', 
    'interference': 'Interference', 'actual_rate': 'Data Rate (Mb/s)', 
    'arrival_rate': 'Arrival Rate (Mb/s)', 'sum_tx_power': 'Total Transmit Power',
    'sm0_cnt': 'Active BSs', 'sm1_cnt': 'BSs in SM1', 'sm2_cnt': 'BSs in SM2', 'sm3_cnt': 'BSs in SM3'})
vars_df['Energy Efficiency (kb/J)'] = vars_df['Data Rate (Mb/s)'] / (
    vars_df['Power Consumption (kW)'] + 1e-6)
win_sz = len(df.loc[(eg_policy, scenario)]) // 168
vars_df = vars_df.rolling(win_sz).mean().iloc[win_sz-1::win_sz]
vars_df['Interference (dB)'] = 10 * np.log10(vars_df.pop('Interference'))
vars_df

# %%
for scenario in vars_df.index.levels[1]:
    _sdf = vars_df.xs(scenario, level=1)
    idx = _sdf.index.get_level_values(1).drop_duplicates()
    for key, ser in _sdf.items():
        _df = ser.unstack(level=0).reindex(idx)
        fig = px.line(_df, title=key, labels={'value': '', 'time': ''}, log_y=key=='Interference')
        fig.update_yaxes(exponentformat='power')  # range=[ymin, ymax]
        # fig.update_layout()
        fig.write_image(f'sim_plots/{group}_{scenario}_{key.replace("/", "p")}.png')
        fig.show()
        # fig.write_image(f'bm_plots/{scenario}-{key}.png', scale=2)
        # _df.plot(title=key)
        # plt.legend(title='')

    rate_df = _sdf[['Data Rate (Mb/s)', 'Arrival Rate (Mb/s)']]
    arr_rates = rate_df['Arrival Rate (Mb/s)'].unstack().values
    # assert all(np.allclose(r1, r2) for r1, r2 in zip(arr_rates, arr_rates[1:]))
    req_rates = rate_df['Data Rate (Mb/s)'].copy().unstack(level=0)
    req_rates['arrival'] = rate_df.loc[eg_policy]['Arrival Rate (Mb/s)']
    fig = px.line(req_rates, labels=dict(value='Rate (Mb/s)', policy='', time=''))
    # fig.write_image(f'bm_plots/{scenario}-data-rate.png', scale=2)
    fig.show()

# %%
stats = vars_df.drop_duplicates().groupby(level=[0,1]).mean()
stats['Power Consumption'] = stats.pop('Power Consumption (kW)') * 1e3
# del stats['avg_antennas']
stats['Actual Sum Rate (Mb/s)'] = stats.pop('Data Rate (Mb/s)')
stats['Arrival Sum Rate (Mb/s)'] = stats.pop('Arrival Rate (Mb/s)')
stats

# %%
files = glob.glob(f'sim_stats/*/{scenario}/bs_stats.csv')
frames = [pd.read_csv(f, index_col=0).mean() for f in files]
agents = [f.split('\\')[1:3] for f in files]
bs_stats = pd.concat(frames, keys=list(map(tuple, agents))).unstack()
bs_stats

# %%
files = glob.glob(f'sim_stats/*/{scenario}/ue_stats.csv')
frames = [pd.read_csv(f) for f in files]
agents = [f.split('\\')[1:3] for f in files]
df = pd.concat(frames, keys=list(map(tuple, agents)), names=['policy', 'scenario'])#.unstack()
df

# %%
df['actual_rate'] = df.done / df.delay
df['req_rate'] = df.demand / df.delay_budget
df['ue_drop_ratio'] = df.dropped > 0
ue_stats = df.groupby(level=[0,1]).agg(['sum', 'mean'])
ue_stats['drop_ratio'] = ue_stats.dropped['sum'] / ue_stats.demand['sum']
cols = ['avg_tx_power', 'avg_interference', 'avg_sinr', 'actual_rate', 'ue_drop_ratio', 'drop_ratio']
ue_stats = ue_stats.drop('sum', axis=1, level=1).droplevel(1, axis=1)[cols]
# ue_stats = ue_stats['']
ue_stats

# %%
ue_stats_per_service = df.groupby(['policy', 'scenario', 'delay_budget']).mean()
ue_stats_per_service

# %%
