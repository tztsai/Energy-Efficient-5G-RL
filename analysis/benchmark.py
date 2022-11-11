# %%
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

scenario = 'B'
files = glob.glob(f'sim_stats/*/{scenario}/trajectory.csv')
df = [pd.read_csv(f, index_col=0).iloc[1:] for f in files]
for f in df:
    f.index = f.index.str.replace(',', '')
agents = [tuple(f.split('\\')[1:3]) for f in files]
df = pd.concat(df, keys=agents, names=['policy', 'scenario'])
df = df0 = df#.rename(index={'fixed': 'always_on'}, level=0)
# df = df.sort_index(level=0, ascending=False)[~df.index.duplicated(keep='last')]

# %%
group = 'offload'
if group == 'baselines':
    policies = 'Always-on Simple MAPPO'.split()
elif group == 'wqos':
    policies = '1.0 2.0 4.0 8.0'.split()
elif group == 'interf':
    policies = 'yes no'.split()
elif group == 'offload':
    policies = 'yes no'.split()
elif group == 'sm':
    policies = '1 2 3'.split()
    # policies = 'simple simple1 simple2'.split()

columns = ['actual_rate',
           'arrival_rate',
           'interference',
           'sum_tx_power',
           'avg_antennas',
           'sm0_cnt',
           'sm1_cnt',
           'sm2_cnt',
           'sm3_cnt',
           'reward',
           'pc_kw',
           'qos_reward',
           'drop_ratio']

if group == 'baselines':
    df = df.rename({'mappo_w_qos=4.0': 'MAPPO', 'fixed': 'Always-on', 'simple1': 'Simple'})
elif group == 'wqos':
    df = df.rename_axis([
        'w_qos', 'scenario', 'time'
    ]).rename({
        'mappo_w_qos=8.0': '8.0',
        'mappo_w_qos=4.0': '4.0',
        'mappo_w_qos=2.0': '2.0',
        'mappo_w_qos=1.0': '1.0',
    })
elif group == 'interf':
    df = df.rename_axis([
        'interference', 'scenario', 'time'
    ]).rename({
        'mappo_w_qos=4.0': 'yes', 
        'mappo_w_qos=4.0_no_interf=True': 'no'})
elif group == 'offload':
    df = df.rename_axis([
        'offloading', 'scenario', 'time'
    ]).rename({
        'mappo_w_qos=4.0': 'yes',
        'mappo_w_qos=4.0_no_offload=True': 'no'})
elif group == 'sm':
    df = df.rename_axis([
        'max sleep depth', 'scenario', 'time'
    ]).rename({'mappo_w_qos=4.0': '3',
                    'mappo_w_qos=4.0_max_sleep=1': '1',
                    'mappo_w_qos=4.0_max_sleep=2': '2'})
eg_policy = df.index[0][0]

# %%
name_maps = {
    'pc_kw': 'Power Consumption (kW)',
    'drop_ratio': 'Drop Ratio',
    'reward': 'Reward',
    'interference': 'Interference',
    'actual_rate': 'Data Rate (Mb/s)',
    'arrival_rate': 'Arrival Rate (Mb/s)',
    'sum_tx_power': 'Total Transmit Power',
    'sm0_cnt': 'Active BSs', 'sm1_cnt': 'BSs in SM1',
    'sm2_cnt': 'BSs in SM2', 'sm3_cnt': 'BSs in SM3'
}
vars_df = df.loc[policies].rename(columns=name_maps).copy()
vars_df['Energy Efficiency (kb/J)'] = vars_df['Data Rate (Mb/s)'] / (
    vars_df['Power Consumption (kW)'] + 1e-6)
win_sz = len(df.loc[(eg_policy, scenario)]) // 168
vars_df = vars_df.rolling(win_sz).mean().iloc[win_sz-1::win_sz]
vars_df['Interference (dB)'] = 10 * np.log10(vars_df.pop('Interference'))
vars_df

# %%
stats = df0.groupby(level=[0,1]).mean().rename(columns=name_maps)
stats

# %%
for scenario in vars_df.index.levels[1]:
    _sdf = vars_df.xs(scenario, level=1)
    idx = _sdf.index.get_level_values(1).drop_duplicates()
    for key, ser in _sdf.items():
        _df = ser.unstack(level=0).reindex(idx).rename({
            'mappo_w_qos=4.0': '4.0',
            'mappo_w_qos=8.0': '8.0',
            'mappo_w_qos=2.0': '2.0',
            'mappo_w_qos=1.0': '1.0',})
        fig = px.line(_df, title=key, labels={'value': '', 'time': ''}, log_y=key=='Interference')
        fig.update_yaxes(exponentformat='power')  # range=[ymin, ymax]
        # fig.update_layout()
        fig.write_image(f'sim_plots/{group}_{scenario}_{key.replace("/", "p")}.png', scale=2)
        # fig.show()
        # _df.plot(title=key)
        # plt.legend(title='')

    # rate_df = _sdf[['Data Rate (Mb/s)', 'Arrival Rate (Mb/s)']]
    # arr_rates = rate_df['Arrival Rate (Mb/s)'].unstack().values
    # # assert all(np.allclose(r1, r2) for r1, r2 in zip(arr_rates, arr_rates[1:]))
    # req_rates = rate_df['Data Rate (Mb/s)'].copy().unstack(level=0)
    # req_rates['arrival'] = rate_df.loc[eg_policy]['Arrival Rate (Mb/s)']
    # fig = px.line(req_rates, labels=dict(w_qos='', policy='', time='', title='Data Rate (Mb/s)'))
    # fig.write_image(f'sim_plots/{scenario}-data-rate.png', scale=2)
    # fig.show()

# %%
def parse_np_series(s):
    l = [np.fromstring(a[1:-1], sep=' ') for a in s]
    return pd.DataFrame(l, index=s.index)

files = glob.glob(f'sim_stats/*/*/bs_stats.csv')
frames = [pd.read_csv(f, index_col=0) for f in files]
agents = [f.split('\\')[1:3] for f in files]
kpis = [
    'avg_pc', 
    'avg_tx_power',
    # 'avg_num_ants',
    'avg_sum_rate',
    # 'avg_req_sum_rate',
    # 'avg_serving_ues', 'avg_queued_ues', 'avg_covered_ues',
    'avg_sleep_ratios', 
    # 'num_rejects', 
    'avg_reject_rate',
    # 'avg_cell_drop_ratio', 'avg_cell_data_rate', 'avg_sleep_switch_fps',
    # 'ant_switches', 'avg_ant_switch_fps', 'disconnects'
]
bs_stats = pd.concat(frames, keys=list(map(tuple, agents)))[kpis]
bs_stats['active_ratio'], bs_stats['sm1_ratio'], bs_stats['sm2_ratio'], bs_stats['sm3_ratio'] = parse_np_series(bs_stats.pop(
    'avg_sleep_ratios')).T.itertuples(index=False)
bs_stats = bs_stats.astype('float').rename_axis(['policy', 'scenario', 'bs'])
bs_stats

# %%
temp_df = bs_stats.loc[('mappo_w_qos=4.0', 'B')]
ks = ['active_ratio', 'sm1_ratio', 'sm2_ratio', 'sm3_ratio']
temp_df[ks].T.plot(kind='bar')
plt.show()
for kpi in temp_df.drop(ks, axis=1).keys():
    temp_df[kpi].plot(kind='bar', title=kpi)
    plt.show()

# %%
files = glob.glob(f'sim_stats/*/*/ue_stats.csv')
frames = [pd.read_csv(f) for f in files]
agents = [f.split('\\')[1:3] for f in files]
df = pd.concat(frames, keys=list(map(tuple, agents)),
               names=['policy', 'scenario'])#.unstack()
df['actual_rate'] = df.done / df.delay / 1e6
df['req_rate'] = df.demand / df.delay_budget / 1e6
df['ue_drop_ratio'] = df.dropped > 0
df['drop_ratio'] = df.dropped / df.demand * 100
ue_stats = df.groupby(level=[0,1]).agg(['sum', 'mean'])
ue_stats['drop_ratio'] = ue_stats.dropped['sum'] / ue_stats.demand['sum']
cols = ['avg_tx_power', 'avg_interference', 'avg_sinr', 'actual_rate', 'req_rate', 'ue_drop_ratio', 'drop_ratio']
ue_stats = ue_stats.drop('sum', axis=1, level=1).droplevel(1, axis=1)[cols]
ue_stats['data_rate_ratio'] = ue_stats.actual_rate / ue_stats.req_rate
ue_stats

# %%
files = glob.glob(f'sim_stats/*/*/net_stats.csv')
frames = [pd.read_csv(f, header=None, index_col=0).T for f in files]
agents = [f.split('\\')[1:3] for f in files]
net_stats = pd.concat(frames, keys=list(map(tuple, agents)),
                      names=['policy', 'scenario']).droplevel(2)
net_stats = net_stats[['avg_pc']]
net_stats

# %%
drop_policies = ['simple', 'simple2', 'random']
selected_cols = ['avg_pc',
                #  'Data Rate (Mb/s)',
                #  'Required Sum Rate (Mb/s)',
                #  'Total Transmit Power',
                 'avg_interference',
                 'data_rate_ratio',
                 'actual_rate',
                 'drop_ratio']
stats_df = pd.concat([net_stats, ue_stats], axis=1)[selected_cols].rename(
    index={'fixed': 'Always-on',
           'mappo_w_qos=1.0': 'MAPPO (w_qos=1.0)',
           'mappo_w_qos=2.0': 'MAPPO (w_qos=2.0)',
           'mappo_w_qos=4.0': 'MAPPO (w_qos=4.0)',
           'mappo_w_qos=8.0': 'MAPPO (w_qos=8.0)',
           'mappo_w_qos=4.0_max_sleep=1': 'MAPPO (deepest_sleep=SM1)',
           'mappo_w_qos=4.0_max_sleep=2': 'MAPPO (deepest_sleep=SM2)',
           'mappo_w_qos=4.0_no_interf=True': 'MAPPO (no interference)',
           'mappo_w_qos=4.0_no_offload=True': 'MAPPO (no offloading)',
           'simple1': 'Auto SM1'
           },
    columns={'avg_pc': 'Power Consumption (W)',
             'avg_interference': 'Interference',
             'actual_rate': 'Avg UE Data Rate (Mb/s)',
             'req_rate': 'Avg UE Required Data Rate (Mb/s)',
             'data_rate_ratio': 'Ratio of actual to required data rate',
             'drop_ratio': 'Drop Ratio'}
).drop(drop_policies, axis=0).astype('float')
stats_df

# %%
for kpi in stats_df.columns:
    stats_df.xs('B', level='scenario')[kpi].plot(kind='bar', title=kpi)
    plt.show()

# %%
ratio_df = stats_df.xs('B', level='scenario').loc[[
    'MAPPO (w_qos=4.0)', 'MAPPO (no offloading)']].rename({'MAPPO (w_qos=4.0)': 'MAPPO (with offloading)'}).T
ratio_df = ratio_df['MAPPO (no offloading)'] / ratio_df['MAPPO (with offloading)'] * 100
ratio_df.plot(kind='bar')

# %%
cols = ['drop_ratio', 'actual_rate']
ue_stats_per_service = df.groupby(['policy', 'scenario', 'delay_budget']).mean()[cols]
ue_stats_per_service
# ue_stats_per_service.plot.scatter(x='delay_budget', y='actual_rate', c='demand', colormap='viridis')

# %%
