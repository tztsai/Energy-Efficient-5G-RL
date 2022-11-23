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
df = df0 = pd.concat(df, keys=agents, names=['policy', 'scenario'])
# df = df.sort_index(level=0, ascending=False)[~df.index.duplicated(keep='last')]

# %%
group = 'baselines'
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

def refactor(df):
    if group == 'baselines':
        df = df.rename({'mappo_w_qos=4.0': 'MAPPO', 'fixed': 'Always-on', 'simple1': 'Auto-SM1', 'simple2': 'Auto-SM2', 'simple': 'Simple'})
    elif group == 'wqos':
        df = df.rename_axis([
            'w_qos', *df.index.names[1:]
        ]).rename({
            'mappo_w_qos=8.0': '8.0',
            'mappo_w_qos=4.0': '4.0',
            'mappo_w_qos=2.0': '2.0',
            'mappo_w_qos=1.0': '1.0',
        })
    elif group == 'interf':
        df = df.rename_axis([
            'interference', *df.index.names[1:]
        ]).rename({
            'mappo_w_qos=4.0': 'considered', 
            'mappo_w_qos=4.0_ignore_interf': 'ignored'})
    elif group == 'offload':
        df = df.rename_axis([
            'offloading', *df.index.names[1:]
        ]).rename({
            'mappo_w_qos=4.0': 'yes',
            'mappo_w_qos=4.0_no_offload': 'no'})
    elif group == 'sm':
        df = df.rename_axis([
            'max sleep depth', *df.index.names[1:]
        ]).rename(
            {'simple1': '1',
             'simple2': '2',
             'simple': '3'})
            # {'mappo_w_qos=4.0': '3',
            #  'mappo_w_qos=4.0_max_sleep=1': '1',
            #  'mappo_w_qos=4.0_max_sleep=2': '2'})
    df = df.rename(index={'A': 'rural', 'B': 'urban', 'C': 'work'}, level=1)
    return df

if group == 'baselines':
    policies = 'Always-on Auto-SM1 MAPPO'.split()
elif group == 'wqos':
    policies = '1.0 4.0 8.0'.split()
elif group == 'interf':
    policies = 'considered ignored'.split()
elif group == 'offload':
    policies = 'yes no'.split()
elif group == 'sm':
    policies = '1 2 3'.split()
    
win_sz = len(df.loc[(df.index[0][0], scenario)]) // 168
df = refactor(df)

# %%
name_maps = {
    'pc_kw': 'Power Consumption (kW)',
    'drop_ratio': 'Drop Ratio',
    'reward': 'Reward',
    'interference': 'Interference',
    'avg_antennas': 'Avg Antennas',
    'actual_rate': 'Data Rate (Mb/s)',
    'arrival_rate': 'Arrival Rate (Mb/s)',
    'sum_tx_power': 'Total Transmit Power',
    'avg_antennas': 'Avg Antennas',
    'sm0_cnt': 'Active BSs', 'sm1_cnt': 'BSs in SM1',
    'sm2_cnt': 'BSs in SM2', 'sm3_cnt': 'BSs in SM3'
}
vars_df = df.loc[policies].rename(columns=name_maps).copy()
vars_df['Energy Efficiency (kb/J)'] = vars_df['Data Rate (Mb/s)'] / (
    vars_df['Power Consumption (kW)'] + 1e-6)
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
        _df.index = pd.MultiIndex.from_tuples(
            [tuple(s.split()) for s in _df.index],
            names=['day', 'time'])
        _df1 = _df.reset_index(level=1).groupby('time').mean()
        fig.update_yaxes(exponentformat='power')  # range=[ymin, ymax]
        fig.write_image(f'sim_plots/{group}_{scenario}_{key.replace("/", "p")}.png', scale=2)
        fig1 = px.line(_df1, title=key, labels={'value': '', 'time': ''})
        fig1.write_image(f'sim_plots/{group}_{scenario}_{key.replace("/", "p")}_daily.png', scale=2)
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
df['rate_ratio'] = df['actual_rate'] / df['req_rate']
df['ue_drop_ratio'] = df.dropped > 0
df['drop_ratio'] = df.dropped / df.demand * 100
ue_stats = df.groupby(level=[0,1]).agg(['sum', 'mean'])
ue_stats['drop_ratio'] = ue_stats.dropped['sum'] / ue_stats.demand['sum'] * 100
tot_traffic = ue_stats.demand['sum']
cols = ['avg_tx_power', 'avg_interference', 'avg_sinr', 'actual_rate', 'req_rate', 'ue_drop_ratio', 'rate_ratio', 'drop_ratio']
ue_stats = ue_stats.drop('sum', axis=1, level=1).droplevel(1, axis=1)[cols]
# ue_stats['data_rate_ratio'] = ue_stats.actual_rate / ue_stats.req_rate
ue_stats

# %%
ue_stats_per_service = df.groupby(['policy', 'scenario', 'delay_budget']).mean()[['drop_ratio', 'actual_rate']]
ue_stats_per_service

# %%
temp_df = refactor(ue_stats_per_service).loc[policies]
for kpi in temp_df.keys():
    temp_df[kpi].xs('urban', level=1).unstack().plot(kind='bar', title=kpi)
    plt.show()

# %%
files = glob.glob(f'sim_stats/*/*/net_stats.csv')
frames = [pd.read_csv(f, header=None, index_col=0).T for f in files]
agents = [f.split('\\')[1:3] for f in files]
net_stats = pd.concat(frames, keys=list(map(tuple, agents)),
                      names=['policy', 'scenario']).droplevel(2)
net_stats = net_stats[['avg_pc', 'energy']].astype('float')
net_stats['energy_efficiency'] = tot_traffic / net_stats.pop('energy') / 1e6
net_stats

# %%
drop_policies = ['simple', 'simple2', 'random', 
                 'mappo_w_qos=4.0_no_interf=True']
selected_cols = ['avg_pc',
                #  'Data Rate (Mb/s)',
                #  'Required Sum Rate (Mb/s)',
                #  'Total Transmit Power',
                #  'avg_interference',
                 'actual_rate',
                 'rate_ratio',
                 'energy_efficiency',
                 'drop_ratio']
renamed_index = {'fixed': 'Always On',
                 'simple1': 'Auto SM1',
                 'mappo_w_qos=1.0': 'MAPPO (w_qos=1.0)',
                 'mappo_w_qos=2.0': 'MAPPO (w_qos=2.0)',
                 'mappo_w_qos=4.0': 'MAPPO (w_qos=4.0 (default))',
                 'mappo_w_qos=8.0': 'MAPPO (w_qos=8.0)',
                 'mappo_w_qos=4.0_max_sleep=1': 'MAPPO (deepest_sleep=SM1)',
                 'mappo_w_qos=4.0_max_sleep=2': 'MAPPO (deepest_sleep=SM2)',
                 'mappo_w_qos=4.0_ignore_interf': 'MAPPO (ignore interference)',
                 'mappo_w_qos=4.0_no_offload': 'MAPPO (no offloading)'}
renamed_cols = {'avg_pc': 'avg total power consumption (W)',
                'avg_interference': 'avg interference',
                'actual_rate': 'avg UE data rate (Mb/s)',
                'req_rate': 'avg UE required data rate (Mb/s)',
                'rate_ratio': 'actual rate / required rate',
                'drop_ratio': 'avg drop ratio (%)',
                'energy_efficiency': 'avg energy efficiency (Mb/J)'}
stats_df = pd.concat([net_stats, ue_stats], axis=1)[selected_cols].astype('float')
stats_df['energy saving (%)'] = (
    stats_df.loc['fixed', 'avg_pc'] - stats_df['avg_pc']
) / stats_df.loc['fixed', 'avg_pc'] * 100
stats_df1 = (stats_df
             .drop(drop_policies, axis=0)
             .rename_axis(['Policy', 'Scenario'])
             .rename(index=renamed_index, columns=renamed_cols)
             .loc[list(renamed_index.values())])
stats_df1.to_csv('sim-stats.csv')
stats_df1

# %%
def copy_text(text):
    s = pd.Series(text)
    s.to_clipboard(index=False, header=False)
    return text

print(copy_text(stats_df.to_latex()))

# %%
temp_df = refactor(stats_df).loc[policies].rename(columns=renamed_cols)
for kpi in temp_df.keys():
    fig = px.bar(temp_df[kpi].unstack(), barmode='group', 
                 labels={'value': kpi})
    fig.write_image('bm_plots/'+f'{group}_{kpi}.png'.replace('/', 'p'), scale=2)
    # fig.show()
    # temp_df[kpi].unstack().plot(kind='bar', title=kpi)
    # plt.savefig(f'{group}_{kpi}.png'.replace('/', 'p'), dpi=256)
    # plt.show()

# for kpi in stats_df.columns:
#     stats_df.xs('B', level='scenario')[kpi].plot(kind='bar', title=kpi)
#     plt.show()

# %%
# ratio_df = stats_df.xs('B', level='scenario').loc[[
#     'MAPPO (w_qos=4.0)', 'MAPPO (no offloading)']].rename({'MAPPO (w_qos=4.0)': 'MAPPO (with offloading)'}).T
# ratio_df = ratio_df['MAPPO (no offloading)'] / ratio_df['MAPPO (with offloading)'] * 100
# ratio_df.plot(kind='bar')

# %%
