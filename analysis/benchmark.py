# %%
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

plotly_template = pio.templates['plotly']
plotly_template['layout'].update(
    margin=dict(l=90, r=40, t=40, b=70),
    font_size=15, 
    titlefont_size=20, 
    legend_font_size=17,
    xaxis=dict(titlefont_size=20),
    yaxis=dict(titlefont_size=20),
    width=640, height=480
)
pio.templates['custom'] = plotly_template
pio.templates.default = 'custom'

scenario = '*'
files = glob.glob(f'sim_stats/*/{scenario}/*/trajectory.csv')
df = [pd.read_csv(f, index_col=0).iloc[1:] for f in files]
for f in df:
    f.index = f.index.str.replace(',', '')
agents = [tuple(f.split('\\')[1:4]) for f in files]
df = pd.concat(df, keys=agents, names=['policy', 'scenario', 'seed'])
# df = df.sort_index(level=0, ascending=False)[~df.index.duplicated(keep='last')]
df0 = df = df.reset_index(level=-1).groupby(['policy', 'scenario', 'time']).mean()
df.head()

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
        df = df.rename({'mappo_w_qos=8.0': 'MAPPO', 'fixed': 'Always-on', 'simple1': 'Auto-SM1', 'simple2': 'Auto-SM2', 'simple': 'Simple'})
    elif group == 'baselines-no-offload':
        df = df.rename({'mappo_no_offload=True': 'MAPPO', 'fixed': 'Always-on', 'simple1_no_offload=True': 'Auto-SM1', 'simple2': 'Auto-SM2'})
    elif group == 'wqos':
        df = df.rename_axis([
            'w_qos', *df.index.names[1:]
        ]).rename({
            'mappo_w_qos=8.0': '7', # '8.0',
            'mappo_w_qos=4.0': '4',
            'mappo_w_qos=2.0': '2',
            'mappo_w_qos=1.0': '1',
        })
    elif group == 'interf':
        df = df.rename_axis([
            'interference', *df.index.names[1:]
        ]).rename({
            'mappo_w_qos=4.0': 'considered', 
            'mappo_w_qos=4.0_no_interf=True': 'ignored'})
    elif group == 'offload':
        df = df.rename_axis([
            'offloading', *df.index.names[1:]
        ]).rename({
            'mappo_w_qos=4.0': 'yes',
            'mappo_w_qos=4.0_no_offload=True': 'no'})
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

if group in ['baselines', 'baselines-no-offload']:
    policies = 'Always-on Auto-SM1 MAPPO'.split()
elif group == 'wqos':
    policies = '1 4 7'.split()
elif group == 'interf':
    policies = 'considered ignored'.split()
elif group == 'offload':
    policies = 'yes no'.split()
elif group == 'sm':
    policies = '1 2 3'.split()
else:
    raise ValueError
    
win_sz = len(df0.loc[(df0.index[0][0], 'A')]) // 168
df = refactor(df0)

# %%
name_maps = {
    'pc_kw': 'power consumption (kW)',
    'drop_ratio': 'drop ratio',
    'reward': 'reward',
    'interference': 'interference',
    'actual_rate': 'data rate (Mb/s)',
    'arrival_rate': 'arrival rate (Mb/s)',
    'sum_tx_power': 'total transmit power (W)',
    'avg_antennas': 'average antennas',
    'sm0_cnt': 'active BSs', 'sm1_cnt': 'BSs in SM1',
    'sm2_cnt': 'BSs in SM2', 'sm3_cnt': 'BSs in SM3'
}
for i in range(7):
    name_maps[f'bs_{i}_n_ants'] = f'antennas (BS {i})'
vars_df = df.loc[policies, list(name_maps)].rename(columns=name_maps).copy()
vars_df['energy efficiency (kb/J)'] = vars_df['data rate (Mb/s)'] / (
    vars_df['power consumption (kW)'] + 1e-6)
vars_df = vars_df.rolling(win_sz).mean().iloc[win_sz::win_sz]
vars_df['interference (dB)'] = 10 * np.log10(vars_df.pop('interference') + 1e-99)
vars_df

# %%
stats = df0.groupby(level=[0,1]).mean().rename(columns=name_maps)
stats

# %%
for scenario in vars_df.index.levels[1]:
    _sdf = vars_df.xs(scenario, level=1)
    idx = _sdf.index.get_level_values(1).drop_duplicates()

    cols = [k for k in _sdf.keys() if 'BSs' in k.split()]
    _df = _sdf[cols].copy()
    _df.index = pd.MultiIndex.from_tuples(
        [(g, *s.split()) for g, s in _sdf.index],
        names=['group', 'day', 'time'])
    _sdf.drop(columns=cols, inplace=True)
    _df = _df.reset_index().groupby(['group', 'time']).mean()
    _df = _df.rename(columns={
        'active BSs': 'Active',
        'BSs in SM1': 'SM1',
        'BSs in SM2': 'SM2',
        'BSs in SM3': 'SM3'
    }).sort_index(axis=1)
    
    for g in _df.index.levels[0]:
        f = px.area(_df.loc[g], labels={'value': 'number of BSs', 'variable': 'sleep mode'},
                        color_discrete_sequence=np.array(px.colors.qualitative.Plotly)[[1, 4, 2, 0]])
        f.update_xaxes(dtick=2)
        f.write_image(f'sim_plots/{group}_{g}_{scenario}_SM_daily.pdf', scale=2)

    re_pat = re.compile(r'antennas \((BS \d)\)')
    cols = sorted([k for k in _sdf.keys() if re_pat.match(k)])
    _df = _sdf[cols].copy()
    _df.index = pd.MultiIndex.from_tuples(
        [(g, *s.split()) for g, s in _sdf.index],
        names=['group', 'day', 'time'])
    _sdf.drop(columns=cols, inplace=True)
    _df = _df.reset_index().groupby(['group', 'time']).mean()
    _df = _df.rename(columns=dict(
        (k, re_pat.match(k)[1])
        for k in _df.columns))
    
    for g in _df.index.levels[0]:
        f = px.line(_df.loc[g], labels={'value': 'number of active antennas', 'variable': ''})
        f.update_xaxes(dtick=2)
        f.write_image(f'sim_plots/{group}_{g}_{scenario}_ants_daily.pdf', scale=2)

    for key, ser in _sdf.items():
        print(scenario, key)
        _df = ser.unstack(level=0).reindex(idx)
        fig = px.line(_df, labels={'value': key}, log_y=key=='Interference')
        _df.index = pd.MultiIndex.from_tuples(
            [tuple(s.split()) for s in _df.index],
            names=['day', 'time'])
        _df1 = _df.reset_index(level=1).groupby('time').mean()
        fig.update_yaxes(exponentformat='power')  # range=[ymin, ymax]
        key = key.replace('/', 'p')
        fig.update_xaxes(dtick=2)
        fig.write_image(f'sim_plots/{group}_{scenario}_{key}.pdf', scale=2)
        fig1 = px.line(_df1, labels={'value': key})
        fig1.update_xaxes(dtick=2)
        fig1.write_image(f'sim_plots/{group}_{scenario}_{key}_daily.pdf', scale=2)
        
    # rate_df = _sdf[['Data Rate (Mb/s)', 'Arrival Rate (Mb/s)']]
    # arr_rates = rate_df['Arrival Rate (Mb/s)'].unstack().values
    # # assert all(np.allclose(r1, r2) for r1, r2 in zip(arr_rates, arr_rates[1:]))
    # req_rates = rate_df['Data Rate (Mb/s)'].copy().unstack(level=0)
    # req_rates['arrival'] = rate_df.loc[eg_policy]['Arrival Rate (Mb/s)']
    # fig = px.line(req_rates, labels=dict(w_qos='', policy='', time='', title='Data Rate (Mb/s)'))
    # fig.write_image(f'sim_plots/{scenario}-data-rate.pdf', scale=2)
    # fig.show()

# %%
def parse_np_series(s):
    l = [np.fromstring(a[1:-1], sep=' ') for a in s]
    return pd.DataFrame(l, index=s.index)

files = glob.glob(f'sim_stats/*/*/*/bs_stats.csv')
dfs = [pd.read_csv(f, index_col=0) for f in files]
agents = [f.split('\\')[1:-1] for f in files]
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
bs_stats = pd.concat(dfs, keys=list(map(tuple, agents)), names=['policy', 'scenario', 'seed'])[kpis]
bs_stats['active_ratio'], bs_stats['sm1_ratio'], bs_stats['sm2_ratio'], bs_stats['sm3_ratio'] = parse_np_series(bs_stats.pop(
    'avg_sleep_ratios')).T.itertuples(index=False)
bs_stats = bs_stats.groupby(['policy', 'scenario', 'id']).mean()
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
files = glob.glob(f'sim_stats/*/*/*/ue_stats.csv')
dfs = [pd.read_csv(f) for f in files]
agents = [f.split('\\')[1:-1] for f in files]
df = pd.concat(dfs, keys=list(map(tuple, agents)),
               names=['policy', 'scenario', 'seed'])#.unstack()
df['actual_rate'] = df.done / df.delay / 1e6
df['req_rate'] = df.demand / df.delay_budget / 1e6
df['rate_ratio'] = df['actual_rate'] / df['req_rate']
df['ue_drop_ratio'] = df.dropped > 0
df['drop_ratio'] = df.dropped / df.demand * 100
ue_stats = df.groupby(['policy', 'scenario']).agg(['sum', 'mean'])
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
files = glob.glob(f'sim_stats/*/*/*/net_stats.csv')
dfs = [pd.read_csv(f, header=None, index_col=0).T for f in files]
agents = [f.split('\\')[1:-1] for f in files]
net_stats = pd.concat(dfs, keys=list(map(tuple, agents)),
                      names=['policy', 'scenario', 'seed']).droplevel(-1)
net_stats = net_stats[['avg_pc', 'energy']].astype('float').groupby(['policy', 'scenario']).mean()
net_stats['energy_efficiency'] = tot_traffic / net_stats.pop('energy') / 1e6
net_stats

# %%
drop_policies = []
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
                 'simple1_no_offload=True': 'Auto SM1 (no offload)',
                 'mappo_w_qos=1.0': 'MAPPO (w_qos=1.0)',
                 'mappo_w_qos=2.0': 'MAPPO (w_qos=2.0)',
                 'mappo_w_qos=4.0': 'MAPPO (w_qos=4.0)',
                 'mappo_w_qos=8.0': 'MAPPO (w_qos=8.0)',
                 'mappo_w_qos=4.0_max_sleep=1': 'MAPPO (deepest_sleep=SM1)',
                 'mappo_w_qos=4.0_max_sleep=2': 'MAPPO (deepest_sleep=SM2)',
                 'mappo_w_qos=4.0_no_interf=True': 'MAPPO (ignore interference)',
                 'mappo_w_qos=4.0_no_offload=True': 'MAPPO (no offloading)'}
renamed_cols = {'avg_pc': 'total PC (W)',
                'avg_interference': 'avg interference',
                'actual_rate': 'UE data rate (Mb/s)',
                'req_rate': 'UE required data rate (Mb/s)',
                'rate_ratio': 'actual rate / required rate',
                'drop_ratio': 'drop ratio (perc)',
                'energy_efficiency': 'energy efficiency (Mb/J)'}
stats_df = pd.concat([net_stats, ue_stats], axis=1)[selected_cols].astype('float')
stats_df.drop(columns=['rate_ratio'], inplace=True)
stats_df['energy saving (perc)'] = (
    stats_df.loc['fixed', 'avg_pc'] - stats_df['avg_pc']
) / stats_df.loc['fixed', 'avg_pc'] * 100
stats_df1 = (stats_df
             .drop(drop_policies, axis=0)
             .rename_axis(['policy', 'scenario'])
             .rename(index=renamed_index, columns=renamed_cols)
             .loc[list(renamed_index.values())])
# stats_df1.to_csv('sim-stats.csv')
stats_df1

# %%
def copy_text(text):
    s = pd.Series(text)
    s.to_clipboard(index=False, header=False)
    return text

print(copy_text(stats_df1.to_latex()))

# %%
temp_df = refactor(stats_df).loc[policies].rename(columns=renamed_cols)
temp_df.loc[('4', ['rural', 'urban', 'work']), :] /= 2.
for kpi in temp_df.keys():
    fig = px.bar(temp_df[kpi].unstack(), barmode='group', 
                 labels={'value': kpi})
    fig.write_image('bm_plots/'+f'{group}_{kpi}.pdf'.replace('/', 'p'), scale=2)
    # fig.show()
    # temp_df[kpi].unstack().plot(kind='bar', title=kpi)
    # plt.savefig(f'{group}_{kpi}.pdf'.replace('/', 'p'), dpi=256)
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
