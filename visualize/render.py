import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from network import config


conn_symbols = ['triangle-down', 'hexagram']
sleep_suffixes = ['', '-open-dot', '-open']
conn_act_symbols = np.array(['circle', 'circle', 'circle-open'])
n_agents = 7
color_sequence = np.array(px.colors.qualitative.Plotly)
oppo_color_sequence = np.array(['#%02X%02X%02X' % tuple(
    255 - int(s[i:i+2], 16) for i in range(1, 7, 2)) for s in color_sequence])
color_sequence = np.hstack([color_sequence[:n_agents], oppo_color_sequence[:n_agents]])


def render(env: 'MultiCellNetEnv', mode='none'):
    net = env.net
    info = env.info_dict()
    
    if env._figure is None:
        env._figure = make_figure(net)
    fig = env._figure

    # plot base stations
    x, y, m, s, c, r, a, i = np.array(
        [[bs.pos[0], bs.pos[1], bs.num_ant, bs.sleep, bs.accept_conn,
          bs.sum_rate, bs._conn_act, i] for i, bs in net.bss.items()]).T
    hover_text_template = """
    id: {id}<br>
    num antennas: {num_ant}<br>
    sleep mode: {sleep}<br>
    connect mode: {conn}<br>
    power consumption: {pc:.2f}<br>
    num ues in service: {num_s}<br>
    num ues in queue: {num_q}<br>
    num ues in coverage: {num_c}<br>
    throughput demand: {thrp_req:.2f}<br>
    throughput ratio: {thrp_ratio:.2f}
    """
    symbols = ['x-open' if s == 3 else conn_symbols[int(c)] + 
               sleep_suffixes[int(s)] for s, c in zip(s, c)]
    hover_texts = [hover_text_template.format(id=i, **bs.info_dict()) for i, bs in net.bss.items()]
    bs_plt = dict(
        type='scatter',
        x=x, y=y, mode='markers', ids=i,
        marker=dict(
            # size=m/6+8, 
            size=18,
            line_width=1, 
            line_color='grey',
            symbol=symbols,
            color=color_sequence[:len(x)]
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        showlegend=False,
    )
    
    # plot cell coverages
    cl_plt = dict(
        type='scatter',
        x=x, y=y, mode='markers', ids=i,
        marker=dict(
            size=config.cellRadius,
            line_width=4 * (a > 0),
            line_color='red',
            # color=[color_sequence[n_agents*(int((a+1)/2))+i]
            #        for i, a in enumerate(a)],
            color=color_sequence,
            # symbol=conn_act_symbols[a.astype(int)],
            opacity=0.01 * np.clip(r/1e7, 0, 30)
        ),
        hoverinfo='skip',
        showlegend=False)
    
    # plot users
    hover_text_template = """
    status: {status}<br>
    base station: {bs_id}<br>
    data rate: {data_rate:.2f}<br>
    demand: {demand:.2f}<br>
    deadline: {deadline:.0f}<br>
    thrp ratio: {thrp_ratio:.2f}<br>
    """
    try:
        x, y, b, s, r, u, i = \
            np.array([[ue.pos[0], ue.pos[1], ue.bs.id if ue.bs else net.num_bs,
                       ue.demand//1e5+2, ue.throughput_ratio, ue.urgent, i] 
                      for i, ue in net.ues.items()]).T
        hover_texts = [hover_text_template.format(**ue.info_dict()) for ue in net.ues.values()]
        b = np.nan_to_num(b, nan=net.num_bs).astype(int)
        symbols = ['circle' + ('-x' if u else '') + ('' if r else '-open') for r, u in zip(r, u)]
        ue_plt = dict(
            type='scatter',
            x=x, y=y, mode='markers', ids=i,
            marker=dict(
                size=s,
                line_width=1,
                line_color='grey',
                symbol=symbols,
                color=color_sequence[b],
                # opacity=np.clip((r+1)/2, 0, 1)
            ),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False)
    except ValueError:
        ue_plt = dict(type='scatter')

    # plot data rates
    fr = fig['frames'][-1] if fig['frames'] else None
    t = fr['data'][3]['x'] + [net._time] if fr else [net._time]
    y_max = fr['layout']['yaxis2']['range'][1] if fr else 0
    rate_plts = []
    for i, key in enumerate(['arrival_rate', 'real_rate', 'required_rate']):
        new_y = info[key]
        if fr:
            y = fr['data'][i+3]['y'] + [new_y]
        else:
            y = [new_y]
        y_max = max(y_max, new_y + 20)
        rate_plts.append(dict(
            type='scatter',
            mode='lines',
            x=t, y=y,
            xaxis='x2',
            yaxis='y2',
            name=key.replace('_', ' '),
        ))
    y2_range = [0, y_max]

    # plot penalty
    pen = -info['reward']
    if fr:
        last_rw_plt = fr['data'][-1]
        y_pen = last_rw_plt['y'] + [pen]
        y3_range = [0, max(fr['layout']['yaxis3']['range'][1], pen + 0.1)]
    else:
        y_pen = [pen]
        y3_range = [0, pen + 0.1]
    rw_plt = dict(
        type='scatter',
        mode='lines',
        x=t, y=y_pen,
        xaxis='x2',
        yaxis='y3',
        name='penalty',
        line_color='indianred',
    )
    
    # plot power consumption
    if fr:
        last_pc_plt = fr['data'][-2]
        y_pc = last_pc_plt['y'] + [info['power_consumption']]
        # y2_range = [0, max(fr['layout']['yaxis2']['range'][1], info['power_consumption'] + 0.1)]
    else:
        y_pc = [info['power_consumption']]
        # y2_range = [0, info['power_consumption'] + 0.1]
    pc_plt = dict(
        type='scatter',
        mode='lines',
        x=t, y=y_pc,
        xaxis='x2',
        yaxis='y3',
        name='power (kW)',
        line_color='peru',
    )
    
    # plot drop penalty
    dr_plt = dict(
        type='scatter',
        x=t+t[::-1], y=y_pen+y_pc[::-1],
        xaxis='x2',
        yaxis='y3',
        name='drop rate (10mb/s)',    
        mode='text',
        fill='toself',
        fillcolor='lightyellow',
    )
    
    # append frame
    steps = env._sim_steps
    data = [bs_plt, ue_plt, cl_plt, *rate_plts, dr_plt, pc_plt, rw_plt]
    layout = {
        'xaxis2': dict( range=[0, net._time] ),
        'yaxis2': dict( range=y2_range ),
        'yaxis3': dict( range=y3_range )
    }
    frame = dict(data=data, name=steps, layout=layout)
    
    fig['data'] = data
    fig['frames'].append(frame)
    fig['customdata'].append(info)
    
    if 'sliders' in fig['layout']:  # append slider step
        fig['layout']['sliders'][0]['steps'].append(dict(
            args=[[steps],
                  {"frame": {"duration": 300, "redraw": False},
                   "mode": "immediate",
                   "transition": {"duration": 300}}],
            label=steps,
            method="animate"
        ))

    if mode == 'human':
        fig = go.Figure(fig)
        fig.show()
    return fig


def animate(env: 'MultiCellNetwork'):
    print('Animating...')
    fig = env._figure
    fig['data'] = fig['frames'][0]['data']
    fig = go.Figure(fig)
    fig.show()
    return fig


def make_figure(net):
    xticks = np.linspace(0, net.area[0], 5)
    yticks = np.linspace(0, net.area[1], 5)[1:]
    return dict(
        data=[],
        frames=[],
        customdata=[],
        layout=dict(
            width=1000, height=600,
            xaxis=dict(range=[0, net.area[0]], tickvals=xticks, 
                       autorange=False, showgrid=False, domain=[0, 0.6]),
            yaxis=dict(range=[0, net.area[1]], tickvals=yticks, 
                       autorange=False, showgrid=False),
            xaxis2=dict(domain=[0.7, 1], autorange=False),
            yaxis2=dict(domain=[0.55, 1], anchor='x2', fixedrange=True),
            yaxis3=dict(domain=[0, 0.45], anchor='x2', fixedrange=True),
            margin=dict(l=25, r=25, b=25, t=25),
            # shapes=[dict(
            #     type="circle",
            #     xref="x", yref="y",
            #     x0=x-r, y0=y-r, x1=x+r, y1=y+r,
            #     fillcolor=color_sequence[i],
            #     line_color=color_sequence[i],
            #     line_width=1,
            #     opacity=0.05,
            #     layer="below"
            #     ) for i, bs in net.bss.items() for x, y, _ in [bs.pos]
            #     for r in [bs.cell_radius]
            # ],
            transition={"duration": 300, "easing": "cubic-in-out"},
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {"duration": 200, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "quadratic-in-out"},
                            # "layout": {"xaxis2": {"range": [0, net._time]}}
                        }],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "type": "buttons",
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 15},
                    "prefix": "Step: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": []
            }]
        )
    )
