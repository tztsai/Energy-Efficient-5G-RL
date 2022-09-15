import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from network import config

sleep_symbols = np.array(['hexagram', 'hexagram-open', 'hexagon-open', 'x-open'])
n_agents = 7
color_sequence = np.array(px.colors.qualitative.Plotly)[:n_agents+1]
oppo_color_sequence = np.array(['#%02X%02X%02X' % tuple(
    255 - int(s[i:i+2], 16) for i in range(1, 7, 2)) for s in color_sequence])

penalty_line_color = 'coral'
drop_penalty_color = 'plum'
power_line_color = 'slateblue'


def render(env: 'MultiCellNetEnv', mode='none'):
    net = env.net
    info = env.info_dict()
    
    if env._figure is None:
        env._figure = make_figure(net)
    fig = env._figure

    # plot base stations
    x, y, m, s, c, r, i = np.array(
        [[bs.pos[0], bs.pos[1], bs.num_ant, bs.sleep, bs.conn_mode,
          bs.sum_rate, i] for i, bs in net.bss.items()]).T
    hover_text_template = """
    id: {id}<br>
    num antennas: {num_antennas}<br>
    sleep mode: {sleep_mode}<br>
    wake up: {wakeup_time}<br>
    responding: {responding}<br>
    power consumption: {pc:.2f}<br>
    ues in service: {num_served}<br>
    ues in queue: {num_queued}<br>
    ues in coverage: {num_covered}<br>
    throughput: {thrp_served:.2f}<br>
    demand rate: {thrp_req_served:.2f}<br>
    queued demand rate: {thrp_req_queued:.2f}<br>
    idle demand rate: {thrp_req_idle:.2f}<br>
    """
    hover_texts = [hover_text_template.format(id=i, **bs.info_dict()) for i, bs in net.bss.items()]
    bs_plt = dict(
        type='scatter',
        x=x, y=y, mode='markers', ids=i,
        marker=dict(
            size=m/8+12,
            line_width=1,
            line_color=color_sequence,
            symbol=sleep_symbols[s.astype(int)],
            color=color_sequence,
            opacity=(s == 0) * 0.5 + 0.5
        ),
        hovertext=hover_texts,
        hoverinfo='text',
        showlegend=False,
    )
    
    # plot cell coverages
    # cl_plt = dict(
    #     type='scatter',
    #     x=x, y=y, mode='markers', ids=i,
    #     marker=dict(
    #         size=config.cellRadius,
    #         line_width=3,
    #         line_color=['red' if c > 0 else color for c, color in zip(c, color_sequence)],
    #         color=['grey' if c < 0 else color for c, color in zip(c, color_sequence)],
    #         symbol='circle',
    #         opacity=0.015 * np.clip(r/1e8, 0, 30) + (c < 0) * 0.06,
    #     ),
    #     hoverinfo='skip',
    #     showlegend=False)
    cell_shapes = [dict(
        type="circle",
        xref="x", yref="y",
        x0=x-r, y0=y-r, x1=x+r, y1=y+r,
        fillcolor='grey' if c < 0 else color,
        line_color='red' if c > 0 else color,
        line_width=3,
        opacity=0.015 * np.clip(v/1e8, 0, 30) + (c < 0) * 0.06,
        layer="below")
        for (i, (c, v, color)) in enumerate(zip(c, r, color_sequence)) 
        for bs in [net.get_bs(i)]
        for x, y, _ in [bs.pos]
        for r in [bs.cell_radius]
    ]
    
    # plot users
    hover_text_template = """
    status: {status}<br>
    base station: {bs_id}<br>
    data rate: {thrp:.2f}<br>
    demand: {demand:.2f}<br>
    deadline: {ddl:.0f}<br>
    """
    try:
        x, y, b, s, r, u, i = \
            np.array([[ue.pos[0], ue.pos[1], ue.bs.id if ue.bs else net.num_bs,
                       ue.demand, ue.data_rate, ue.urgent, i] 
                      for i, ue in net.ues.items()]).T
        hover_texts = [hover_text_template.format(**ue.info_dict()) for ue in net.ues.values()]
        b = np.nan_to_num(b, nan=net.num_bs).astype(int)
        symbols = ['circle' + ('-x' if u else '') + ('' if r else '-open') for r, u in zip(r, u)]
        ue_plt = dict(
            type='scatter',
            x=x, y=y, mode='markers', ids=i,
            marker=dict(
                size=s/1.5e5+2,
                line_width=0,
                # line_color='grey',
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
    ws = 80  # window size
    fr = fig['frames'][-1] if fig['frames'] else None
    t = fr['data'][2]['x'] + [net._time] if fr else [net._time]
    t = t[-ws:]
    rate_plts = []
    y_max = 0
    for i, key in enumerate(['arrival_rate', 'actual_rate', 'required_rate']):
        new_y = info[key] / 8
        if fr:
            y = fr['data'][i+2]['y'] + [new_y]
        else:
            y = [new_y]
        y = y[-ws:]
        y_max = max(y_max, max(y) * 1.05)
        rate_plts.append(dict(
            type='scatter',
            mode='lines',
            x=t, y=y,
            xaxis='x2',
            yaxis='y2',
            name=key.replace('_', ' ')+' (MB/s)',
        ))
    y2_range = [0, y_max]

    # plot penalty
    pen = -info['reward']
    if fr:
        y_pen = fr['data'][-1]['y'] + [pen]
    else:
        y_pen = [pen]
    y_pen = y_pen[-ws:]
    y3_range = [0, max(y_pen) * 1.05]
    rw_plt = dict(
        type='scatter',
        mode='lines',
        x=t, y=y_pen,
        xaxis='x2',
        yaxis='y3',
        name='penalty',
        line_color=penalty_line_color,
    )
    
    # plot power consumption
    if fr:
        y_pc = fr['data'][-2]['y'] + [info['power_consumption']]
        # y2_range = [0, max(fr['layout']['yaxis2']['range'][1], info['power_consumption'] + 0.1)]
    else:
        y_pc = [info['power_consumption']]
        # y2_range = [0, info['power_consumption'] + 0.1]
    y_pc = y_pc[-ws:]
    pc_plt = dict(
        type='scatter',
        mode='lines',
        x=t, y=y_pc,
        xaxis='x2',
        yaxis='y3',
        name='power (kW)',
        line_color=power_line_color,
    )
    
    # plot drop penalty
    # if fr:
    #     y_pc = fr['data'][-3]['y'] + [info['power_consumption']]
    #     # y2_range = [0, max(fr['layout']['yaxis2']['range'][1], info['power_consumption'] + 0.1)]
    # else:
    #     y_pc = [info['power_consumption']]
    #     # y2_range = [0, info['power_consumption'] + 0.1]
    # y_pc = y_pc[-ws:]
    dr_plt = dict(
        type='scatter',
        x=t[::-1]+t, y=y_pen[::-1]+y_pc,
        xaxis='x2',
        yaxis='y3',
        name='drop rate (MB/s)',    
        mode='text',
        fill='toself',
        fillcolor=drop_penalty_color,
    )
    
    # append frame
    time = info['time']
    data = [bs_plt, ue_plt, *rate_plts, dr_plt, pc_plt, rw_plt]
    layout = {
        'xaxis2': dict( range=[t[0], t[-1]] ),
        'yaxis2': dict( range=y2_range ),
        'yaxis3': dict( range=y3_range ),
        'shapes': cell_shapes
    }
    frame = dict(data=data, name=time, layout=layout)
    
    fig['data'] = data
    fig['frames'].append(frame)
    fig['customdata'].append(info)
    
    if 'sliders' in fig['layout']:  # append slider step
        fig['layout']['sliders'][0]['steps'].append(dict(
            args=[[time],
                  {"frame": {"duration": 300, "redraw": False},
                   "mode": "immediate",
                   "transition": {"duration": 300}}],
            label=info['time'],
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
    # frames = fig.pop('frames')
    # fig = go.FigureWidget(fig)
    # yield fig
    # updateable_attrs = {'x', 'y', 'marker', 'ids'}
    # for fr in frames:
    #     for fig_tr, fr_tr in zip(fig.data, fr['data']):
    #         for attr, val in fr_tr.items():
    #             if attr in updateable_attrs:
    #                 setattr(fig_tr, attr, val)
    #     yield


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
            transition={"duration": 300, "easing": "cubic-in-out"},
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {"duration": 300, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 300, "easing": "cubic-in-out"},
                            # "layout": {"xaxis2": {"range": [0, net._time]}}
                        }],
                        "args2": [[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}],
                        "label": "Play",
                        "method": "animate"
                    },
                    # {
                    #     "args": [[None], {"frame": {"duration": 0, "redraw": False},
                    #                       "mode": "immediate",
                    #                       "transition": {"duration": 0}}],
                    #     "label": "Pause",
                    #     "method": "animate"
                    # }
                ],
                "type": "buttons",
                "direction": "left",
                "pad": {"t": 36},
                "showactive": False,
                "x": 0.035,
                "xanchor": "left",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                # "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 15},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.96,
                "x": 0.024,
                "y": 0,
                "steps": []
            }]
        )
    )
