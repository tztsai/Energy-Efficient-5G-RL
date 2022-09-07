import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from network import config


conn_symbols = ['triangle-down', 'hexagram']
sleep_suffixes = ['', '-open-dot', '-open']
ue_symbols = ['open-circle', 'circle']
conn_act_symbols = np.array(['circle', 'circle', 'circle-open'])
n_agents = 7
color_sequence = np.array(px.colors.qualitative.Plotly)
oppo_color_sequence = np.array(['#%02X%02X%02X' % tuple(
    255 - int(s[i:i+2], 16) for i in range(1, 7, 2)) for s in color_sequence])
color_sequence = np.hstack([color_sequence[:n_agents], oppo_color_sequence[:n_agents]])


def render(env: 'MultiCellNetEnv', mode='human'):
    net = env.net
    
    if env._figure is None:
        env._figure = make_figure(net)

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
    bs_plt = go.Scatter(
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
        showlegend=False)
    
    # plot cell coverages
    cl_plt = go.Scatter(
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
        symbols = [ue_symbols[bool(r)] + ('-x' if u else '') for r, u in zip(r, u)]
        ue_plt = go.Scatter(
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
        ue_plt = go.Scatter(x=[], y=[])
    
    fig = append_frame(env, [bs_plt, ue_plt, cl_plt])
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
            width=600, height=600,
            xaxis=dict(range=[0, net.area[0]], tickvals=xticks,
                       autorange=False, showgrid=False),
            yaxis=dict(range=[0, net.area[1]], tickvals=yticks,
                       autorange=False, showgrid=False),
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
                        "args": [None, {"frame": {"duration": 300, "redraw": False},
                                        "fromcurrent": True,
                                        "transition": {"duration": 300, 
                                                       "easing": "quadratic-in-out"}}],
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
                    "prefix": "time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 200, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": []
            }]
        )
    )

def append_frame(env, data):
    # time = env._steps_info[-1]['time']
    time = env._sim_steps
    frame = dict(data=data, name=time)
    fig = env._figure
    fig['data'] = data
    fig['frames'].append(frame)
    fig['customdata'].append(env.net.info_dict())
    if 'sliders' in fig['layout']:
        fig['layout']['sliders'][0]['steps'].append(dict(
            args=[[time],
                  {"frame": {"duration": 300, "redraw": False},
                   "mode": "immediate",
                   "transition": {"duration": 300}}],
            label=time,
            method="animate"
        ))
    return fig
