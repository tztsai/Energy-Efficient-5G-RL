import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx
from dash.exceptions import PreventUpdate
from dash.dependencies import ClientsideFunction
from utils import deep_update
from network import config


sleep_symbols = np.array(['hexagram', 'hexagram-open', 'hexagon-open', 'x-open'])
n_agents = 7
color_sequence = np.array(px.colors.qualitative.Plotly)[:n_agents+1]
oppo_color_sequence = np.array(['#%02X%02X%02X' % tuple(
    255 - int(s[i:i+2], 16) for i in range(1, 7, 2)) for s in color_sequence])

delay_penalty_color = 'slateblue'
drop_penalty_color = 'plum'
pc_penalty_color = 'coral'


def render(env: 'MultiCellNetEnv', mode='frame'):
    """ Render the environment using Plotly.
    Render modes:
    - show: create, store, and show Plotly figure
    - dash: update Graph in Dash app
    - any other True value: create and store a dict to be used to render as a Plotly frame
    - any False value: do nothing
    """
    if not mode: return
    
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
    num antennas: {n_ants}<br>
    sleep mode: {sleep_mode}<br>
    wake up: {wakeup_time}<br>
    conn mode: {conn_mode}<br>
    power consumption: {pc:.2f}<br>
    ues in service: {active_ues}<br>
    ues in queue: {queued_ues}<br>
    ues in coverage: {covered_ues}<br>
    sum rate: {sum_rate:.2f}<br>
    required rate: {req_rate:.2f}<br>
    """
    hover_texts = [hover_text_template.format(id=i, **bs.info_dict()) for i, bs in net.bss.items()]
    bs_plt = dict(
        type='scatter',
        x=x, y=y, mode='markers', ids=i,
        marker=dict(
            size=m/10+14,
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
    data rate: {rate:.2f}<br>
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
                size=s/3e5+3,
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

    fr = fig['frames'][-1] if fig['frames'] else []
    
    if net._stats_updated:
        # plot data rates
        ws = 60  # window size
        t = (fr and fr['data'][2]['x']) + [net.world_time/3600] # [net.world_time_repr.split(', ')[1]]
        t = t[-ws:]
        rate_plts = []
        y_max = 0
        for i, key in enumerate(['arrival_rate', 'actual_rate', 'required_rate']):
            new_y = info[key]
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
                name=key.replace('_', ' ')+' (Mb/s)',
            ))
        y2_range = [0, y_max]

        # plot penalties
        dl = info['weighted_delay']
        dr = info['weighted_drop']
        pc = info['weighted_pc']
        
        y31 = (fr and fr['data'][-3]['y']) + [dl + pc + dr]
        y31 = y31[-ws:]
        y3_range = [0, max(y31) * 1.05]
        pc_plt = dict(
            type='scatter',
            mode='lines',
            x=t, y=y31,
            xaxis='x2',
            yaxis='y3',
            name='drop rate',
            fill='tozeroy',
            line_color=pc_penalty_color,
        )
        
        y32 = (fr and fr['data'][-2]['y']) + [dl + pc]
        y32 = y32[-ws:]
        dr_plt = dict(
            type='scatter',
            mode='lines',
            x=t, y=y32,
            xaxis='x2',
            yaxis='y3',
            name='power (kW)',
            fill='tozeroy',
            line_color=drop_penalty_color,
        )

        y33 = (fr and fr['data'][-1]['y']) + [dl]
        y33 = y33[-ws:]
        dl_plt = dict(
            type='scatter',
            mode='lines',
            x=t, y=y33,
            xaxis='x2',
            yaxis='y3',
            name='delay',
            fill='tozeroy',
            line_color=delay_penalty_color,
        )
        
        data = [
            bs_plt, ue_plt, 
            *rate_plts, pc_plt, dr_plt, dl_plt
        ]
        layout = {
            # 'xaxis2': dict( range=[t[0], t[-1]] ),
            'yaxis2': dict( range=y2_range ),
            'yaxis3': dict( range=y3_range ),
            'shapes': cell_shapes
        }
    else:
        data = [bs_plt, ue_plt]
        layout = {}
        if fr is not None:
            data.extend(fr['data'][2:])
            layout.update(fr['layout'])
        layout['shapes'] = cell_shapes

    time = info['time']
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

    if mode == 'show':
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


def make_figure(net, mode='plotly'):
    xticks = np.linspace(0, net.area[0], 5)
    yticks = np.linspace(0, net.area[1], 5)[1:]
    fig = dict(
        data=[],
        frames=[],
        customdata=[],
        layout=dict(
            width=1000, height=600,
            xaxis=dict(range=[0, net.area[0]], tickvals=xticks, 
                       autorange=False, showgrid=False, domain=[0, 0.6]),
            yaxis=dict(range=[0, net.area[1]], tickvals=yticks, 
                       autorange=False, showgrid=False),
            xaxis2=dict(domain=[0.7, 1], #autorange=False,
                        tickangle=45, nticks=4),
            yaxis2=dict(domain=[0.55, 1], anchor='x2'),
            yaxis3=dict(domain=[0, 0.45], anchor='x2'),
            margin=dict(l=25, r=25, b=25, t=25),
            transition={"duration": 300, "easing": "cubic-in-out"},
        ))
    if mode == 'plotly':  # otherwise dash
        fig['layout'].update(
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
                        "label": "Play",  # press to switch between play and pause
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
            }])
    return fig


def dash_app(env, args):
    app = Dash(type(env).__name__)

    figure = env._figure

    T = args.num_env_steps
    app.layout = html.Div([
        # html.H4('5G Network Simulation'),
        dcc.Graph(id="graph", figure=go.Figure(figure)),
        html.Div([
            html.Button('Play', id="run-pause", n_clicks=0, className='column'),
            html.P(id="step-info", className='column')], className='row'),
        dcc.Interval(id='clock', interval=300),
        dcc.Slider(
            id='slider',
            min=0, max=T, step=1, value=0,
            marks={t: f'{t:.2f}' for t in np.linspace(0, T, num=6)},
        ),
        # dcc.Store(id='storage', data=env._figure)
    ])

    # app.clientside_callback(
    #     ClientsideFunction(namespace='clientside', function_name='update'),
    #     Output("graph", "figure"),
    #     Output("step-info", "children"),
    #     Output("run-pause", "value"),
    #     Output("slider", "value"),
    #     Input("slider", "value"),
    #     Input("run-pause", "n_clicks"),
    #     Input("clock", "n_intervals"),
    #     Input("storage", "data")
    # )

    @app.callback(
        Output("graph", "figure"),
        Output("step-info", "children"),
        Output("run-pause", "value"),
        Output("slider", "value"),
        Input("slider", "value"),
        Input("run-pause", "n_clicks"),
        Input("clock", "n_intervals"),
        Input("graph", "figure")
    )
    def update_plot(time, clicks, ticks, fig):
        running = clicks % 2
        if ctx.triggered_id != 'clock':
            raise PreventUpdate  # avoid loop
        elif not running:
            raise PreventUpdate
        t_max = len(fig['frames']) - 1
        if running and time < t_max:
            time += 1
        if time > t_max:
            time = t_max
        frame = fig['frames'][time]
        fig['data'] = frame['data']
        deep_update(fig['layout'], frame['layout'])
        text = "Step: {}  Time: {}".format(time, frame['name'])
        return fig, text, ('Stop' if running else 'Play'), time

    return app
