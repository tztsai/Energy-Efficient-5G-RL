from utils import *
from network import config
import plotly.express as px
import plotly.graph_objects as go

sleep_symbols = np.array(['hexagram', 'hexagram-open', 'hexagon-open', 'x-open'])
n_agents = 7
color_sequence = np.array(px.colors.qualitative.Plotly)[:n_agents+1]
oppo_color_sequence = np.array(['#%02X%02X%02X' % tuple(
    255 - int(s[i:i+2], 16) for i in range(1, 7, 2)) for s in color_sequence])

delay_penalty_color = 'slateblue'
drop_penalty_color = 'coral'
pc_penalty_color = 'plum'


def render(env: 'MultiCellNetEnv', mode='frame'):
    """ Render the environment using Plotly.
    Render modes:
    - show: create, store, and show Plotly figure
    - dash: update Graph in Dash app
    - frame: create and store a dict to be used to render as a Plotly frame
    - any false value: do nothing
    """
    if not mode: return
    
    net = env.net
    info = env.info_dict()
    
    if env._figure is None:
        env._figure = make_figure(net, add_anim_btn=(mode == 'frame'))
    fig = env._figure
    
    last_fr = fig['frames'][-1] if fig['frames'] else []
    frame = dict(data=[], layout={}, name=info['time'])
    if last_fr:
        frame['layout'] = last_fr['layout'].copy()
    
    # render_csi(net, frame, last_fr, kpis=['SINR'])
    render_bss(net, frame)
    render_ues(net, frame)
    render_data_rates(net, info, frame, last_fr)
    render_penalties(net, info, frame, last_fr)

    fig['frames'].append(frame)
    # fig['customdata'].append(info)
    
    if 'sliders' in fig['layout']:  # append slider step
        fig['layout']['sliders'][0]['steps'].append(dict(
            args=[[info['time']],
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

bs_info_template = """<br>
id: {id}<br>
pc: {pc:.2f} W<br>
antennas: {n_ants}<br>
sleep mode: {sleep_mode}<br>
wakeup time: {wakeup_time} ms<br>
accept conn: {responding}<br>
ues in service: {serving_ues}<br>
ues in queue: {queued_ues}<br>
ues in coverage: {covered_ues}<br>
sum rate: {sum_rate:.1f} Mb/s<br>
sum rate req: {req_sum_rate:.1f} Mb/s<br>
""".strip()
def render_bss(net, frame, show_id=False):
    i, x, y, m, s, r = np.array([[
        bs.id, bs.pos[0], bs.pos[1], bs.num_ant, bs.sleep,
        bs.responding] for bs in net.bss.values()]).T
    hover_texts = [bs_info_template.format(id=i, **bs.info_dict()) for i, bs in net.bss.items()]
    frame['data'].append(dict(
        type='scatter',
        x=x, y=y, mode='markers+text', ids=i,
        marker=dict(
            size=m/8+12,
            line_width=1,
            line_color=color_sequence,
            symbol=sleep_symbols[s.astype(int)],
            color=color_sequence,
            opacity=(r > 0) * 0.6 + 0.4
        ),
        text=list(range(net.num_bs)) if show_id else None,
        # textposition="middle center",
        # textfont=dict(color='white'),
        hovertext=hover_texts,
        hoverlabel=dict(font_size=10),
        hoverinfo='text',
        showlegend=False,
    ))


ue_info_template = """<br>
status: {status}<br>
base station: {bs_id}<br>
data rate: {rate:.2f} Mb/s<br>
demand: {demand:.2f} kb<br>
time limit: {ddl:.0f} ms<br>
""".strip()
def render_ues(net, frame):
    if net.num_ue == 0:
        return frame['data'].append({})
    i, x, y, b, s, r, l = np.array(
        [[ue.id, ue.pos[0], ue.pos[1], ue.bs.id if ue.bs else -1, ue.demand,
          ue.data_rate, ue.time_limit] for ue in net.ues.values()]).T
    hover_texts = [ue_info_template.format(**ue.info_dict()) for ue in net.ues.values()]
    symbols = ['x-thin' if l <= 3e-3 else ('circle' if r else 'circle-open') for r, l in zip(r, l)]
    # 'x' if UE is going to be dropped, 'circle' if UE is being served, 'circle-open' otherwise
    frame['data'].append(dict(
        type='scatter',
        x=x, y=y, mode='markers', ids=i,
        marker=dict(
            size=s/3e5 + 4 + 10*(l<=3e-3),
            line_width=0,
            # line_color='grey',
            symbol=symbols,
            color=color_sequence[b.astype(int)],
            # opacity=np.clip((r+1)/2, 0, 1)
        ),
        hovertext=hover_texts,
        hoverlabel=dict(font_size=10),
        hoverinfo='text',
        showlegend=False))

def render_cells(net, frame):
    c, r = np.array([[bs.conn_mode, bs.sum_rate] for bs in net.bss.values()]).T
    frame['layout'].setdefault('shapes', []).extend(dict(
        type="circle",
        xref="x", yref="y",
        x0=x-r, y0=y-r, x1=x+r, y1=y+r,
        fillcolor='grey' if c < 0 else color,
        line_color='red' if c > 0 else color,
        line_width=3,
        opacity=0.015 * np.clip(v/1e8, 0, 30), # + (c < 0) * 0.06,
        layer="below")
        for i, (c, v, color) in enumerate(zip(c, r, color_sequence))
        for bs in [net.get_bs(i)]
        for x, y, _ in [bs.pos]
        for r in [bs.cell_radius])

def render_csi(net, frame, last_frame=None, kpis=['S', 'I', 'SINR']):
    if net._stats_updated:
        csi_df = net.test_network_channel()
    layouts = dict()
    for var in kpis:
        if net._stats_updated:
            df = csi_df[var].unstack().T
            im = px.imshow(df, labels=dict(color='dB', x='', y=''),
                           width=600, aspect='equal', origin='lower')
            im.update_xaxes(showgrid=False)
            im.update_yaxes(showgrid=False)
            im.update_layout(margin=dict(l=30, r=55, t=65, b=40))
            trace = im.data[0].to_plotly_json()
            trace['name'] = var
            trace['visible'] = var == 'SINR'
            layouts[var] = im.layout
            if var == 'SINR':
                frame['layout']['coloraxis'] = im.layout['coloraxis']
        elif last_frame:
            trace = next(t for t in last_frame['data'] if t['name'] == var)
        frame['data'].append(trace)
    return frame, layouts
    # frame['_layouts'] = layouts

    # frame['layout']['updatemenus'] = [{
    #     'buttons': [{'args': [{'visible': [True] * len(frame['data'])}],
    #                  'label': 'All',
    #                  'method': 'restyle'},
    #                 {'args': [{'visible': [False] * len(frame['data'])}],
    #                  'label': 'None',
    #                  'method': 'restyle'}] + [
    #                     {'args': [{'visible': [False] * len(frame['data'])}],
    #                      'label': key,
    #                      'method': 'restyle'}
    #                     for key in csi_ds.data_vars.keys()],
    #     'direction': 'down',
    #     'showactive': True,
    #     'type': 'dropdown',
    #     'x': 0.1,
    #     'xanchor': 'left',
    #     'y': 1.2,
    #     'yanchor': 'top'
    # }]
        
def render_data_rates(net, info, frame, last_frame=[], ws=300):
    i0 = len(frame['data'])
    
    if not net._stats_updated:
        frame['data'].extend(last_frame['data'][i0:i0+3])
        return
    
    t = (last_frame and last_frame['data'][i0]['x'])[1-ws:] + [net.world_time/3600]
    y_max = 0
    for i, key in enumerate(['arrival_rate', 'actual_rate', 'required_rate']):
        new_y = info[key]
        y = (last_frame and last_frame['data'][i0+i]['y'])[1-ws:] + [new_y]
        y_max = max(y_max, max(y) * 1.05)
        frame['data'].append(dict(
            type='scatter',
            mode='lines',
            x=t, y=y,
            xaxis='x2',
            yaxis='y2',
            name=key.replace('_', ' ')#+' (Mb/s)',
        ))
    frame['layout']['xaxis2'] = dict(range=[t[0], t[-1]])
    frame['layout']['yaxis2'] = dict(range=[0, y_max])

def render_penalties(net, info, frame, last_frame=[], ws=300):
    i0 = len(frame['data'])
    
    if not net._stats_updated:
        frame['data'].extend(last_frame['data'][i0:i0+3])
        return
    
    # dl = info['weighted_delay']
    # dr = info['weighted_drop']
    qos = info['qos_reward']
    pc = info['pc_kw']

    t = (last_frame and last_frame['data'][i0]['x'])[1-ws:] + [net.world_time/3600]
    y31 = (last_frame and last_frame['data'][i0]['y'])[1-ws:] + [pc] # [dl + pc + dr]
    y3_range = [0, max(y31) * 1.05]
    frame['data'].append(dict(
        type='scatter',
        mode='lines',
        x=t, y=y31,
        xaxis='x2',
        yaxis='y3',
        name='power (kW)', #'drop rate',
        fill='tozeroy',
        line_color=pc_penalty_color
    ))
    frame['layout']['yaxis3'] = dict(range=y3_range)

    y32 = (last_frame and last_frame['data'][i0+1]['y'])[1-ws:] + [pc - qos] #[dl + pc]
    frame['data'].append(dict(
        type='scatter',
        mode='lines',
        x=t, y=y32,
        xaxis='x2',
        yaxis='y3',
        name='penalty',
        fill='tozeroy',
        line_color=drop_penalty_color
    ))

    # y33 = (last_frame and last_frame['data'][i0+2]['y'])[1-ws:] + [dl]
    # frame['data'].append(dict(
    #     type='scatter',
    #     mode='lines',
    #     x=t, y=y33,
    #     xaxis='x2',
    #     yaxis='y3',
    #     name='delay',
    #     fill='tozeroy',
    #     line_color=delay_penalty_color,
    # ))

def make_figure(net, size=(1000, 600), 
                add_anim_btn=False, add_subplots=True):
    xticks = np.linspace(0, net.area[0], 5)
    yticks = np.linspace(0, net.area[1], 5)[1:]
    fig = dict(
        data=[],
        frames=[],
        customdata=[],
        layout=dict(
            width=size[0], height=size[1],
            xaxis=dict(range=[0, net.area[0]], tickvals=xticks, 
                       autorange=False, showgrid=False),
            yaxis=dict(range=[0, net.area[1]], tickvals=yticks, 
                       autorange=False, showgrid=False),
            margin=dict(l=25, r=25, b=25, t=25),
            transition={"duration": 300, "easing": "cubic-in-out"},
        ))
    if add_subplots:
        fig['layout']['xaxis']['domain'] = [0, 0.6]
        fig['layout'].update(
            xaxis2=dict(domain=[0.7, 1],
                        # range=[0, 24],
                        tickangle=45, nticks=4),
            yaxis2=dict(domain=[0.55, 1], anchor='x2',
                        title_text='Mb/s', title_standoff=5),
            yaxis3=dict(domain=[0, 0.45], anchor='x2'))
    if add_anim_btn:  # otherwise dash
        fig['layout'].update(
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {
                            "frame": {"duration": 150, "redraw": False},
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
                    # dict(
                    #     args=["type", "surface"],
                    #     label="3D Surface",
                    #     method="restyle"
                    # ),
                    # dict(
                    #     args=["type", "heatmap"],
                    #     label="Heatmap",
                    #     method="restyle"
                    # )
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
                "transition": {"duration": 200, "easing": "cubic-in-out"},
                "pad": {"b": 10, "t": 50},
                "len": 0.96,
                "x": 0.024,
                "y": 0,
                "steps": []
            }])
    return fig


def create_dash_app(env, args):
    from dash import Dash, dcc, html, Input, Output, ctx
    from dash.exceptions import PreventUpdate
    from dash.dependencies import ClientsideFunction

    app = Dash(type(env).__name__)

    figure = env._figure
    slider_ticks = np.linspace(0, args.num_env_steps, num=6)

    app.layout = html.Div([
        # html.H4('5G Network Simulation'),
        dcc.Graph(id="graph", figure=go.Figure(figure)),
        html.Div([
            html.Button('Play', id="run-pause", n_clicks=0, className='column'),
            html.P(id="step-info", className='column')], className='row'),
        dcc.Interval(id='clock', interval=300),
        dcc.Slider(
            id='slider',
            min=slider_ticks[0], max=slider_ticks[-1], step=1, value=0,
            marks={t: f'{t:.2f}' for t in slider_ticks},
        ),
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
