# %%
import sys
import pickle
import random
import atexit
from dash import Dash, dcc, html, Input, Output, ctx
sys.path.append('..')

from network.network import MultiCellNetwork
from network.base_station import BaseStation
from visualize.render import *

net = MultiCellNetwork()
net.reset()

# %%
def rand_act(net):
        bs = net.bss[random.randrange(net.num_bs)]
        bs.num_ant = random.randrange(0, bs.max_antennas + 1, 8)
        # bs.num_ue = random.choice([0, bs.num_ant // 2, max(0, bs.num_ant - 2)])
        # bs.conn_mode = random.randrange(0, 2)
        
def monte_carlo(N=100):
    for _ in tqdm(range(N)):
        rand_act(net)
        net.test_network_channel()

def save_csi_cache():
    print('Saving CSI cache...')
    with open('csi_cache.pkl', 'wb') as f:
        pickle.dump(net._csi_cache, f, pickle.HIGHEST_PROTOCOL)
    print('Done.')

atexit.register(save_csi_cache)

try:
    with open('csi_cache.pkl', 'rb') as f:
        net._csi_cache = pickle.load(f)
except:
    pass

# monte_carlo(100)


# %%
figure = make_figure(net, size=(700, 600), add_subplots=False)
render_bss(net, figure)
render_csi(net, figure)

# %%
app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id="graph", figure=go.Figure(figure)),
    html.Div([
        dcc.RadioItems(
            id='options', value='SINR',
            options=['S', 'I', 'SINR']
        ),
        dcc.Dropdown(
            id='bs-selector', value=0,
            options=[{'label': x, 'value': x} for x in net.bss.keys()],
        ),
        dcc.Slider(
            id='ant-slider', min=0, max=BaseStation.max_antennas,
            value=64, step=4
        ),
        dcc.Checklist(
            id='bs-switch', options=list(range(net.num_bs)), inline=True,
            value=list(range(net.num_bs))
        ),
    ]),
    dcc.Interval(id='interval', interval=60000, n_intervals=0),
])

frames = []

@app.callback(
    Output("graph", "figure"),
    Output("ant-slider", "value"),
    Input("options", "value"),
    Input("bs-selector", "value"),
    Input("ant-slider", "value"),
    Input("bs-switch", "value"),
    Input("interval", "n_intervals"),
)
def update_plot(opt, bs_id, n_ants, on_bss, time):
    if ctx.triggered_id == 'interval':
        pass
        # rand_act(net)
    elif ctx.triggered_id == 'ant-slider':
        if n_ants == net.get_bs(bs_id).num_ant:
            raise PreventUpdate
        net.get_bs(bs_id).num_ant = n_ants
    elif ctx.triggered_id == 'bs-switch':
        for i, bs in net.bss.items():
            bs.conn_mode = int(i in on_bss)
            if bs.conn_mode < 1:
                bs.num_ant = 0
    if ctx.triggered_id in ('ant-slider', 'bs-switch', 'interval'):
        figure['data'].clear()
        render_csi(net, figure)
        render_bss(net, figure)
    fig = figure.copy()
    layouts = fig.pop('_layouts')
    for t in fig['data']:
        if 'name' in t:
            t['visible'] = t['name'] == opt
    fig['layout'] = layouts[opt]
    titles = dict(S='Signal', I='Interference', SINR='SINR')
    fig['layout']['title'] = dict(
        text=titles[opt], x=0.4, y=0.94, xanchor='center', yanchor='top')
    frames.append(fig)
    fig = go.Figure(fig)
    ants = ''.join(map(str, [bs.num_ant//8 for bs in net.bss.values()]))
    modes = ''.join(map(str, [bs.conn_mode for bs in net.bss.values()]))
    figname = f'sinr_plots/{ants}_{modes}_{opt}.png'
    fig.write_image(figname, scale=2)
    return fig, net.get_bs(bs_id).num_ant

def make_anim(frames=frames):
    layout = frames[0]['layout'].to_plotly_json().copy()
    frames = [go.Frame(data=f['data'], layout=f['layout']) for f in frames]
    layout['updatemenus'] =  [dict(
        type="buttons",
        buttons=[dict(label="Play", method="animate", args=[None])])]
    fig = go.Figure(
        data=frames[0]['data'],
        frames=frames,
        layout=layout)
    fig.write_html('anim.html', auto_play=True)
    
# atexit.register(make_anim)
app.run()
