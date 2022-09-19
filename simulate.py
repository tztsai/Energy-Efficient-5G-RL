#!/usr/bin/env python
# %%
import torch
from utils import *
from agents import *
from arguments import *
from env import MultiCellNetEnv
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx
from dash.exceptions import PreventUpdate
from dash.dependencies import ClientsideFunction
# %reload_ext autoreload
# %autoreload 2

accelerate = 1000

parser = get_config()
parser.add_argument("-A", '--agent', type=str, default='mappo',
                    help='type of agent used in simulation')
parser.add_argument("--perf_save_path", default="results/performance.csv",
                    help="path to save the performance of the simulation")
parser.add_argument("--log_path",
                    help="path to save the log of the simulation")
parser.add_argument("--render_interval", type=int, default=4,
                    help="interval of rendering")
parser.add_argument("--days", type=int, default=7,
                    help="number of days to simulate")

env_parser = get_env_config()

parser.set_defaults(log_level='NOTICE')
env_parser.set_defaults(accelerate=accelerate)

args, env_args = parser.parse_known_args()
env_args = env_parser.parse_args(env_args)

args.num_env_steps = args.days * 24 * 3600 * 50 // env_args.accelerate

# %%
set_log_level(args.log_level)

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# %% [markdown]
# ## Simulation Parameters

# %%
def get_env_kwargs(args):
    return {k: v for k, v in vars(args).items() if v is not None}

def get_latest_model_dir(args, run_dir):
    assert run_dir.exists(), "Run directory does not exist: {}".format(run_dir)
    if args.model_dir is not None:
        return run_dir / args.model_dir
    p = 'wandb/run*/files/' if args.use_wandb else 'run*/models/'
    print(run_dir)
    return max(run_dir.glob(p), key=os.path.getmtime)

env = MultiCellNetEnv(**get_env_kwargs(env_args), seed=args.seed)
env.print_info()
env.net.traffic_model.print_info()

obs_space = env.observation_space[0]
cent_obs_space = env.cent_observation_space
action_space = env.action_space[0]

run_dir = get_run_dir(args, env_args)

if args.log_path is None:
    fn = '{}_{}_{}_acc-{}.log'.format(
        args.agent, env_args.scenario, 
        re.sub('(, |:)', '-', env.net.world_time_repr),
        env.net.accelerate)
    args.log_path = 'logs/' + fn

set_log_file(args.log_path)

# match args.agent.lower():
if args.agent == 'mappo':
    model_dir = args.model_dir or get_latest_model_dir(args, run_dir)
    agent = MappoPolicy(args, obs_space, cent_obs_space, action_space, model_dir=model_dir)
elif args.agent == 'fixed':
    agent = AlwaysOnPolicy(action_space, env.num_agents)
elif args.agent == 'random':
    agent = RandomPolicy(action_space, env.num_agents)
elif args.agent == 'adaptive':
    agent = AdaptivePolicy(action_space, env.num_agents)
elif args.agent == 'sleepy':
    agent = SleepyPolicy(action_space, env.num_agents)
else:
    raise ValueError('invalid agent type')

# %%
from datetime import datetime

obs, _, _ = env.reset()

if args.use_render:
    env.render()
else:
    render_interval = None

def step_env(obs):
    actions = agent.act(obs) if env.need_action else None
    obs, _, reward, done, _, _ = env.step(
        actions, render_interval=render_interval)
    return obs, reward[0], done

T = args.num_env_steps

def simulate(obs=obs):
    step_rewards = []
    for i in trange(T, file=sys.stdout):
        obs, reward, done = step_env(obs)
        step_rewards.append(reward)
    rewards = pd.Series(np.squeeze(step_rewards), name='reward')
    info = rewards.describe()
    print(info)
    info.index = ['reward_' + str(i) for i in info.index]
    info['agent'] = args.agent
    info['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info['scenario'] = env_args.scenario
    info['total_steps'] = T
    info['accelerate'] = env_args.accelerate
    info['w_pc'] = env.w_pc
    info['w_drop'] = env.w_drop
    save_path = args.perf_save_path
    info.to_frame().T.set_index('time').to_csv(
        save_path, mode='a', header=not os.path.exists(str(save_path)))
    if args.use_render and not args.use_dash:
        return env.animate()

simulate()
env.close()

# %%
if not args.use_dash: exit()

app = Dash(__name__)

figure = env._figure
figure['layout'].pop('sliders')
figure['layout'].pop('updatemenus')

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

# threading.Thread(target=simulate).start()
app.run_server(debug=True)
