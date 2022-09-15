#!/usr/bin/env python
# %%
import torch
from utils import *
from agents import *
from arguments import get_config
from env import MultiCellNetEnv
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx
from dash.exceptions import PreventUpdate
from dash.dependencies import ClientsideFunction
# %reload_ext autoreload
# %autoreload 2

acceleration = 60000  # 1 substep = 1 minute
substeps = 20
days = 7
n_steps = 3 * 24 * days

# %%
def parse_env_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--area_size", type=float,
                        help="width of the square area in meters")
    parser.add_argument("-S", "--traffic_type", type=str, default="A",
                        help="type of traffic to generate")
    parser.add_argument("--start_time", type=str,
                        help="start time of the simulation")
    parser.add_argument("--accel_rate", type=float, default=acceleration,
                        help="acceleration rate of the simulation")
    return parser.parse_args(args)

parser = get_config()
parser.add_argument("-A", '--agent', type=str, default='mappo',
                    help='type of agent used in simulation')

parser.set_defaults(num_env_steps=n_steps)

args, env_args = parser.parse_known_args()
env_args = parse_env_args(env_args)

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

def get_run_dir(args, env_args):
    return Path(os.path.dirname(os.path.abspath(__file__))) / "results" \
        / args.env_name / env_args.traffic_type / args.algorithm_name / args.experiment_name

def get_latest_model_dir(args, run_dir):
    assert run_dir.exists(), "Run directory does not exist: {}".format(run_dir)
    if args.model_dir is not None:
        return run_dir / args.model_dir
    p = 'wandb/run*/files/' if args.use_wandb else 'run*/models/'
    print(run_dir)
    return max(run_dir.glob(p), key=os.path.getmtime)

env = MultiCellNetEnv(**get_env_kwargs(env_args), seed=args.seed)
env.print_info()
obs_space = env.observation_space[0]
cent_obs_space = env.cent_observation_space
action_space = env.action_space[0]

run_dir = get_run_dir(args, env_args)

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
else:
    raise ValueError('invalid agent type')

# %%
from datetime import datetime

obs, _, _ = env.reset()
if args.use_render:
    env.render()

def step_env(obs):
    actions = agent.act(obs) if env.need_action else None
    obs, _, reward, done, _, _ = env.step(actions, substeps=substeps)
    if args.use_render:
        # if env._episode_steps > 20:
        #     env.render(mode='human')
        #     exit()
        env.render()
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
    info['scenario'] = env_args.traffic_type
    info['total_steps'] = T
    info['accel_rate'] = env_args.accel_rate
    info['w_pc'] = env.w_pc
    info['w_drop'] = env.w_drop
    save_path = Path(__file__).parent / "results" / 'records.csv'
    info.to_frame().T.set_index('time').to_csv(
        save_path, mode='a', header=not save_path.exists())
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
