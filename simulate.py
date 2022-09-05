# %%
import torch
from utils import *
from agents import *
from config import get_config
from env import Green5GNetEnv
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# %reload_ext autoreload
# %autoreload 2

# %%
def parse_env_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--area_size", type=float,
                        help="width of the square area in meters")
    parser.add_argument("--traffic_type", default="urban",
                        type=str, help="type of traffic to generate")
    parser.add_argument("--start_time", type=str,
                        help="start time of the simulation")
    parser.add_argument("--accel_rate", type=float,
                        help="acceleration rate of the simulation")
    parser.add_argument("--act_interval", type=int,
                        help="number of simulation steps between two actions")
    return parser.parse_known_args(args)[0]

def get_env_kwargs(args):
    return {k: v for k, v in vars(args).items() if v is not None}

def get_latest_model_dir(args, env_args, use_wandb=True):
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
        / args.env_name / env_args.traffic_type / args.algorithm_name / args.experiment_name
    assert run_dir.exists(), "Run directory does not exist: {}".format(run_dir)
    if args.model_dir is not None:
        return run_dir / args.model_dir
    p = 'wandb/run*/files' if use_wandb else 'run*/models'
    return max(run_dir.glob(p), key=os.path.getmtime)

parser = get_config()
args = [
    "-T", "100",
    "--start_time", "307800",
    "--log_level", "WARN",
    "--use_render", 
    # "--use_dash", 
    # "--model_dir", "wandb/run-20220825_231102-3q9eju6l/files"
]
args, env_args = parser.parse_known_args(args)
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
# agent = FixedPolicy([5, 4, 4], 7)
# agent = RandomPolicy([5, 4, 4], 7)
env = Green5GNetEnv(**get_env_kwargs(env_args), seed=args.seed)
spaces = env.observation_space[0], env.cent_observation_space, env.action_space[0]
model_dir = get_latest_model_dir(args, env_args)
agent = MappoPolicy(args, *spaces, model_dir=model_dir)

# %%
step_rewards = []
substeps = 2
obs, _, _ = env.reset()
if args.use_render:
    env.render(mode='none')

def step_env():
    global obs
    actions = agent.act(obs, deterministic=False) if env.need_action else None
    obs, _, reward, done, _, _ = env.step(actions, substeps=substeps)
    step_rewards.append(reward[0])
    if args.use_render:
        env.render(mode='none')
    return obs, reward[0], done

T = args.num_env_steps
pbar = tqdm(range(T), file=sys.stdout)

if not args.use_dash:
    for i in pbar:
        step_env()
    if args.use_render:
        env.animate()
    print(pd.Series(np.squeeze(step_rewards)).describe())
    exit()
    
# %%
app = Dash(__name__)

figure = env._figure
figure['layout'].pop('sliders')
figure['layout'].pop('updatemenus')

app.layout = html.Div([
    html.H4('5G Network Simulation'),
    dcc.Graph(id="graph", figure=figure),
    html.P(id="step-info"),
    dcc.Slider(
        id='time-slider',
        min=0, max=T, step=1, value=0,
        marks={t: f'{t:.2f}' for t in np.linspace(0, T, num=6)},
    ),
])

@app.callback(
    Output("graph", "figure"),
    Output("step-info", "children"),
    Output("time-slider", "value"),
    Input("time-slider", "value"))
def update_plot(time):
    if time > env._sim_steps / substeps:
        step_env()
        pbar.update()
        time = env._sim_steps // substeps
    fig = env._figure
    fig['data'] = fig['frames'][time]['data']
    text = "Step: {}, Time: {}".format(env._sim_steps, env._steps_info[time]['time'])
    return fig, text, time

app.run_server(debug=True)
