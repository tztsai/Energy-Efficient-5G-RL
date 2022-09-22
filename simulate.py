#!/usr/bin/env python
# %%
import torch
from utils import *
from agents import *
from arguments import *
from env import MultiCellNetEnv
from visualize.render import create_dash_app
# %reload_ext autoreload
# %autoreload 2

sim_days = 7
accelerate = 6000
render_interval = 4

parser = get_config()
parser.add_argument("-A", '--agent', type=str, default='mappo',
                    help='type of agent used in simulation')
parser.add_argument("--perf_save_path", default="results/performance.csv",
                    help="path to save the performance of the simulation")
parser.add_argument("--render_interval", type=int, default=render_interval,
                    help="interval of rendering")
parser.add_argument("--days", type=int, default=sim_days,
                    help="number of days to simulate")

env_parser = get_env_config()

parser.set_defaults(log_level='NOTICE')
env_parser.set_defaults(accelerate=accelerate)

args, env_args = parser.parse_known_args()
env_args = env_parser.parse_args(env_args)

args.num_env_steps = args.days * 24 * 3600 * 50 // env_args.accelerate
env_args.steps_info_path = f'analysis/{env_args.scenario}-{args.agent}-steps.csv'

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
    return max(run_dir.glob(p), key=os.path.getmtime)

env = MultiCellNetEnv(**get_env_kwargs(env_args), seed=args.seed)
env.print_info()
env.net.traffic_model.print_info()

obs_space = env.observation_space[0]
cent_obs_space = env.cent_observation_space
action_space = env.action_space[0]

run_dir = get_run_dir(args, env_args)

if args.sim_log_path is None:
    fn = '{}_{}_{}_acc-{}.log'.format(
        args.agent, env_args.scenario,
        re.sub('(, |:)', '-', env.net.world_time_repr),
        env.net.accelerate)
    args.sim_log_path = 'logs/' + fn

set_log_file(args.sim_log_path)

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

obs, _, _ = env.reset(args.use_render)

def step_env(obs):
    actions = agent.act(obs, deterministic=False) if env.need_action else None
    obs, _, reward, done, _, _ = env.step(
        actions, render_mode=args.use_render, render_interval=render_interval)
    return obs, reward[0], done

def simulate(obs=obs):
    step_rewards = []
    for i in trange(args.num_env_steps, file=sys.stdout):
        obs, reward, done = step_env(obs)
        step_rewards.append(reward)
    rewards = pd.Series(np.squeeze(step_rewards), name='reward')
    info = rewards.describe()
    info.index = ['reward_' + str(i) for i in info.index]
    info['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info['agent'] = args.agent
    info['scenario'] = env_args.scenario
    info['total_steps'] = args.num_env_steps
    info['accelerate'] = env_args.accelerate
    info['traffic_density'] = env.net.traffic_model.density_mean
    info['w_pc'] = env.w_pc
    info['w_drop'] = env.w_drop
    info['w_delay'] = env.w_delay
    print(info)
    save_path = args.perf_save_path
    info.to_frame().T.set_index('time').to_csv(
        save_path, mode='a', header=not os.path.exists(str(save_path)))
    if args.use_render and not args.use_dash:
        return env.animate()

simulate()
env.close()

# %%
if not args.use_dash: exit()

# threading.Thread(target=simulate).start()
app = create_dash_app(env, args)
app.run_server(debug=True)
