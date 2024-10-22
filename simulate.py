#!/usr/bin/env python
# %%
import yaml
import torch
from utils import *
from agents import *
from arguments import *
from env import MultiCellNetEnv
from visualize.render import create_dash_app
# %reload_ext autoreload
# %autoreload 2

sim_days = 7
accelerate = 3000
render_interval = 4
model_params = dict(w_qos=4, no_interf=False, max_sleep=3, no_offload=False)

parser = get_config()
parser.add_argument("--perf_save_path", default="results/performance.csv",
                    help="path to save the performance of the simulation")
parser.add_argument("--render_interval", type=int, default=render_interval,
                    help="interval of rendering")
parser.add_argument("--days", type=int, default=sim_days,
                    help="number of days to simulate")
# parser.add_argument("--count-flops", action="store_true",
#                     help="count flops of the model")
parser.add_argument("--stochastic", action="store_true",
                    help="whether to use stochastic policy")

env_parser = get_env_config()
    
parser.set_defaults(log_level='NOTICE', group_name='RANDOM')
env_parser.set_defaults(accelerate=accelerate)

args, env_args = parser.parse_known_args()
env_args, rl_args = env_parser.parse_known_args(env_args)

AGENT = args.algorithm_name

if args.experiment_name == 'test': args.use_wandb = False
args.num_env_steps = args.days * 24 * 3600 * 50 // env_args.accelerate
env_args.stats_dir = f'analysis/sim_stats/{AGENT}'

# %%
set_log_level(args.log_level)

# seed
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

# %% [markdown]
# ## Simulation Parameters

# %%
def make_env(args, seed=None):
    kwds = vars(args).copy()
    kwds['no_interf'] = False  # always has interference at test time
    env = MultiCellNetEnv(seed=seed, **{
        k: v for k, v in kwds.items() if v is not None})
    pars = inspect.signature(MultiCellNetEnv.__init__).parameters
    [setattr(args, k, pars[k].default) for k, v in kwds.items() if v is None]
    return env

def get_model_dir(args, env_args, run_dir, version=''):
    assert run_dir.exists(), "Run directory does not exist: {}".format(run_dir)
    if args.model_dir is not None:
        return run_dir / args.model_dir
    p = 'wandb/run-*%s/files/' if args.use_wandb else '%s/models/'
    dirs = run_dir.glob(p % version)
    for d in sorted(dirs, key=os.path.getmtime, reverse=True):
        if env_args.no_interf ^ ('no_interf' in str(d)):
            continue
        config_path = d/'config.yaml'
        if not config_path.exists():
            warn("no config file in %s" % d)
            return d
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
            if all(getattr(env_args, k) == cfg[k]['value']
                   for k in model_params if k in cfg):
                return d
    raise FileNotFoundError("no such model directory")

env = make_env(env_args, seed=args.seed)

obs_space = env.observation_space[0]
cent_obs_space = env.cent_observation_space
action_space = env.action_space[0]

run_dir = get_run_dir(args, env_args)

if args.sim_log_path is None:
    # fn = '{}_{}_{}_acc-{}.log'.format(
    #     AGENT, env_args.scenario,
    #     re.sub('(, |:)', '-', env.net.world_time_repr),
    #     env.net.accelerate)
    fn = 'simulation.log'
    args.sim_log_path = 'logs/' + fn

if AGENT == 'mappo':
    from trainers.mappo_trainer import get_mappo_config
    parser = get_mappo_config()
    rl_args = parser.parse_args(rl_args)
    model_dir = args.model_dir or get_model_dir(args, env_args, run_dir, version=args.run_version)
    agent = MappoPolicy(rl_args, obs_space, cent_obs_space, action_space, model_dir=model_dir)
elif AGENT == 'dqn':
    model_dir = args.model_dir or get_model_dir(args, env_args, run_dir, version=args.run_version)
    agent = DQNPolicy(obs_space, action_space, model_dir=model_dir, model_version=args.model_version)
elif AGENT == 'fixed':
    agent = AlwaysOnPolicy(action_space, env.num_agents)
elif AGENT == 'random':
    agent = RandomPolicy(action_space, env.num_agents)
elif AGENT == 'simple':
    agent = SimplePolicy(action_space, env.num_agents)
elif AGENT == 'simple1':
    agent = SimplePolicySM1Only(action_space, env.num_agents)
elif AGENT == 'simple2':
    agent = SimplePolicyNoSM3(action_space, env.num_agents)
elif AGENT == 'sleepy':
    agent = SleepyPolicy(action_space, env.num_agents)
else:
    raise ValueError('invalid agent type')
print('Policy:', type(agent).__name__)

for par, defval in model_params.items():
    val = getattr(env_args, par)
    if val != defval:
        env.stats_dir += f'_{par}={val}'
if args.model_version:
    env.stats_dir += '_eps=%s' % args.model_version

print(env.full_stats_dir)
if os.path.exists(env.full_stats_dir):
    print('Simulation already done.')
    exit()

env.print_info()
env.net.traffic_model.print_info()

set_log_file(args.sim_log_path)

# %%
from datetime import datetime

render_mode = args.use_render and ('dash' if args.use_dash else 'frame')

# from hiddenlayer import build_graph
# build_graph(agent.actor, torch.tensor(obs))

# %%
def simulate():
    step_rewards = []
    # warm up
    # print('Warming up...')
    # for _ in range(warmup_steps):
    #     env.step()
    obs, _, _ = env.reset(render_mode)
    for i in trange(args.num_env_steps, file=sys.stdout):
        actions = agent.act(obs, deterministic=not args.stochastic)
        obs, _, rewards, done, _, _ = env.step(
            actions, render_mode=render_mode, render_interval=render_interval)
        step_rewards.append(np.mean(rewards))
    rewards = pd.Series(np.squeeze(step_rewards), name='reward')
    
    info = rewards.describe()
    info.index = ['reward_' + str(i) for i in info.index]
    info['time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info['agent'] = AGENT
    info['total_steps'] = args.num_env_steps
    info['accelerate'] = env_args.accelerate
    info['scenario'] = env.net.traffic_scenario
    info['traffic_density'] = env.net.traffic_model.density_mean
    # info['w_pc'] = env.w_pc
    info['w_qos'] = env.w_qos
    info['w_xqos'] = env.w_xqos
    print(info)
    
    save_path = args.perf_save_path
    info.to_frame().T.set_index('time').to_csv(
        save_path, mode='a', header=not os.path.exists(str(save_path)))
    if args.use_render and not args.use_dash:
        return env.animate()

simulate()
env.close()

# %%
if args.use_dash:
    # threading.Thread(target=simulate).start()
    app = create_dash_app(env, args)
    app.run_server(debug=True)
